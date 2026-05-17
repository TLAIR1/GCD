"""
graph_coop.py
=============
Unified Graph-CoOp trainer implementing all four graph-based modifications:

  Mod 1 — Graph Laplacian Regularisation   (loss term)
  Mod 2 — Graph Label Smoothing             (loss term)
  Mod 3 — GCN Class-Embedding Refinement   (architecture)
  Mod 4 — Graph-Guided Context Decomp.     (architecture, new)

PLACEMENT
---------
Save as  CoOp/trainers/graph_coop.py
Then register it in  CoOp/trainers/__init__.py  by adding:
    from .graph_coop import GraphCoOp          (one line, no other changes)

Run with:
    python train.py --config-file configs/trainers/CoOp/rn50_ep200.yaml \\
        TRAINER.NAME GraphCoOp \\
        TRAINER.GRAPHCOOP.USE_GCN True \\
        TRAINER.GRAPHCOOP.USE_GLS True \\
        TRAINER.GRAPHCOOP.USE_LAP True \\
        TRAINER.GRAPHCOOP.USE_GCD False \\
        TRAINER.GRAPHCOOP.ALPHA 0.1 \\
        TRAINER.GRAPHCOOP.LAMBDA_LAP 0.01 \\
        TRAINER.GRAPHCOOP.THRESHOLD 0.3

Dependencies (all in CoOp/trainers/):
    graph_utils.py   — graph construction helpers
    gcn_module.py    — ClassGCN
    gcd_module.py    — GCDContextLearner
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import os.path as osp

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .graph_utils import (
    build_class_graph,
    build_diffusion_matrix,
    build_gcn_adjacency,
    graph_label_smooth,
    laplacian_loss,
)
from .gcn_module import ClassGCN
from .gcd_module import GCDContextLearner

_tokenizer = _Tokenizer()


# ---------------------------------------------------------------------------
# Helpers (copied / adapted from original CoOp coop.py)
# ---------------------------------------------------------------------------

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model


# ---------------------------------------------------------------------------
# Standard CoOp PromptLearner (used when USE_GCD=False)
# ---------------------------------------------------------------------------

class PromptLearner(nn.Module):
    """
    Identical to the original CoOp PromptLearner.
    Kept here so graph_coop.py is self-contained.
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.GRAPHCOOP.N_CTX
        ctx_init = cfg.TRAINER.GRAPHCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, (
            f"cfg_imsize ({cfg_imsize}) must equal clip_imsize ({clip_imsize})"
        )

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        else:
            if cfg.TRAINER.GRAPHCOOP.CSC:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        prompt_prefix = " ".join(["X"] * n_ctx)
        self.ctx = nn.Parameter(ctx_vectors)
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = cfg.TRAINER.GRAPHCOOP.CSC

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])       # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.GRAPHCOOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half = self.n_ctx // 2
            prompts = torch.cat(
                [prefix, ctx[:, :half, :], suffix[:, :1, :],
                 ctx[:, half:, :], suffix[:, 1:, :]], dim=1
            )
        else:
            raise ValueError(f"Unknown class_token_position: {self.class_token_position}")

        return prompts


# ---------------------------------------------------------------------------
# GCDPromptLearner — like PromptLearner but uses GCDContextLearner
# ---------------------------------------------------------------------------

class GCDPromptLearner(nn.Module):
    """
    Wraps GCDContextLearner to produce the same (K, 77, d) token tensor
    that the CLIP text encoder expects.
    """

    def __init__(self, cfg, classnames, clip_model, phi: torch.Tensor):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.GRAPHCOOP.N_CTX
        n_proto = cfg.TRAINER.GRAPHCOOP.N_PROTO
        ctx_init = cfg.TRAINER.GRAPHCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.gcd = GCDContextLearner(
            n_ctx=n_ctx,
            ctx_dim=ctx_dim,
            n_cls=n_cls,
            n_proto=n_proto,
            phi=phi,
            ctx_init=ctx_init,
            token_embedding=clip_model.token_embedding,
        ).type(dtype)

        # Build prefix / suffix token buffers (same as PromptLearner)
        prompt_prefix = " ".join(["X"] * n_ctx)
        classnames_clean = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(_tokenizer.encode(n)) for n in classnames_clean]
        prompts = [prompt_prefix + " " + n + "." for n in classnames_clean]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.tokenized_prompts = tokenized_prompts
        self.n_cls = n_cls
        self.n_ctx = n_ctx

    def forward(self):
        # ctx: (K, M, d) from GCDContextLearner
        ctx = self.gcd()                      # (K, M, d)
        prompts = torch.cat(
            [self.token_prefix, ctx, self.token_suffix], dim=1
        )                                     # (K, 77, d)
        return prompts


# ---------------------------------------------------------------------------
# CustomCLIP — wraps PromptLearner (or GCDPromptLearner) + CLIP encoders
# ---------------------------------------------------------------------------

class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model, graph_data, phi):
        super().__init__()
        self.cfg = cfg
        self.use_gcn = cfg.TRAINER.GRAPHCOOP.USE_GCN
        self.use_gcd = cfg.TRAINER.GRAPHCOOP.USE_GCD

        # ---- prompt learner ----
        if self.use_gcd:
            self.prompt_learner = GCDPromptLearner(
                cfg, classnames, clip_model, phi
            )
        else:
            self.prompt_learner = PromptLearner(cfg, classnames, clip_model)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder_model = clip_model  # need full model for encode_text
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # ---- GCN (Modification 3) ----
        # ctx_dim is the transformer hidden dim (512 for RN50).
        # But text_features after text_projection has shape (K, output_dim)
        # where output_dim = text_projection.shape[1] — this differs by backbone:
        #   RN50: 1024,  RN101: 512,  ViT-B/32: 512,  ViT-B/16: 512
        # We must use output_dim for the GCN, not ctx_dim.
        output_dim = clip_model.text_projection.shape[1]
        if self.use_gcn:
            d_hidden = cfg.TRAINER.GRAPHCOOP.GCN_HIDDEN
            self.gcn = ClassGCN(d=output_dim, d_hidden=d_hidden).type(self.dtype)
            self.register_buffer(
                "A_hat",
                build_gcn_adjacency(graph_data["A"]).type(self.dtype)
            )

    def encode_text(self, prompts):
        """Run CLIP text encoder on (K, 77, d) prompt embeddings."""
        x = prompts + self.text_encoder_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)                      # (77, K, d)
        x = self.text_encoder_model.transformer(x)
        x = x.permute(1, 0, 2)                      # (K, 77, d)
        x = self.text_encoder_model.ln_final(x).type(self.dtype)
        # tokenized_prompts lives on CPU after __init__; move to match x
        tok = self.tokenized_prompts.to(x.device)
        x = x[
            torch.arange(x.shape[0], device=x.device),
            tok.argmax(dim=-1)
        ] @ self.text_encoder_model.text_projection
        return x                                     # (K, d)

    def forward(self, image):
        prompts = self.prompt_learner()              # (K, 77, d)

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = F.normalize(image_features, dim=-1)

        text_features = self.encode_text(prompts)   # (K, d)
        text_features = F.normalize(text_features, dim=-1)

        # ---- GCN refinement (Modification 3) ----
        if self.use_gcn:
            text_features = self.gcn(text_features, self.A_hat)
            # GCN already re-normalises inside ClassGCN.forward()

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T   # (B, K)

        return logits, text_features


# ---------------------------------------------------------------------------
# GraphCoOp Trainer
# ---------------------------------------------------------------------------

@TRAINER_REGISTRY.register()
class GraphCoOp(TrainerX):
    """
    Graph-CoOp: all four graph-based modifications to Context Optimization.

    Config keys (all under TRAINER.GRAPHCOOP):
    -----------------------------------------------------------------------
    N_CTX            int    16      number of context tokens
    CTX_INIT         str    ""      init words (e.g. "a photo of a")
    CSC              bool   False   class-specific context (std CoOp)
    CLASS_TOKEN_POSITION str "end"  'end' | 'middle'
    PREC             str    "fp16"  'fp16' | 'fp32' | 'amp'

    USE_GLS          bool   True    Modification 2: graph label smoothing
    USE_LAP          bool   True    Modification 1: Laplacian regularisation
    USE_GCN          bool   True    Modification 3: GCN refinement
    USE_GCD          bool   False   Modification 4: GCD context decomposition

    ALPHA            float  0.1     label smoothing strength
    LAMBDA_LAP       float  0.01    Laplacian loss weight
    THRESHOLD        float  0.3     edge threshold for graph construction
    GCN_HIDDEN       int    256     GCN hidden dimension
    N_PROTO          int    8       number of GCD prototypes (R)
    GAMMA_DIFF       float  1.0     GCD diffusion strength (gamma)
    -----------------------------------------------------------------------
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.GRAPHCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        prec = cfg.TRAINER.GRAPHCOOP.PREC
        if prec == "fp32" or prec == "amp":
            clip_model.float()

        # ----------------------------------------------------------------
        # Build class graph (once, frozen)
        # ----------------------------------------------------------------
        print("Building semantic class graph from CLIP text embeddings ...")
        # Temporarily move clip_model to device for the graph-building forward pass
        clip_model.to(self.device)
        with torch.no_grad():
            bare_prompts = [
                f"a photo of a {name.replace('_', ' ')}."
                for name in classnames
            ]
            tokenised = torch.cat(
                [clip.tokenize(p) for p in bare_prompts]
            ).to(self.device)
            bare_feats = clip_model.encode_text(tokenised)
            bare_feats = F.normalize(bare_feats.float(), dim=-1).cpu()
        # Move back to CPU; CustomCLIP will re-place it via .to(self.device) later
        clip_model.cpu()

        graph_data = build_class_graph(
            bare_feats,
            threshold=cfg.TRAINER.GRAPHCOOP.THRESHOLD,
        )
        # Move graph tensors to device
        for k in graph_data:
            graph_data[k] = graph_data[k].to(self.device)

        phi = None
        if cfg.TRAINER.GRAPHCOOP.USE_GCD:
            gamma = cfg.TRAINER.GRAPHCOOP.GAMMA_DIFF
            phi = build_diffusion_matrix(graph_data["L"], gamma=gamma)
            phi = phi.to(self.device)
            print(
                f"GCD: built diffusion matrix Phi (gamma={gamma}, "
                f"K={len(classnames)}, R={cfg.TRAINER.GRAPHCOOP.N_PROTO})"
            )

        # ----------------------------------------------------------------
        # Build CustomCLIP
        # ----------------------------------------------------------------
        print("Building custom CLIP with graph modifications ...")
        self.model = CustomCLIP(cfg, classnames, clip_model, graph_data, phi)

        # Store graph tensors as float32 on device.
        # In fp16 training we cast them to text_features.dtype inside
        # forward_backward so the trace computation stays numerically stable.
        self.graph_L       = graph_data["L"].float()
        self.graph_A_tilde = graph_data["A_tilde"].float()

        # ----------------------------------------------------------------
        # Freeze CLIP; only prompt learner (+ optional GCN) stays trainable
        # ----------------------------------------------------------------
        print("Turning off gradients in CLIP ...")
        for name, param in self.model.named_parameters():
            # Trainable: prompt_learner params and (optionally) gcn params
            if "prompt_learner" not in name and "gcn" not in name:
                param.requires_grad_(False)

        # logit_scale is a raw Parameter stored directly on CustomCLIP
        # (not inside prompt_learner or gcn), so the loop above misses it.
        # Freeze it explicitly — it belongs to the frozen CLIP backbone.
        if hasattr(self.model, "logit_scale"):
            self.model.logit_scale.requires_grad_(False)

        # ----------------------------------------------------------------
        # Optimiser, scheduler, scaler
        # ----------------------------------------------------------------
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.scaler = GradScaler() if prec == "amp" else None
        self.register_model("model", self.model, self.optim, self.sched)

        # Multi-GPU
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n={device_count}), using DataParallel")
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.prec = prec

        n_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        cfg = self.cfg

        use_gls    = cfg.TRAINER.GRAPHCOOP.USE_GLS
        use_lap    = cfg.TRAINER.GRAPHCOOP.USE_LAP
        alpha      = cfg.TRAINER.GRAPHCOOP.ALPHA
        lambda_lap = cfg.TRAINER.GRAPHCOOP.LAMBDA_LAP

        # ---------- fp16 path ----------
        if self.prec == "fp16":
            logits, text_features = self.model(image)

            if use_gls and alpha > 0:
                y_soft = graph_label_smooth(
                    label, self.graph_A_tilde,
                    alpha=alpha, num_classes=logits.shape[1],
                    dtype=text_features.dtype,
                )
                log_prob = F.log_softmax(logits, dim=-1)
                loss_cls = -(y_soft * log_prob).sum(dim=-1).mean()
            else:
                loss_cls = F.cross_entropy(logits, label)

            if use_lap and lambda_lap > 0:
                L = self.graph_L.to(dtype=text_features.dtype, device=self.device)
                loss_lap = laplacian_loss(text_features, L)
                loss = loss_cls + lambda_lap * loss_lap
            else:
                loss_lap = torch.tensor(0.0, device=self.device)
                loss = loss_cls

            self.model_zero_grad("model")
            loss.backward()
            self.model_update("model")

        # ---------- amp path ----------
        elif self.prec == "amp":
            with autocast():
                logits, text_features = self.model(image)

                if use_gls and alpha > 0:
                    y_soft = graph_label_smooth(
                        label, self.graph_A_tilde,
                        alpha=alpha, num_classes=logits.shape[1],
                        dtype=text_features.dtype,
                    )
                    log_prob = F.log_softmax(logits, dim=-1)
                    loss_cls = -(y_soft * log_prob).sum(dim=-1).mean()
                else:
                    loss_cls = F.cross_entropy(logits, label)

                if use_lap and lambda_lap > 0:
                    L = self.graph_L.to(dtype=text_features.dtype, device=self.device)
                    loss_lap = laplacian_loss(text_features, L)
                    loss = loss_cls + lambda_lap * loss_lap
                else:
                    loss_lap = torch.tensor(0.0, device=self.device)
                    loss = loss_cls

            self.model_zero_grad("model")
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        # ---------- fp32 path ----------
        else:
            logits, text_features = self.model(image)

            if use_gls and alpha > 0:
                y_soft = graph_label_smooth(
                    label, self.graph_A_tilde,
                    alpha=alpha, num_classes=logits.shape[1],
                    dtype=text_features.dtype,
                )
                log_prob = F.log_softmax(logits, dim=-1)
                loss_cls = -(y_soft * log_prob).sum(dim=-1).mean()
            else:
                loss_cls = F.cross_entropy(logits, label)

            if use_lap and lambda_lap > 0:
                L = self.graph_L.to(dtype=text_features.dtype, device=self.device)
                loss_lap = laplacian_loss(text_features, L)
                loss = loss_cls + lambda_lap * loss_lap
            else:
                loss_lap = torch.tensor(0.0, device=self.device)
                loss = loss_cls

            self.model_zero_grad("model")
            loss.backward()
            self.model_update("model")

        # ---------- logging ----------
        with torch.no_grad():
            acc = compute_accuracy(logits, label)[0].item()

        loss_summary = {
            "loss":     loss.item(),
            "loss_cls": loss_cls.item(),
            "loss_lap": loss_lap.item(),
            "acc":      acc,
        }

        # Write individual loss components to tensorboard.
        # Dassl versions differ: older uses self._writer, newer uses self.writer.
        _writer = getattr(self, "writer", None) or getattr(self, "_writer", None)
        if _writer is not None:
            n_iter = self.epoch * self.num_batches + self.batch_idx
            _writer.add_scalar("train/loss_cls",   loss_cls.item(), n_iter)
            _writer.add_scalar("train/loss_lap",   loss_lap.item(), n_iter)
            _writer.add_scalar("train/loss_total", loss.item(),     n_iter)
            _writer.add_scalar("train/acc",        acc,             n_iter)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference(self, image):
        """
        Inference-only forward: returns logits without computing any loss.
        Called by Dassl's TrainerX.test() via model_inference().
        """
        logits, _ = self.model(image)
        return logits

    # ------------------------------------------------------------------
    def model_inference(self, input):
        return self.inference(input)

    # ------------------------------------------------------------------
    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label

    # ------------------------------------------------------------------
    def after_epoch(self):
        """Log current LR to tensorboard at the end of each epoch."""
        last_epoch = (self.epoch + 1) == self.max_epoch
        if self.val_loader is not None and not last_epoch:
            self.test(split="val")

        _writer = getattr(self, "writer", None) or getattr(self, "_writer", None)
        if _writer is not None:
            _writer.add_scalar("train/lr", self.get_current_lr(), self.epoch)

    # ------------------------------------------------------------------
    def get_current_lr(self):
        return self.optim.param_groups[0]["lr"]

    # ------------------------------------------------------------------
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note: no pre-trained model provided; training from scratch")
            return

        names = self.get_model_names()
        model_file = (
            "model-best.pth.tar"
            if epoch is None
            else f"model.pth.tar-{epoch}"
        )

        for name in names:
            path = osp.join(directory, name, model_file)
            if not osp.exists(path):
                raise FileNotFoundError(f"No model found at {path}")
            checkpoint = load_checkpoint(path)
            state_dict = checkpoint["state_dict"]
            epoch_val  = checkpoint["epoch"]
            print(f"Loading weights from {path} (saved at epoch {epoch_val})")

            # strict=False lets us load a standard CoOp checkpoint and
            # only warm-start the prompt_learner, ignoring GCN/GCD keys.
            model = self._models[name]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys  : {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")
