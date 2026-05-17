"""
gcd_module.py
=============
Graph-Guided Context Decomposition (GCD) — Modification 4.

PLACEMENT: Save as  CoOp/trainers/gcd_module.py
No changes to any existing file required for this module alone.
It is imported and instantiated inside graph_coop.py.

Key idea
--------
Instead of a single shared context (CoOp) or one context per class (CSC),
GCD learns R prototype context vectors and, for each class, computes a
graph-smooth blend of them.  The blend coefficients are derived from the
graph diffusion matrix Phi = (I + gamma*L)^{-1}, so semantically close
classes automatically receive similar context corrections.

Parameterisation
----------------
  ctx_global   : (M, d)      — shared base context (as in standard CoOp)
  prototypes   : (R, M, d)   — R learnable context corrections
  W_alpha      : (K, R)      — projects graph fingerprint → mixing weights

Forward per class i:
  c_i   = softmax( Phi[i] @ W_alpha )      (R,)   mixing coefficients
  V_i   = ctx_global + sum_r c_i[r] * P_r  (M, d) class-specific context
  t_i   = [V_i ; CLASS_i]                          prompt to text encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# clip is installed as the openai/CLIP package; import at module level
try:
    import clip as openai_clip
except ImportError:
    openai_clip = None


class GCDContextLearner(nn.Module):
    """
    Replaces CoOp's PromptLearner for the GCD variant.

    Args:
        n_ctx      : number of context tokens M (default 16)
        ctx_dim    : CLIP word-embedding dimension d (512 for RN50)
        n_cls      : number of classes K
        n_proto    : number of prototype contexts R (default 8)
        ctx_init   : optional string for initialising ctx_global
        phi        : (K, K) pre-computed diffusion matrix (frozen)
        token_embedding: CLIP's token embedding layer (for ctx_init)
    """

    def __init__(
        self,
        n_ctx: int,
        ctx_dim: int,
        n_cls: int,
        n_proto: int,
        phi: torch.Tensor,
        ctx_init: str = "",
        token_embedding: nn.Embedding = None,
    ):
        super().__init__()

        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.n_cls = n_cls
        self.n_proto = n_proto

        # ---- register Phi as a non-trainable buffer ----
        self.register_buffer("phi", phi.float())     # (K, K)

        # ---- shared global context (as in CoOp unified) ----
        if ctx_init and token_embedding is not None:
            # initialise from word embeddings, same as CoOp
            ctx_init_words = ctx_init.replace("_", " ")
            if openai_clip is None:
                raise ImportError(
                    "openai/CLIP not found. Install with: "
                    "pip install git+https://github.com/openai/CLIP.git"
                )
            tokenised = openai_clip.tokenize(ctx_init_words)
            with torch.no_grad():
                init_emb = token_embedding(tokenised)   # (1, 77, d)
            n_init = min(len(ctx_init_words.split()), n_ctx)
            ctx_vectors = init_emb[0, 1: 1 + n_init]   # (n_init, d)
            # pad to n_ctx if needed — match dtype of ctx_vectors (may be fp16)
            if n_init < n_ctx:
                pad = torch.zeros(n_ctx - n_init, ctx_dim, dtype=ctx_vectors.dtype)
                ctx_vectors = torch.cat([ctx_vectors, pad], dim=0)
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx_global = nn.Parameter(ctx_vectors)    # (M, d)

        # ---- R prototype context corrections ----
        # shape (R, M, d); small init so they start as near-zero corrections
        proto_vectors = torch.zeros(n_proto, n_ctx, ctx_dim)
        nn.init.normal_(proto_vectors, std=0.01)
        self.prototypes = nn.Parameter(proto_vectors)  # (R, M, d)

        # ---- class-to-prototype mixing weights ----
        # (K, R); initialised to uniform so softmax starts at 1/R for all classes
        W_alpha_init = torch.zeros(n_cls, n_proto)
        self.W_alpha = nn.Parameter(W_alpha_init)      # (K, R)

    # ------------------------------------------------------------------
    def forward(self) -> torch.Tensor:
        """
        Compute all K class-specific context tensors.

        Returns:
            contexts : (K, M, d)  one context sequence per class
        """
        # mixing coefficients for each class
        # (K, K) @ (K, R) -> (K, R)  then softmax over prototype dimension
        logits = self.phi @ self.W_alpha               # (K, R)
        c = F.softmax(logits, dim=-1)                  # (K, R)

        # weighted sum of prototypes: (K, R) x (R, M, d) -> (K, M, d)
        # einsum: k r, r m d -> k m d
        correction = torch.einsum("kr,rmd->kmd", c, self.prototypes)

        # add global base context (broadcast over K)
        contexts = self.ctx_global.unsqueeze(0) + correction   # (K, M, d)

        return contexts                                # (K, M, d)

    # ------------------------------------------------------------------
    def parameter_count(self) -> dict:
        """Utility: break down learnable parameter counts."""
        return {
            "ctx_global (M*d)": self.ctx_global.numel(),
            "prototypes (R*M*d)": self.prototypes.numel(),
            "W_alpha (K*R)": self.W_alpha.numel(),
            "total": (
                self.ctx_global.numel()
                + self.prototypes.numel()
                + self.W_alpha.numel()
            ),
        }
