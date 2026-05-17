"""
graph_utils.py
==============
Shared graph-construction utilities for all Graph-CoOp modifications.

PLACEMENT: Save as  CoOp/trainers/graph_utils.py
No changes to any existing file required for this module alone.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Class-graph construction
# ---------------------------------------------------------------------------

def build_class_graph(text_features: torch.Tensor, threshold: float = 0.3) -> dict:
    """
    Build a semantic class graph from CLIP frozen text embeddings.

    Args:
        text_features: (K, d) L2-normalised text embeddings of class names,
                       produced by CLIP's frozen text encoder BEFORE any
                       learned context is added.  In practice you call this
                       once with prompts like "a photo of a <classname>".
        threshold:     Edges are kept only when cosine similarity >= threshold.

    Returns a dict with keys:
        A      : (K, K) raw adjacency matrix (float32, on same device as input)
        A_tilde: (K, K) row-normalised adjacency (for label propagation / GCN)
        L      : (K, K) graph Laplacian  D - A
        D      : (K,)   degree vector
    """
    # text_features is already L2-normalised by CLIP
    # cosine similarity = dot product for unit vectors
    with torch.no_grad():
        sim = text_features @ text_features.T          # (K, K)
        sim.fill_diagonal_(0.0)                        # no self-loops

        A = sim.clone()
        A[A < threshold] = 0.0                        # threshold sparse

        # Row-normalised adjacency  A_tilde[i,j] = A[i,j] / (sum_k A[i,k] + eps)
        row_sum = A.sum(dim=1, keepdim=True).clamp(min=1e-6)
        A_tilde = A / row_sum

        # Laplacian
        D = A.sum(dim=1)                              # (K,)
        L = torch.diag(D) - A                         # (K, K)

    return {"A": A, "A_tilde": A_tilde, "L": L, "D": D}


# ---------------------------------------------------------------------------
# 2. Graph diffusion matrix  Phi = (I + gamma * L)^{-1}
# ---------------------------------------------------------------------------

def build_diffusion_matrix(L: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    Compute the closed-form graph diffusion matrix used by GCD.

    Phi = (I + gamma * L)^{-1}

    Row i of Phi is the diffused fingerprint of class i across the graph.
    Classes that are close in the graph receive similar rows, which is
    exactly the smoothness prior we want for prototype mixing.

    Args:
        L    : (K, K) graph Laplacian from build_class_graph()
        gamma: diffusion strength  (default 1.0; range [0.1, 5.0] typical)

    Returns:
        Phi  : (K, K) float32 tensor on the same device as L
    """
    K = L.shape[0]
    I = torch.eye(K, dtype=L.dtype, device=L.device)
    M = I + gamma * L                                 # (K, K)
    Phi = torch.linalg.inv(M)                         # (K, K)
    return Phi


# ---------------------------------------------------------------------------
# 3. GCN normalised adjacency  A_hat = D^{-1/2} (A + I) D^{-1/2}
# ---------------------------------------------------------------------------

def build_gcn_adjacency(A: torch.Tensor) -> torch.Tensor:
    """
    Symmetrically normalised adjacency with self-loops, as in Kipf & Welling.

    A_hat = D_tilde^{-1/2} * (A + I) * D_tilde^{-1/2}

    Args:
        A: (K, K) raw adjacency (no self-loops, non-negative)

    Returns:
        A_hat: (K, K) float32, same device
    """
    K = A.shape[0]
    A_tilde = A + torch.eye(K, dtype=A.dtype, device=A.device)   # add self-loops
    D_tilde = A_tilde.sum(dim=1)                                   # (K,)
    D_inv_sqrt = torch.diag(D_tilde.pow(-0.5))
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return A_hat


# ---------------------------------------------------------------------------
# 4. Graph label smoothing
# ---------------------------------------------------------------------------

def graph_label_smooth(
    labels: torch.Tensor,
    A_tilde: torch.Tensor,
    alpha: float = 0.1,
    num_classes: int = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    One step of label propagation to soften one-hot targets.

    y_soft = (1 - alpha_eff) * one_hot  +  alpha_eff * A_tilde^T * one_hot

    For isolated classes (no neighbours) alpha_eff is set to 0 automatically
    so that y_soft always sums to 1.

    Args:
        labels     : (B,) long tensor of ground-truth class indices
        A_tilde    : (K, K) row-normalised adjacency from build_class_graph()
        alpha      : smoothing strength in [0, 1]
        num_classes: K  (inferred from A_tilde if None)
        dtype      : output dtype — pass text_features.dtype so fp16 training
                     never triggers a silent upcast or a device-dtype mismatch

    Returns:
        y_soft: (B, K) soft target distribution, sums to 1 per row
    """
    K = A_tilde.shape[0] if num_classes is None else num_classes
    B = labels.shape[0]
    device = labels.device

    # Always work in the caller's dtype to avoid silent upcasts
    A = A_tilde.to(device=device, dtype=dtype)

    one_hot = torch.zeros(B, K, dtype=dtype, device=device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)              # (B, K)

    # neighbour[i] = A_tilde[labels[i], :] — the outgoing neighbours of class i.
    # A_tilde is row-normalised (rows sum to 1), NOT column-normalised.
    # Correct: one_hot @ A  →  picks row labels[i] of A_tilde  → sums to 1
    # Wrong:   one_hot @ A.T → picks col labels[i] of A_tilde  → does NOT sum to 1
    neighbour = one_hot @ A                                     # (B, K)

    # Isolated classes: zero row-sum in A_tilde → set alpha_eff = 0
    row_has_neighbours = (A.sum(dim=1) > 0).to(dtype)          # (K,)
    alpha_eff = alpha * row_has_neighbours[labels]              # (B,)
    alpha_eff = alpha_eff.unsqueeze(1)                          # (B, 1)

    y_soft = (1.0 - alpha_eff) * one_hot + alpha_eff * neighbour
    return y_soft                                               # (B, K)


# ---------------------------------------------------------------------------
# 5. Laplacian regularisation loss
# ---------------------------------------------------------------------------

def laplacian_loss(W: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    L_lap = Tr(W^T L W) / (K * d)

    Penalises semantically similar classes for having different weight vectors.
    Gradients flow back through W to the context vectors.

    Args:
        W : (K, d) class weight matrix (text encoder outputs, post-normalisation)
        L : (K, K) graph Laplacian

    Returns:
        scalar loss (already normalised by K*d)
    """
    # Tr(W^T L W) = sum_{i,j} L_{ij} * (w_i . w_j)
    K, d = W.shape
    loss = torch.trace(W.T @ L @ W) / (K * d)
    return loss
