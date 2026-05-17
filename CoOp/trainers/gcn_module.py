"""
gcn_module.py
=============
Lightweight two-layer GCN for class-embedding refinement (Modification 3).

PLACEMENT: Save as  CoOp/trainers/gcn_module.py
No changes to any existing file required for this module alone.
It is imported by graph_coop.py (the unified trainer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassGCN(nn.Module):
    """
    Two-layer GCN that refines the (K, d) matrix of CLIP class weight vectors
    by aggregating information from semantically related classes.

    Architecture
    ------------
    H^(1) = ReLU( A_hat @ W^(0) @ Theta_1 )         shape: (K, d_hidden)
    H^(2) = ReLU( A_hat @ H^(1) @ Theta_2 )          shape: (K, d)
    W'    = L2_normalise( W^(0) + H^(2) )             shape: (K, d)

    The residual connection ensures the GCN only needs to learn *corrections*
    to the existing class embeddings, stabilising early training.

    Parameters learned: Theta_1 (d x d_hidden) and Theta_2 (d_hidden x d).
    A_hat (the normalised adjacency) is fixed throughout training.

    Args:
        d       : CLIP embedding dimension (512 for RN50 / ViT-B)
        d_hidden: hidden dimension of the GCN (default 256)
    """

    def __init__(self, d: int = 512, d_hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(d, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d, bias=False)

        # Xavier initialisation keeps activations in a healthy range
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, W: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            W    : (K, d)  class weight vectors from the text encoder
            A_hat: (K, K)  symmetrically-normalised adjacency with self-loops
                           (produced by graph_utils.build_gcn_adjacency)

        Returns:
            W_prime: (K, d)  refined and L2-normalised class weight vectors
        """
        # Layer 1
        H1 = F.relu(A_hat @ self.fc1(W))        # (K, d_hidden)

        # Layer 2
        H2 = F.relu(A_hat @ self.fc2(H1))       # (K, d)

        # Residual + L2 normalise
        W_prime = F.normalize(W + H2, dim=-1)    # (K, d)

        return W_prime
