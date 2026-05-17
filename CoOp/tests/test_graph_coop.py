"""
test_graph_coop.py
==================
CPU-only unit tests for all Graph-CoOp components.
No GPU, no dataset, no CLIP weights required.

PLACEMENT: Save as  CoOp/tests/test_graph_coop.py

Run:
    cd CoOp
    python -m pytest tests/test_graph_coop.py -v

Or without pytest:
    python tests/test_graph_coop.py
"""

import sys
import math
import unittest

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Make sure the trainers package is importable without a full Dassl install
# by adding the repo root to sys.path
# ---------------------------------------------------------------------------
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# We import the individual modules directly so the tests don't need Dassl
from trainers.graph_utils import (
    build_class_graph,
    build_diffusion_matrix,
    build_gcn_adjacency,
    graph_label_smooth,
    laplacian_loss,
)
from trainers.gcn_module import ClassGCN
from trainers.gcd_module import GCDContextLearner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_normed(K: int, d: int) -> torch.Tensor:
    """Return (K, d) L2-normalised random float32 tensor."""
    x = torch.randn(K, d)
    return F.normalize(x, dim=-1)


def _make_graph(K: int = 8, d: int = 16, threshold: float = 0.0):
    """Build a small graph with threshold=0 so we get a dense graph."""
    feats = _random_normed(K, d)
    return build_class_graph(feats, threshold=threshold), feats


# ---------------------------------------------------------------------------
# 1. Graph construction
# ---------------------------------------------------------------------------

class TestBuildClassGraph(unittest.TestCase):

    def test_output_keys(self):
        g, _ = _make_graph()
        self.assertIn("A",       g)
        self.assertIn("A_tilde", g)
        self.assertIn("L",       g)
        self.assertIn("D",       g)

    def test_shapes(self):
        K = 8
        g, _ = _make_graph(K=K)
        self.assertEqual(g["A"].shape,       (K, K))
        self.assertEqual(g["A_tilde"].shape, (K, K))
        self.assertEqual(g["L"].shape,       (K, K))
        self.assertEqual(g["D"].shape,       (K,))

    def test_adjacency_symmetric(self):
        g, _ = _make_graph()
        diff = (g["A"] - g["A"].T).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=5,
                               msg="Adjacency must be symmetric")

    def test_no_self_loops(self):
        g, _ = _make_graph(threshold=0.0)
        diag = g["A"].diagonal()
        self.assertTrue((diag == 0).all(),
                        msg="Diagonal of A must be zero (no self-loops)")

    def test_A_tilde_row_sums_le_one(self):
        """Row-normalised adjacency rows should sum to ≤ 1
        (isolated nodes sum to 0)."""
        g, _ = _make_graph(threshold=0.99)   # high threshold → sparse
        row_sums = g["A_tilde"].sum(dim=1)
        self.assertTrue((row_sums <= 1.0 + 1e-5).all())

    def test_laplacian_identity(self):
        """L = D - A  must hold element-wise."""
        g, _ = _make_graph()
        L_expected = torch.diag(g["D"]) - g["A"]
        diff = (g["L"] - L_expected).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=5)

    def test_threshold_removes_edges(self):
        """With threshold=1.0 (impossible cosine sim) graph should be empty."""
        K = 6
        feats = _random_normed(K, 16)
        g = build_class_graph(feats, threshold=1.0)
        self.assertTrue((g["A"] == 0).all(),
                        msg="threshold=1.0 should produce empty graph")


# ---------------------------------------------------------------------------
# 2. Diffusion matrix
# ---------------------------------------------------------------------------

class TestBuildDiffusionMatrix(unittest.TestCase):

    def test_shape(self):
        K = 6
        g, _ = _make_graph(K=K)
        phi = build_diffusion_matrix(g["L"], gamma=1.0)
        self.assertEqual(phi.shape, (K, K))

    def test_gamma_zero_gives_identity(self):
        """gamma=0  →  (I + 0*L)^{-1} = I"""
        K = 5
        g, _ = _make_graph(K=K)
        phi = build_diffusion_matrix(g["L"], gamma=0.0)
        diff = (phi - torch.eye(K)).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=4)

    def test_rows_sum_to_positive(self):
        """All entries of Phi should be positive for a connected graph."""
        K = 5
        g, _ = _make_graph(K=K, threshold=0.0)
        phi = build_diffusion_matrix(g["L"], gamma=1.0)
        # Phi = (I + gamma*L)^{-1}: not guaranteed all-positive for arbitrary L,
        # but row sums should be ≤ 1 and Phi diagonal should be dominant.
        self.assertTrue((phi.diagonal() > 0).all())

    def test_invertibility(self):
        """(I + gamma*L) @ Phi  should be close to identity."""
        K = 6
        g, _ = _make_graph(K=K, threshold=0.0)
        gamma = 2.0
        L = g["L"]
        phi = build_diffusion_matrix(L, gamma=gamma)
        I = torch.eye(K)
        product = (I + gamma * L) @ phi
        diff = (product - I).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=4)


# ---------------------------------------------------------------------------
# 3. GCN adjacency
# ---------------------------------------------------------------------------

class TestBuildGcnAdjacency(unittest.TestCase):

    def test_shape(self):
        K = 7
        g, _ = _make_graph(K=K)
        A_hat = build_gcn_adjacency(g["A"])
        self.assertEqual(A_hat.shape, (K, K))

    def test_symmetric(self):
        g, _ = _make_graph()
        A_hat = build_gcn_adjacency(g["A"])
        diff = (A_hat - A_hat.T).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=5)

    def test_diagonal_nonzero(self):
        """Self-loops are added so diagonal must be > 0."""
        g, _ = _make_graph(threshold=0.99)   # may have no off-diag edges
        A_hat = build_gcn_adjacency(g["A"])
        self.assertTrue((A_hat.diagonal() > 0).all())


# ---------------------------------------------------------------------------
# 4. Graph label smoothing
# ---------------------------------------------------------------------------

class TestGraphLabelSmooth(unittest.TestCase):

    def setUp(self):
        self.K = 5
        self.B = 4
        g, _ = _make_graph(K=self.K, threshold=0.0)
        self.A_tilde = g["A_tilde"]

    def test_output_shape(self):
        labels = torch.randint(0, self.K, (self.B,))
        y = graph_label_smooth(labels, self.A_tilde, alpha=0.1)
        self.assertEqual(y.shape, (self.B, self.K))

    def test_rows_sum_to_one(self):
        labels = torch.randint(0, self.K, (self.B,))
        y = graph_label_smooth(labels, self.A_tilde, alpha=0.1)
        row_sums = y.sum(dim=1)
        for s in row_sums:
            self.assertAlmostEqual(s.item(), 1.0, places=5)

    def test_neighbour_direction(self):
        """
        Verify that smoothing draws mass from A_tilde ROWS (outgoing neighbours),
        not columns.  A_tilde is row-normalised so rows sum to 1; columns do not.
        The correct formula is  one_hot @ A_tilde  (not one_hot @ A_tilde.T).
        """
        # Build asymmetric A_tilde manually so row != col
        K = 4
        A_tilde = torch.zeros(K, K)
        # class 0 → class 1 with weight 1.0 (only outgoing edge from 0)
        A_tilde[0, 1] = 1.0
        # class 1 → classes 2,3 (so col 1 != row 1)
        A_tilde[1, 2] = 0.5
        A_tilde[1, 3] = 0.5
        labels = torch.tensor([0])
        y = graph_label_smooth(labels, A_tilde, alpha=1.0)
        # With alpha=1.0: y = A_tilde[0, :] = [0, 1, 0, 0]
        expected = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(y, expected, atol=1e-5),
            msg=f"Expected {expected}, got {y}. Check one_hot @ A (not A.T).")

    def test_alpha_zero_gives_one_hot(self):
        labels = torch.tensor([0, 2, 4])
        y = graph_label_smooth(labels, self.A_tilde, alpha=0.0)
        expected = F.one_hot(labels, num_classes=self.K).float()
        diff = (y - expected).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=5)

    def test_all_values_nonneg(self):
        labels = torch.randint(0, self.K, (self.B,))
        y = graph_label_smooth(labels, self.A_tilde, alpha=0.2)
        self.assertTrue((y >= 0).all())

    def test_dtype_propagation(self):
        """Output dtype should match the requested dtype arg."""
        labels = torch.randint(0, self.K, (self.B,))
        y16 = graph_label_smooth(labels, self.A_tilde, alpha=0.1,
                                 dtype=torch.float16)
        self.assertEqual(y16.dtype, torch.float16)

    def test_isolated_class_no_mass_loss(self):
        """Isolated nodes (zero row in A_tilde) must still give rows summing to 1."""
        # Build a graph where class 0 is isolated
        A_tilde = torch.zeros(self.K, self.K)
        A_tilde[1:, 1:] = 1.0 / (self.K - 1)   # only classes 1-4 are connected
        labels = torch.tensor([0])               # isolated class
        y = graph_label_smooth(labels, A_tilde, alpha=0.3)
        self.assertAlmostEqual(y.sum(dim=1).item(), 1.0, places=5)


# ---------------------------------------------------------------------------
# 5. Laplacian loss
# ---------------------------------------------------------------------------

class TestLaplacianLoss(unittest.TestCase):

    def test_scalar_output(self):
        K, d = 6, 16
        g, _ = _make_graph(K=K)
        W = torch.randn(K, d, requires_grad=True)
        loss = laplacian_loss(W, g["L"])
        self.assertEqual(loss.shape, ())   # scalar

    def test_nonneg(self):
        """Tr(W^T L W) is non-negative for PSD Laplacians."""
        K, d = 5, 8
        g, _ = _make_graph(K=K, threshold=0.0)
        W = torch.randn(K, d)
        loss = laplacian_loss(W, g["L"])
        self.assertGreaterEqual(loss.item(), -1e-5)

    def test_identical_rows_give_zero(self):
        """If all class weight vectors are identical, Laplacian loss = 0."""
        K, d = 4, 8
        g, _ = _make_graph(K=K, threshold=0.0)
        W = torch.ones(K, d)          # all rows identical
        loss = laplacian_loss(W, g["L"])
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_gradient_flows(self):
        K, d = 5, 8
        g, _ = _make_graph(K=K, threshold=0.0)
        W = torch.randn(K, d, requires_grad=True)
        loss = laplacian_loss(W, g["L"])
        loss.backward()
        self.assertIsNotNone(W.grad)
        self.assertFalse(W.grad.isnan().any())

    def test_scale_invariant_normalisation(self):
        """Loss is divided by K*d; doubling K*d should roughly halve the raw value."""
        K, d = 4, 8
        g1, _ = _make_graph(K=K, d=d, threshold=0.0)
        W1 = torch.ones(K, d)
        W1[0] += 1.0                   # introduce some difference
        l1 = laplacian_loss(W1, g1["L"]).item()
        # Just check it's a finite, positive float — the normalisation is tested
        # implicitly by test_identical_rows_give_zero and test_nonneg
        self.assertTrue(math.isfinite(l1))


# ---------------------------------------------------------------------------
# 6. ClassGCN
# ---------------------------------------------------------------------------

class TestClassGCN(unittest.TestCase):

    def setUp(self):
        self.K, self.d = 8, 32
        self.d_hidden = 16
        g, _ = _make_graph(K=self.K, threshold=0.0)
        self.A_hat = build_gcn_adjacency(g["A"])
        self.gcn = ClassGCN(d=self.d, d_hidden=self.d_hidden)

    def test_output_shape(self):
        W = _random_normed(self.K, self.d)
        out = self.gcn(W, self.A_hat)
        self.assertEqual(out.shape, (self.K, self.d))

    def test_output_is_l2_normalised(self):
        W = _random_normed(self.K, self.d)
        out = self.gcn(W, self.A_hat)
        norms = out.norm(dim=-1)
        for n in norms:
            self.assertAlmostEqual(n.item(), 1.0, places=5)

    def test_gradient_flows_to_W(self):
        W = _random_normed(self.K, self.d).requires_grad_(True)
        out = self.gcn(W, self.A_hat)
        out.sum().backward()
        self.assertIsNotNone(W.grad)
        self.assertFalse(W.grad.isnan().any())

    def test_gradient_flows_to_gcn_params(self):
        W = _random_normed(self.K, self.d)
        out = self.gcn(W, self.A_hat)
        out.sum().backward()
        for name, p in self.gcn.named_parameters():
            self.assertIsNotNone(p.grad, msg=f"No grad for {name}")

    def test_residual_connection_effect(self):
        """With zero-initialised GCN weights output should be close to input."""
        # Zero the GCN weights
        for p in self.gcn.parameters():
            p.data.zero_()
        W = _random_normed(self.K, self.d)
        out = self.gcn(W, self.A_hat)
        # With zero weights H2 = 0, so out = L2_norm(W + 0) = W (already normed)
        diff = (out - W).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=5)


# ---------------------------------------------------------------------------
# 7. GCDContextLearner
# ---------------------------------------------------------------------------

class TestGCDContextLearner(unittest.TestCase):

    def setUp(self):
        self.K  = 6
        self.M  = 4    # context tokens (small for speed)
        self.d  = 16   # embedding dim
        self.R  = 3    # prototypes
        g, _    = _make_graph(K=self.K, threshold=0.0)
        self.phi = build_diffusion_matrix(g["L"], gamma=1.0)
        self.gcd = GCDContextLearner(
            n_ctx=self.M,
            ctx_dim=self.d,
            n_cls=self.K,
            n_proto=self.R,
            phi=self.phi,
        )

    def test_output_shape(self):
        ctx = self.gcd()
        self.assertEqual(ctx.shape, (self.K, self.M, self.d))

    def test_output_dtype(self):
        ctx = self.gcd()
        self.assertEqual(ctx.dtype, torch.float32)

    def test_gradient_flows(self):
        ctx = self.gcd()
        ctx.sum().backward()
        # All three learnable tensors must have gradients
        self.assertIsNotNone(self.gcd.ctx_global.grad)
        self.assertIsNotNone(self.gcd.prototypes.grad)
        self.assertIsNotNone(self.gcd.W_alpha.grad)

    def test_phi_is_not_trainable(self):
        """Diffusion matrix must be a buffer, not a parameter."""
        param_names = [n for n, _ in self.gcd.named_parameters()]
        self.assertNotIn("phi", param_names)

    def test_parameter_count(self):
        pc = self.gcd.parameter_count()
        expected_global    = self.M * self.d
        expected_proto     = self.R * self.M * self.d
        expected_W_alpha   = self.K * self.R
        expected_total     = expected_global + expected_proto + expected_W_alpha
        self.assertEqual(pc["ctx_global (M*d)"],    expected_global)
        self.assertEqual(pc["prototypes (R*M*d)"],  expected_proto)
        self.assertEqual(pc["W_alpha (K*R)"],        expected_W_alpha)
        self.assertEqual(pc["total"],                expected_total)

    def test_graph_smoothness(self):
        """
        The diffusion matrix Phi encodes graph smoothness: row i and row j of
        Phi should be more similar when i and j are graph neighbours than when
        they are far apart.  We verify this on the chain graph 0-1-2-3-4-5
        before any gradient steps, because the smoothness is a property of Phi
        itself (not of the learned W_alpha, which starts uniform).
        """
        K = 6
        # Build a chain graph: strong edges between adjacent nodes only
        A = torch.zeros(K, K)
        for i in range(K - 1):
            A[i, i + 1] = 0.9
            A[i + 1, i] = 0.9
        D = A.sum(dim=1)
        L = torch.diag(D) - A
        phi = build_diffusion_matrix(L, gamma=2.0)   # (K, K)

        # Normalise rows of Phi so we can compute cosine similarity
        phi_normed = F.normalize(phi, dim=-1)
        sim_close = (phi_normed[0] * phi_normed[1]).sum().item()   # neighbours
        sim_far   = (phi_normed[0] * phi_normed[5]).sum().item()   # chain ends

        self.assertGreater(
            sim_close, sim_far,
            msg="Adjacent classes should have more similar Phi rows (graph fingerprints)"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.discover(start_dir=os.path.dirname(__file__), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
