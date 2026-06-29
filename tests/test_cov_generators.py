"""
Tests that every covariance generator produces PSD matrices with non-trivial
off-diagonal entries (non-null covariance across output dimensions), and that
generate_episode produces Y with non-trivial cross-dimensional correlations.

Run from project root:
    conda run -n multivariate-icl pytest tests/test_cov_generators.py -v -s

Generators under test:
  Copula (unit-diagonal correlation matrices):
    LinearProjKernel, IsotropicModulatedKernel, SimpleAggKernel, TabICLFeatureKernel
  Full covariance:
    KernelCovGen
  Episode-level (tested through Y):
    mlp, fixed_cov, anchor, hyperplane_multimodal modes of generate_episode
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)

from data_gen import (
    GlobalAnchorCovGen,
    GlobalFixedNets,
    IsotropicModulatedKernel,
    KernelCovGen,
    LinearProjKernel,
    SimpleAggKernel,
    TabICLFeatureKernel,
    generate_episode,
    sample_tabular_x,
)

DEVICE = "cpu"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _recover_cov(L: torch.Tensor) -> torch.Tensor:
    """C = L @ L^T from lower-triangular Cholesky factor L ∈ (B, T, d, d)."""
    return torch.matmul(L, L.transpose(-1, -2))


def _off_diag_mask(d: int) -> torch.Tensor:
    return ~torch.eye(d, dtype=torch.bool)


def _off_diag_abs_mean(C: torch.Tensor) -> float:
    """Mean absolute value of all off-diagonal entries across (B, T, m≠n)."""
    d = C.shape[-1]
    mask = _off_diag_mask(d)
    return C[..., mask].abs().mean().item()


def _off_diag_std_across_t(C: torch.Tensor) -> float:
    """Mean std of each off-diagonal entry across the T axis — measures x-dependence."""
    d = C.shape[-1]
    mask = _off_diag_mask(d)
    off = C[:, :, mask]  # (B, T, d*(d-1))
    return off.std(dim=1).mean().item()


def _check_cov_quality(
    C: torch.Tensor,
    name: str,
    is_copula: bool,
    min_off_diag: float = 0.05,
    min_off_diag_std: float = 0.01,
    check_x_variation: bool = True,
) -> None:
    """Common quality checks on a batch (B, T, d, d) of symmetric PD matrices."""
    B, T, d, _ = C.shape

    # Finite
    assert C.isfinite().all(), f"{name}: C contains non-finite values"

    # Symmetric (up to floating point)
    sym_err = (C - C.transpose(-1, -2)).abs().max().item()
    assert sym_err < 1e-5, f"{name}: C is not symmetric (max asymmetry={sym_err:.2e})"

    # PSD: all eigenvalues strictly positive
    eigs = torch.linalg.eigvalsh(C.reshape(B * T, d, d))
    min_eig = eigs.min().item()
    print(f"\n  [{name}] min eigenvalue = {min_eig:.2e}")
    assert min_eig > 0, f"{name}: C has non-positive eigenvalue {min_eig:.2e}"

    # Copula: diagonal must be exactly 1 (correlation matrix)
    if is_copula:
        diag = C.diagonal(dim1=-2, dim2=-1)  # (B, T, d)
        diag_dev = (diag - 1.0).abs().max().item()
        print(f"  [{name}] max |diag - 1| = {diag_dev:.2e}")
        assert diag_dev < 1e-3, (
            f"{name}: copula generator diagonal deviates from 1 (max={diag_dev:.2e})"
        )

    # Non-trivial off-diagonal (non-null cross-covariance)
    off_mean = _off_diag_abs_mean(C)
    print(f"  [{name}] mean |off-diagonal| = {off_mean:.4f}  (threshold > {min_off_diag})")
    assert off_mean > min_off_diag, (
        f"{name}: near-identity covariance — mean |off-diagonal|={off_mean:.4f}"
        f" ≤ {min_off_diag}. Dimensions appear nearly uncorrelated."
    )

    # Off-diagonal entries vary across instances → generator is x-dependent
    if check_x_variation:
        off_std = _off_diag_std_across_t(C)
        print(f"  [{name}] std of off-diagonal across T = {off_std:.4f}  (threshold > {min_off_diag_std})")
        assert off_std > min_off_diag_std, (
            f"{name}: correlation structure is constant across instances"
            f" (std across T = {off_std:.4f} ≤ {min_off_diag_std}). No x-dependence."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Shared test fixture parameters
# ──────────────────────────────────────────────────────────────────────────────

B, T, p, d = 4, 64, 10, 4


@pytest.fixture(scope="module")
def X_sample():
    torch.manual_seed(0)
    return sample_tabular_x(B, T, p, DEVICE, n_train=T // 2)


# ──────────────────────────────────────────────────────────────────────────────
# 1. LinearProjKernel (the primary target of this investigation)
# ──────────────────────────────────────────────────────────────────────────────


class TestLinearProjKernel:
    """LinearProjKernel produces non-trivial x-dependent correlation matrices.

    Expected: mean |off-diagonal| ≈ E[|cosine between two random q-dim unit
    vectors|] ≈ sqrt(2/(π·q)) ≈ 0.4 for q=4. Threshold is conservative at 0.1.
    """

    def _make(self, **kw):
        defaults = dict(embed_dim=4, max_feats=3, nugget=1e-4)
        defaults.update(kw)
        return LinearProjKernel(**defaults)

    def test_shape(self, X_sample):
        torch.manual_seed(1)
        L = self._make()(X_sample, d)
        assert L.shape == (B, T, d, d), f"Expected ({B},{T},{d},{d}), got {L.shape}"

    def test_finite(self, X_sample):
        torch.manual_seed(2)
        L = self._make()(X_sample, d)
        assert L.isfinite().all(), "L contains non-finite values"

    def test_cov_quality(self, X_sample):
        torch.manual_seed(3)
        L = self._make()(X_sample, d)
        C = _recover_cov(L)
        _check_cov_quality(C, "LinearProjKernel", is_copula=True, min_off_diag=0.1)

    def test_embed_dim_1(self, X_sample):
        """q=1: each embedding is scalar → cosine = ±1 → extreme correlations."""
        torch.manual_seed(4)
        L = self._make(embed_dim=1)(X_sample, d)
        C = _recover_cov(L)
        off_mean = _off_diag_abs_mean(C)
        print(f"\n  [LinearProjKernel q=1] mean |off-diagonal| = {off_mean:.4f}")
        # With q=1 correlations are near ±1 (definitely non-null)
        assert off_mean > 0.5, f"q=1 should produce near-±1 correlations, got {off_mean:.4f}"

    def test_different_seeds_produce_different_C(self, X_sample):
        """Two independent calls should produce different correlation structures."""
        torch.manual_seed(10)
        C1 = _recover_cov(self._make()(X_sample, d))
        torch.manual_seed(20)
        C2 = _recover_cov(self._make()(X_sample, d))
        max_diff = (C1 - C2).abs().max().item()
        print(f"\n  [LinearProjKernel] max |C1 - C2| across seeds = {max_diff:.4f}")
        assert max_diff > 0.01, "Two independent calls produced identical correlation matrices"

    def test_cov_varies_with_x(self, X_sample):
        """Correlation matrix should differ across instances (x-dependence)."""
        torch.manual_seed(5)
        L = self._make()(X_sample, d)
        C = _recover_cov(L)
        # Check that C[b, t1, :, :] ≠ C[b, t2, :, :] for t1 ≠ t2
        C_t0 = C[:, 0, :, :]   # (B, d, d)
        C_t1 = C[:, 1, :, :]   # (B, d, d)
        diff = (C_t0 - C_t1).abs().max().item()
        print(f"\n  [LinearProjKernel] max |C(x_0) - C(x_1)| = {diff:.4f}")
        assert diff > 1e-4, "Correlation matrix does not vary across instances"


# ──────────────────────────────────────────────────────────────────────────────
# 2. IsotropicModulatedKernel
# ──────────────────────────────────────────────────────────────────────────────


class TestIsotropicModulatedKernel:

    @pytest.mark.parametrize("kernel_type", ["rbf", "matern12", "matern32", "matern52", "random"])
    def test_cov_quality(self, X_sample, kernel_type):
        torch.manual_seed(42)
        gen = IsotropicModulatedKernel(kernel_type=kernel_type, nugget=1e-4)
        L = gen(X_sample, d)
        assert L.shape == (B, T, d, d)
        assert L.isfinite().all()
        C = _recover_cov(L)
        _check_cov_quality(C, f"IsoModKernel[{kernel_type}]", is_copula=True, min_off_diag=0.05)


# ──────────────────────────────────────────────────────────────────────────────
# 3. SimpleAggKernel
# ──────────────────────────────────────────────────────────────────────────────


class TestSimpleAggKernel:

    @pytest.mark.parametrize("kernel_type", ["cosine", "dot_product", "rbf", "matern12", "random"])
    def test_cov_quality(self, X_sample, kernel_type):
        torch.manual_seed(42)
        gen = SimpleAggKernel(
            kernel_type=kernel_type,
            nugget=1e-4,
            embed_dim=4,
            max_feats=3,
            lengthscale_lo=0.5,
            lengthscale_hi=5.0,
        )
        L = gen(X_sample, d)
        assert L.shape == (B, T, d, d)
        assert L.isfinite().all()
        C = _recover_cov(L)
        _check_cov_quality(C, f"SimpleAggKernel[{kernel_type}]", is_copula=True, min_off_diag=0.05)


# ──────────────────────────────────────────────────────────────────────────────
# 4. TabICLFeatureKernel  (use linear-only pool to keep tests fast)
# ──────────────────────────────────────────────────────────────────────────────


class TestTabICLFeatureKernel:

    @pytest.mark.parametrize("kernel_type", ["rbf", "matern32", "random"])
    def test_cov_quality(self, X_sample, kernel_type):
        torch.manual_seed(42)
        gen = TabICLFeatureKernel(
            kernel_type=kernel_type,
            nugget=1e-4,
            embed_dim=4,
            max_feats=3,
            lengthscale_lo=0.5,
            lengthscale_hi=5.0,
            func_pool=["linear", "mlp"],  # restrict to fast functions
        )
        L = gen(X_sample, d)
        assert L.shape == (B, T, d, d)
        assert L.isfinite().all()
        C = _recover_cov(L)
        _check_cov_quality(C, f"TabICLFeatureKernel[{kernel_type}]", is_copula=True, min_off_diag=0.05)


# ──────────────────────────────────────────────────────────────────────────────
# 5. KernelCovGen  (full covariance, not copula → no diagonal≈1 check)
# ──────────────────────────────────────────────────────────────────────────────


class TestKernelCovGen:

    @pytest.mark.parametrize("kernel_type", ["rbf", "exponential", "matern32", "random"])
    def test_cov_quality(self, X_sample, kernel_type):
        torch.manual_seed(42)
        gen = KernelCovGen(
            kernel_type=kernel_type,
            latent_dim=1,
            mlp_hidden=16,
            nugget=1e-4,
        )
        L = gen(X_sample, d)
        assert L.shape == (B, T, d, d)
        assert L.isfinite().all()
        C = _recover_cov(L)
        # KernelCovGen is not a copula — diagonal varies with x, no unit-diagonal check
        _check_cov_quality(C, f"KernelCovGen[{kernel_type}]", is_copula=False, min_off_diag=0.05)


# ──────────────────────────────────────────────────────────────────────────────
# 6. generate_episode end-to-end: Y must carry non-trivial cross-correlations
# ──────────────────────────────────────────────────────────────────────────────


def _empirical_cross_corr(Y_train: torch.Tensor) -> torch.Tensor:
    """Empirical cross-correlation matrix of Y_train ∈ (B, n_train, d).

    Y_train is already z-normalised per dimension, so diagonal ≈ 1.
    Returns (B, d, d) correlation matrices.
    """
    # Y_train is (B, n_train, d); already zero-mean & unit-std per dim (train split)
    n = Y_train.shape[1]
    C_emp = torch.einsum("bti,btj->bij", Y_train, Y_train) / (n - 1)
    return C_emp


class TestGenerateEpisodeYCorrelations:
    """For each generate_episode mode, check that Y_train has non-trivial
    cross-dimensional correlations (mean |off-diagonal| > 0.05)."""

    B, p, d, r = 4, 10, 4, 4
    n_train, n_test = 256, 32
    diag_alpha = 0.5

    def _check_y_correlations(self, Y_train: torch.Tensor, name: str, threshold: float = 0.05):
        C_emp = _empirical_cross_corr(Y_train)
        d = Y_train.shape[-1]
        mask = _off_diag_mask(d)
        off_mean = C_emp[:, mask].abs().mean().item()
        print(f"\n  [{name}] mean |empirical cross-corr off-diag| = {off_mean:.4f}  (threshold > {threshold})")
        assert off_mean > threshold, (
            f"{name}: Y_train has near-zero cross-correlations (mean |off-diag|={off_mean:.4f}). "
            f"The covariance generator may not be inducing meaningful cross-dimensional structure."
        )

    def test_mlp_mode(self):
        """Default MLP covariance (no kernel_cov_gen)."""
        torch.manual_seed(0)
        X_tr, Y_tr, *_ = generate_episode(
            self.B, self.p, self.d, self.r, self.n_train, self.n_test, DEVICE,
            diag_alpha=self.diag_alpha,
        )
        self._check_y_correlations(Y_tr, "mlp")

    def test_fixed_cov_mode(self):
        """Fixed (per-dataset, x-independent) covariance."""
        torch.manual_seed(1)
        X_tr, Y_tr, *_ = generate_episode(
            self.B, self.p, self.d, self.r, self.n_train, self.n_test, DEVICE,
            diag_alpha=self.diag_alpha,
            fixed_cov=True, fixed_cov_n_anchors=4,
        )
        self._check_y_correlations(Y_tr, "fixed_cov")

    def test_anchor_mode(self):
        """Anchor-based x-dependent covariance."""
        torch.manual_seed(2)
        anchor_gen = GlobalAnchorCovGen(K=4, r=self.r, tau=1.0, device=DEVICE)
        X_tr, Y_tr, *_ = generate_episode(
            self.B, self.p, self.d, self.r, self.n_train, self.n_test, DEVICE,
            diag_alpha=self.diag_alpha,
            anchor_gen=anchor_gen,
        )
        self._check_y_correlations(Y_tr, "anchor")

    def test_hyperplane_multimodal_mode(self):
        """Piecewise-constant covariance (K hyperplane regions)."""
        torch.manual_seed(3)
        X_tr, Y_tr, *_ = generate_episode(
            self.B, self.p, self.d, self.r, self.n_train, self.n_test, DEVICE,
            diag_alpha=self.diag_alpha,
            hyperplane_multimodal=True,
            hyperplane_multimodal_scale_lo=0.5,
            hyperplane_multimodal_scale_hi=4.0,
            hyperplane_multimodal_n_groups=3,
        )
        self._check_y_correlations(Y_tr, "hyperplane_multimodal")

    def test_linear_proj_kernel_mode(self):
        """LinearProjKernel used through generate_episode."""
        torch.manual_seed(4)
        gen = LinearProjKernel(embed_dim=4, max_feats=3, nugget=1e-4)
        X_tr, Y_tr, *_ = generate_episode(
            self.B, self.p, self.d, self.r, self.n_train, self.n_test, DEVICE,
            diag_alpha=self.diag_alpha,
            kernel_cov_gen=gen,
        )
        self._check_y_correlations(Y_tr, "linear_proj_kernel")

    def test_iso_kernel_mode(self):
        """IsotropicModulatedKernel used through generate_episode."""
        torch.manual_seed(5)
        gen = IsotropicModulatedKernel(kernel_type="rbf", nugget=1e-4)
        X_tr, Y_tr, *_ = generate_episode(
            self.B, self.p, self.d, self.r, self.n_train, self.n_test, DEVICE,
            diag_alpha=self.diag_alpha,
            kernel_cov_gen=gen,
        )
        self._check_y_correlations(Y_tr, "iso_kernel")

    def test_simple_agg_kernel_mode(self):
        """SimpleAggKernel (cosine) used through generate_episode."""
        torch.manual_seed(6)
        gen = SimpleAggKernel(kernel_type="cosine", embed_dim=4, max_feats=3)
        X_tr, Y_tr, *_ = generate_episode(
            self.B, self.p, self.d, self.r, self.n_train, self.n_test, DEVICE,
            diag_alpha=self.diag_alpha,
            kernel_cov_gen=gen,
        )
        self._check_y_correlations(Y_tr, "simple_agg_kernel")

    def test_kernel_cov_gen_mode(self):
        """KernelCovGen (MLP-parameterised GP) used through generate_episode."""
        torch.manual_seed(7)
        gen = KernelCovGen(kernel_type="rbf", latent_dim=1, mlp_hidden=16, nugget=1e-4)
        X_tr, Y_tr, *_ = generate_episode(
            self.B, self.p, self.d, self.r, self.n_train, self.n_test, DEVICE,
            diag_alpha=self.diag_alpha,
            kernel_cov_gen=gen,
        )
        self._check_y_correlations(Y_tr, "kernel_cov_gen")
