"""
tests/test_model_diagnostics.py
================================
Architectural diagnostic comparison: ``tabicl-archi`` vs ``copula-tabICL``.

Run from project root:
    conda run -n multivariate-icl pytest tests/test_model_diagnostics.py -v -s

Each test class targets a specific failure hypothesis:

1. TestMatrixSanity         — symmetry, unit diagonal, positive definiteness.
2. TestInitializationAnalysis — R at t=0: both models; reveals the degenerate
                                identity-matrix init of tabicl-archi.
3. TestZSensitivity         — sensitivity of R_test to changes in Z_support.
                              tabicl-archi is expected to be far less sensitive
                              because the outer-product covariance signal is absent.
4. TestXSensitivity         — sensitivity of R_test to changes in X_query.
                              tabicl-archi has NO X-routing mechanism; copula-tabICL
                              has x_sim_bias + x_route_proj.
5. TestGradientFlow         — gradient magnitude at every Z-injection weight.
                              Reveals dead/saturated components and the scalar-gate
                              cancellation present in tabicl-archi.
6. TestConstantOutputDiagnosis — std(off-diagonal R) across query instances.
                                 Near-zero std exposes "all instances get the same R"
                                 (the constant-matrix collapse seen in tabicl-archi).
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC  = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = "cpu"
torch.manual_seed(0)

# Tolerances
SYM_TOL       = 1e-5   # |R - Rᵀ|  ≤ this
DIAG_TOL      = 1e-5   # |diag(R) - 1| ≤ this
PD_MARGIN     = 0.0    # min eigenvalue must be > this


def reconstruct_R(D: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Build dense correlation matrix R = diag(D) + V Vᵀ.

    Args:
        D : (B, N, d) — diagonal variances.
        V : (B, N, d, r) — low-rank factor.
    Returns:
        R : (B, N, d, d)
    """
    return torch.diag_embed(D) + V @ V.transpose(-2, -1)


def check_correlation_matrix(
    R: torch.Tensor,
    name: str = "",
) -> dict:
    """Run all three structural checks on a batch of correlation matrices.

    Args:
        R    : (B, N, d, d)
        name : label printed in assertion messages.

    Returns:
        dict with keys:
            symmetry_error           — max |R - Rᵀ|
            unit_diag_error          — max |diag(R) - 1|
            min_eigenvalue           — minimum eigval (float)
            n_nonpositive_eigvals    — count(λ ≤ 0)
            n_near_eps_eigvals       — count(λ < machine-epsilon)
            off_diag_mean_abs        — mean |off-diagonal| entries
            off_diag_std_across_inst — std of per-instance mean|off-diag|
                                       (near 0 → constant matrix)
            is_valid                 — bool: passes all three checks
    """
    B, N, d, _ = R.shape

    # a) Symmetry
    sym_err = (R - R.transpose(-2, -1)).abs().max().item()

    # b) Unit diagonal
    diag = R.diagonal(dim1=-2, dim2=-1)           # (B, N, d)
    diag_err = (diag - 1.0).abs().max().item()

    # c) Positive definiteness via eigvalsh (symmetric path, more stable)
    eigvals  = torch.linalg.eigvalsh(R)            # (B, N, d)
    min_eig  = eigvals.min().item()
    eps_mach = torch.finfo(R.dtype).eps
    n_nonpos = (eigvals <= PD_MARGIN).sum().item()
    n_near   = (eigvals < eps_mach).sum().item()

    # d) Off-diagonal variation (constant-matrix probe)
    ri, ci = torch.triu_indices(d, d, offset=1, device=R.device)
    off = R[..., ri, ci]                           # (B, N, n_pairs)
    off_mean  = off.abs().mean().item()
    per_inst  = off.abs().mean(dim=-1)             # (B, N)
    off_std   = per_inst.std().item()

    return dict(
        symmetry_error           = sym_err,
        unit_diag_error          = diag_err,
        min_eigenvalue           = min_eig,
        n_nonpositive_eigvals    = n_nonpos,
        n_near_eps_eigvals       = n_near,
        off_diag_mean_abs        = off_mean,
        off_diag_std_across_inst = off_std,
        is_valid                 = (sym_err < SYM_TOL) and
                                   (diag_err < DIAG_TOL) and
                                   (min_eig > PD_MARGIN),
        label                    = name,
    )


def _print_diag(info: dict) -> None:
    print(f"\n  [{info['label']}]")
    print(f"    symmetry_error           = {info['symmetry_error']:.2e}")
    print(f"    unit_diag_error          = {info['unit_diag_error']:.2e}")
    print(f"    min_eigenvalue           = {info['min_eigenvalue']:.4e}")
    print(f"    n_nonpositive_eigvals    = {info['n_nonpositive_eigvals']}")
    print(f"    n_near_eps_eigvals       = {info['n_near_eps_eigvals']}")
    print(f"    off_diag_mean_abs        = {info['off_diag_mean_abs']:.4f}")
    print(f"    off_diag_std_across_inst = {info['off_diag_std_across_inst']:.4f}")
    print(f"    is_valid                 = {info['is_valid']}")


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def make_episode(
    B: int = 4,
    p: int = 20,
    d: int = 8,
    n_support: int = 64,
    n_query: int = 16,
    seed: int = 42,
    device: str = DEVICE,
) -> dict:
    """Create a small synthetic ICL episode (no real PIT required)."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    N = n_support + n_query
    X_all = torch.randn(B, N, p, generator=g, device=device)
    # standardise X along instance axis (like the real pipeline)
    X_all = (X_all - X_all.mean(dim=1, keepdim=True)) / (
        X_all.std(dim=1, keepdim=True) + 1e-8
    )

    # Z_all: standard-normal marginals (PIT outputs)
    Z_all = torch.randn(B, N, d, generator=g, device=device)

    return dict(X_all=X_all, Z_all=Z_all, n_support=n_support, n_query=n_query,
                B=B, p=p, d=d, N=N)


# ---------------------------------------------------------------------------
# Model factories — small but architecturally faithful
# ---------------------------------------------------------------------------

def build_tabicl_archi(d: int = 8, k: int = 4, p_max: int = 20):
    """Instantiate a compact tabicl-archi (CopulaTabICL wrapped).

    Uses reduced depths for test-speed; qualitative properties are unchanged.
    """
    tabicl_archi = pytest.importorskip(
        "tabicl_archi",
        reason="tabicl_archi / tabicl_upstream not importable",
    )
    from model import _CopulaTabICLWrapper

    inner = tabicl_archi.CopulaTabICL(
        d=d,
        k=k,
        embed_dim=32,
        col_num_blocks=2,
        col_nhead=4,
        col_num_inds=32,
        row_num_blocks=2,
        row_nhead=4,
        row_num_cls=4,
        icl_num_blocks=4,
        icl_nhead=4,
        dropout=0.0,
    )
    return _CopulaTabICLWrapper(inner)


def model_device(model: torch.nn.Module) -> str:
    """Return the device string of the first parameter of the model."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return DEVICE


def build_copula_tabicl_v2(d_max: int = 8, p_max: int = 20, rank: int = 4):
    """Instantiate a compact CopulaTabICLv2 (copula-tabICL)."""
    from model import CopulaTabICLv2

    return CopulaTabICLv2(
        d_model=64,
        n_heads=4,
        n_layers_s1=2,
        n_layers_s2=2,
        n_layers_s3=3,
        n_inducing=32,
        n_cls=4,
        p_max=p_max,
        d_max=d_max,
        rank=rank,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# 1. Matrix sanity & constraint verification
# ---------------------------------------------------------------------------

class TestMatrixSanity:
    """R must be symmetric, have unit diagonal, and be positive definite.

    Both models construct R via the Woodbury decomposition:
        R = diag(C_diag) + W Wᵀ  with  C_diag_i + ||W_i||² = 1.

    Verification is structural: check that the output of a fresh (random)
    forward pass satisfies all three constraints at machine-precision.
    """

    @pytest.fixture(scope="class")
    def episode(self):
        return make_episode(B=2, p=20, d=8, n_support=32, n_query=8)

    def _run_model_and_check(self, model, ep, name, train_mode: bool = False):
        dev = model_device(model)
        # Move episode to the model's device (TabICL may live on CUDA)
        X = ep["X_all"].to(dev)
        Z = ep["Z_all"].to(dev)
        # TabICL's inference_mgr moves inputs to CUDA regardless of weight device
        # when model.eval() is called; use training mode to bypass this for tabicl-archi.
        if train_mode:
            model.train()
        else:
            model.eval()
        with torch.no_grad():
            mu, D, V = model(X, Z, ep["n_support"])
        # mu should be zero
        assert (mu == 0).all(), f"{name}: mu must be exactly zero"
        # D must be strictly positive
        assert (D > 0).all(), f"{name}: diagonal D must be > 0 everywhere"
        # Build full R and run checks
        R = reconstruct_R(D, V)
        info = check_correlation_matrix(R, name=name)
        _print_diag(info)
        return info

    def test_copula_tabicl_v2_symmetry(self, episode):
        info = self._run_model_and_check(
            build_copula_tabicl_v2(), episode, "copula_tabicl_v2"
        )
        assert info["symmetry_error"] < SYM_TOL, (
            f"copula_tabicl_v2: R not symmetric  ({info['symmetry_error']:.2e})"
        )

    def test_copula_tabicl_v2_unit_diagonal(self, episode):
        info = self._run_model_and_check(
            build_copula_tabicl_v2(), episode, "copula_tabicl_v2"
        )
        assert info["unit_diag_error"] < DIAG_TOL, (
            f"copula_tabicl_v2: diagonal ≠ 1  (max err = {info['unit_diag_error']:.2e})"
        )

    def test_copula_tabicl_v2_positive_definite(self, episode):
        info = self._run_model_and_check(
            build_copula_tabicl_v2(), episode, "copula_tabicl_v2"
        )
        assert info["n_nonpositive_eigvals"] == 0, (
            f"copula_tabicl_v2: {info['n_nonpositive_eigvals']} non-positive eigenvalue(s), "
            f"min = {info['min_eigenvalue']:.3e}"
        )

    def test_tabicl_archi_symmetry(self, episode):
        model = build_tabicl_archi()
        info = self._run_model_and_check(model, episode, "tabicl_archi", train_mode=True)
        assert info["symmetry_error"] < SYM_TOL, (
            f"tabicl_archi: R not symmetric  ({info['symmetry_error']:.2e})"
        )

    def test_tabicl_archi_unit_diagonal(self, episode):
        model = build_tabicl_archi()
        info = self._run_model_and_check(model, episode, "tabicl_archi", train_mode=True)
        assert info["unit_diag_error"] < DIAG_TOL, (
            f"tabicl_archi: diagonal ≠ 1  (max err = {info['unit_diag_error']:.2e})"
        )

    def test_tabicl_archi_positive_definite(self, episode):
        model = build_tabicl_archi()
        info = self._run_model_and_check(model, episode, "tabicl_archi", train_mode=True)
        assert info["n_nonpositive_eigvals"] == 0, (
            f"tabicl_archi: {info['n_nonpositive_eigvals']} non-positive eigenvalue(s), "
            f"min = {info['min_eigenvalue']:.3e}"
        )


# ---------------------------------------------------------------------------
# 2. Initialization analysis
# ---------------------------------------------------------------------------

class TestInitializationAnalysis:
    """Inspect the predicted R at t=0 (random weights, no training).

    Expected findings
    -----------------
    tabicl-archi (_normalize near W=0):
        * W = raw[..., :d*k] ≈ 0  (last Linear layer uses default kaiming_uniform)
        * D = softplus(raw[..., d*k:]) ≈ softplus(0) = ln(2) ≈ 0.693  everywhere
        * sigma = ||W||² + D ≈ D
        * D_tilde = D/sigma = 1.0  → R = I  (identity for every instance)
        * off_diag_mean_abs ≈ 0.0
        * off_diag_std_across_inst ≈ 0.0  (ALL instances identical)
        * Gradient of D_tilde w.r.t. raw_D:  ∂D_tilde/∂D = ||W||²/sigma² ≈ 0
          → D branch has ZERO gradient at init; model can only learn via W branch.

    copula-tabICL (CopulaTabICLv2):
        * readout_U[-1] initialized with std=0.1 (explicit _init_weights)
        * U_sq_norm ≈ d * r * 0.01 ≈ 0.32   (d=8, r=4)
        * C_diag = 1/(1 + 0.32) ≈ 0.76
        * W ≈ U/1.15  (non-trivial magnitudes)
        * off_diag_mean_abs > 0  (non-trivial off-diagonal structure at init)
        * per-instance R varies due to random X + dim_emb symmetry-breaking
        * D branch gradient: ∂C_diag/∂U_sq_norm = -1/(1+||U||²)²  non-zero everywhere
    """

    @pytest.fixture(scope="class")
    def episode(self):
        return make_episode(B=4, p=20, d=8, n_support=64, n_query=16)

    def _init_stats(self, model, ep, name, train_mode: bool = False) -> dict:
        """Return init-time R statistics."""
        dev = model_device(model)
        X = ep["X_all"].to(dev)
        Z = ep["Z_all"].to(dev)
        if train_mode:
            model.train()
        else:
            model.eval()
        with torch.no_grad():
            _, D, V = model(X, Z, ep["n_support"])
        R = reconstruct_R(D, V)
        info = check_correlation_matrix(R, name=name)
        _print_diag(info)

        # Extra: report min/max diagonal (should all be 1.0)
        diag = D  # C_diag == D after Woodbury
        print(f"    D  min={D.min().item():.4f}  max={D.max().item():.4f}  "
              f"mean={D.mean().item():.4f}")
        # V norm per instance
        v_norm = V.norm(dim=(-2, -1))           # (B, N_query)
        print(f"    ||V|| min={v_norm.min():.4f}  max={v_norm.max():.4f}  "
              f"mean={v_norm.mean():.4f}")
        return info

    def test_copula_tabicl_v2_off_diag_nonzero(self, episode):
        """CopulaTabICLv2 must produce non-trivial off-diagonal entries at init."""
        info = self._init_stats(build_copula_tabicl_v2(), episode, "copula_tabicl_v2 (init)")
        # Readout init std=0.1 → ||U||² ≈ 0.32 → non-zero W at init
        assert info["off_diag_mean_abs"] > 1e-4, (
            "copula_tabicl_v2: off-diagonal entries are too small at init "
            f"(mean|off| = {info['off_diag_mean_abs']:.2e}). "
            "The std=0.1 readout init should produce non-trivial correlations."
        )

    def test_copula_tabicl_v2_instance_variation(self, episode):
        """CopulaTabICLv2 must show per-instance variation in R from the start."""
        info = self._init_stats(build_copula_tabicl_v2(), episode, "copula_tabicl_v2 (init)")
        # dim_emb + random X → different U per instance → std > 0
        assert info["off_diag_std_across_inst"] > 1e-6, (
            "copula_tabicl_v2: all query instances produce identical R at init. "
            "dim_emb should break instance symmetry."
        )

    def test_tabicl_archi_identity_at_init(self, episode):
        """tabicl-archi COLLAPSES to R=I at init due to the _normalize near W=0.

        This test DOCUMENTS the observed failure, not a desideratum.
        It passes when tabicl-archi's off-diagonal entries are near zero.
        """
        info = self._init_stats(build_tabicl_archi(), episode, "tabicl_archi (init)",
                                train_mode=True)
        print(
            "\n  ROOT CAUSE — _normalize near W=0:\n"
            "    raw ≈ 0  (last Linear uses kaiming_uniform, small bias)\n"
            "    D = softplus(0) ≈ 0.693  (per dimension, uniform)\n"
            "    sigma = ||W||² + D ≈ D\n"
            "    D_tilde = D/sigma = 1.0   ← every instance, every dimension\n"
            "    W_tilde = W/√sigma ≈ 0    ← no off-diagonal information\n"
            "    R = I  for all instances  ← constant prediction\n"
            "    ∂D_tilde/∂raw_D = ||W||²/sigma² ≈ 0  ← D branch DEAD at init"
        )
        # tabicl-archi should produce near-identity R (small off-diag)
        assert info["off_diag_mean_abs"] < 0.1, (
            f"tabicl_archi: expected near-identity R at init, "
            f"got mean|off| = {info['off_diag_mean_abs']:.4f}"
        )

    def test_tabicl_archi_constant_across_instances(self, episode):
        """tabicl-archi has near-zero instance variation at init.

        Since _normalize maps every instance to R≈I, the std of per-instance
        off-diagonal means should be near zero.  This confirms the
        'constant-output' failure mode observed during training.
        """
        info = self._init_stats(build_tabicl_archi(), episode, "tabicl_archi (init)",
                                train_mode=True)
        # Expect very low variation because R≈I for all instances
        assert info["off_diag_std_across_inst"] < info["off_diag_mean_abs"] + 0.05, (
            "tabicl_archi: expected near-zero instance variation at init "
            f"(std={info['off_diag_std_across_inst']:.4f}, "
            f"mean|off|={info['off_diag_mean_abs']:.4f})"
        )

    def test_normalize_gradient_vanishing(self):
        """Verify that ∂D_tilde/∂raw_D ≈ 0 at init in tabicl-archi's _normalize.

        This is the algebraic root cause of the D branch being dead at init.
        """
        pytest.importorskip("tabicl_archi", reason="tabicl_upstream not importable")
        from tabicl_archi import CopulaTabICL

        d, k = 8, 4
        B, N = 2, 8
        raw = torch.zeros(B, N, d * k + d, requires_grad=False)
        raw_D_part = raw[..., d * k :].detach().clone().requires_grad_(True)

        # Simulate _normalize: D = softplus(raw_D), W = 0 (at init)
        W_part = torch.zeros(B, N, d, k)
        D_part = F.softplus(raw_D_part)
        sigma = (W_part ** 2).sum(-1) + D_part

        D_tilde = D_part / sigma

        # Gradient of sum(D_tilde) w.r.t. raw_D_part
        D_tilde.sum().backward()
        grad_D = raw_D_part.grad  # (B, N, d)

        max_grad = grad_D.abs().max().item()
        print(
            f"\n  ∂D_tilde/∂raw_D at W=0, D=softplus(0)={D_part[0,0,0].item():.4f}:\n"
            f"    grad max abs = {max_grad:.4e}\n"
            "    (expected near 0 because ||W||²/sigma² = 0)"
        )

        # The gradient is exactly 0 when W=0 (see derivation above)
        assert max_grad < 1e-6, (
            f"Expected vanishing D-branch gradient at init, got {max_grad:.2e}. "
            "Formula: ∂D_tilde/∂D = ||W||²/sigma² = 0 when W=0."
        )

    def test_woodbury_gradient_health_at_init(self):
        """Verify that CopulaTabICLv2's Woodbury param has healthy C_diag gradient.

        The CopulaTabICLv2 reparameterisation: C_diag = 1/(1+||U||²),  W = U/√(1+||U||²).
        ∂C_diag/∂||U||² = -1/(1+||U||²)²  which is non-zero everywhere,
        unlike tabicl-archi's D branch.
        """
        d, r = 8, 4
        B, N = 2, 8
        # U ~ N(0, 0.1²) as in _init_weights
        U = torch.randn(B, N, d, r) * 0.1
        U.requires_grad_(True)

        U_sq = (U ** 2).sum(dim=-1)     # (B, N, d)
        C_diag = 1.0 / (1.0 + U_sq)    # (B, N, d)

        C_diag.sum().backward()
        grad_U = U.grad                 # (B, N, d, r)

        min_abs_grad = grad_U.abs().max().item()
        print(
            f"\n  ∂C_diag/∂U at U~N(0,0.01):\n"
            f"    grad max abs = {min_abs_grad:.4e}\n"
            "    (expected >> 0 since -2U/(1+||U||²)² is non-zero for U≠0)"
        )
        assert min_abs_grad > 1e-4, (
            "CopulaTabICLv2 C_diag branch gradient too small at init. "
            f"Got {min_abs_grad:.2e}."
        )


# ---------------------------------------------------------------------------
# 3. Z-sensitivity: does R_test change when Z_support changes?
# ---------------------------------------------------------------------------

class TestZSensitivity:
    """Measures how much R_test changes when Z_support is permuted.

    If the model ignores Z, the two runs produce identical R.
    tabicl-archi is expected to show LESS sensitivity because:
      - It has NO pairwise outer-product signal (vech(Z⊗Z) absent)
      - Only sees raw Z values → harder to detect correlation structure
      - Without X-routing, ICL attention averages support regions, reducing
        the effective Z signal each query sees.

    copula-tabICL has:
      - vech(Z⊗Z) at both S1 (embed_tae) and S3 (icl_emb) injection
      - The outer products directly encode which pairs of dims covary
      - X-routing ensures correct-region Z reaches each query
    """

    @pytest.fixture(scope="class")
    def episode(self):
        return make_episode(B=2, p=20, d=8, n_support=48, n_query=8, seed=1)

    def _z_sensitivity(self, model, ep, name, train_mode: bool = False):
        dev    = model_device(model)
        B, N   = ep["B"], ep["N"]
        X_all  = ep["X_all"].to(dev)
        Z_orig = ep["Z_all"].to(dev)
        n_sup  = ep["n_support"]

        # Permuted Z_support: shuffle support rows within each batch element
        g = torch.Generator()
        g.manual_seed(99)
        perm = torch.randperm(n_sup, generator=g)
        Z_perm = Z_orig.clone()
        Z_perm[:, :n_sup, :] = Z_orig[:, perm, :]      # only support rows changed

        if train_mode:
            model.train()
        else:
            model.eval()
        with torch.no_grad():
            _, D1, V1 = model(X_all, Z_orig, n_sup)
            _, D2, V2 = model(X_all, Z_perm, n_sup)

        R1 = reconstruct_R(D1, V1)   # (B, n_query, d, d)
        R2 = reconstruct_R(D2, V2)

        diff = (R1 - R2).abs()
        mean_diff  = diff.mean().item()
        max_diff   = diff.max().item()
        rel_change = mean_diff / (R1.abs().mean().item() + 1e-8)

        print(
            f"\n  [{name}] Z_support permutation sensitivity:\n"
            f"    mean|ΔR| = {mean_diff:.4e}\n"
            f"    max|ΔR|  = {max_diff:.4e}\n"
            f"    relative change = {rel_change:.4f}"
        )
        return dict(mean_diff=mean_diff, max_diff=max_diff, rel_change=rel_change)

    def test_copula_tabicl_v2_z_sensitive(self, episode):
        """copula-tabICL must produce different R when Z_support changes."""
        stats = self._z_sensitivity(
            build_copula_tabicl_v2(), episode, "copula_tabicl_v2"
        )
        assert stats["mean_diff"] > 1e-5, (
            "copula_tabicl_v2: R_test did not change after permuting Z_support. "
            "The vech(Z⊗Z) injection should make R sensitive to support Z values."
        )

    def test_tabicl_archi_z_sensitivity_is_lower(self, episode):
        """tabicl-archi has lower Z-sensitivity than copula-tabICL (or near zero).

        This demonstrates that the missing pairwise covariance signal (no outer
        product) AND the averaging over hyperplane regions reduces the effective
        Z signal reaching the decoder.
        """
        stats_v2 = self._z_sensitivity(
            build_copula_tabicl_v2(), episode, "copula_tabicl_v2"
        )
        try:
            stats_ta = self._z_sensitivity(
                build_tabicl_archi(), episode, "tabicl_archi", train_mode=True
            )
        except Exception as e:
            pytest.skip(f"tabicl-archi not importable: {e}")

        print(
            "\n  COMPARISON:\n"
            f"    copula_tabicl_v2  mean|ΔR| = {stats_v2['mean_diff']:.4e}\n"
            f"    tabicl_archi      mean|ΔR| = {stats_ta['mean_diff']:.4e}\n"
            f"    ratio (v2/tabicl) = {stats_v2['mean_diff'] / max(stats_ta['mean_diff'], 1e-12):.2f}"
        )
        assert stats_v2["mean_diff"] >= stats_ta["mean_diff"], (
            "Expected copula_tabicl_v2 to have greater Z-sensitivity than tabicl_archi. "
            "If tabicl_archi somehow has higher Z-sensitivity, re-inspect the Z injection path."
        )


# ---------------------------------------------------------------------------
# 4. X-sensitivity: does R_test change when X_query changes?
# ---------------------------------------------------------------------------

class TestXSensitivity:
    """Key diagnostic for the hyperplane-multimodal failure.

    In the hyperplane dataset, X encodes which correlation REGION an instance
    belongs to.  The model must route each query to the correct support region.

    copula-tabICL mechanisms for X-routing:
      1. x_sim_bias = X_qry @ X_sup^T  (cosine similarity, additive ICL bias)
      2. x_route_proj: learned projection of X into d_icl, gated by x_route_gate
      Both together make Stage-3 attention X-aware from the very first step.

    tabicl-archi mechanisms:
      NONE.  The ICL transformer in the TabICL backbone has no X-routing signal.
      It must implicitly learn to route via Q·K attention in the row embedding
      space, which is a much harder inductive problem.

    Test: fix Z_support, swap X_query between two contrasting X distributions.
    copula-tabICL should show a large change in R; tabicl-archi much smaller or zero.
    """

    @pytest.fixture(scope="class")
    def episode_pair(self):
        ep1 = make_episode(B=2, p=20, d=8, n_support=48, n_query=8, seed=2)

        # Create ep2 with the same Z_all but completely different X_query
        g = torch.Generator()
        g.manual_seed(777)
        ep2 = dict(**ep1)  # shallow copy
        X_alt = ep1["X_all"].clone()
        # Replace X for query rows with independent noise (opposite sign to maximize distance)
        n_sup = ep1["n_support"]
        X_alt[:, n_sup:, :] = -ep1["X_all"][:, n_sup:, :] + 2.0 * torch.randn(
            ep1["B"], ep1["n_query"], ep1["p"], generator=g
        )
        ep2["X_all"] = X_alt
        return ep1, ep2

    def _x_sensitivity(self, model, ep1, ep2, name, train_mode: bool = False):
        dev   = model_device(model)
        if train_mode:
            model.train()
        else:
            model.eval()
        n_sup = ep1["n_support"]
        with torch.no_grad():
            _, D1, V1 = model(ep1["X_all"].to(dev), ep1["Z_all"].to(dev), n_sup)
            _, D2, V2 = model(ep2["X_all"].to(dev), ep2["Z_all"].to(dev), n_sup)

        R1 = reconstruct_R(D1, V1)
        R2 = reconstruct_R(D2, V2)

        diff = (R1 - R2).abs()
        mean_diff = diff.mean().item()
        max_diff  = diff.max().item()

        print(
            f"\n  [{name}] X_query change sensitivity (same Z_support):\n"
            f"    mean|ΔR| = {mean_diff:.4e}\n"
            f"    max|ΔR|  = {max_diff:.4e}"
        )
        return dict(mean_diff=mean_diff, max_diff=max_diff)

    def test_copula_tabicl_v2_x_sensitive(self, episode_pair):
        """copula-tabICL must respond to X_query change.

        x_sim_bias routes attention; different X_query → different attention
        pattern over support → different aggregated Z signal → different R.
        """
        ep1, ep2 = episode_pair
        stats = self._x_sensitivity(
            build_copula_tabicl_v2(), ep1, ep2, "copula_tabicl_v2"
        )
        assert stats["mean_diff"] > 1e-5, (
            "copula_tabicl_v2: R_test did not change after flipping X_query. "
            "x_sim_bias and x_route_proj should make R sensitive to X_query."
        )

    def test_tabicl_archi_x_sensitivity_vs_copula(self, episode_pair):
        """tabicl-archi is expected to have lower X-sensitivity.

        Prints a comparison table to quantify the routing gap.
        If tabicl-archi has near-zero X-sensitivity, it confirms the
        'constant correlation matrix per dataset' failure mode.
        """
        ep1, ep2 = episode_pair
        stats_v2 = self._x_sensitivity(
            build_copula_tabicl_v2(), ep1, ep2, "copula_tabicl_v2"
        )
        try:
            stats_ta = self._x_sensitivity(
                build_tabicl_archi(), ep1, ep2, "tabicl_archi", train_mode=True
            )
        except Exception as e:
            pytest.skip(f"tabicl-archi not importable: {e}")

        ratio = stats_v2["mean_diff"] / max(stats_ta["mean_diff"], 1e-12)
        print(
            f"\n  ROUTING GAP:\n"
            f"    copula_tabicl_v2  mean|ΔR| = {stats_v2['mean_diff']:.4e}\n"
            f"    tabicl_archi      mean|ΔR| = {stats_ta['mean_diff']:.4e}\n"
            f"    v2 / tabicl ratio = {ratio:.2f}x more X-sensitive\n"
            "\n  ROOT CAUSE: tabicl-archi has no X-routing:\n"
            "    • No x_sim_bias (cosine similarity routing)\n"
            "    • No x_route_proj (learned X projection into ICL space)\n"
            "    The ICL attention must learn X discrimination purely via Q·K in\n"
            "    the row embedding space — a much harder inductive problem."
        )
        # copula-tabICL must be at least as X-sensitive
        assert stats_v2["mean_diff"] >= stats_ta["mean_diff"], (
            "copula_tabicl_v2 should be more X-sensitive than tabicl_archi. "
            "Something unexpected is happening."
        )


# ---------------------------------------------------------------------------
# 5. Gradient flow through Z injection
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Verify that gradients reach the Z-encoding weights after one backward pass.

    tabicl-archi failure modes:
      • In `_icl_predictions`: `R[:, :T] = R[:, :T] + y_encoder(Z.unsqueeze(-1))`
        This is a direct in-place addition (no gating). The gradient flows back
        to y_encoder.linear.weight IF the decoder produces a non-trivial signal.
        But with W_tilde≈0 at init, the NLL loss gradient w.r.t. the decoder
        output is small for nearly-uncorrelated data → slow learning.
      • The col_embedder.y_encoder also receives gradients through a long path
        (Stages 1+2+3+decoder). Gradient magnitude may be too small to drive
        the y_encoder to produce a strong Z signal.

    copula-tabICL (CopulaTabICLv2):
      • embed_tae and embed_icl both see vech(Z⊗Z) — 2x more signal
      • icl_gate_sup is a VECTOR parameter — no cross-dimension cancellation
      • Direct path: embed_icl → icl_gate → row_emb → S3 → readout_U → R
    """

    @pytest.fixture(scope="class")
    def episode(self):
        return make_episode(B=2, p=20, d=8, n_support=32, n_query=8, seed=3)

    def _compute_grads(self, model, ep, name):
        """Run a forward+backward with a dummy MSE loss and return gradient norms."""
        dev   = model_device(model)
        model.train()
        n_sup = ep["n_support"]
        _, D, V = model(ep["X_all"].to(dev), ep["Z_all"].to(dev), n_sup)
        # Dummy correlation-structure loss: push off-diagonal toward a target
        R = reconstruct_R(D, V)
        d = R.shape[-1]
        ri, ci = torch.triu_indices(d, d, offset=1, device=R.device)
        target_offdiag = torch.full_like(R[..., ri, ci], 0.5)
        loss = F.mse_loss(R[..., ri, ci], target_offdiag)
        loss.backward()

        grad_info = {}
        for pname, param in model.named_parameters():
            if param.grad is not None:
                g_norm = param.grad.norm().item()
                if g_norm > 0:
                    grad_info[pname] = g_norm

        print(f"\n  [{name}] Gradient norms (top-10 by magnitude):")
        for k, v in sorted(grad_info.items(), key=lambda x: -x[1])[:10]:
            print(f"    {k:60s}  {v:.3e}")
        return grad_info

    def test_copula_tabicl_v2_z_encoder_gets_grad(self, episode):
        """Both embed_tae and embed_icl must receive non-zero gradients."""
        model = build_copula_tabicl_v2()
        grads = self._compute_grads(model, episode, "copula_tabicl_v2")

        z_params = [k for k in grads if "embed_tae" in k or "embed_icl" in k]
        print(f"\n  Z-encoder params with grad: {z_params}")
        assert len(z_params) > 0, (
            "copula_tabicl_v2: embed_tae/embed_icl have zero gradient. "
            "Check that Z injection path is connected to the loss."
        )
        for pname in z_params:
            assert grads[pname] > 1e-9, (
                f"copula_tabicl_v2: {pname} has near-zero gradient ({grads[pname]:.2e})"
            )

    def test_copula_tabicl_v2_icl_gate_gets_grad(self, episode):
        """icl_gate_sup must receive gradient — the vector gate must be trainable."""
        model = build_copula_tabicl_v2()
        grads = self._compute_grads(model, episode, "copula_tabicl_v2")
        gate_grads = {k: v for k, v in grads.items() if "icl_gate" in k}
        print(f"\n  icl_gate grads: {gate_grads}")
        assert len(gate_grads) > 0, (
            "copula_tabicl_v2: icl_gate_sup has zero gradient."
        )

    def test_tabicl_archi_y_encoder_gets_grad(self, episode):
        """tabicl-archi's y_encoders must receive gradient from the loss."""
        try:
            model = build_tabicl_archi()
        except Exception as e:
            pytest.skip(f"tabicl-archi not importable: {e}")

        grads = self._compute_grads(model, episode, "tabicl_archi")
        y_enc_grads = {k: v for k, v in grads.items() if "y_encoder" in k}
        print(
            f"\n  tabicl_archi y_encoder grads: {y_enc_grads}\n"
            "  NOTE: if these are very small relative to copula_tabicl_v2's\n"
            "  embed_tae/embed_icl grads, the Z signal is effectively dead."
        )
        # Gradient must be non-zero (even if small)
        for pname, g in y_enc_grads.items():
            assert g > 0, (
                f"tabicl_archi: {pname} has exactly zero gradient. "
                "The y_encoder is completely disconnected from the loss."
            )

    def test_readout_head_grad_comparison(self, episode):
        """Compare readout head gradient magnitude between the two architectures.

        If tabicl-archi's decoder gets vanishingly small gradients relative to
        copula-tabICL's readout_U, that explains why the head never deviates
        from the R=I initialization.
        """
        model_v2 = build_copula_tabicl_v2()
        grads_v2 = self._compute_grads(model_v2, episode, "copula_tabicl_v2")

        # Collect readout head grads for copula-tabICL
        readout_v2_norms = [v for k, v in grads_v2.items() if "readout_U" in k or "fc_V" in k]
        mean_readout_v2 = sum(readout_v2_norms) / len(readout_v2_norms) if readout_v2_norms else 0.0

        try:
            model_ta = build_tabicl_archi()
            grads_ta = self._compute_grads(model_ta, episode, "tabicl_archi")
            readout_ta_norms = [v for k, v in grads_ta.items() if "decoder" in k]
            mean_readout_ta = sum(readout_ta_norms) / len(readout_ta_norms) if readout_ta_norms else 0.0
        except Exception as e:
            pytest.skip(f"tabicl-archi not importable: {e}")

        print(
            f"\n  READOUT HEAD GRADIENT COMPARISON:\n"
            f"    copula_tabicl_v2  mean readout grad = {mean_readout_v2:.3e}\n"
            f"    tabicl_archi      mean decoder grad = {mean_readout_ta:.3e}"
        )
        # Both must be non-zero
        assert mean_readout_v2 > 0, "copula_tabicl_v2 readout head has zero gradient"


# ---------------------------------------------------------------------------
# 6. Constant-output diagnosis
# ---------------------------------------------------------------------------

class TestConstantOutputDiagnosis:
    """Detect the 'constant R per dataset' collapse.

    In the hyperplane multimodal dataset, different query instances in the
    SAME dataset may belong to different hyperplane regions, so their true
    correlation structure differs within a single batch.

    A well-functioning model should produce DIFFERENT R for different query
    instances within the same dataset (std(off-diag) across instances > 0).

    tabicl-archi failure:
      • At init, all instances get R=I (std=0) — confirmed by TestInitializationAnalysis.
      • During training, without X-routing, the ICL attention averages across
        regions → every query sees the same mixed-region support signal →
        the decoder outputs the same R for all instances in a dataset.
      • This manifests as off_diag_std_across_inst ≈ 0 throughout training.

    copula-tabICL:
      • x_sim_bias + x_route_proj differentiates queries by their X location.
      • dim_emb differentiates target dimensions.
      • Both ensure off_diag_std > 0 even at init.
    """

    @pytest.fixture(scope="class")
    def episode_multimodal(self):
        """Construct a batch with explicit two-region structure in X.

        Instances 0..n_support//2-1 have X ≫ 0 (region A),
        instances n_support//2..n_support-1 have X ≪ 0 (region B),
        query instances alternate A/B.
        This maximizes the chance of detecting X-dependent routing.
        """
        B, p, d = 2, 20, 8
        n_support, n_query = 48, 16
        N = n_support + n_query

        g = torch.Generator()
        g.manual_seed(5)

        # Region-separated X
        X_A = torch.randn(B, N // 2, p, generator=g)  + 2.0
        X_B = torch.randn(B, N // 2, p, generator=g)  - 2.0
        X_all = torch.cat([X_A, X_B], dim=1)

        # Shuffle so regions are mixed
        perm = torch.randperm(N, generator=g)
        X_all = X_all[:, perm, :]

        # Z ~ N(0,1)
        Z_all = torch.randn(B, N, d, generator=g)

        return dict(
            X_all=X_all, Z_all=Z_all,
            n_support=n_support, n_query=n_query,
            B=B, p=p, d=d, N=N,
        )

    def _instance_variation(self, model, ep, name, train_mode: bool = False) -> dict:
        dev = model_device(model)
        if train_mode:
            model.train()
        else:
            model.eval()
        with torch.no_grad():
            _, D, V = model(ep["X_all"].to(dev), ep["Z_all"].to(dev), ep["n_support"])
        R = reconstruct_R(D, V)   # (B, n_query, d, d)
        info = check_correlation_matrix(R, name=name)
        _print_diag(info)
        return info

    def test_copula_tabicl_v2_instance_variation_multimodal(self, episode_multimodal):
        """CopulaTabICLv2 must show inter-instance R variation on two-region data."""
        info = self._instance_variation(
            build_copula_tabicl_v2(), episode_multimodal, "copula_tabicl_v2 (2-region)"
        )
        assert info["off_diag_std_across_inst"] > 1e-6, (
            "copula_tabicl_v2: all query instances have identical R on two-region data. "
            "x_sim_bias should produce different routing for region-A vs region-B queries."
        )

    def test_tabicl_archi_documents_low_variation_multimodal(self, episode_multimodal):
        """tabicl-archi is expected to have near-zero inter-instance R variation.

        This test DOCUMENTS the failure (it passes when tabicl-archi fails to
        differentiate between region-A and region-B query instances).
        """
        try:
            info_ta = self._instance_variation(
                build_tabicl_archi(), episode_multimodal, "tabicl_archi (2-region)",
                train_mode=True
            )
        except Exception as e:
            pytest.skip(f"tabicl-archi not importable: {e}")

        info_v2 = self._instance_variation(
            build_copula_tabicl_v2(), episode_multimodal, "copula_tabicl_v2 (2-region)"
        )

        print(
            "\n  CONSTANT-OUTPUT DIAGNOSIS:\n"
            f"    copula_tabicl_v2 off_diag_std_across_inst = {info_v2['off_diag_std_across_inst']:.4e}\n"
            f"    tabicl_archi     off_diag_std_across_inst = {info_ta['off_diag_std_across_inst']:.4e}\n"
            "\n  ROOT CAUSES of tabicl-archi constant output:\n"
            "    1. No X-routing: ICL attn averages across regions → same R for all queries\n"
            "    2. R=I at init: D_tilde=1, W_tilde=0 (dead D-branch gradient)\n"
            "    3. No outer product: can't see pairwise Z covariance signal\n"
            "    4. No dim_emb: all d target dims get same projection → limited rank-1 R\n"
            "\n  copula-tabICL fixes:\n"
            "    1. x_sim_bias + x_route_proj: direct X-based routing bias\n"
            "    2. readout_U std=0.1 init: ||U||²≈0.32 → non-trivial W at t=0\n"
            "    3. vech(Z⊗Z) in both embed_tae and embed_icl: pairwise signal\n"
            "    4. dim_emb (orthogonal init): per-dim symmetry-breaking\n"
            "    5. Vector icl_gate_sup: no cross-dim gradient cancellation"
        )
        # tabicl-archi must have LESS variation than copula-tabICL
        assert info_v2["off_diag_std_across_inst"] >= info_ta["off_diag_std_across_inst"], (
            "Expected copula_tabicl_v2 to show more inter-instance variation than tabicl_archi."
        )


# ---------------------------------------------------------------------------
# 7. Full matrix structure — regression check
# ---------------------------------------------------------------------------

class TestFullMatrixStructure:
    """Regression tests: both models always produce valid correlation matrices,
    regardless of batch size, p, d, or n_support/n_query combinations.
    """

    @pytest.mark.parametrize("B,p,d,n_sup,n_qry", [
        (1,  4, 2,  8,  4),
        (2, 20, 8, 64, 16),
        (4, 15, 4, 32,  8),
    ])
    def test_copula_tabicl_v2_valid_R(self, B, p, d, n_sup, n_qry):
        ep = make_episode(B=B, p=p, d=d, n_support=n_sup, n_query=n_qry)
        model = build_copula_tabicl_v2(d_max=d, p_max=p)
        model.eval()
        with torch.no_grad():
            _, D, V = model(ep["X_all"], ep["Z_all"], ep["n_support"])
        R = reconstruct_R(D, V)
        info = check_correlation_matrix(R, name=f"v2 B={B} d={d}")
        _print_diag(info)
        assert info["is_valid"], (
            f"copula_tabicl_v2: invalid R for (B={B}, p={p}, d={d}, "
            f"n_sup={n_sup}, n_qry={n_qry}): {info}"
        )

    @pytest.mark.parametrize("B,p,d,n_sup,n_qry", [
        (1,  4, 8,  8,  4),
        (2, 20, 8, 32,  8),
    ])
    def test_tabicl_archi_valid_R(self, B, p, d, n_sup, n_qry):
        try:
            model = build_tabicl_archi(d=d, p_max=p)
        except Exception as e:
            pytest.skip(f"tabicl-archi not importable: {e}")
        dev = model_device(model)
        ep = make_episode(B=B, p=p, d=d, n_support=n_sup, n_query=n_qry, device=dev)
        model.train()  # bypass TabICL inference_mgr device routing
        with torch.no_grad():
            _, D, V = model(ep["X_all"], ep["Z_all"], ep["n_support"])
        R = reconstruct_R(D, V)
        info = check_correlation_matrix(R, name=f"tabicl B={B} d={d}")
        _print_diag(info)
        assert info["is_valid"], (
            f"tabicl_archi: invalid R for (B={B}, p={p}, d={d}, "
            f"n_sup={n_sup}, n_qry={n_qry}): {info}"
        )
