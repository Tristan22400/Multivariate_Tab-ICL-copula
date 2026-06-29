"""
Acceptance tests for src/dataset_filter.py.

Run from project root:
    conda run -n multivariate-icl pytest tests/test_dataset_filter.py -v -s

Tests:
  1. Woodbury / det-lemma correctness vs dense linear algebra (tol 1e-6)
  2. Null dataset (Z independent of X) → filter rejects in ≥ 80 % of seeds
  3. Signal dataset (R(x) varies with x) → filter keeps
  4. Leakage probe: _allow_leak=True makes the null dataset spuriously pass
  5. Over-filter safeguard: select_for_pretraining retains the configured simple fraction
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)

from dataset_filter import (
    FilterConfig,
    FilterResult,
    LowRankCov,
    filter_dataset,
    project_to_low_rank,
    select_for_pretraining,
)


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _compound_sym(d: int, rho: float) -> np.ndarray:
    """Compound-symmetry correlation matrix with off-diagonal rho."""
    R = np.full((d, d), rho)
    np.fill_diagonal(R, 1.0)
    return R


def _sample_null(n: int, d: int, rho: float, rng: np.random.Generator) -> tuple:
    """X independent of Z; Z ~ N(0, R) with R = compound_sym(d, rho)."""
    p = 6
    X = rng.standard_normal((n, p))
    R = _compound_sym(d, rho)
    L = np.linalg.cholesky(R)
    Z = rng.standard_normal((n, d)) @ L.T
    return X, Z


def _sample_signal(
    n: int, d: int, rho0: float, rho1: float, rng: np.random.Generator
) -> tuple:
    """
    R(x) = (1-t)*R0 + t*R1 where t = sigmoid(3 * x[:,0]).
    Signal: correlation structure changes smoothly from R0 to R1.
    """
    p = 6
    X = rng.standard_normal((n, p))
    t = 1.0 / (1.0 + np.exp(-3.0 * X[:, 0]))  # (n,) in (0,1)
    R0 = _compound_sym(d, rho0)
    R1 = _compound_sym(d, rho1)
    Z = np.zeros((n, d))
    for i in range(n):
        R_i = (1.0 - t[i]) * R0 + t[i] * R1
        L_i = np.linalg.cholesky(R_i)
        Z[i] = L_i @ rng.standard_normal(d)
    return X, Z


def _random_pd(d: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    A = rng.standard_normal((d, d)) * scale
    return A @ A.T + np.eye(d) * 0.5


# ─── Test 1: Woodbury / det-lemma correctness ───────────────────────────────────

class TestLowRankCovCorrectness:
    """LowRankCov.logdet and .quad_form must match dense linear algebra to 1e-6."""

    @pytest.mark.parametrize("d,r", [(5, 1), (8, 3), (10, 4), (20, 4)])
    def test_logdet(self, d: int, r: int) -> None:
        rng = np.random.default_rng(0)
        S = _random_pd(d, rng)
        cov = project_to_low_rank(S, r, diag_floor=1e-6, eig_floor=1e-8)
        Sigma = cov.B @ cov.B.T + np.diag(cov.Dg)
        expected = float(np.linalg.slogdet(Sigma)[1])
        assert abs(cov.logdet() - expected) < 1e-6, (
            f"logdet mismatch: got {cov.logdet():.8f}, expected {expected:.8f}"
        )

    @pytest.mark.parametrize("d,r", [(5, 1), (8, 3), (10, 4), (20, 4)])
    def test_quad_form(self, d: int, r: int) -> None:
        rng = np.random.default_rng(1)
        S = _random_pd(d, rng)
        cov = project_to_low_rank(S, r, diag_floor=1e-6, eig_floor=1e-8)
        Sigma = cov.B @ cov.B.T + np.diag(cov.Dg)
        Sigma_inv = np.linalg.inv(Sigma)
        m = 10
        z = rng.standard_normal((m, d))
        expected = np.array([float(z[i] @ Sigma_inv @ z[i]) for i in range(m)])
        actual = cov.quad_form(z)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0,
                                   err_msg="quad_form deviates from dense z @ inv(Σ) @ z")

    def test_quad_form_scalar_input(self) -> None:
        rng = np.random.default_rng(2)
        d, r = 6, 2
        S = _random_pd(d, rng)
        cov = project_to_low_rank(S, r, diag_floor=1e-6, eig_floor=1e-8)
        Sigma_inv = np.linalg.inv(cov.B @ cov.B.T + np.diag(cov.Dg))
        z = rng.standard_normal(d)
        expected = float(z @ Sigma_inv @ z)
        actual = cov.quad_form(z)
        assert isinstance(actual, float)
        assert abs(actual - expected) < 1e-6


# ─── Test 2: Null dataset → uninformative ───────────────────────────────────────

class TestNullDataset:
    N_SEEDS = 20
    N_KEEP_THRESHOLD = 4   # at most 4/20 runs should erroneously keep a null dataset

    def test_null_mostly_rejected(self) -> None:
        """Z independent of X: filter should reject in ≥ 80 % of seeds."""
        n_kept = 0
        for seed in range(self.N_SEEDS):
            rng = np.random.default_rng(seed + 100)
            X, Z = _sample_null(n=500, d=4, rho=0.6, rng=rng)
            config = FilterConfig(
                seed=seed,
                K=3,
                B_bootstrap=100,
                min_neighborhood_size=30,
                n_trees=32,
            )
            result = filter_dataset(X, Z, config=config)
            if result.keep:
                n_kept += 1

        assert n_kept <= self.N_KEEP_THRESHOLD, (
            f"Null dataset kept in {n_kept}/{self.N_SEEDS} runs "
            f"(expected ≤ {self.N_KEEP_THRESHOLD})"
        )


# ─── Test 3: Signal dataset → kept ──────────────────────────────────────────────

class TestSignalDataset:
    def test_signal_kept(self) -> None:
        """R(x) varies with x: filter should keep the dataset."""
        rng = np.random.default_rng(42)
        # rho0=0.7 (strong positive) vs rho1=-0.25 (mild negative); d=4 safe range
        X, Z = _sample_signal(n=600, d=4, rho0=0.7, rho1=-0.25, rng=rng)
        config = FilterConfig(
            seed=42,
            K=3,
            B_bootstrap=200,
            min_neighborhood_size=30,
            n_trees=32,
        )
        result = filter_dataset(X, Z, config=config)
        assert result.keep, (
            f"Signal dataset not kept: delta_mean={result.delta_mean:.4f}, "
            f"win_rate={result.bootstrap_win_rate:.3f}"
        )
        assert result.delta_mean > 0, "delta_mean should be positive for a signal dataset"

    def test_signal_multiple_seeds(self) -> None:
        """Signal dataset should be kept in ≥ 80 % of seeds."""
        N_SEEDS = 10
        n_kept = 0
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed + 200)
            X, Z = _sample_signal(n=600, d=4, rho0=0.7, rho1=-0.25, rng=rng)
            config = FilterConfig(seed=seed, K=3, B_bootstrap=100,
                                  min_neighborhood_size=30, n_trees=32)
            if filter_dataset(X, Z, config=config).keep:
                n_kept += 1
        assert n_kept >= int(0.8 * N_SEEDS), (
            f"Signal kept in only {n_kept}/{N_SEEDS} seeds"
        )


# ─── Test 4: Leakage probe ──────────────────────────────────────────────────────

class TestLeakageProbe:
    """
    Demonstrate that the cross-fitting / train-only-neighborhood discipline is
    load-bearing: disabling it (_allow_leak=True) makes a null dataset pass.

    This test is a regression guard — if future refactors silently remove the
    out-of-sample protection, the leak variant will stop spuriously passing and
    the test will catch it.
    """

    N_SEEDS = 20
    # With no leak: null should be rejected most of the time
    MAX_KEPT_NO_LEAK = 5     # at most 5/20
    # With leak:    null should be kept most of the time (spurious pass)
    MIN_KEPT_LEAK = 12       # at least 12/20

    def test_leak_vs_no_leak_on_null(self) -> None:
        n_kept_no_leak = 0
        n_kept_leak = 0

        for seed in range(self.N_SEEDS):
            rng = np.random.default_rng(seed + 300)
            X, Z = _sample_null(n=400, d=4, rho=0.6, rng=rng)

            cfg_clean = FilterConfig(
                seed=seed, K=3, B_bootstrap=100,
                min_neighborhood_size=25, n_trees=32,
                _allow_leak=False,
            )
            cfg_leak = FilterConfig(
                seed=seed, K=3, B_bootstrap=100,
                min_neighborhood_size=25, n_trees=32,
                _allow_leak=True,
            )
            if filter_dataset(X, Z, config=cfg_clean).keep:
                n_kept_no_leak += 1
            if filter_dataset(X, Z, config=cfg_leak).keep:
                n_kept_leak += 1

        assert n_kept_no_leak <= self.MAX_KEPT_NO_LEAK, (
            f"Without leak: null kept in {n_kept_no_leak}/{self.N_SEEDS} runs "
            f"(expected ≤ {self.MAX_KEPT_NO_LEAK}); cross-fitting may be broken"
        )
        assert n_kept_leak >= self.MIN_KEPT_LEAK, (
            f"With leak: null kept in only {n_kept_leak}/{self.N_SEEDS} runs "
            f"(expected ≥ {self.MIN_KEPT_LEAK}); leakage probe may not be effective"
        )


# ─── Test 5: Over-filter safeguard ──────────────────────────────────────────────

class TestOverFilterSafeguard:
    def _make_results(
        self,
        n_null: int,
        n_signal: int,
        null_score: float = -0.5,
        signal_score: float = 1.5,
    ) -> list:
        results = [
            FilterResult(
                keep=False,
                delta_mean=null_score + i * 0.001,  # spread slightly for percentile sampling
                bootstrap_win_rate=0.1,
                score=null_score + i * 0.001,
                pit_calibration={"per_dim": {}, "pit_suspect": False},
                diagnostics={},
            )
            for i in range(n_null)
        ]
        results += [
            FilterResult(
                keep=True,
                delta_mean=signal_score,
                bootstrap_win_rate=0.99,
                score=signal_score,
                pit_calibration={"per_dim": {}, "pit_suspect": False},
                diagnostics={},
            )
            for _ in range(n_signal)
        ]
        return results

    def test_retain_simple_fraction(self) -> None:
        """select_for_pretraining retains ≈ retain_simple_fraction * n simple datasets.

        Setup: 80 null (score -0.5) + 20 signal (score 1.5), total 100.
        target_keep_fraction=0.20 → exactly 20 initial keeps (the signal datasets).
        Then retain_simple_fraction=0.12 → 12 more from the 80 rejected nulls.
        Total null datasets in selected should be ≈ 12.
        """
        n_null, n_signal = 80, 20
        retain_frac = 0.12
        results = self._make_results(n_null, n_signal)
        n_total = len(results)

        selected = select_for_pretraining(
            results,
            target_keep_fraction=0.20,   # keep only the 20 signal datasets initially
            retain_simple_fraction=retain_frac,
            seed=0,
        )
        assert len(selected) == n_total

        # Only null datasets (indices 0..79) were rejected; count how many were re-added
        null_indices = list(range(n_null))
        n_retained_simple = sum(selected[i] for i in null_indices)

        expected = int(round(retain_frac * n_total))   # 12
        assert abs(n_retained_simple - expected) <= 2, (
            f"Retained {n_retained_simple} simple datasets, expected ≈ {expected} "
            f"(retain_frac={retain_frac}, n={n_total})"
        )

    def test_signal_datasets_preserved(self) -> None:
        """Signal-like datasets (high score) should always be kept."""
        n_null, n_signal = 80, 20
        results = self._make_results(n_null, n_signal)
        selected = select_for_pretraining(
            results,
            target_keep_fraction=0.65,
            retain_simple_fraction=0.12,
            seed=0,
        )
        signal_indices = list(range(n_null, n_null + n_signal))
        assert all(selected[i] for i in signal_indices), (
            "Some high-score (signal) datasets were incorrectly filtered out"
        )

    def test_total_kept_fraction_reasonable(self) -> None:
        """Overall kept fraction should be ≥ target_keep_fraction (before simple retains)."""
        n_null, n_signal = 70, 30
        results = self._make_results(n_null, n_signal)
        selected = select_for_pretraining(
            results,
            target_keep_fraction=0.65,
            retain_simple_fraction=0.12,
            seed=0,
        )
        kept_frac = sum(selected) / len(selected)
        # Total kept = target + simple retains ≥ target
        assert kept_frac >= 0.65 - 0.01, (
            f"Total kept fraction {kept_frac:.3f} is below target 0.65"
        )


# ─── Miscellaneous edge-case tests ──────────────────────────────────────────────

class TestEdgeCases:
    def test_d1_trivially_kept(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3))
        Z = rng.standard_normal((50, 1))
        result = filter_dataset(X, Z, config=FilterConfig())
        assert result.keep

    def test_small_n(self) -> None:
        """Should not crash with small n."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((30, 3))
        Z = rng.standard_normal((30, 4))
        config = FilterConfig(K=3, min_neighborhood_size=8, n_trees=16, B_bootstrap=50)
        result = filter_dataset(X, Z, config=config)
        assert isinstance(result.keep, bool)
        assert np.isfinite(result.delta_mean)

    def test_pit_calibration_populated(self) -> None:
        """pit_calibration should contain per-dim KS stats for every dimension."""
        rng = np.random.default_rng(9)
        X, Z = _sample_null(n=200, d=3, rho=0.4, rng=rng)
        config = FilterConfig(K=3, B_bootstrap=50, n_trees=16, min_neighborhood_size=20)
        result = filter_dataset(X, Z, config=config)
        assert "per_dim" in result.pit_calibration
        assert "pit_suspect" in result.pit_calibration
        per_dim = result.pit_calibration["per_dim"]
        assert len(per_dim) == 3
        for j in range(3):
            assert "ks_stat" in per_dim[j]
            assert "ks_pval" in per_dim[j]
            assert 0.0 <= per_dim[j]["ks_stat"] <= 1.0
            assert 0.0 <= per_dim[j]["ks_pval"] <= 1.0

    def test_diagnostics_keys(self) -> None:
        rng = np.random.default_rng(11)
        X, Z = _sample_null(n=100, d=3, rho=0.3, rng=rng)
        config = FilterConfig(K=3, B_bootstrap=20, n_trees=8, min_neighborhood_size=15)
        result = filter_dataset(X, Z, config=config)
        for key in ("n", "d", "r", "p", "K", "eps", "n_singular_neighborhoods"):
            assert key in result.diagnostics, f"Missing diagnostics key: {key!r}"
