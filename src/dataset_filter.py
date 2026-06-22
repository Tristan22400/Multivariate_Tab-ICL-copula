"""
Dataset filter for copula pretraining: tests whether copula dependence varies with x.

Method
------
Operate directly on Z = Φ⁻¹(U) latents (the data the copula model actually consumes).
Fit an OAS-shrunk unconditional baseline and a generic ExtraTrees local-covariance
detector. Both are projected into the same low-rank+diagonal class (rank r) so the
comparison asks "does conditioning help *within the structure our model can represent*."
Score cross-fitted Δ = NLL_base − NLL_detector on out-of-sample rows; bootstrap-test
whether the mean Δ is positive.

Why Z (deployment-matched latents)
-----------------------------------
The copula model is trained on Z = Φ⁻¹(U) produced by the frozen TabICL marginals.
Filtering in Z-space ensures we judge the data the model will actually see, not
ground-truth marginals that may differ from what TabICL estimates.

Why ExtraTrees (not our own copula model)
------------------------------------------
Using our own model as the detector would cause circular selection — the filter would
prefer datasets our model already handles well and penalise novel structure. ExtraTrees
provides a generic nonlinear partition and acts as a kNN-in-tree-space smoother.
We deliberately avoid entrywise Cholesky regression as an alternative, because
Cholesky entries do not average on the PSD manifold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from scipy import stats
from scipy.linalg import solve_triangular
from sklearn.covariance import OAS
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


# ─── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class FilterConfig:
    """All tunables for the conditional-dependence filter."""
    eps: float = 1e-4               # clamp U to [eps, 1-eps] before ppf
    r: Optional[int] = None         # low-rank dim; None → min(d-1, 4)
    K: int = 5                      # cross-fitting folds
    B_bootstrap: int = 200          # bootstrap resamples
    margin: float = 0.0             # Δ̄^(b) must exceed this to count as a win
    min_neighborhood_size: int = 32 # min co-leaf neighbors for local OAS
    n_trees: int = 64               # ExtraTrees estimators
    bootstrap_win_threshold: float = 0.95
    ks_level: float = 0.01          # KS p-value threshold for PIT calibration gate
    diag_floor: float = 1e-6        # floor for Dg entries in low-rank projection
    eig_floor: float = 1e-8         # floor for eigenvalues of shrunk covariance
    seed: int = 0
    _allow_leak: bool = False       # TEST ONLY: disables cross-fitting discipline


# ─── Low-rank covariance ────────────────────────────────────────────────────────

class LowRankCov:
    """
    Σ = B Bᵀ + Dg, with B: (d, r) and Dg: (d,) strictly positive.

    All linear algebra uses the Woodbury identity / matrix determinant lemma —
    no dense d×d matrix is ever formed or inverted.
    """

    def __init__(self, B: np.ndarray, Dg: np.ndarray) -> None:
        self.B = B          # (d, r)
        self.Dg = Dg        # (d,) positive
        self.d, self.r = B.shape
        # Capacitance matrix M = I_r + Bᵀ Dg⁻¹ B  (r×r, always PD since Dg>0)
        DginvB = B / Dg[:, None]               # (d, r)
        M = np.eye(self.r) + B.T @ DginvB      # (r, r)
        self._L_M = np.linalg.cholesky(M)      # lower triangular (r, r)
        self._DginvB = DginvB                   # cached

    def logdet(self) -> float:
        """log|Σ| via the matrix determinant lemma: sum log Dg + log|M|."""
        log_dg = np.sum(np.log(self.Dg))
        log_M = 2.0 * np.sum(np.log(np.diag(self._L_M)))
        return float(log_dg + log_M)

    def quad_form(self, z: np.ndarray) -> np.ndarray:
        """
        zᵀ Σ⁻¹ z via Woodbury for z: (m, d) → (m,), or z: (d,) → scalar.

        Σ⁻¹ = Dg⁻¹ − Dg⁻¹ B M⁻¹ Bᵀ Dg⁻¹
        """
        scalar = z.ndim == 1
        z = np.atleast_2d(z)                            # (m, d)
        a = z / self.Dg[None, :]                        # (m, d), = z Dg⁻¹
        raw = (a * z).sum(-1)                            # (m,), = zᵀ Dg⁻¹ z
        q = a @ self.B                                  # (m, r)
        # Solve L_M y = qᵀ  →  y: (r, m)
        y = solve_triangular(self._L_M, q.T, lower=True)
        correction = (y * y).sum(0)                     # (m,)
        out = raw - correction
        return float(out[0]) if scalar else out

    def solve(self, z: np.ndarray) -> np.ndarray:
        """Σ⁻¹ z via Woodbury for z: (m, d) → (m, d), or z: (d,) → (d,)."""
        scalar = z.ndim == 1
        z = np.atleast_2d(z)
        a = z / self.Dg[None, :]
        q = a @ self.B                                  # (m, r)
        y = solve_triangular(self._L_M, q.T, lower=True)
        y2 = solve_triangular(self._L_M.T, y, lower=False)
        correction = (self._DginvB @ y2).T             # (m, d)
        out = a - correction
        return out[0] if scalar else out


def project_to_low_rank(
    S: np.ndarray,
    r: int,
    diag_floor: float,
    eig_floor: float,
) -> LowRankCov:
    """
    Project a PSD matrix S onto the low-rank+diagonal class: Σ = BBᵀ + Dg.

    Uses the top-r eigen-decomposition; floors eigenvalues and diagonal residual.
    """
    eigvals, eigvecs = np.linalg.eigh(S)    # ascending order, (d,) and (d, d)
    eigvals = eigvals[-r:]                  # top-r, (r,)
    eigvecs = eigvecs[:, -r:]               # (d, r)
    eigvals = np.maximum(eigvals, eig_floor)
    B = eigvecs * np.sqrt(eigvals)[None, :] # (d, r)
    Dg = np.diag(S) - np.sum(B ** 2, axis=1)
    Dg = np.maximum(Dg, diag_floor)
    return LowRankCov(B, Dg)


# ─── NLL helper ─────────────────────────────────────────────────────────────────

def _nll(cov: LowRankCov, z: np.ndarray) -> np.ndarray:
    """Per-row Gaussian NLL (constant dropped): 0.5 logdet + 0.5 zᵀ Σ⁻¹ z."""
    return 0.5 * cov.logdet() + 0.5 * cov.quad_form(z)


# ─── Tree-space neighborhood ────────────────────────────────────────────────────

def _build_neighborhood(
    query_leaves: np.ndarray,   # (n_trees,)
    train_leaves: np.ndarray,   # (n_tr, n_trees)
    min_size: int,
) -> np.ndarray:
    """
    Return indices of ≥ min_size train points with highest leaf co-occurrence.

    Co-occurrence = number of trees in which the query and the train point
    share the same leaf.  Falls back to the nearest-by-count points if the
    set of truly co-occurring points is smaller than min_size.
    """
    co_occur = (train_leaves == query_leaves[None, :]).sum(axis=1)  # (n_tr,)
    order = np.argsort(-co_occur)                   # descending
    n_nonzero = int((co_occur > 0).sum())
    n_take = min(max(n_nonzero, min_size), len(train_leaves))
    return order[:n_take]


# ─── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    keep: bool
    delta_mean: float
    bootstrap_win_rate: float
    score: float            # = delta_mean; continuous informativeness measure
    pit_calibration: dict   # {"per_dim": {j: {"ks_stat", "ks_pval"}}, "pit_suspect": bool}
    diagnostics: dict       # n, d, r, p, K, eps, n_singular_neighborhoods, nbhd_*


# ─── Core filter ────────────────────────────────────────────────────────────────

def filter_dataset(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    config: FilterConfig,
) -> FilterResult:
    """
    Filter a single dataset: X (n, p) features, Z (n, d) copula latents.

    Z must already be standard-normal (Φ⁻¹(U)), i.e. produced by the TabICL
    PIT pipeline.  If you have raw (X, Y), use filter_dataset_from_Y instead.
    """
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    n, d = Z.shape
    p = X.shape[1]

    r = config.r if config.r is not None else min(d - 1, 4)
    r = max(r, 1)

    if d < 2:
        return FilterResult(
            keep=True, delta_mean=0.0, bootstrap_win_rate=1.0, score=0.0,
            pit_calibration={"per_dim": {}, "pit_suspect": False},
            diagnostics={"n": n, "d": d, "r": r, "p": p, "note": "d<2, trivially kept"},
        )

    K = max(min(config.K, n // 2), 2)

    deltas = np.zeros(n)
    nbhd_sizes: list = []
    n_singular = 0
    U_held_out = np.zeros((n, d))

    def _process_fold(
        X_fit: np.ndarray,
        Z_fit: np.ndarray,
        X_query: np.ndarray,
        Z_query: np.ndarray,
        query_global_idx: np.ndarray,
    ) -> None:
        nonlocal n_singular

        # Unconditional baseline: OAS on the fit split
        oas = OAS(assume_centered=True).fit(Z_fit)
        cov_base = project_to_low_rank(
            oas.covariance_, r, config.diag_floor, config.eig_floor
        )

        # Conditional detector: ExtraTrees purely for leaf assignments
        forest = ExtraTreesRegressor(
            n_estimators=config.n_trees,
            random_state=config.seed,
            n_jobs=1,
        ).fit(X_fit, Z_fit)

        leaves_fit = forest.apply(X_fit)        # (n_fit, n_trees)
        leaves_query = forest.apply(X_query)    # (n_q, n_trees)

        for j in range(len(Z_query)):
            nbhd = _build_neighborhood(
                leaves_query[j], leaves_fit, config.min_neighborhood_size
            )
            nbhd_sizes.append(len(nbhd))

            try:
                oas_loc = OAS(assume_centered=True).fit(Z_fit[nbhd])
                cov_det = project_to_low_rank(
                    oas_loc.covariance_, r, config.diag_floor, config.eig_floor
                )
            except Exception:
                cov_det = cov_base
                n_singular += 1

            z_j = Z_query[j : j + 1]           # shape (1, d)
            deltas[query_global_idx[j]] = float(
                _nll(cov_base, z_j)[0] - _nll(cov_det, z_j)[0]
            )

    if config._allow_leak:
        # TEST ONLY: fit on all rows, neighborhood includes the query itself
        _process_fold(X, Z, X, Z, np.arange(n))
        U_held_out[:] = stats.norm.cdf(Z)
    else:
        kf = KFold(n_splits=K, shuffle=True, random_state=config.seed)
        for train_idx, test_idx in kf.split(X):
            _process_fold(
                X[train_idx], Z[train_idx],
                X[test_idx], Z[test_idx],
                test_idx,
            )
            U_held_out[test_idx] = stats.norm.cdf(Z[test_idx])

    # Bootstrap decision
    rng = np.random.default_rng(config.seed + 1)
    boot_means = np.fromiter(
        (rng.choice(deltas, size=n, replace=True).mean()
         for _ in range(config.B_bootstrap)),
        dtype=float,
        count=config.B_bootstrap,
    )
    win_rate = float((boot_means > config.margin).mean())
    keep = win_rate >= config.bootstrap_win_threshold

    # PIT calibration gate (flags miscalibration, never auto-rejects)
    pit_per_dim: dict = {}
    pit_suspect = False
    for j in range(d):
        ks_stat, ks_pval = stats.kstest(U_held_out[:, j], "uniform")
        pit_per_dim[j] = {"ks_stat": float(ks_stat), "ks_pval": float(ks_pval)}
        if ks_pval < config.ks_level:
            pit_suspect = True

    return FilterResult(
        keep=keep,
        delta_mean=float(deltas.mean()),
        bootstrap_win_rate=win_rate,
        score=float(deltas.mean()),
        pit_calibration={"per_dim": pit_per_dim, "pit_suspect": pit_suspect},
        diagnostics={
            "n": n, "d": d, "r": r, "p": p, "K": K, "eps": config.eps,
            "n_singular_neighborhoods": n_singular,
            "nbhd_size_mean": float(np.mean(nbhd_sizes)) if nbhd_sizes else 0.0,
            "nbhd_size_min": int(np.min(nbhd_sizes)) if nbhd_sizes else 0,
            "nbhd_size_max": int(np.max(nbhd_sizes)) if nbhd_sizes else 0,
        },
    )


def filter_dataset_from_Y(
    X: np.ndarray,
    Y: np.ndarray,
    marginal_pit: Callable,
    *,
    config: FilterConfig,
) -> FilterResult:
    """
    Spec-compatible wrapper: marginal_pit(Y, X) → U ∈ (0,1)^(n,d).

    Clamps U to [eps, 1-eps] then converts Z = Φ⁻¹(U) and calls filter_dataset.
    """
    U = np.asarray(marginal_pit(Y, X), dtype=float)
    U = np.clip(U, config.eps, 1.0 - config.eps)
    Z = stats.norm.ppf(U)
    return filter_dataset(X, Z, config=config)


# ─── Over-filter safeguard ──────────────────────────────────────────────────────

def select_for_pretraining(
    results: List[FilterResult],
    *,
    target_keep_fraction: float = 0.65,
    seed: int = 0,
) -> List[bool]:
    """
    Calibrate a score threshold to keep ~target_keep_fraction of datasets by score rank.

    Returns a list[bool] aligned with results (True = include in pretraining).
    """
    n = len(results)
    scores = np.array([r.score for r in results])

    # Calibrate: select exactly k_target datasets by score rank.
    # argsort-based selection handles ties without over-keeping.
    k_target = max(1, int(round(target_keep_fraction * n)))
    order = np.argsort(-scores, kind="stable")     # descending
    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[order[:k_target]] = True
    n_kept = int(keep_mask.sum())

    logger.info(
        "select_for_pretraining: kept %d/%d (%.1f%%)",
        n_kept, n, 100.0 * n_kept / n,
    )

    return list(keep_mask.tolist())


# ─── Episode adapter ────────────────────────────────────────────────────────────

def filter_episode_arrays(
    X: np.ndarray,
    Z: np.ndarray,
    config: FilterConfig,
) -> List[FilterResult]:
    """
    Filter a batch of datasets.

    Accepts X/Z shaped (B, n, p)/(B, n, d) or (n, p)/(n, d) (single dataset).
    Returns a list of B FilterResults.
    """
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if X.ndim == 2:
        X = X[None]
        Z = Z[None]
    return [filter_dataset(X[b], Z[b], config=config) for b in range(X.shape[0])]


def filter_episode_file(
    path: str,
    config: FilterConfig,
    split: str = "train",
) -> List[FilterResult]:
    """
    Load a .pt episode dict and filter each dataset in the batch.

    Expected keys: X_{split} (B, n, p) and Z_{split} (B, n, d).
    torch is used only to read the file; arrays are converted to numpy immediately.
    """
    import torch  # deferred: torch is not a core dependency of this module

    ep = torch.load(path, map_location="cpu", weights_only=False)
    x_key = f"X_{split}"
    z_key = f"Z_{split}"
    if x_key not in ep or z_key not in ep:
        raise KeyError(
            f"Episode file {path!r} does not contain keys {x_key!r} and {z_key!r}. "
            f"Available keys: {list(ep.keys())}"
        )
    X_np = ep[x_key].float().numpy()
    Z_np = ep[z_key].float().numpy()
    return filter_episode_arrays(X_np, Z_np, config)
