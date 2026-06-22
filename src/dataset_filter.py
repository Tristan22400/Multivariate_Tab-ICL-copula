"""
Dataset filter for copula pretraining: tests whether copula dependence varies with x.

Method
------
K-fold cross-fitting on Z = Φ⁻¹(U) latents.  For each test point we find its
k nearest neighbours in X-space (torch.cdist kNN) and compute the sample
covariance of their Z-values as a local estimate.  The score Δ = NLL_base −
NLL_local measures whether the local model fits better than the unconditional
baseline; a positive Δ indicates conditional dependence.

All computation (kNN + covariance + NLL) is batched on GPU across the full
episode batch B.  Bootstrap tests whether mean Δ > 0.  select_for_pretraining
then calibrates a global threshold to keep the target fraction of datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
from scipy import stats
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────────────────────

@dataclass
class FilterConfig:
    """All tunables for the conditional-dependence filter."""
    eps: float = 1e-4
    r: Optional[int] = None              # low-rank dim; None → min(d-1, 4)
    K: int = 5                           # cross-fitting folds
    B_bootstrap: int = 200
    margin: float = 0.0
    min_neighborhood_size: int = 32      # k for kNN
    n_trees: int = 64                    # unused (kept for config compatibility)
    bootstrap_win_threshold: float = 0.95
    ks_level: float = 0.01
    diag_floor: float = 1e-6
    eig_floor: float = 1e-8
    seed: int = 0
    _allow_leak: bool = False            # TEST ONLY


# ─── Result dataclass ────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    keep: bool
    delta_mean: float
    bootstrap_win_rate: float
    score: float
    pit_calibration: dict
    diagnostics: dict


# ─── GPU primitives ──────────────────────────────────────────────────────────────

def _sample_cov(Z: torch.Tensor) -> torch.Tensor:
    """(..., n, d) → (..., d, d)  plain sample covariance, assume_centered=True."""
    return torch.einsum("...ni,...nj->...ij", Z, Z) / Z.shape[-2]


def _project_lr_batched(
    S: torch.Tensor,
    r: int,
    diag_floor: float,
    eig_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., d, d) → B_mat (..., d, r), Dg (..., d)  via top-r eigendecomposition."""
    eigvals, eigvecs = torch.linalg.eigh(S)            # ascending
    eigvals = eigvals[..., -r:].clamp(min=eig_floor)   # (..., r)
    eigvecs = eigvecs[..., -r:]                         # (..., d, r)
    B_mat = eigvecs * eigvals.unsqueeze(-2).sqrt()      # (..., d, r)
    Dg = S.diagonal(dim1=-2, dim2=-1) - (B_mat ** 2).sum(-1)
    return B_mat, Dg.clamp(min=diag_floor)


def _nll_batched(
    B_mat: torch.Tensor,
    Dg: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """
    (..., d, r), (..., d), (..., d) → (...)  per-row Gaussian NLL via Woodbury.

    Σ = B Bᵀ + Dg;  NLL = 0.5 log|Σ| + 0.5 zᵀ Σ⁻¹ z  (additive constant dropped)
    """
    r = B_mat.shape[-1]
    log_dg = Dg.log().sum(-1)
    DginvB = B_mat / Dg.unsqueeze(-1)                                      # (..., d, r)
    M = torch.eye(r, device=B_mat.device, dtype=B_mat.dtype) + \
        torch.einsum("...di,...dj->...ij", DginvB, B_mat)                  # (..., r, r)
    L = torch.linalg.cholesky(M)
    logdet = log_dg + 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    a = z / Dg
    q = torch.einsum("...d,...dr->...r", a, B_mat).unsqueeze(-1)           # (..., r, 1)
    y = torch.linalg.solve_triangular(L, q, upper=False).squeeze(-1)       # (..., r)
    return 0.5 * (logdet + (a * z).sum(-1) - (y * y).sum(-1))


# ─── Episode-level GPU filter ────────────────────────────────────────────────────

def filter_episode_gpu(
    X: np.ndarray,
    Z: np.ndarray,
    config: FilterConfig,
    device: str = "cuda",
) -> List[FilterResult]:
    """
    Filter B datasets jointly on GPU using batched kNN + sample covariance.

    X: (B, n, p), Z: (B, n, d) — numpy arrays from a .pt episode file.
    Returns a list of B FilterResults.
    """
    X_np = np.asarray(X, dtype=np.float32)
    Z_np = np.asarray(Z, dtype=np.float32)
    if X_np.ndim == 2:
        X_np = X_np[None]
        Z_np = Z_np[None]

    B, n, d = Z_np.shape
    p = X_np.shape[-1]
    r = config.r if config.r is not None else min(d - 1, 4)
    r = max(r, 1)
    k = min(config.min_neighborhood_size, n // 2)

    if d < 2:
        return [FilterResult(
            keep=True, delta_mean=0.0, bootstrap_win_rate=1.0, score=0.0,
            pit_calibration={"per_dim": {}, "pit_suspect": False},
            diagnostics={"n": n, "d": d, "r": r, "p": p, "note": "d<2, trivially kept"},
        ) for _ in range(B)]

    K = max(min(config.K, n // 2), 2)
    dev = torch.device(device)

    X_t = torch.from_numpy(X_np).to(dev)   # (B, n, p)
    Z_t = torch.from_numpy(Z_np).to(dev)   # (B, n, d)

    deltas = torch.zeros(B, n, device=dev, dtype=torch.float32)
    U_held = np.zeros((B, n, d), dtype=np.float32)

    kf = KFold(n_splits=K, shuffle=True, random_state=config.seed)
    for train_idx, test_idx in kf.split(range(n)):
        tr_t = torch.as_tensor(train_idx, device=dev, dtype=torch.long)
        te_t = torch.as_tensor(test_idx,  device=dev, dtype=torch.long)
        n_tr, n_te = len(train_idx), len(test_idx)

        X_tr = X_t[:, tr_t, :]             # (B, n_tr, p)
        Z_tr = Z_t[:, tr_t, :]             # (B, n_tr, d)
        X_te = X_t[:, te_t, :]             # (B, n_te, p)
        Z_te = Z_t[:, te_t, :]             # (B, n_te, d)

        # ── Baseline covariance: one per dataset ─────────────────────────────
        S_base = _sample_cov(Z_tr)                                         # (B, d, d)
        B_base, Dg_base = _project_lr_batched(
            S_base, r, config.diag_floor, config.eig_floor,
        )
        B_base_e  = B_base.unsqueeze(1).expand(B, n_te, d, r).reshape(B * n_te, d, r)
        Dg_base_e = Dg_base.unsqueeze(1).expand(B, n_te, d).reshape(B * n_te, d)

        # ── kNN: per-fold X standardisation then cdist ───────────────────────
        mu_x  = X_tr.mean(dim=1, keepdim=True)                # (B, 1, p)
        std_x = X_tr.std(dim=1, keepdim=True).clamp(min=1e-8)
        X_tr_n = (X_tr - mu_x) / std_x
        X_te_n = (X_te - mu_x) / std_x

        k_actual = min(k, n_tr)
        D   = torch.cdist(X_te_n, X_tr_n)                     # (B, n_te, n_tr)
        idx = D.topk(k_actual, largest=False, dim=-1).indices  # (B, n_te, k)

        # ── Gather Z-neighbours ───────────────────────────────────────────────
        b_exp = (torch.arange(B, device=dev)
                 .view(B, 1, 1)
                 .expand(B, n_te, k_actual))
        nbhd = Z_tr[b_exp, idx]                                # (B, n_te, k, d)

        # ── Local covariance: all B×n_te neighbourhoods at once ──────────────
        S_det = _sample_cov(nbhd.reshape(B * n_te, k_actual, d))  # (B*n_te, d, d)
        B_det, Dg_det = _project_lr_batched(
            S_det, r, config.diag_floor, config.eig_floor,
        )

        # ── NLL delta ────────────────────────────────────────────────────────
        z_flat   = Z_te.reshape(B * n_te, d)
        nll_base = _nll_batched(B_base_e, Dg_base_e, z_flat)  # (B*n_te,)
        nll_det  = _nll_batched(B_det,    Dg_det,    z_flat)
        deltas[:, te_t] = (nll_base - nll_det).reshape(B, n_te)

        U_held[:, test_idx, :] = stats.norm.cdf(Z_te.cpu().numpy())

    # ── Bootstrap + PIT calibration (CPU) ────────────────────────────────────
    deltas_cpu = deltas.cpu().numpy()   # (B, n)
    rng = np.random.default_rng(config.seed + 1)
    results: List[FilterResult] = []

    for b in range(B):
        d_b = deltas_cpu[b]
        boot_means = np.fromiter(
            (rng.choice(d_b, size=n, replace=True).mean()
             for _ in range(config.B_bootstrap)),
            dtype=float,
            count=config.B_bootstrap,
        )
        win_rate = float((boot_means > config.margin).mean())

        pit_per_dim: dict = {}
        pit_suspect = False
        for j in range(d):
            ks_stat, ks_pval = stats.kstest(U_held[b, :, j], "uniform")
            pit_per_dim[j] = {"ks_stat": float(ks_stat), "ks_pval": float(ks_pval)}
            if ks_pval < config.ks_level:
                pit_suspect = True

        results.append(FilterResult(
            keep=win_rate >= config.bootstrap_win_threshold,
            delta_mean=float(d_b.mean()),
            bootstrap_win_rate=win_rate,
            score=float(d_b.mean()),
            pit_calibration={"per_dim": pit_per_dim, "pit_suspect": pit_suspect},
            diagnostics={
                "n": n, "d": d, "r": r, "p": p, "K": K,
                "k_nbhd": k_actual, "backend": "gpu_knn",
            },
        ))

    return results


# ─── Public entry points ─────────────────────────────────────────────────────────

def filter_episode_file(
    path: str,
    config: FilterConfig,
    split: str = "train",
    device: str = "cuda",
) -> List[FilterResult]:
    """
    Load a .pt episode dict and filter each dataset in the batch on GPU.

    Expected keys: X_{split} (B, n, p) and Z_{split} (B, n, d).
    Pass device='cpu' to fall back to single-threaded CPU (slow, for testing).
    """
    ep = torch.load(path, map_location="cpu", weights_only=False)
    x_key, z_key = f"X_{split}", f"Z_{split}"
    if x_key not in ep or z_key not in ep:
        raise KeyError(
            f"Episode file {path!r} missing keys {x_key!r}/{z_key!r}. "
            f"Available: {list(ep.keys())}"
        )
    X_np = ep[x_key].float().numpy()
    Z_np = ep[z_key].float().numpy()
    return filter_episode_gpu(X_np, Z_np, config, device=device)


# ─── Global threshold calibration ────────────────────────────────────────────────

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
    k_target = max(1, int(round(target_keep_fraction * n)))
    order = np.argsort(-scores, kind="stable")
    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[order[:k_target]] = True
    n_kept = int(keep_mask.sum())
    logger.info(
        "select_for_pretraining: kept %d/%d (%.1f%%)",
        n_kept, n, 100.0 * n_kept / n,
    )
    return list(keep_mask.tolist())
