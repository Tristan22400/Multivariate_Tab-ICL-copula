"""
train_on_datasets.py — Attention-based in-context copula estimator vs CopulaTransformer.

Loads one pre-computed episode from dataset_dir (b=0 of the batch) and:

  1. Fits an AttentionCopulaEstimator T_φ(x*, {(x_i, z_i)}) in z-space with early
     stopping on a fixed held-out validation split.
  2. Evaluates a set of classical baselines (moment, shrunk moment, Nadaraya-Watson
     with RBF/Epanechnikov/Laplace kernels).
  3. Evaluates the pretrained CopulaTransformer.
  4. Logs the full ablation ladder to W&B and console.

Usage:
    python src/train_on_datasets.py \\
        --config conf/config.yaml \\
        --ckpt   ./checkpoints/copula_transformer/step_0029999_final.pt \\
        [--episode_idx 0]
"""

from __future__ import annotations

import argparse
import copy
import io
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from data_gen import select_group_representative_indices
from loss import woodbury_nll
from model import build_copula_tabicl_v2, build_copula_transformer
from pit import load_tabicl, run_pit
from viz import plot_corr_grid, plot_prediction_comparison

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _build_ICL_model(cfg) -> nn.Module:
    return build_copula_transformer(cfg)


# ---------------------------------------------------------------------------
# Copula NLL (full d×d Cholesky — d=8 makes O(d³) negligible)
# ---------------------------------------------------------------------------


def copula_nll_full(z: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Gaussian copula NLL = 0.5*(log|R| + z^T R^{-1} z - z^T z).

    Args:
        z : (..., d)
        R : (..., d, d)  — SPD correlation matrix, diag = 1
    Returns:
        scalar — mean NLL over all leading dims
    """
    jitter = 1e-6 * torch.eye(R.shape[-1], dtype=R.dtype, device=R.device)
    R_j = R + jitter
    L = torch.linalg.cholesky(R_j)  # (..., d, d) lower triangular
    log_det = 2.0 * L.diagonal(dim1=-2, dim2=-1).clamp(min=1e-12).log().sum(-1)  # (...)
    # R^{-1} z via two triangular solves
    z_rinv_z = torch.cholesky_solve(z.unsqueeze(-1), L).squeeze(-1).mul(z).sum(-1)
    nll = 0.5 * (log_det + z_rinv_z - z.pow(2).sum(-1))
    return nll.mean()


# ---------------------------------------------------------------------------
# Correlation matrix helpers
# ---------------------------------------------------------------------------


def moment_estimator(Z: torch.Tensor, delta: float = 1e-6) -> torch.Tensor:
    """Empirical correlation matrix from Z: (n, d) -> (d, d)."""
    n, d = Z.shape
    S = Z.T @ Z / n  # (d, d)
    S = S + delta * torch.eye(d, dtype=Z.dtype, device=Z.device)
    std = S.diagonal().clamp(min=1e-8).sqrt()
    R = S / (std.unsqueeze(1) * std.unsqueeze(0))
    return R


def _xi_cond_stats(R: torch.Tensor) -> dict[str, float]:
    """Xi-conditioning diagnostics for a batch of correlation matrices.

    Args:
        R : (n, d, d) correlation matrices — one per query point xi.
    Returns:
        mean_abs_offdiag  : avg |R_ij| for i≠j — overall correlation strength.
        std_offdiag_xi    : mean std of each off-diagonal entry across query
                            points — 0 for xi-invariant models (moment), >0
                            when the model adapts its predictions to xi.
        var_frob_from_I   : variance of ||R - I||_F^2 across instances —
                            a single-number summary of xi-conditioning.
    """
    n, d, _ = R.shape
    idx_i, idx_j = torch.tril_indices(d, d, offset=-1, device=R.device)
    offdiag = R[:, idx_i, idx_j]          # (n, n_pairs)
    mean_abs = offdiag.abs().mean().item()
    # std across query instances for each pair, then averaged over pairs
    std_xi = offdiag.std(dim=0).mean().item() if n > 1 else 0.0
    I = torch.eye(d, device=R.device).unsqueeze(0)
    frob_sq = (R - I).pow(2).sum(dim=(-1, -2))  # (n,)
    var_frob = frob_sq.var().item() if n > 1 else 0.0
    return {
        "mean_abs_offdiag": mean_abs,
        "std_offdiag_xi": std_xi,
        "var_frob_from_I": var_frob,
    }


def _woodbury_diag_stats(D: torch.Tensor, V: torch.Tensor) -> dict[str, float]:
    """Variance stats for the diagonal of Sigma = diag(D) + VV^T across xi.

    Args:
        D : (n, d)    — diagonal variances per query point.
        V : (n, d, r) — low-rank factors per query point.
    Returns:
        mean_diag_var     : mean of diag(Sigma) across dims and instances.
        std_diag_var_xi   : mean std of diag(Sigma) across query instances —
                            measures how much the marginal variance varies with xi.
    """
    diag_var = D + (V**2).sum(-1)          # (n, d)  — diag(diag(D) + VV^T)
    mean_dv = diag_var.mean().item()
    std_dv_xi = diag_var.std(dim=0).mean().item() if D.shape[0] > 1 else 0.0
    return {"mean_diag_var": mean_dv, "std_diag_var_xi": std_dv_xi}


def shrinkage_grid_cv(
    Z_ctx: torch.Tensor,
    Z_val: torch.Tensor,
    n_grid: int = 20,
) -> tuple[torch.Tensor, float]:
    """Find optimal isotropic shrinkage λ on val set.

    Returns best R = (1-λ)R_mom + λI and best λ value.
    """
    R_mom = moment_estimator(Z_ctx)
    d = R_mom.shape[0]
    I = torch.eye(d, dtype=R_mom.dtype, device=R_mom.device)
    best_lambda = 0.0
    best_nll = float("inf")
    grid = torch.linspace(0.0, 0.95, n_grid)
    for lam in grid:
        R_s = (1.0 - lam) * R_mom + lam * I
        # broadcast over val instances: Z_val (m, d) -> m individual NLL calls
        R_exp = R_s.unsqueeze(0).expand(Z_val.shape[0], -1, -1)  # (m, d, d)
        nll = copula_nll_full(Z_val, R_exp).item()
        if nll < best_nll:
            best_nll = nll
            best_lambda = lam.item()
    R_best = (1.0 - best_lambda) * R_mom + best_lambda * I
    return R_best, best_lambda


def nw_corr(
    X_ctx: torch.Tensor,
    Z_ctx: torch.Tensor,
    X_qry: torch.Tensor,
    kernel: str,
    delta: float = 1e-6,
) -> torch.Tensor:
    """Nadaraya-Watson correlation estimator with median-bandwidth kernel.

    Args:
        X_ctx : (n, p)
        Z_ctx : (n, d)
        X_qry : (m, p)
        kernel: "rbf" | "epanechnikov" | "laplace"
    Returns:
        R : (m, d, d) — per-query correlation matrices
    """
    n, d = Z_ctx.shape
    m = X_qry.shape[0]

    # Pairwise squared distances (m, n)
    diff = X_qry.unsqueeze(1) - X_ctx.unsqueeze(0)  # (m, n, p)
    sq_dist = diff.pow(2).sum(-1)  # (m, n)
    dist = sq_dist.clamp(min=0.0).sqrt()  # (m, n)

    # Median bandwidth (computed from all pairwise train distances)
    all_dists = torch.pdist(X_ctx)
    h = all_dists.median().clamp(min=1e-6)

    if kernel == "rbf":
        log_w = -sq_dist / (2.0 * h**2)
    elif kernel == "epanechnikov":
        u = sq_dist / h**2
        log_w = torch.where(u < 1.0, (1.0 - u).log(), torch.full_like(u, -1e9))
    elif kernel == "laplace":
        log_w = -dist / h
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    w = F.softmax(log_w, dim=-1)  # (m, n) — normalized weights

    # Weighted outer product sum: R_q = Σ_i w_qi * z_i z_i^T
    # Z_ctx: (n, d) -> (1, n, d, 1) and (1, n, 1, d)
    outer = Z_ctx.unsqueeze(-1) * Z_ctx.unsqueeze(-2)  # (n, d, d)
    R = torch.einsum("mn,nij->mij", w, outer)  # (m, d, d)

    # Enforce diagonal = 1 (correlation normalization)
    R = R + delta * torch.eye(d, dtype=R.dtype, device=R.device).unsqueeze(0)
    std = R.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()  # (m, d)
    R = R / (std.unsqueeze(-1) * std.unsqueeze(-2))

    return R


# ---------------------------------------------------------------------------
# UCI real-world dataset loaders
# ---------------------------------------------------------------------------


def _normalize_X_cols(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    cat_cols: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-normalize continuous columns of X using training mean/std."""
    n_cols = X_tr.shape[1]
    cont = [j for j in range(n_cols) if cat_cols is None or j not in cat_cols]
    X_tr_out = X_tr.copy().astype(np.float32)
    X_te_out = X_te.copy().astype(np.float32)
    for j in cont:
        mu = X_tr[:, j].mean()
        sigma = X_tr[:, j].std()
        if sigma < 1e-9:
            sigma = 1.0
        X_tr_out[:, j] = (X_tr[:, j] - mu) / sigma
        X_te_out[:, j] = (X_te[:, j] - mu) / sigma
    return X_tr_out, X_te_out


def _to_f32(arr, device: torch.device) -> torch.Tensor:
    if hasattr(arr, "values"):
        arr = arr.values
    return torch.tensor(arr, dtype=torch.float32, device=device)


def _load_enb(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Energy Efficiency — UCI 242, d=2 (Heating Load, Cooling Load)."""
    from sklearn.model_selection import train_test_split
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=242)
    X = ds.data.features.values.astype(np.float32)
    y = ds.data.targets.values.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_te = _normalize_X_cols(X_tr, X_te)
    return _to_f32(X_tr, device), _to_f32(y_tr, device), _to_f32(X_te, device), _to_f32(y_te, device), False


def _load_student(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Student Performance — UCI 320, d=2 (G1, G2). dequantize=True (integer grades)."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=320)
    X_df = ds.data.features.copy()
    y_df = ds.data.targets[["G1", "G2"]].copy()
    valid = ~(X_df.isnull().any(axis=1) | y_df.isnull().any(axis=1))
    X_df = X_df[valid].reset_index(drop=True)
    y_df = y_df[valid].reset_index(drop=True)
    cat_names = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    X_df = pd.get_dummies(X_df, columns=cat_names, drop_first=False).astype(np.float32)
    X = X_df.values
    y = y_df.values.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    ohe_cols = [j for j in range(X_tr.shape[1]) if set(np.unique(X_tr[:, j])).issubset({0.0, 1.0})]
    X_tr, X_te = _normalize_X_cols(X_tr, X_te, cat_cols=ohe_cols)
    return _to_f32(X_tr, device), _to_f32(y_tr, device), _to_f32(X_te, device), _to_f32(y_te, device), True


def _load_comms_crime(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Communities & Crime — UCI 211, d=5 (murders, rapes, robberies, assaults, burglaries)."""
    from sklearn.model_selection import train_test_split
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=211)
    X_df = ds.data.features
    num_cols = X_df.select_dtypes(include="number").columns
    X_df = X_df[num_cols].fillna(X_df[num_cols].median())
    target_cols = ["murders", "rapes", "robberies", "assaults", "burglaries"]
    y_df = ds.data.targets[target_cols].dropna()
    X_df = X_df.loc[y_df.index]
    X = X_df.values.astype(np.float32)
    y = y_df.values.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_te = _normalize_X_cols(X_tr, X_te)
    return _to_f32(X_tr, device), _to_f32(y_tr, device), _to_f32(X_te, device), _to_f32(y_te, device), False


# ---------------------------------------------------------------------------
# Real-world copula evaluation (no oracle)
# ---------------------------------------------------------------------------


def _run_realworld_dataset(
    name: str,
    X_train: torch.Tensor,
    Z_train: torch.Tensor,
    X_test: torch.Tensor,
    Z_test: torch.Tensor,
    cfg,
    ICL_model: nn.Module,
    n_steps: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Train AttentionCopulaEstimator and evaluate all copula estimators on one real-world dataset.

    No oracle is available; the sole metric is the Gaussian copula NLL on the held-out test set,
    which is a proper scoring rule that requires no ground-truth correlation matrix.
    """
    torch.manual_seed(seed)

    acfg = cfg.attn_copula
    n_train, p = X_train.shape
    n_test, d  = Z_test.shape

    # ---- Val / pool split ----
    n_val = max(1, int(round(0.2 * n_train)))
    perm = torch.randperm(n_train, device=device)
    val_idx, pool_idx = perm[:n_val], perm[n_val:]
    X_val,  Z_val  = X_train[val_idx],  Z_train[val_idx]
    X_pool, Z_pool = X_train[pool_idx], Z_train[pool_idx]
    n_pool = X_pool.shape[0]

    # ---- Train AttentionCopulaEstimator ----
    model = AttentionCopulaEstimator(
        p=p, d=d,
        m=int(acfg.m),
        n_heads=int(acfg.n_heads),
        n_layers=int(acfg.n_layers),
        dropout=float(acfg.dropout),
        alpha=float(acfg.alpha),
        lambda_min=float(acfg.lambda_min),
        lambda_max=float(acfg.lambda_max),
        include_outer=bool(acfg.include_outer),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=float(acfg.lr), weight_decay=float(acfg.weight_decay))
    warmup_steps = int(acfg.get("warmup_steps", 0))
    cosine_steps = max(n_steps - warmup_steps, 1)
    _cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=float(acfg.lr_min))
    if warmup_steps > 0:
        _warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[_warmup, _cosine], milestones=[warmup_steps])
    else:
        scheduler = _cosine

    best_val_nll = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    steps_since_improvement = 0
    patience  = int(acfg.patience)
    val_every = int(acfg.val_every)
    clip_grad = float(acfg.clip_grad)

    model.train()
    for step in range(n_steps):
        n_sup = max(1, int(round(0.8 * n_pool)))
        perm_pool = torch.randperm(n_pool, device=device)
        X_s, Z_s = X_pool[perm_pool[:n_sup]], Z_pool[perm_pool[:n_sup]]
        X_q, Z_q = X_pool[perm_pool[n_sup:]], Z_pool[perm_pool[n_sup:]]
        R_pred = model(X_s, Z_s, X_q)
        loss = copula_nll_full(Z_q, R_pred)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()

        if step % val_every == 0:
            model.eval()
            with torch.no_grad():
                R_val, _, _ = model(X_pool, Z_pool, X_val, return_gates=True)
                val_nll = copula_nll_full(Z_val, R_val).item()
            model.train()
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_state = copy.deepcopy(model.state_dict())
                steps_since_improvement = 0
            else:
                steps_since_improvement += val_every
            if steps_since_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    # ---- Evaluate all estimators on Z_test ----
    with torch.no_grad():
        R_attn, _, _ = model(X_train, Z_train, X_test, return_gates=True)
    attn_nll = copula_nll_full(Z_test, R_attn).item()

    R_mom = moment_estimator(Z_train)
    moment_nll = copula_nll_full(Z_test, R_mom.unsqueeze(0).expand(n_test, -1, -1)).item()

    R_shrunk, _ = shrinkage_grid_cv(Z_train, Z_val)
    shrunk_nll = copula_nll_full(Z_test, R_shrunk.unsqueeze(0).expand(n_test, -1, -1)).item()

    nw_nll: dict[str, float] = {}
    for kern in ("rbf", "epanechnikov", "laplace"):
        nw_nll[kern] = copula_nll_full(Z_test, nw_corr(X_train, Z_train, X_test, kernel=kern)).item()

    # ICL model — ICLCorrNetV2 handles d < d_max by padding Z_sup internally
    with torch.no_grad():
        X_all = torch.cat([X_train, X_test], dim=0).unsqueeze(0)
        Z_all = torch.cat([Z_train, torch.zeros_like(Z_test)], dim=0).unsqueeze(0)
        _, d_ICL, V_ICL = ICL_model(X_all, Z_all, n_support=n_train)
    d_ICL, V_ICL = d_ICL.squeeze(0), V_ICL.squeeze(0)
    Sigma_ICL = torch.diag_embed(d_ICL) + V_ICL @ V_ICL.transpose(-2, -1)
    std_ICL   = Sigma_ICL.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    R_ICL     = Sigma_ICL / (std_ICL.unsqueeze(-1) * std_ICL.unsqueeze(-2))
    ICL_nll   = copula_nll_full(Z_test, R_ICL).item()

    return {
        "copula_nll/moment":        moment_nll,
        "copula_nll/shrunk_moment": shrunk_nll,
        "copula_nll/nw_rbf":        nw_nll["rbf"],
        "copula_nll/nw_epan":       nw_nll["epanechnikov"],
        "copula_nll/nw_laplace":    nw_nll["laplace"],
        "copula_nll/attn":          attn_nll,
        "copula_nll/ICL":           ICL_nll,
    }


def _print_realworld_table(results: dict[str, dict[str, float]]) -> None:
    """Print copula NLL comparison table across real-world datasets."""
    _METHODS = [
        ("moment",        "Moment"),
        ("shrunk_moment", "ShrunkMoment"),
        ("nw_rbf",        "NW-RBF"),
        ("nw_epan",       "NW-Epan"),
        ("nw_laplace",    "NW-Lap"),
        ("attn",          "Attention"),
        ("ICL",           "ICL model"),
    ]
    ds_names = list(results.keys())
    col_w = 14
    total_w = 22 + col_w * len(ds_names)

    print(f"\n  {'─' * total_w}")
    print(f"  Real-world copula NLL (z-space) — lower is better")
    print(f"  {'─' * total_w}")
    print(f"  {'Method':<20}" + "".join(f"{n:>{col_w}}" for n in ds_names))
    print(f"  {'─' * 20}" + "─" * (col_w * len(ds_names)))
    for key, label in _METHODS:
        row = f"  {label:<20}"
        for ds in ds_names:
            v = results[ds].get(f"copula_nll/{key}", float("nan"))
            row += f"{v:>{col_w}.4f}"
        print(row)
    print(f"  {'─' * total_w}\n")


# ---------------------------------------------------------------------------
# AttentionCopulaEstimator
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SelfAttentionBlock(nn.Module):
    """Pre-norm multi-head self-attention + FFN (permutation-equivariant)."""

    def __init__(self, m: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(m)
        self.norm2 = nn.LayerNorm(m)
        self.attn = nn.MultiheadAttention(m, n_heads, dropout=dropout, batch_first=True)
        d_ff = max(round(8 / 3 * m / 64) * 64, 64)
        self.ff = nn.Sequential(
            nn.Linear(m, d_ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_ff, m)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n, m)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class _CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention: query attends to context."""

    def __init__(self, m: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(m)
        self.norm_kv = nn.LayerNorm(m)
        self.attn = nn.MultiheadAttention(m, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: (B, n_q, m),  kv: (B, n_ctx, m)
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        h, _ = self.attn(q_n, kv_n, kv_n, need_weights=False)
        return q + self.drop(h)


class AttentionCopulaEstimator(nn.Module):
    """Amortized set-to-correlation operator.

    T_φ(x*, {(x_i, z_i)}) → R* ∈ R_d (valid correlation matrix)

    Implements the residual formula:
        R* = (1 - λ*) [(1 - γ*) R_mom + γ* R_attn] + λ* I

    where R_mom is the empirical moment estimator and R_attn comes from
    a Set Transformer + cross-attention + row-normalized Cholesky head.
    """

    def __init__(
        self,
        p: int,
        d: int,
        m: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
        alpha: float = 3.0,
        lambda_min: float = 1e-4,
        lambda_max: float = 0.9,
        include_outer: bool = True,
    ) -> None:
        super().__init__()
        self.d = d
        self.m = m
        self.alpha = alpha
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.include_outer = include_outer

        n_offdiag = d * (d - 1) // 2

        # Feature encoder g_x: shared for context rows and queries
        self.g_x = _MLP(p, m, m, dropout)

        # Row encoder: [g_x(x_i), z_i, vech(z_i z_i^T)?] -> m
        row_in = m + d + (d * (d + 1) // 2 if include_outer else 0)
        self.row_enc = _MLP(row_in, m, m, dropout)

        # Self-attention over context rows
        self.self_attn_blocks = nn.ModuleList(
            [_SelfAttentionBlock(m, n_heads, dropout) for _ in range(n_layers)]
        )

        # Query projection: g_x(x*) -> query embedding
        self.W_q = nn.Linear(m, m)

        # Cross-attention: query attends to encoded context
        self.cross_attn = _CrossAttentionBlock(m, n_heads, dropout)

        # Output head: h* -> (a_raw, eta_lambda, eta_gamma)
        self.head = nn.Linear(m, n_offdiag + 2)

        self._init_weights()

    def _init_weights(self) -> None:
        # Cholesky head: initialize near zero so R_attn ≈ I at start
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # γ* ≈ 0.05 at start (rely on moment estimator early in training)
        self.head.bias.data[-1] = -3.0
        # λ* ≈ midpoint at start (mild shrinkage)
        # eta_lambda at 0 → σ(0) = 0.5 → λ* = 0.5*(lambda_min + lambda_max)
        self.head.bias.data[-2] = 0.0

    def _cholesky_corr(self, a: torch.Tensor) -> torch.Tensor:
        """Row-normalized Cholesky: a (..., n_offdiag) -> R (..., d, d).

        Builds lower-triangular L with unit row norms, then R = L L^T.
        Guarantees R_ii = 1 and R ≻ 0.
        """
        d = self.d
        *batch, _ = a.shape
        L = torch.zeros(*batch, d, d, dtype=a.dtype, device=a.device)
        idx = 0
        for j in range(d):
            if j == 0:
                L[..., j, j] = 1.0
            else:
                off = a[..., idx : idx + j]  # (..., j)
                idx += j
                sq_sum = off.pow(2).sum(-1)  # (...)
                s = (1.0 + sq_sum).sqrt()
                L[..., j, j] = 1.0 / s
                L[..., j, :j] = off / s.unsqueeze(-1)
        R = L @ L.transpose(-2, -1)
        return R

    def forward(
        self,
        X_ctx: torch.Tensor,
        Z_ctx: torch.Tensor,
        X_qry: torch.Tensor,
        return_gates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            X_ctx : (B, n, p) or (n, p) — context features
            Z_ctx : (B, n, d) or (n, d) — context z-scores
            X_qry : (B, m, p) or (m, p) — query features
            return_gates: if True, also return (gamma, lambda) tensors
        Returns:
            R : (B, m, d, d) or (m, d, d) — valid correlation matrices
        """
        squeeze = X_ctx.dim() == 2
        if squeeze:
            X_ctx = X_ctx.unsqueeze(0)
            Z_ctx = Z_ctx.unsqueeze(0)
            X_qry = X_qry.unsqueeze(0)

        B, n, p = X_ctx.shape
        m_q = X_qry.shape[1]
        d = self.d

        # ---- Moment estimator R_mom per batch element ----
        R_mom_list = []
        for b in range(B):
            R_mom_list.append(moment_estimator(Z_ctx[b]))
        R_mom = torch.stack(R_mom_list, dim=0)  # (B, d, d)
        R_mom = R_mom.unsqueeze(1).expand(B, m_q, d, d)  # (B, m, d, d)

        # ---- Encode context rows ----
        ex = self.g_x(X_ctx.reshape(B * n, p)).reshape(B, n, self.m)  # (B, n, m)

        if self.include_outer:
            outer = Z_ctx.unsqueeze(-1) * Z_ctx.unsqueeze(-2)  # (B, n, d, d)
            idx_i, idx_j = torch.tril_indices(d, d, offset=0, device=Z_ctx.device)
            vech = outer[..., idx_i, idx_j]  # (B, n, d*(d+1)//2)
            row_in = torch.cat([ex, Z_ctx, vech], dim=-1)  # (B, n, m+d+d*(d+1)//2)
        else:
            row_in = torch.cat([ex, Z_ctx], dim=-1)  # (B, n, m+d)

        r = self.row_enc(row_in.reshape(B * n, -1)).reshape(B, n, self.m)  # (B, n, m)

        # ---- Self-attention over context ----
        for block in self.self_attn_blocks:
            r = block(r)

        # ---- Query cross-attention ----
        eq = self.g_x(X_qry.reshape(B * m_q, p)).reshape(B, m_q, self.m)  # (B, m, m)
        q_emb = self.W_q(eq)  # (B, m, m)
        h = self.cross_attn(q_emb, r)  # (B, m, m)

        # ---- Output head ----
        n_offdiag = d * (d - 1) // 2
        out = self.head(h)  # (B, m, n_offdiag+2)
        a_raw = out[..., :n_offdiag]
        eta_lambda = out[..., -2]
        eta_gamma = out[..., -1]

        # ---- Cholesky correlation head ----
        a_scaled = self.alpha * torch.tanh(a_raw)
        R_attn = self._cholesky_corr(a_scaled)  # (B, m, d, d)

        # ---- Scalar gates ----
        lam = self.lambda_min + (self.lambda_max - self.lambda_min) * torch.sigmoid(
            eta_lambda
        )  # (B, m)
        gamma = torch.sigmoid(eta_gamma)  # (B, m)

        # ---- Mix and shrink ----
        I = torch.eye(d, dtype=R_attn.dtype, device=R_attn.device)
        I = I.view(1, 1, d, d).expand(B, m_q, d, d)

        lam_ = lam.unsqueeze(-1).unsqueeze(-1)
        gam_ = gamma.unsqueeze(-1).unsqueeze(-1)

        R_mix = (1.0 - gam_) * R_mom + gam_ * R_attn
        R_out = (1.0 - lam_) * R_mix + lam_ * I

        if squeeze:
            R_out = R_out.squeeze(0)
            if return_gates:
                return R_out, gamma.squeeze(0), lam.squeeze(0)
            return R_out

        if return_gates:
            return R_out, gamma, lam
        return R_out


# ---------------------------------------------------------------------------
# Worker: train + eval one dataset (runs in a thread on the shared device)
# ---------------------------------------------------------------------------


def _run_dataset(
    ds_idx: int,
    ep_path: str,
    cfg: object,
    ICL_model: nn.Module,
    n_steps: int,
    seed: int,
    device: torch.device,
) -> tuple[int, dict[str, float]]:
    """Train AttentionCopulaEstimator and evaluate all methods for one episode.

    Shares the pre-loaded ICL_model (eval-only, no grad) with other threads.
    Returns (ds_idx, metrics_dict).
    """
    torch.manual_seed(seed + ds_idx)

    acfg = cfg.attn_copula

    # ---- Load episode -------------------------------------------------------
    ep = torch.load(ep_path, map_location=device, weights_only=True)
    X_train = ep["X_train"][0].to(device)  # (256, p)
    Z_train = ep["Z_train"][0].to(device)  # (256, d)
    X_test  = ep["X_test"][0].to(device)   # (n_te, p)
    Z_test  = ep["Z_test"][0].to(device)   # (n_te, d)
    log_p_test = ep["log_p_test"][0].to(device)
    Y_test     = ep["Y_test"][0].to(device)
    oracle_mu  = ep["oracle_mu"][0].to(device)
    oracle_D   = ep["oracle_D"][0].to(device)
    oracle_V   = ep["oracle_V"][0].to(device)

    n_train, p = X_train.shape
    n_test,  d = Z_test.shape

    # ---- Fixed data splits ------------------------------------------------
    # 20% held-out val from training set (fixed once for the whole run)
    n_val = max(1, int(round(0.2 * n_train)))
    perm = torch.randperm(n_train, device=device)
    val_idx  = perm[:n_val]
    pool_idx = perm[n_val:]

    X_val,  Z_val  = X_train[val_idx],  Z_train[val_idx]   # (~51, .)
    X_pool, Z_pool = X_train[pool_idx], Z_train[pool_idx]  # (~205, .)
    n_pool = X_pool.shape[0]

    # ====================================================================
    # Phase 1 — Train AttentionCopulaEstimator
    # ====================================================================
    model = AttentionCopulaEstimator(
        p=p,
        d=d,
        m=int(acfg.m),
        n_heads=int(acfg.n_heads),
        n_layers=int(acfg.n_layers),
        dropout=float(acfg.dropout),
        alpha=float(acfg.alpha),
        lambda_min=float(acfg.lambda_min),
        lambda_max=float(acfg.lambda_max),
        include_outer=bool(acfg.include_outer),
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=float(acfg.lr),
        weight_decay=float(acfg.weight_decay),
    )
    warmup_steps = int(acfg.get("warmup_steps", 0))
    cosine_steps = max(n_steps - warmup_steps, 1)
    _cosine = CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=float(acfg.lr_min)
    )
    if warmup_steps > 0:
        _warmup = LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[_warmup, _cosine], milestones=[warmup_steps]
        )
    else:
        scheduler = _cosine

    best_val_nll = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    steps_since_improvement = 0
    patience  = int(acfg.patience)
    val_every = int(acfg.val_every)
    clip_grad = float(acfg.clip_grad)

    model.train()
    for step in range(n_steps):
        # 80/20 split of train pool
        n_sup = max(1, int(round(0.8 * n_pool)))
        perm_pool = torch.randperm(n_pool, device=device)
        sup_idx = perm_pool[:n_sup]
        qry_idx = perm_pool[n_sup:]

        X_s, Z_s = X_pool[sup_idx], Z_pool[sup_idx]
        X_q, Z_q = X_pool[qry_idx], Z_pool[qry_idx]

        R_pred = model(X_s, Z_s, X_q)  # (n_qry, d, d)
        loss = copula_nll_full(Z_q, R_pred)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()

        if step % val_every == 0:
            model.eval()
            with torch.no_grad():
                R_val, _, _ = model(X_pool, Z_pool, X_val, return_gates=True)
                val_nll = copula_nll_full(Z_val, R_val).item()
            model.train()

            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_state = copy.deepcopy(model.state_dict())
                steps_since_improvement = 0
            else:
                steps_since_improvement += val_every

            if steps_since_improvement >= patience:
                break

    # Restore best checkpoint
    model.load_state_dict(best_state)
    model.eval()

    # ====================================================================
    # Phase 2 — Evaluate all estimators on Z_test
    # ====================================================================
    with torch.no_grad():
        # Attention estimator: full Z_train as context
        R_attn, gamma_final, lam_final = model(
            X_train, Z_train, X_test, return_gates=True
        )
        attn_nll = copula_nll_full(Z_test, R_attn).item()
        cond_nums = torch.linalg.cond(R_attn)  # (n_te,)
        gamma_f = gamma_final.mean().item()
        lam_f = lam_final.mean().item()

    # Moment estimator (no neural component, global)
    R_mom_full = moment_estimator(Z_train)  # (d, d)
    R_mom_exp = R_mom_full.unsqueeze(0).expand(n_test, -1, -1)  # (n_te, d, d)
    moment_nll = copula_nll_full(Z_test, R_mom_exp).item()

    # Shrunk moment: λ chosen by CV on val set
    R_shrunk, best_lam_cv = shrinkage_grid_cv(Z_train, Z_val)
    R_shrunk_exp = R_shrunk.unsqueeze(0).expand(n_test, -1, -1)
    shrunk_nll = copula_nll_full(Z_test, R_shrunk_exp).item()

    # Nadaraya-Watson baselines (keep R tensors for plotting)
    nw_nll: dict[str, float] = {}
    nw_R: dict[str, torch.Tensor] = {}
    for kern in ("rbf", "epanechnikov", "laplace"):
        R_nw = nw_corr(X_train, Z_train, X_test, kernel=kern)
        nw_nll[kern] = copula_nll_full(Z_test, R_nw).item()
        nw_R[kern] = R_nw

    # ====================================================================
    # Phase 3 — Pretrained ICL model
    # ====================================================================
    with torch.no_grad():
        X_all = torch.cat([X_train, X_test], dim=0).unsqueeze(0)
        Z_all = torch.cat([Z_train, torch.zeros_like(Z_test)], dim=0).unsqueeze(0)
        mu_ICL, d_ICL, V_ICL = ICL_model(X_all, Z_all, n_support=n_train)
        mu_ICL = mu_ICL.squeeze(0)
        d_ICL  = d_ICL.squeeze(0)
        V_ICL  = V_ICL.squeeze(0)

    marginal_nll_te = -log_p_test.sum(dim=-1).mean().item()

    # Convert Woodbury params (D, V) → full correlation matrix: R = Corr(diag(D) + VV^T)
    def _woodbury_to_corr(D: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """(n, d), (n, d, r) -> (n, d, d) correlation matrices."""
        Sigma = torch.diag_embed(D) + V @ V.transpose(-2, -1)  # (n, d, d)
        std = Sigma.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()  # (n, d)
        R = Sigma / (std.unsqueeze(-1) * std.unsqueeze(-2))
        return R

    R_ICL    = _woodbury_to_corr(d_ICL, V_ICL)   # (n_te, d, d)
    R_oracle = _woodbury_to_corr(oracle_D, oracle_V)  # (n_te, d, d)

    # Note on "oracle" semantics in this script:
    #   * pseudo_oracle_corrY_copula_nll = copula_nll_full(Z_test, corr(Σ_Y)) is *not* a hard
    #     lower bound. Z_test was produced by PIT with TabICL's *estimated* marginals,
    #     so Z_test is not exactly N(0, corr(Σ_Y)); an in-context model can legitimately
    #     beat this number by undoing TabICL's marginal distortion or by conditioning
    #     on x*. We keep it as a reference under the "corr(Σ_Y)" Gaussian-copula model.
    #   * The only true lower bound on joint Y-NLL is woodbury_nll(Y_test, μ_Y, D_Y, V_Y),
    #     the negative log of the data-generating density. That is `nll_oracle_y` below.
    nll_ICL_z = copula_nll_full(Z_test, R_ICL).item()
    nll_pseudo_oracle_corrY_z = copula_nll_full(Z_test, R_oracle).item()
    nll_ICL_y = nll_ICL_z + marginal_nll_te

    # True Y-space oracle (lower bound on joint-Y NLL). Computed directly from the
    # data-generating Gaussian on Y_test — bypasses PIT, so TabICL's marginal
    # estimation error does not enter.
    nll_oracle_y = woodbury_nll(
        Y_test.unsqueeze(0),
        oracle_mu.unsqueeze(0),
        oracle_D.unsqueeze(0),
        oracle_V.unsqueeze(0),
    ).item()

    # ====================================================================
    # Three-way error decomposition
    # ====================================================================
    # Total excess = A (marginal error) + B (pure copula error) + C (bleed)
    #
    #   A = marginal_nll_te - marginal_nll_oracle
    #         → how well TabICL estimates the marginal densities
    #
    #   B = copula_nll(R_ICL, Z_true) - copula_nll(R_oracle, Z_true)  ≥ 0
    #         → pure copula error; KL(N(0,R_oracle) || N(0,R_ICL)) in expectation
    #         → zero only when ICL predicts R_oracle exactly
    #
    #   C = copula_nll(R_ICL, Z_hat) - copula_nll(R_ICL, Z_true)
    #         → how much TabICL's marginal distortion inflates the copula NLL
    #         → ≈ 0 when marginal estimation is accurate
    #
    # Z_true  — PIT with oracle (true) marginals: z*_j = (y_j - mu_j) / sigma_j
    # Z_hat   — current Z_test: PIT with TabICL's estimated marginals
    #
    # Sanity: A + B + C = (nll_ICL_z + marginal_nll_te) - (nll_oracle_copula + marginal_nll_oracle)
    #                   = ICL_joint_nll_y - (oracle_copula_floor + oracle_marginal_floor)

    # Oracle marginal std:  sigma_j^2 = D_j + ||V_j||^2
    sigma2_ora = oracle_D + oracle_V.pow(2).sum(-1)       # (n_te, d)
    sigma_ora  = sigma2_ora.clamp(min=1e-8).sqrt()

    Z_true = (Y_test - oracle_mu) / sigma_ora             # (n_te, d)

    marginal_nll_oracle = 0.5 * (
        math.log(2.0 * math.pi)
        + sigma2_ora.log()
        + (Y_test - oracle_mu).pow(2) / sigma2_ora
    ).sum(-1).mean().item()

    nll_oracle_copula_truez = copula_nll_full(Z_true, R_oracle).item()  # floor
    nll_ICL_truez           = copula_nll_full(Z_true, R_ICL).item()

    err_A_marginal     = marginal_nll_te    - marginal_nll_oracle   # TabICL marginal error
    err_B_copula_pure  = nll_ICL_truez      - nll_oracle_copula_truez  # KL(R_oracle||R_ICL) ≥ 0
    err_C_bleed        = nll_ICL_z          - nll_ICL_truez           # marginal→copula bleed

    # ====================================================================
    # Phase 4 — Collect metrics
    # ====================================================================
    # joint_nll_y = copula_nll_z + marginal_nll_te  (same decomposition as ICL)
    # marginal_nll_te is the same for all copula methods (it depends only on marginals)
    eval_metrics = {
        # ---- Copula NLL (z-space, copula component only) --------------------
        "eval/moment_copula_nll": moment_nll,
        "eval/shrunk_moment_copula_nll": shrunk_nll,
        "eval/nw_rbf_copula_nll": nw_nll["rbf"],
        "eval/nw_epan_copula_nll": nw_nll["epanechnikov"],
        "eval/nw_laplace_copula_nll": nw_nll["laplace"],
        "eval/attn_copula_nll": attn_nll,
        "eval/ICL_copula_nll": nll_ICL_z,
        # corr(Σ_Y) Gaussian copula on PIT-Z — NOT a hard lower bound (see note above).
        "eval/pseudo_oracle_corrY_copula_nll": nll_pseudo_oracle_corrY_z,
        # ---- Joint NLL (y-space) = copula_nll + marginal_nll_te ------------
        "eval/moment_joint_nll_y": moment_nll + marginal_nll_te,
        "eval/shrunk_moment_joint_nll_y": shrunk_nll + marginal_nll_te,
        "eval/nw_rbf_joint_nll_y": nw_nll["rbf"] + marginal_nll_te,
        "eval/nw_epan_joint_nll_y": nw_nll["epanechnikov"] + marginal_nll_te,
        "eval/nw_laplace_joint_nll_y": nw_nll["laplace"] + marginal_nll_te,
        "eval/attn_joint_nll_y": attn_nll + marginal_nll_te,
        "eval/ICL_joint_nll_y": nll_ICL_y,
        # Sklar route: biased by TabICL's marginal density (log_p_test) — not tight.
        "eval/pseudo_oracle_corrY_joint_nll_y": nll_pseudo_oracle_corrY_z + marginal_nll_te,
        # True Y-space oracle: hard lower bound on any joint-Y NLL.
        "eval/oracle_joint_nll_y": nll_oracle_y,
        # ---- Shared marginal term (same for all copula methods) -------------
        "eval/marginal_nll_te": marginal_nll_te,
        # ---- Three-way error decomposition ----------------------------------
        # ICL copula NLL evaluated on oracle-PIT Z_true (removes marginal distortion)
        "eval/ICL_copula_nll_truez":      nll_ICL_truez,
        # Oracle copula floor on Z_true  (= 0.5*log|R_oracle| in expectation)
        "eval/oracle_copula_nll_truez":   nll_oracle_copula_truez,
        # True oracle marginal NLL (lower bound on marginal term)
        "eval/marginal_nll_oracle":       marginal_nll_oracle,
        # A: TabICL marginal estimation error (independent of copula model)
        "eval/err_A_marginal":            err_A_marginal,
        # B: pure ICL copula error = KL(N(0,R_oracle)||N(0,R_ICL)) in expectation  (≥ 0)
        "eval/err_B_copula_pure":         err_B_copula_pure,
        # C: how much TabICL's marginal distortion inflates the copula NLL
        "eval/err_C_marginal_bleed":      err_C_bleed,
        # ---- Diagnostics ----------------------------------------------------
        "eval/attn_R_cond_num_mean": cond_nums.mean().item(),
        "eval/attn_gamma_final": gamma_f,
        "eval/attn_lambda_final": lam_f,
        "eval/shrunk_moment_best_lambda": best_lam_cv,
    }

    # ====================================================================
    # Xi-conditioning table — does each model's covariance vary with xi?
    # ====================================================================
    # For correlation-matrix models: std_offdiag_xi > 0 ↔ model is xi-conditioned.
    # For Woodbury (D, V) models: std_diag_var_xi > 0 ↔ marginal variance varies with xi.
    # Moment / ShrunkMoment give a single global R → xi-invariant (std=0 by construction).

    # Broadcast global R matrices to (n_test, d, d) for uniform API
    R_mom_exp_stat = R_mom_full.unsqueeze(0).expand(n_test, -1, -1)
    R_shrunk_exp_stat = R_shrunk.unsqueeze(0).expand(n_test, -1, -1)

    xi_corr_models: dict[str, torch.Tensor] = {
        "moment":     R_mom_exp_stat,
        "shrunk_mom": R_shrunk_exp_stat,
        "nw_rbf":     nw_R["rbf"],
        "nw_epan":    nw_R["epanechnikov"],
        "nw_laplace": nw_R["laplace"],
        "attn":       R_attn,
        "ICL":        R_ICL,
        "oracle":     R_oracle,
    }
    xi_corr_stats: dict[str, dict[str, float]] = {
        name: _xi_cond_stats(R) for name, R in xi_corr_models.items()
    }
    # Woodbury (D, V) models have actual diagonal variances per xi
    xi_wb_stats: dict[str, dict[str, float]] = {
        "oracle": _woodbury_diag_stats(oracle_D, oracle_V),
        "ICL":    _woodbury_diag_stats(d_ICL, V_ICL),
    }

    for name, s in xi_corr_stats.items():
        for k, v in s.items():
            eval_metrics[f"xi_cond/{name}_{k}"] = v
    for name, s in xi_wb_stats.items():
        for k, v in s.items():
            eval_metrics[f"xi_cond/{name}_{k}"] = v

    return ds_idx, eval_metrics, (
        ep, R_mom_full, R_shrunk, nw_R, R_attn, R_oracle,
        mu_ICL, d_ICL, V_ICL, R_ICL, Z_train, X_train,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel per-dataset AttentionCopulaEstimator vs pretrained ICL model"
    )
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--episode_idx", type=int, default=0,
                        help="Starting episode index; comparison.n_datasets episodes are used")
    parser.add_argument(
        "--steps", type=int, default=None, help="Override attn_copula.steps"
    )
    parser.add_argument("--wandb_project", default="copula-attn-vs-ICL")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_workers", type=int, default=None,
        help="Concurrent threads per device (default: 4 on GPU, cpu_count on CPU)"
    )
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip covariance plot generation (faster for testing)")
    parser.add_argument("--tabicl_ckpt", default=None,
                        help="TabICL checkpoint name for real-world PIT (default: cfg.tabicl.ckpt)")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    print(f"Device: {device}")

    cfg = OmegaConf.load(args.config)
    acfg = cfg.attn_copula
    n_comparison_datasets = int(cfg.get("comparison", {}).get("n_datasets", 50))
    n_steps = args.steps if args.steps is not None else int(acfg.steps)
    dataset_dir = cfg.training.dataset_dir

    # Default: 4 concurrent threads on GPU (small models fit easily), cpu_count on CPU
    if args.n_workers is not None:
        n_workers = args.n_workers
    elif device.type == "cuda":
        n_workers = 4
    else:
        n_workers = os.cpu_count() or 4

    print(f"Comparing {n_comparison_datasets} datasets | {n_workers} concurrent threads on {device}")

    # ---- Load ICL model once (shared read-only across threads) -------------
    print(f"\n--- Loading ICL model from {args.ckpt} ---")
    ICL_ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ICL_cfg  = ICL_ckpt.get("cfg", cfg)
    if isinstance(ICL_cfg, dict):
        ICL_cfg = OmegaConf.create(ICL_cfg)

    ICL_model = _build_ICL_model(ICL_cfg).to(device)
    ICL_model.load_state_dict(ICL_ckpt["model_state"])
    ICL_model.eval()
    n_ICL_params = sum(p_.numel() for p_ in ICL_model.parameters())
    print(f"ICL model parameters: {n_ICL_params:,}")

    # Warm-up: trigger lazy initialisations of torch.linalg and ICL_model before
    # threads start.  torch.linalg ops (cholesky, cond) use a lazy-wrapper that
    # raises "should be called at most once" if two threads hit it simultaneously.
    with torch.no_grad():
        _d = ICL_ckpt["cfg"].model.d if hasattr(ICL_ckpt.get("cfg", {}), "model") else 8
        _dummy = torch.eye(_d, device=device).unsqueeze(0)
        torch.linalg.cholesky(_dummy)
        torch.linalg.cond(_dummy)
        del _dummy
    _ep0 = torch.load(
        os.path.join(cfg.training.dataset_dir, f"episode_{args.episode_idx:06d}.pt"),
        map_location=device, weights_only=True,
    )
    with torch.no_grad():
        _X = torch.cat([_ep0["X_train"][0], _ep0["X_test"][0]], dim=0).to(device).unsqueeze(0)
        _Z = torch.cat([_ep0["Z_train"][0], torch.zeros_like(_ep0["Z_test"][0])], dim=0).to(device).unsqueeze(0)
        ICL_model(_X, _Z, n_support=_ep0["X_train"].shape[1])
    del _ep0, _X, _Z

    # ---- TabICL (for real-world PIT) ----------------------------------------
    tabicl_ckpt_name = args.tabicl_ckpt or str(cfg.tabicl.ckpt)
    print(f"\n--- Loading TabICL ({tabicl_ckpt_name}) for real-world PIT ---")
    tabicl_model = load_tabicl(tabicl_ckpt_name, device)

    # ---- W&B ---------------------------------------------------------------
    wandb_run = None
    try:
        import wandb

        run_name = args.wandb_name or (
            f"N{n_comparison_datasets}_m{acfg.m}_L{acfg.n_layers}_drop{acfg.dropout}"
        )
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                **dict(OmegaConf.to_container(acfg, resolve=True)),
                "n_comparison_datasets": n_comparison_datasets,
                "episode_start": args.episode_idx,
            },
        )
        print(f"W&B run: {wandb_run.url}")
    except Exception as e:
        print(f"W&B unavailable ({e}), logging to console only.")

    # ====================================================================
    # Parallel training and evaluation across datasets
    # ====================================================================
    episode_indices = list(range(args.episode_idx, args.episode_idx + n_comparison_datasets))
    all_metrics: list[dict[str, float]] = [{}] * n_comparison_datasets
    plot_artifacts = None  # kept only from dataset 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _run_dataset,
                ds_idx=i,
                ep_path=os.path.join(dataset_dir, f"episode_{ep_i:06d}.pt"),
                cfg=cfg,
                ICL_model=ICL_model,
                n_steps=n_steps,
                seed=args.seed,
                device=device,
            ): i
            for i, ep_i in enumerate(episode_indices)
        }

        completed = 0
        for future in as_completed(futures):
            local_i = futures[future]
            try:
                ds_idx, metrics, artifacts = future.result()
            except Exception as exc:
                import traceback
                print(f"[dataset {local_i}] FAILED: {exc}")
                traceback.print_exc()
                continue

            all_metrics[ds_idx] = metrics
            if ds_idx == 0:
                plot_artifacts = artifacts
            completed += 1

            # Log per-dataset metrics to W&B
            if wandb_run is not None:
                prefixed = {f"ds{ds_idx:03d}/{k}": v for k, v in metrics.items()}
                wandb_run.log(prefixed, step=ds_idx)

            attn_nll = metrics.get("eval/attn_copula_nll", float("nan"))
            ICL_nll  = metrics.get("eval/ICL_copula_nll",  float("nan"))
            print(
                f"[{completed:>3d}/{n_comparison_datasets}] ds={ds_idx:03d}"
                f"  attn={attn_nll:.4f}  ICL={ICL_nll:.4f}"
            )

    # ====================================================================
    # Aggregate metrics across datasets
    # ====================================================================
    valid = [m for m in all_metrics if m]
    if not valid:
        print("No datasets completed successfully.")
        return

    metric_keys = list(valid[0].keys())
    agg_metrics: dict[str, float] = {}
    for k in metric_keys:
        vals = [m[k] for m in valid if k in m]
        if vals:
            agg_metrics[f"agg/mean_{k}"] = float(np.mean(vals))
            agg_metrics[f"agg/std_{k}"]  = float(np.std(vals))

    print(f"\n--- Aggregate summary over {len(valid)} datasets ---")
    copula_keys = [k for k in agg_metrics if "copula_nll" in k and k.startswith("agg/mean")]
    joint_keys  = [k for k in agg_metrics if "joint_nll_y" in k and k.startswith("agg/mean")]
    col_w = max((len(k) for k in copula_keys + joint_keys), default=0) + 2
    for header, keys in [
        ("Copula NLL (z-space) — mean ± std", copula_keys),
        ("Joint  NLL (y-space) — mean ± std", joint_keys),
    ]:
        print(f"\n  [{header}]")
        for k in keys:
            k_std  = k.replace("agg/mean_", "agg/std_")
            mean_v = agg_metrics[k]
            std_v  = agg_metrics.get(k_std, float("nan"))
            print(f"  {k:<{col_w}}: {mean_v:.4f} ± {std_v:.4f}")

    if wandb_run is not None:
        wandb_run.log(agg_metrics, step=n_comparison_datasets)

    # ====================================================================
    # Phase 5a — Real-world dataset evaluation (TabICL PIT)
    # ====================================================================
    _RW_LOADERS = [
        ("ENB",        _load_enb),
        ("StudentMat", _load_student),
        ("CommsCrime", _load_comms_crime),
    ]
    rw_results: dict[str, dict[str, float]] = {}
    print("\n--- Real-world dataset evaluation (TabICL PIT) ---")
    try:
        for ds_name, loader_fn in _RW_LOADERS:
            print(f"  [{ds_name}] loading dataset and running PIT...")
            X_tr, Y_tr, X_te, Y_te, dequantize = loader_fn(device)
            Z_tr, Z_te, _ = run_pit(
                tabicl_model, X_tr, Y_tr, X_te, Y_te,
                pit_batch_size=int(cfg.tabicl.pit_batch_size),
                eps=float(cfg.tabicl.pit_eps),
                dequantize=dequantize,
            )
            print(f"  [{ds_name}] training AttentionCopulaEstimator and evaluating...")
            rw_results[ds_name] = _run_realworld_dataset(
                ds_name, X_tr, Z_tr, X_te, Z_te,
                cfg, ICL_model, n_steps, args.seed, device,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {f"realworld/{ds_name}/{k}": v for k, v in rw_results[ds_name].items()},
                    step=n_comparison_datasets,
                )
        _print_realworld_table(rw_results)
    except ImportError as exc:
        print(f"  Skipped real-world datasets: missing dependency ({exc})")
        print("  Install with: conda run -n multivariate-icl pip install ucimlrepo scikit-learn")
    except Exception as exc:
        import traceback
        print(f"  Real-world evaluation failed: {exc}")
        traceback.print_exc()

    # ====================================================================
    # Phase 5 — Covariance plots (dataset 0 only)
    # ====================================================================
    if not args.no_plots and plot_artifacts is not None:
        import matplotlib.pyplot as plt
        from sklearn.covariance import OAS

        print("\n--- Generating covariance plots (dataset 0) ---")
        ep, R_mom_full, R_shrunk, nw_R, R_attn, R_oracle, \
            mu_ICL, d_ICL, V_ICL, R_ICL, Z_train, X_train = plot_artifacts

        # All estimators in ablation order
        estimators = {
            "Moment":      R_mom_full,            # (d, d) — broadcast
            "ShrunkMom":   R_shrunk,              # (d, d) — broadcast
            "NW-RBF":      nw_R["rbf"],           # (n_te, d, d)
            "NW-Epan":     nw_R["epanechnikov"],  # (n_te, d, d)
            "NW-Lap":      nw_R["laplace"],       # (n_te, d, d)
            "Attention":   R_attn,                # (n_te, d, d)
            "ICL model":   R_ICL,                 # (n_te, d, d)
        }

        n_test = R_attn.shape[0]
        fig_grid = plot_corr_grid(
            estimators=estimators,
            oracle_R=R_oracle,
            title=f"Correlation estimator comparison — episode {args.episode_idx}",
        )

        # Also keep the detailed ICL Woodbury plot for reference
        oas_z = OAS().fit(Z_train.cpu().numpy())
        oracle_groups_raw = ep.get("oracle_groups", None)  # (B, n_test) or None
        groups_b0 = oracle_groups_raw[0] if oracle_groups_raw is not None else None
        ICL_instance_indices = select_group_representative_indices(
            groups_b0, max_n=3, n_total=n_test
        )
        oracle_mu = ep["oracle_mu"][0].to(device)
        oracle_D  = ep["oracle_D"][0].to(device)
        oracle_V  = ep["oracle_V"][0].to(device)
        fig_ICL = plot_prediction_comparison(
            mu_pred=mu_ICL.unsqueeze(0),
            D_pred=d_ICL.unsqueeze(0),
            V_pred=V_ICL.unsqueeze(0),
            mu_true=oracle_mu.unsqueeze(0),
            D_true=oracle_D.unsqueeze(0),
            V_true=oracle_V.unsqueeze(0),
            sigma_oas=oas_z.covariance_,
            dataset_label="ICL model — predicted vs oracle (Z-space)",
            instance_indices=ICL_instance_indices,
        )

        if wandb_run is not None:
            import wandb as wandb_mod

            wandb_run.log(
                {
                    "eval/corr_grid":  wandb_mod.Image(fig_grid),
                    "eval/ICL_detail": wandb_mod.Image(fig_ICL),
                },
                step=n_comparison_datasets,
            )
            print("Covariance plots logged to W&B.")

        plt.close(fig_grid)
        plt.close(fig_ICL)

    if wandb_run is not None:
        wandb_run.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
