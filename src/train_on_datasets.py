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
import math
import os
import random
import sys

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


def _build_ct_model(cfg) -> nn.Module:
    mcfg = cfg.model
    if hasattr(mcfg, "n_layers_s1"):
        return build_copula_tabicl_v2(cfg)
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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attention copula estimator vs pretrained CopulaTransformer"
    )
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument(
        "--steps", type=int, default=None, help="Override attn_copula.steps"
    )
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--wandb_project", default="copula-attn-vs-ct")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = (
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    print(f"Device: {device}")

    cfg = OmegaConf.load(args.config)
    acfg = cfg.attn_copula
    dataset_dir = cfg.training.dataset_dir
    ep_path = os.path.join(dataset_dir, f"episode_{args.episode_idx:06d}.pt")
    print(f"Loading episode: {ep_path}")

    ep = torch.load(ep_path, map_location=device, weights_only=True)
    X_train = ep["X_train"][0].to(device)  # (256, p)
    Z_train = ep["Z_train"][0].to(device)  # (256, d)
    X_test = ep["X_test"][0].to(device)  # (n_te, p)
    Z_test = ep["Z_test"][0].to(device)  # (n_te, d)
    log_p_test = ep["log_p_test"][0].to(device)
    Y_test = ep["Y_test"][0].to(device)
    oracle_mu = ep["oracle_mu"][0].to(device)
    oracle_D = ep["oracle_D"][0].to(device)
    oracle_V = ep["oracle_V"][0].to(device)

    n_train, p = X_train.shape
    n_test, d = Z_test.shape
    print(
        f"Episode {args.episode_idx}: n_train={n_train}, n_test={n_test}, p={p}, d={d}"
    )

    # ---- Fixed data splits ------------------------------------------------
    # 20% held-out val from training set (fixed once for the whole run)
    n_val = max(1, int(round(0.2 * n_train)))
    perm = torch.randperm(n_train, device=device)
    val_idx = perm[:n_val]
    pool_idx = perm[n_val:]

    X_val, Z_val = X_train[val_idx], Z_train[val_idx]  # (~51, .)
    X_pool, Z_pool = X_train[pool_idx], Z_train[pool_idx]  # (~205, .)
    n_pool = X_pool.shape[0]

    # ---- W&B ---------------------------------------------------------------
    wandb_run = None
    try:
        import wandb

        run_name = args.wandb_name or (
            f"ep{args.episode_idx}_m{acfg.m}_L{acfg.n_layers}_drop{acfg.dropout}"
        )
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=dict(OmegaConf.to_container(acfg, resolve=True)),
        )
        print(f"W&B run: {wandb_run.url}")
    except Exception as e:
        print(f"W&B unavailable ({e}), logging to console only.")

    # ====================================================================
    # Phase 1 — Train AttentionCopulaEstimator
    # ====================================================================
    n_steps = args.steps if args.steps is not None else int(acfg.steps)
    print(
        f"\n--- Training AttentionCopulaEstimator ({n_steps} steps, early stopping) ---"
    )

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

    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"AttentionCopulaEstimator parameters: {n_params:,}")

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
    patience = int(acfg.patience)
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
                R_val, gamma_val, lam_val = model(
                    X_pool, Z_pool, X_val, return_gates=True
                )
                val_nll = copula_nll_full(Z_val, R_val).item()
                gamma_mean = gamma_val.mean().item()
                lam_mean = lam_val.mean().item()
            model.train()

            improved = val_nll < best_val_nll
            if improved:
                best_val_nll = val_nll
                best_state = copy.deepcopy(model.state_dict())
                steps_since_improvement = 0
            else:
                steps_since_improvement += val_every

            # --- xi-conditioning diagnostics on val set ---
            xi_val = _xi_cond_stats(R_val)

            lr_now = scheduler.get_last_lr()[0]
            print(
                f"[step {step:>5d}]  train={loss.item():.4f}  val={val_nll:.4f}"
                f"{'*' if improved else ' '}  γ={gamma_mean:.3f}  λ={lam_mean:.3f}"
                f"  lr={lr_now:.2e}"
                f"  |ρ|={xi_val['mean_abs_offdiag']:.3f}"
                f"  std_ξ={xi_val['std_offdiag_xi']:.4f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "attn/train_copula_nll": loss.item(),
                        "attn/val_copula_nll": val_nll,
                        "attn/gamma_mean": gamma_mean,
                        "attn/lambda_mean": lam_mean,
                        "attn/lr": lr_now,
                        "attn/val_mean_abs_offdiag": xi_val["mean_abs_offdiag"],
                        "attn/val_std_offdiag_xi": xi_val["std_offdiag_xi"],
                        "attn/val_var_frob_from_I": xi_val["var_frob_from_I"],
                    },
                    step=step,
                )

            if steps_since_improvement >= patience:
                print(
                    f"Early stopping at step {step} (no improvement for {patience} steps)."
                )
                break

    # Restore best checkpoint
    model.load_state_dict(best_state)
    print(f"Restored best checkpoint (val_nll={best_val_nll:.4f})")

    # ====================================================================
    # Phase 2 — Evaluate all estimators on Z_test
    # ====================================================================
    print("\n--- Evaluating all estimators on test set ---")
    model.eval()

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
    # Phase 3 — Pretrained CopulaTransformer
    # ====================================================================
    print(f"\n--- Loading CopulaTransformer from {args.ckpt} ---")
    ct_ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ct_cfg = ct_ckpt.get("cfg", cfg)
    if isinstance(ct_cfg, dict):
        ct_cfg = OmegaConf.create(ct_cfg)

    ct_model = _build_ct_model(ct_cfg).to(device)
    ct_model.load_state_dict(ct_ckpt["model_state"])
    ct_model.eval()
    n_ct_params = sum(p_.numel() for p_ in ct_model.parameters())
    print(f"CopulaTransformer parameters: {n_ct_params:,}")

    with torch.no_grad():
        X_all = torch.cat([X_train, X_test], dim=0).unsqueeze(0)
        Z_all = torch.cat([Z_train, torch.zeros_like(Z_test)], dim=0).unsqueeze(0)
        mu_ct, d_ct, V_ct = ct_model(X_all, Z_all, n_support=n_train)
        mu_ct = mu_ct.squeeze(0)
        d_ct = d_ct.squeeze(0)
        V_ct = V_ct.squeeze(0)

    marginal_nll_te = -log_p_test.sum(dim=-1).mean().item()

    # Convert Woodbury params (D, V) → full correlation matrix: R = Corr(diag(D) + VV^T)
    def _woodbury_to_corr(D: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """(n, d), (n, d, r) -> (n, d, d) correlation matrices."""
        Sigma = torch.diag_embed(D) + V @ V.transpose(-2, -1)  # (n, d, d)
        std = Sigma.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()  # (n, d)
        R = Sigma / (std.unsqueeze(-1) * std.unsqueeze(-2))
        return R

    R_ct = _woodbury_to_corr(d_ct, V_ct)  # (n_te, d, d)
    R_oracle = _woodbury_to_corr(oracle_D, oracle_V)  # (n_te, d, d)

    # Note on "oracle" semantics in this script:
    #   * pseudo_oracle_corrY_copula_nll = copula_nll_full(Z_test, corr(Σ_Y)) is *not* a hard
    #     lower bound. Z_test was produced by PIT with TabICL's *estimated* marginals,
    #     so Z_test is not exactly N(0, corr(Σ_Y)); an in-context model can legitimately
    #     beat this number by undoing TabICL's marginal distortion or by conditioning
    #     on x*. We keep it as a reference under the "corr(Σ_Y)" Gaussian-copula model.
    #   * The only true lower bound on joint Y-NLL is woodbury_nll(Y_test, μ_Y, D_Y, V_Y),
    #     the negative log of the data-generating density. That is `nll_oracle_y` below.
    nll_ct_z = copula_nll_full(Z_test, R_ct).item()
    nll_pseudo_oracle_corrY_z = copula_nll_full(Z_test, R_oracle).item()
    marginal_nll_te = -log_p_test.sum(dim=-1).mean().item()
    nll_ct_y = nll_ct_z + marginal_nll_te

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
    #   B = copula_nll(R_ct, Z_true) - copula_nll(R_oracle, Z_true)  ≥ 0
    #         → pure copula error; KL(N(0,R_oracle) || N(0,R_ct)) in expectation
    #         → zero only when CT predicts R_oracle exactly
    #
    #   C = copula_nll(R_ct, Z_hat) - copula_nll(R_ct, Z_true)
    #         → how much TabICL's marginal distortion inflates the copula NLL
    #         → ≈ 0 when marginal estimation is accurate
    #
    # Z_true  — PIT with oracle (true) marginals: z*_j = (y_j - mu_j) / sigma_j
    # Z_hat   — current Z_test: PIT with TabICL's estimated marginals
    #
    # Sanity: A + B + C = (nll_ct_z + marginal_nll_te) - (nll_oracle_copula + marginal_nll_oracle)
    #                   = ct_joint_nll_y - (oracle_copula_floor + oracle_marginal_floor)

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
    nll_ct_truez            = copula_nll_full(Z_true, R_ct).item()

    err_A_marginal     = marginal_nll_te    - marginal_nll_oracle   # TabICL marginal error
    err_B_copula_pure  = nll_ct_truez       - nll_oracle_copula_truez  # KL(R_oracle||R_ct) ≥ 0
    err_C_bleed        = nll_ct_z           - nll_ct_truez           # marginal→copula bleed

    # ====================================================================
    # Phase 4 — Log final metrics
    # ====================================================================
    # joint_nll_y = copula_nll_z + marginal_nll_te  (same decomposition as CT)
    # marginal_nll_te is the same for all copula methods (it depends only on marginals)
    final_step = n_steps
    eval_metrics = {
        # ---- Copula NLL (z-space, copula component only) --------------------
        "eval/moment_copula_nll": moment_nll,
        "eval/shrunk_moment_copula_nll": shrunk_nll,
        "eval/nw_rbf_copula_nll": nw_nll["rbf"],
        "eval/nw_epan_copula_nll": nw_nll["epanechnikov"],
        "eval/nw_laplace_copula_nll": nw_nll["laplace"],
        "eval/attn_copula_nll": attn_nll,
        "eval/ct_copula_nll": nll_ct_z,
        # corr(Σ_Y) Gaussian copula on PIT-Z — NOT a hard lower bound (see note above).
        "eval/pseudo_oracle_corrY_copula_nll": nll_pseudo_oracle_corrY_z,
        # ---- Joint NLL (y-space) = copula_nll + marginal_nll_te ------------
        "eval/moment_joint_nll_y": moment_nll + marginal_nll_te,
        "eval/shrunk_moment_joint_nll_y": shrunk_nll + marginal_nll_te,
        "eval/nw_rbf_joint_nll_y": nw_nll["rbf"] + marginal_nll_te,
        "eval/nw_epan_joint_nll_y": nw_nll["epanechnikov"] + marginal_nll_te,
        "eval/nw_laplace_joint_nll_y": nw_nll["laplace"] + marginal_nll_te,
        "eval/attn_joint_nll_y": attn_nll + marginal_nll_te,
        "eval/ct_joint_nll_y": nll_ct_y,
        # Sklar route: biased by TabICL's marginal density (log_p_test) — not tight.
        "eval/pseudo_oracle_corrY_joint_nll_y": nll_pseudo_oracle_corrY_z + marginal_nll_te,
        # True Y-space oracle: hard lower bound on any joint-Y NLL.
        "eval/oracle_joint_nll_y": nll_oracle_y,
        # ---- Shared marginal term (same for all copula methods) -------------
        "eval/marginal_nll_te": marginal_nll_te,
        # ---- Three-way error decomposition ----------------------------------
        # CT copula NLL evaluated on oracle-PIT Z_true (removes marginal distortion)
        "eval/ct_copula_nll_truez":       nll_ct_truez,
        # Oracle copula floor on Z_true  (= 0.5*log|R_oracle| in expectation)
        "eval/oracle_copula_nll_truez":   nll_oracle_copula_truez,
        # True oracle marginal NLL (lower bound on marginal term)
        "eval/marginal_nll_oracle":       marginal_nll_oracle,
        # A: TabICL marginal estimation error (independent of copula model)
        "eval/err_A_marginal":            err_A_marginal,
        # B: pure CT copula error = KL(N(0,R_oracle)||N(0,R_ct)) in expectation  (≥ 0)
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
        "moment":      R_mom_exp_stat,
        "shrunk_mom":  R_shrunk_exp_stat,
        "nw_rbf":      nw_R["rbf"],
        "nw_epan":     nw_R["epanechnikov"],
        "nw_laplace":  nw_R["laplace"],
        "attn":        R_attn,
        "ct":          R_ct,
        "oracle":      R_oracle,
    }
    xi_corr_stats: dict[str, dict[str, float]] = {
        name: _xi_cond_stats(R) for name, R in xi_corr_models.items()
    }
    # Woodbury (D, V) models have actual diagonal variances per xi
    xi_wb_stats: dict[str, dict[str, float]] = {
        "oracle": _woodbury_diag_stats(oracle_D, oracle_V),
        "ct":     _woodbury_diag_stats(d_ct, V_ct),
    }

    # Flatten into eval_metrics for W&B
    for name, s in xi_corr_stats.items():
        for k, v in s.items():
            eval_metrics[f"xi_cond/{name}_{k}"] = v
    for name, s in xi_wb_stats.items():
        for k, v in s.items():
            eval_metrics[f"xi_cond/{name}_{k}"] = v

    print("\n--- Summary ---")
    col_w = max(len(k) for k in eval_metrics) + 2
    copula_keys = [k for k in eval_metrics if "copula_nll" in k]
    joint_keys = [k for k in eval_metrics if "joint_nll_y" in k]
    xi_keys = [k for k in eval_metrics if k.startswith("xi_cond/")]
    diag_keys = [
        k for k in eval_metrics
        if k not in copula_keys and k not in joint_keys and k not in xi_keys
    ]
    for header, keys in [
        ("Copula NLL (z-space)", copula_keys),
        ("Joint NLL  (y-space)", joint_keys),
        ("Diagnostics         ", diag_keys),
    ]:
        print(f"\n  [{header}]")
        for k in keys:
            print(f"  {k:<{col_w}}: {eval_metrics[k]:.4f}")

    # ---- Xi-conditioning table (console only — more readable as a table) ----
    model_names = list(xi_corr_stats.keys())
    print(f"\n  [Xi-conditioning: does the predicted covariance vary with xi?]")
    print(f"  {'model':<14}  {'|ρ|_mean':>9}  {'std_ξ(ρ)':>10}  {'var_frob':>10}", end="")
    print(f"  {'mean_var':>10}  {'std_var_ξ':>10}")
    print("  " + "-" * 68)
    for name in model_names:
        cs = xi_corr_stats[name]
        wb = xi_wb_stats.get(name, {})
        mv = wb.get("mean_diag_var", float("nan"))
        sv = wb.get("std_diag_var_xi", float("nan"))
        print(
            f"  {name:<14}  {cs['mean_abs_offdiag']:>9.4f}  {cs['std_offdiag_xi']:>10.5f}"
            f"  {cs['var_frob_from_I']:>10.5f}  {mv:>10.4f}  {sv:>10.5f}"
        )

    if wandb_run is not None:
        wandb_run.log(eval_metrics, step=final_step)

    # ====================================================================
    # Phase 5 — Covariance plots
    # ====================================================================
    import matplotlib.pyplot as plt

    print("\n--- Generating covariance plots ---")

    # All estimators in ablation order
    estimators = {
        "Moment": R_mom_full,  # (d, d) — broadcast
        "ShrunkMom": R_shrunk,  # (d, d) — broadcast
        "NW-RBF": nw_R["rbf"],  # (n_te, d, d)
        "NW-Epan": nw_R["epanechnikov"],  # (n_te, d, d)
        "NW-Lap": nw_R["laplace"],  # (n_te, d, d)
        "Attention": R_attn,  # (n_te, d, d)
        "Copula Transformer": R_ct,  # (n_te, d, d)
    }

    fig_grid = plot_corr_grid(
        estimators=estimators,
        oracle_R=R_oracle,
        title=f"Correlation estimator comparison — episode {args.episode_idx}",
    )

    # Also keep the detailed CT Woodbury plot for reference
    from sklearn.covariance import OAS

    oas_z = OAS().fit(Z_train.cpu().numpy())
    oracle_groups_raw = ep.get("oracle_groups", None)  # (B, n_test) or None
    groups_b0 = oracle_groups_raw[0] if oracle_groups_raw is not None else None
    ct_instance_indices = select_group_representative_indices(
        groups_b0, max_n=3, n_total=n_test
    )
    fig_ct = plot_prediction_comparison(
        mu_pred=mu_ct.unsqueeze(0),
        D_pred=d_ct.unsqueeze(0),
        V_pred=V_ct.unsqueeze(0),
        mu_true=oracle_mu.unsqueeze(0),
        D_true=oracle_D.unsqueeze(0),
        V_true=oracle_V.unsqueeze(0),
        sigma_oas=oas_z.covariance_,
        dataset_label="CopulaTransformer — predicted vs oracle (Z-space)",
        instance_indices=ct_instance_indices,
    )

    if wandb_run is not None:
        import wandb as wandb_mod

        wandb_run.log(
            {
                "eval/corr_grid": wandb_mod.Image(fig_grid),
                "eval/ct_detail": wandb_mod.Image(fig_ct),
            },
            step=final_step,
        )
        print("Covariance plots logged to W&B.")

    plt.close(fig_grid)
    plt.close(fig_ct)

    if wandb_run is not None:
        wandb_run.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
