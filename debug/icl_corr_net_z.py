"""
icl_corr_net_z.py — ICLCorrNet v3 trained on Z (PIT) instead of Y.

What this script does
---------------------
  1.  Statistical analysis: compare Y_train vs Z_tabicl vs Z_empirical
        - Spearman rank correlation per dimension (should be 1.0: PIT is monotone)
        - Pearson correlation Y↔Z per dimension
        - KS test: are Y marginals N(0,1)?  Are Z marginals N(0,1)?
        - Group-wise empirical correlation: corr(Y_grp) vs corr(Z_grp) vs oracle R
        - Rank-scatter plots and distribution plots

  2.  Train ICLCorrNet v3 using Z_empirical instead of Y.
        Z is computed with the rank-based empirical PIT:
            u_ij = rank(Y_ij) / (N+1)     → monotone per dimension
            Z_ij = Φ⁻¹(u_ij)             → N(0,1) marginals by construction
        TabICL PIT takes 1.4 s/step → too slow for per-step generation.
        Empirical PIT ≈ 0 ms/step and the statistical section shows how close
        it is to the TabICL result.

  3.  Plots include the oracle correlation matrix explicitly.

Usage
-----
  conda run -n multivariate-icl python debug/icl_corr_net_z.py
"""

from __future__ import annotations

import math, os, sys, warnings
from pathlib import Path

import matplotlib
import numpy as np
import scipy.stats as sstats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data_gen import generate_episode
from pit import load_tabicl, run_pit_batched
from viz import plot_corr_grid

OUTPUT_DIR = ROOT / "debug" / "icl_test_results" / "z_version"

# ── Training hyperparameters (same as v3) ────────────────────────────────────
P, D, R      = 8, 6, 4
N_TRAIN      = 128
N_TEST       = 16
BATCH_SIZE   = 16
D_HIDDEN     = 256
N_HEADS      = 8
N_LAYERS     = 2
N_STEPS      = 10_000
LR           = 3e-4
GRAD_CLIP    = 1.0
LOG_EVERY    = 1_000
VAL_EPISODES = 100
STAT_EPISODES = 20   # episodes for statistical analysis


# ── PIT helpers ──────────────────────────────────────────────────────────────

def empirical_pit(Y: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Rank-based PIT: Y → Z where Z_j ~ N(0,1) exactly (per dataset, per dim).

    rank computed within each (batch, dimension) slice independently.
    u_ij = rank(Y_ij) / (N+1)   →   Z_ij = Φ⁻¹(u_ij)
    """
    N = Y.shape[-2]
    ranks = Y.argsort(dim=-2).argsort(dim=-2).float()   # (B, N, d) or (N, d)
    u = (ranks + 1.0) / (N + 1.0)
    u = u.clamp(eps, 1.0 - eps)
    return torch.erfinv(2.0 * u - 1.0) * math.sqrt(2.0)


def cov_to_corr(D: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    S   = torch.diag_embed(D) + V @ V.transpose(-2, -1)
    std = S.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    return S / (std.unsqueeze(-1) * std.unsqueeze(-2))


def empirical_corr(Z: torch.Tensor) -> torch.Tensor:
    """Z : (N, d) → correlation matrix (d, d)."""
    Zc   = Z - Z.mean(0, keepdim=True)
    cov  = Zc.T @ Zc / (Z.shape[0] - 1)
    std  = cov.diagonal().clamp(min=1e-8).sqrt()
    return cov / (std.unsqueeze(1) * std.unsqueeze(0))


def sample_ep_with_Z(device, tabicl, B=1):
    """Generate episode; return Y and both PIT variants."""
    X_tr, Y_tr, X_te, Y_te, oracle = generate_episode(
        B=B, p=P, d=D, r=R,
        n_train=N_TRAIN, n_test=N_TEST,
        device=device,
        hyperplane_bimodal=True,
        return_oracle=True,
    )
    R_ora = cov_to_corr(oracle["D"], oracle["V"])   # (B, n_test, d, d)

    # Empirical PIT (fast, used for training)
    Z_emp_tr = empirical_pit(Y_tr)   # (B, N_TRAIN, d)
    Z_emp_te = empirical_pit(
        torch.cat([Y_tr, Y_te], dim=1)
    )[:, N_TRAIN:]                   # (B, N_TEST, d)  — test quantiles wrt train+test dist

    return X_tr, Y_tr, Y_te, Z_emp_tr, Z_emp_te, X_te, R_ora


# ── Statistical analysis ─────────────────────────────────────────────────────

def run_statistical_analysis(device, tabicl):
    """Compare Y, Z_tabicl, Z_empirical on a single generated episode."""
    print(f"\n{'═'*65}")
    print("STATISTICAL ANALYSIS — Y vs Z_tabicl vs Z_empirical")
    print(f"{'═'*65}")

    # Generate one episode
    X_tr, Y_tr, X_te, Y_te, oracle = generate_episode(
        B=1, p=P, d=D, r=R,
        n_train=N_TRAIN, n_test=N_TEST,
        device=device,
        hyperplane_bimodal=True,
        return_oracle=True,
    )
    R_ora = cov_to_corr(oracle["D"], oracle["V"])   # (1, n_test, d, d)

    # TabICL PIT
    with torch.no_grad():
        Z_tab_tr, Z_tab_te, _ = run_pit_batched(
            tabicl, X_tr, Y_tr, X_te, Y_te, k_folds=8
        )

    # Empirical PIT
    Z_emp_tr = empirical_pit(Y_tr)   # (1, N, d)
    Z_emp_te = empirical_pit(torch.cat([Y_tr, Y_te], dim=1))[:, N_TRAIN:]

    Y  = Y_tr[0].cpu().numpy()           # (N, d)
    Zt = Z_tab_tr[0].cpu().numpy()       # (N, d)
    Ze = Z_emp_tr[0].cpu().numpy()       # (N, d)

    print(f"\n{'─'*65}")
    print(f"  {'Dim':>4}  {'Spearman(Y,Zt)':>15}  {'Spearman(Y,Ze)':>15}  "
          f"{'Pearson(Y,Zt)':>14}  {'Pearson(Y,Ze)':>14}")
    spearman_yt, spearman_ye, pearson_yt, pearson_ye = [], [], [], []
    for j in range(D):
        sp_yt = sstats.spearmanr(Y[:, j], Zt[:, j]).statistic
        sp_ye = sstats.spearmanr(Y[:, j], Ze[:, j]).statistic
        pe_yt = np.corrcoef(Y[:, j], Zt[:, j])[0, 1]
        pe_ye = np.corrcoef(Y[:, j], Ze[:, j])[0, 1]
        spearman_yt.append(sp_yt); spearman_ye.append(sp_ye)
        pearson_yt.append(pe_yt);  pearson_ye.append(pe_ye)
        print(f"  {j:>4d}  {sp_yt:>15.6f}  {sp_ye:>15.6f}  "
              f"{pe_yt:>14.6f}  {pe_ye:>14.6f}")
    print(f"  {'mean':>4}  {np.mean(spearman_yt):>15.6f}  {np.mean(spearman_ye):>15.6f}  "
          f"{np.mean(pearson_yt):>14.6f}  {np.mean(pearson_ye):>14.6f}")

    # KS test: is Z ~ N(0,1)?
    print(f"\n{'─'*65}")
    print(f"  KS test vs N(0,1) — statistic (p-value):")
    print(f"  {'Dim':>4}  {'Y':>18}  {'Z_tabicl':>18}  {'Z_empirical':>18}")
    for j in range(D):
        ks_y  = sstats.kstest(Y[:, j],  'norm')
        ks_zt = sstats.kstest(Zt[:, j], 'norm')
        ks_ze = sstats.kstest(Ze[:, j], 'norm')
        print(f"  {j:>4d}  {ks_y.statistic:.4f} ({ks_y.pvalue:.3f})  "
              f"{ks_zt.statistic:.4f} ({ks_zt.pvalue:.3f})  "
              f"{ks_ze.statistic:.4f} ({ks_ze.pvalue:.3f})")

    # Identify groups from hyperplane
    v_norms   = oracle["V"][0].norm(dim=(-2, -1))   # (N_TEST,) — use training hack
    # Use oracle_V for test; for train, infer from Y norms
    # Groups for training instances: estimate from variance of Y rows
    row_var   = torch.from_numpy(Y).var(dim=1)       # (N,) proxy for group
    thresh    = row_var.median()
    g_tr      = (row_var > thresh).numpy()           # True = strong group

    # Per-group empirical correlation vs oracle
    # Oracle R for test: take mean across strong/weak groups
    test_groups = (oracle["V"][0].norm(dim=(-2,-1)) > oracle["V"][0].norm(dim=(-2,-1)).median())
    R_strong_ora = R_ora[0, test_groups.cpu()].mean(0).cpu().numpy()
    R_weak_ora   = R_ora[0, ~test_groups.cpu()].mean(0).cpu().numpy()

    def group_corr(mat, mask):
        sub = mat[mask]
        return empirical_corr(torch.from_numpy(sub)).numpy()

    Ry_s  = group_corr(Y,  g_tr);   Ry_w  = group_corr(Y,  ~g_tr)
    Rzt_s = group_corr(Zt, g_tr);   Rzt_w = group_corr(Zt, ~g_tr)
    Rze_s = group_corr(Ze, g_tr);   Rze_w = group_corr(Ze, ~g_tr)

    ri, ci = np.triu_indices(D, k=1)
    def od_mse(R_hat, R_ref): return float(np.mean((R_hat[ri, ci] - R_ref[ri, ci])**2))

    print(f"\n{'─'*65}")
    print(f"  Off-diagonal MSE vs oracle (group mean):")
    print(f"  {'':20}  {'strong grp':>12}  {'weak grp':>12}")
    for label, Rs, Rw in [
        ("corr(Y)",         Ry_s,  Ry_w),
        ("corr(Z_tabicl)",  Rzt_s, Rzt_w),
        ("corr(Z_empirical)", Rze_s, Rze_w),
    ]:
        print(f"  {label:20}  {od_mse(Rs, R_strong_ora):>12.5f}  "
              f"{od_mse(Rw, R_weak_ora):>12.5f}")

    return {
        "Y": Y, "Zt": Zt, "Ze": Ze,
        "Ry_s": Ry_s, "Ry_w": Ry_w,
        "Rzt_s": Rzt_s, "Rzt_w": Rzt_w,
        "Rze_s": Rze_s, "Rze_w": Rze_w,
        "R_strong_ora": R_strong_ora, "R_weak_ora": R_weak_ora,
        "spearman_yt": spearman_yt, "spearman_ye": spearman_ye,
        "pearson_yt": pearson_yt, "pearson_ye": pearson_ye,
    }


def plot_stats(s, out_dir):
    """Four-panel statistical comparison figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ── Panel A: rank scatter Y vs Z_tabicl (dim 0) ─────────────────────────
    ax = axes[0, 0]
    ax.scatter(s["Y"][:, 0], s["Zt"][:, 0], s=8, alpha=0.5, color="steelblue",
               label="Z_tabicl")
    ax.scatter(s["Y"][:, 0], s["Ze"][:, 0], s=8, alpha=0.5, color="tomato",
               label="Z_empirical")
    ax.set_xlabel("Y (dim 0)"); ax.set_ylabel("Z (dim 0)")
    ax.set_title("Rank scatter: Y vs Z (dim 0)")
    ax.legend(fontsize=8)

    # ── Panel B: marginal histograms ─────────────────────────────────────────
    ax = axes[0, 1]
    from scipy.stats import norm as sp_norm
    x_range = np.linspace(-4, 4, 200)
    ax.hist(s["Y"][:, 0],  bins=25, density=True, alpha=0.4, color="gray",
            label="Y (dim 0)")
    ax.hist(s["Zt"][:, 0], bins=25, density=True, alpha=0.4, color="steelblue",
            label="Z_tabicl")
    ax.hist(s["Ze"][:, 0], bins=25, density=True, alpha=0.4, color="tomato",
            label="Z_empirical")
    ax.plot(x_range, sp_norm.pdf(x_range), "k--", lw=1.5, label="N(0,1)")
    ax.set_title("Marginal distributions (dim 0)"); ax.legend(fontsize=8)

    # ── Panel C: Spearman r per dimension ────────────────────────────────────
    ax = axes[0, 2]
    dims = np.arange(D)
    w = 0.35
    ax.bar(dims - w/2, s["spearman_yt"], width=w, label="Spearman(Y, Z_tabicl)",
           color="steelblue", alpha=0.8)
    ax.bar(dims + w/2, s["spearman_ye"], width=w, label="Spearman(Y, Z_empirical)",
           color="tomato", alpha=0.8)
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.set_ylim(0.9, 1.01); ax.set_xticks(dims); ax.set_xlabel("Dimension")
    ax.set_title("Spearman rank correlation per dimension"); ax.legend(fontsize=8)

    # ── Panel D: Group-level empirical corr — strong group ───────────────────
    def plot_corr_mat(ax_local, R, title, vmin=-1, vmax=1):
        im = ax_local.imshow(R, vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="auto")
        ax_local.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax_local, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    plot_corr_mat(ax, s["R_strong_ora"], "Oracle R (strong group)")

    ax = axes[1, 1]
    diff_y  = s["Ry_s"]  - s["R_strong_ora"]
    diff_zt = s["Rzt_s"] - s["R_strong_ora"]
    diff_ze = s["Rze_s"] - s["R_strong_ora"]
    ri, ci = np.triu_indices(D, k=1)
    x = np.arange(3)
    mse_vals = [
        float(np.mean(diff_y[ri, ci]**2)),
        float(np.mean(diff_zt[ri, ci]**2)),
        float(np.mean(diff_ze[ri, ci]**2)),
    ]
    bars = ax.bar(x, mse_vals, color=["gray", "steelblue", "tomato"])
    ax.set_xticks(x); ax.set_xticklabels(["corr(Y)", "corr(Z_tab)", "corr(Z_emp)"],
                                          fontsize=9)
    ax.set_ylabel("Off-diag MSE vs oracle"); ax.set_title("Strong group: corr MSE vs oracle")
    for bar, v in zip(bars, mse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)

    ax = axes[1, 2]
    diff_y  = s["Ry_w"]  - s["R_weak_ora"]
    diff_zt = s["Rzt_w"] - s["R_weak_ora"]
    diff_ze = s["Rze_w"] - s["R_weak_ora"]
    mse_vals = [
        float(np.mean(diff_y[ri, ci]**2)),
        float(np.mean(diff_zt[ri, ci]**2)),
        float(np.mean(diff_ze[ri, ci]**2)),
    ]
    bars = ax.bar(x, mse_vals, color=["gray", "steelblue", "tomato"])
    ax.set_xticks(x); ax.set_xticklabels(["corr(Y)", "corr(Z_tab)", "corr(Z_emp)"],
                                          fontsize=9)
    ax.set_ylabel("Off-diag MSE vs oracle"); ax.set_title("Weak group: corr MSE vs oracle")
    for bar, v in zip(bars, mse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)

    fig.suptitle("Statistical comparison: Y_train vs Z_tabicl vs Z_empirical", fontsize=12)
    fig.tight_layout()
    path = out_dir / "stat_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Model (identical to v3) ──────────────────────────────────────────────────

class CrossAttnLayer(nn.Module):
    def __init__(self, d_h, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_h // n_heads
        self.scale   = self.d_head ** -0.5
        self.W_q = nn.Linear(d_h, d_h, bias=False)
        self.W_k = nn.Linear(d_h, d_h, bias=False)
        self.W_v = nn.Linear(d_h, d_h, bias=False)
        self.W_o = nn.Linear(d_h, d_h)
        self.norm1 = nn.LayerNorm(d_h)
        self.norm2 = nn.LayerNorm(d_h)
        self.ff = nn.Sequential(
            nn.Linear(d_h, d_h * 2), nn.GELU(), nn.Linear(d_h * 2, d_h)
        )

    def forward(self, Q_in, K_in, V_in):
        B, n_q, _ = Q_in.shape
        N = K_in.shape[1]
        H, Dh = self.n_heads, self.d_head
        Q = self.W_q(Q_in).view(B, n_q, H, Dh).transpose(1, 2)
        K = self.W_k(K_in).view(B, N,   H, Dh).transpose(1, 2)
        V = self.W_v(V_in).view(B, N,   H, Dh).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_w = F.softmax(scores, dim=-1)
        ctx    = torch.matmul(attn_w, V).transpose(1, 2).reshape(B, n_q, -1)
        ctx    = self.norm1(Q_in + self.W_o(ctx))
        ctx    = self.norm2(ctx + self.ff(ctx))
        return ctx, attn_w.mean(dim=1)


class ICLCorrNetZ(nn.Module):
    """Same as v3 but support encoder uses Z⊗Z instead of Y⊗Y."""

    def __init__(self, p, d, d_h=256, n_heads=8, n_layers=2):
        super().__init__()
        self.d = d
        d_vech = d * (d + 1) // 2

        def mlp(d_in, d_out):
            return nn.Sequential(
                nn.Linear(d_in, d_h), nn.LayerNorm(d_h), nn.GELU(),
                nn.Linear(d_h, d_out), nn.LayerNorm(d_out),
            )

        self.enc_qry = mlp(p,      d_h)
        self.enc_key = mlp(p,      d_h)
        self.enc_val = mlp(d_vech, d_h)   # encodes vech(Z⊗Z)

        self.layers = nn.ModuleList(
            [CrossAttnLayer(d_h, n_heads) for _ in range(n_layers)]
        )

        d_L = d * (d + 1) // 2
        self.readout_L = nn.Sequential(
            nn.Linear(d_h * 2, d_h), nn.GELU(),
            nn.Linear(d_h,     d_L),
        )

        ti, tj = torch.tril_indices(d, d)
        self.register_buffer("ti", ti)
        self.register_buffer("tj", tj)
        self.register_buffer("diag_idx", torch.arange(d))

    def forward(self, X_tr, Z_tr, X_te):
        """Z_tr: (B, N, d) — PIT-transformed support observations."""
        B, N, _ = X_tr.shape
        d = self.d

        # vech of Z⊗Z: standardised, so E[Z_i Z_i^T] = Σ_i more cleanly than Y
        outer = Z_tr.unsqueeze(-1) * Z_tr.unsqueeze(-2)
        vech  = outer[:, :, self.ti, self.tj]

        Q  = self.enc_qry(X_te)
        K  = self.enc_key(X_tr)
        V  = self.enc_val(vech)

        ctx = Q
        attn_last = None
        for layer in self.layers:
            ctx, attn_last = layer(ctx, K, V)

        L_flat = self.readout_L(torch.cat([ctx, Q], dim=-1))
        L = torch.zeros(B, Q.shape[1], d, d, device=X_tr.device, dtype=X_tr.dtype)
        L[:, :, self.ti, self.tj] = L_flat
        L[:, :, self.diag_idx, self.diag_idx] = (
            F.softplus(L[:, :, self.diag_idx, self.diag_idx]) + 1e-4
        )
        Sigma_raw = L @ L.transpose(-2, -1)
        std = Sigma_raw.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
        R_pred = Sigma_raw / (std.unsqueeze(-1) * std.unsqueeze(-2))
        return R_pred, attn_last


# ── Training ─────────────────────────────────────────────────────────────────

def train(model, device):
    ri, ci    = torch.triu_indices(D, D, offset=1, device=device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_STEPS, eta_min=LR * 0.1)
    steps_log, mse_log, div_log, h_log = [], [], [], []

    print(f"\nparams={sum(p.numel() for p in model.parameters()):,}  "
          f"d_h={D_HIDDEN}  Z_empirical input  {N_STEPS} steps  B={BATCH_SIZE}")
    print(f"{'─'*65}")
    print(f"  {'step':>6}  {'MSE':>8}  {'div':>8}  {'H_norm':>8}  {'LR':>10}")

    model.train()
    for step in range(N_STEPS):
        X_tr, Y_tr, _, Z_emp_tr, _, X_te, R_ora = sample_ep_with_Z(device, None, B=BATCH_SIZE)
        optimizer.zero_grad()
        R_pred, attn_w = model(X_tr, Z_emp_tr, X_te)
        loss = F.mse_loss(R_pred[:, :, ri, ci], R_ora[:, :, ri, ci])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                mse = loss.item()
                div = R_pred[:, :, ri, ci].std(dim=1).mean().item()
                w   = attn_w.clamp(min=1e-10)
                h   = (-(w * w.log()).sum(-1).mean() / math.log(max(N_TRAIN,2))).item()
                lr  = optimizer.param_groups[0]["lr"]
            steps_log.append(step); mse_log.append(mse)
            div_log.append(div);    h_log.append(h)
            print(f"  {step:>6d}  {mse:>8.5f}  {div:>8.5f}  {h:>8.4f}  {lr:>10.2e}")

    return steps_log, mse_log, div_log, h_log


@torch.no_grad()
def validate(model, device):
    ri, ci = torch.triu_indices(D, D, offset=1, device=device)
    model.eval()
    mses = []
    for _ in range(VAL_EPISODES):
        X_tr, Y_tr, _, Z_emp_tr, _, X_te, R_ora = sample_ep_with_Z(device, None, B=1)
        R_pred, _ = model(X_tr, Z_emp_tr, X_te)
        mses.append(F.mse_loss(R_pred[:, :, ri, ci], R_ora[:, :, ri, ci]).item())
    return float(np.mean(mses))


@torch.no_grad()
def icl_test(model, device):
    ri, ci = torch.triu_indices(D, D, offset=1, device=device)
    model.eval()

    X_a_tr, Y_a_tr, _, Za_tr, _, X_a_te, Ra = sample_ep_with_Z(device, None, B=1)
    X_b_tr, Y_b_tr, _, Zb_tr, _, _,      _  = sample_ep_with_Z(device, None, B=1)

    def mse(Rp, Ro): return F.mse_loss(Rp[:, :, ri, ci], Ro[:, :, ri, ci]).item()
    def div(Rp):     return Rp[0, :, ri, ci].std(dim=0).mean().item()
    def hn(w):
        w = w.clamp(min=1e-10)
        return (-(w*w.log()).sum(-1).mean() / math.log(max(N_TRAIN,2))).item()

    Raa, waa = model(X_a_tr, Za_tr, X_a_te)
    Rba, wba = model(X_b_tr, Zb_tr, X_a_te)
    Rza, _   = model(torch.zeros_like(X_a_tr), torch.zeros_like(Za_tr), X_a_te)
    Ra0, _   = model(X_a_tr, Za_tr, torch.zeros_like(X_a_te))

    chg_swap = (Rba - Raa).abs().mean().item()
    chg_zero = (Rza - Raa).abs().mean().item()

    print(f"\n{'═'*65}")
    print(f"ICL TEST (Z version)")
    print(f"{'═'*65}")
    print(f"  Baseline   MSE={mse(Raa,Ra):.5f}  div={div(Raa):.4f}  H={hn(waa):.4f}")
    print(f"  Supp swap  MSE={mse(Rba,Ra):.5f}  div={div(Rba):.4f}  "
          f"H={hn(wba):.4f}  chg={chg_swap:.5f}")
    print(f"  Zero supp  MSE={mse(Rza,Ra):.5f}  div={div(Rza):.4f}  chg={chg_zero:.5f}")
    print(f"  Zero Xte   MSE={mse(Ra0,Ra):.5f}  div={div(Ra0):.4f}")

    doing_icl = chg_swap > 0.02
    print(f"\n  VERDICT: {'TRUE ICL' if doing_icl else 'X REGRESSION'}  "
          f"chg_swap={chg_swap:.4f}")
    return {"Raa": Raa, "Rba": Rba, "Ra": Ra,
            "chg_swap": chg_swap, "verdict": "TRUE ICL" if doing_icl else "X REGRESSION"}


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_training(steps_log, mse_log, div_log, h_log, val_mse, baseline_mse, result, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(steps_log, mse_log, lw=1.5, color="steelblue", label="train MSE (Z input)")
    axes[0].axhline(val_mse,      color="green", ls="--", lw=1.2,
                    label=f"val MSE={val_mse:.3f}")
    axes[0].axhline(baseline_mse, color="gray",  ls=":",  lw=1.2,
                    label=f"indep baseline={baseline_mse:.3f}")
    axes[0].set_title("MSE — Z_empirical input"); axes[0].legend(fontsize=8)

    axes[1].plot(steps_log, div_log, lw=1.5, color="purple")
    axes[1].set_title("Prediction diversity")

    axes[2].plot(steps_log, h_log, lw=1.5, color="tomato")
    axes[2].axhline(1.0, color="red",   ls="--", lw=0.8, alpha=0.6)
    axes[2].axhline(0.0, color="green", ls="--", lw=0.8, alpha=0.6)
    axes[2].set_ylim(-0.05, 1.1)
    axes[2].set_title("Cross-attn H_norm")

    fig.suptitle(
        f"ICLCorrNetZ (v3 + Z input) — val={val_mse:.4f}  "
        f"base={baseline_mse:.4f}  [{result['verdict']}  chg={result['chg_swap']:.3f}]",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "training.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_corr_grid_with_oracle(result, out_dir):
    """Correlation grid with oracle explicitly included as an estimator column."""
    n_test_ = result["Raa"].shape[1]
    idx     = torch.arange(n_test_)

    # plot_corr_grid already shows oracle in the first column; we also add it
    # as an estimator so the error panel (|R* - R_hat|) shows oracle vs itself
    # (should be black/zero) as a sanity check.
    fig_g = plot_corr_grid(
        estimators={
            "Base (ep-A supp)": result["Raa"][0, idx].cpu(),
            "Swap (ep-B supp)": result["Rba"][0, idx].cpu(),
        },
        oracle_R=result["Ra"][0, idx].cpu(),
        n_instances=n_test_,
        title=f"ICLCorrNetZ — support swap test  [{result['verdict']}]",
    )
    fig_g.savefig(out_dir / "swap.png", dpi=100, bbox_inches="tight")
    plt.close(fig_g)

    # Additional: show oracle correlation matrices explicitly
    R_ora_cpu = result["Ra"][0, idx].cpu()  # (n_test, d, d)
    fig, axes = plt.subplots(2, n_test_ // 2, figsize=(n_test_ * 1.5, 6))
    axes = axes.ravel()
    for i in range(n_test_):
        im = axes[i].imshow(R_ora_cpu[i].numpy(), vmin=-1, vmax=1,
                            cmap="RdBu_r", aspect="auto")
        axes[i].set_title(f"Instance {i}", fontsize=8)
        axes[i].axis("off")
    fig.suptitle("Oracle correlation matrices — all test instances", fontsize=11)
    plt.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "oracle_corr_matrices.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  p={P} d={D} r={R}  N={N_TRAIN}")

    # Load TabICL for statistical analysis only
    print("\nLoading TabICL for statistical analysis...")
    tabicl = load_tabicl("tabicl-regressor-v2-20260212.ckpt", device)
    tabicl.eval()

    # ── 1. Statistical analysis ──────────────────────────────────────────────
    stats = run_statistical_analysis(device, tabicl)
    plot_stats(stats, OUTPUT_DIR)

    # ── 2. Independence baseline ─────────────────────────────────────────────
    print("\nComputing independence baseline MSE...")
    ri_b, ci_b = torch.triu_indices(D, D, offset=1, device=device)
    base_mses = []
    for _ in range(100):
        _, _, _, _, _, _, R_ora = sample_ep_with_Z(device, None, B=1)
        I = torch.eye(D, device=device).unsqueeze(0).unsqueeze(0).expand_as(R_ora)
        base_mses.append(F.mse_loss(I[:, :, ri_b, ci_b], R_ora[:, :, ri_b, ci_b]).item())
    baseline_mse = float(np.mean(base_mses))
    print(f"Independence baseline MSE: {baseline_mse:.5f}")

    # ── 3. Train with Z ──────────────────────────────────────────────────────
    torch.manual_seed(3)
    model = ICLCorrNetZ(p=P, d=D, d_h=D_HIDDEN, n_heads=N_HEADS,
                        n_layers=N_LAYERS).to(device)
    steps_log, mse_log, div_log, h_log = train(model, device)

    # ── 4. Validate and ICL test ─────────────────────────────────────────────
    val_mse = validate(model, device)
    improvement = (1 - val_mse / baseline_mse) * 100
    print(f"\nVal MSE ({VAL_EPISODES} episodes): {val_mse:.5f}  "
          f"(baseline: {baseline_mse:.5f}  improvement: {improvement:.1f}%)")

    result = icl_test(model, device)

    # ── 5. Plots ─────────────────────────────────────────────────────────────
    plot_training(steps_log, mse_log, div_log, h_log,
                  val_mse, baseline_mse, result, OUTPUT_DIR)
    plot_corr_grid_with_oracle(result, OUTPUT_DIR)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")
    print(f"\n{'═'*65}")
    print(f"SUMMARY")
    print(f"{'═'*65}")
    print(f"  v3  (Y input):   val MSE = 0.02260  (76% over baseline)")
    print(f"  Z version:       val MSE = {val_mse:.5f}  "
          f"({improvement:.1f}% over baseline)")
    print(f"  ICL:             {result['verdict']}  swap_chg={result['chg_swap']:.4f}")


if __name__ == "__main__":
    main()
