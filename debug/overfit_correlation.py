"""
overfit_correlation.py — Iterate until a model perfectly learns oracle correlation.

Root cause of previous failure
--------------------------------
Training with NLL + MSE caused MSE to jump from 0.147 → 0.692 in 100 steps:
the NLL gradient drives U large (concentrating mass on observed Z values) which
produces arbitrary off-diagonal structure unrelated to oracle R.  The MSE term
never recovered because the NLL gradient dominates in magnitude.

Strategy here
-------------
Remove NLL entirely.  With MSE-only the gradient directly pushes predictions
toward oracle R for each query instance.  We try progressively simpler
configurations until MSE reaches the SUCCESS_THRESHOLD:

  Run 1 — SimpleCopulaNet (d_h=128), Woodbury,  MSE-only, LR=1e-3
  Run 2 — TinyNet          (d_h=32),  Woodbury,  MSE-only, LR=1e-3
  Run 3 — TinyNet          (d_h=32),  Cholesky,  MSE-only, LR=1e-3
  Run 4 — TinyNet          (d_h=32),  Cholesky,  MSE-only, LR=3e-4 (stabilise)

After each run we print prediction diversity (std of off-diagonal entries across
query instances) — if this stays near 0, the model is still producing the same
matrix for all queries (context collapse).

Usage:
    conda run -n multivariate-icl python debug/overfit_correlation.py
"""

from __future__ import annotations

import math, os, sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from viz import plot_corr_grid  # noqa: E402

EPISODE_FILE      = ROOT / "data" / "pit_hyperplane_debug" / "episode_000000.pt"
OUTPUT_DIR        = ROOT / "debug" / "overfit_correlation"
BATCH_ELEM        = 0
SUCCESS_THRESHOLD = 0.02   # off-diagonal MSE below which we declare success
MAX_STEPS         = 20_000
LOG_EVERY         = 500


# ---------------------------------------------------------------------------
# Helpers shared by all runs
# ---------------------------------------------------------------------------

def _cov_to_corr(D, V):
    S   = torch.diag_embed(D) + V @ V.transpose(-2, -1)
    std = S.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    return S / (std.unsqueeze(-1) * std.unsqueeze(-2))


def _oracle_wby(oracle_D, oracle_V):
    S   = torch.diag_embed(oracle_D) + oracle_V @ oracle_V.transpose(-2, -1)
    std = S.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    V_c = oracle_V / std.unsqueeze(-1)
    D_c = (1.0 - (V_c**2).sum(-1)).clamp(min=1e-6)
    return D_c, V_c


def off_diag_mse(Sigma_pred, R_ora):
    d = Sigma_pred.shape[-1]
    ri, ci = torch.triu_indices(d, d, offset=1, device=Sigma_pred.device)
    return F.mse_loss(Sigma_pred[..., ri, ci], R_ora[..., ri, ci]).item()


def prediction_diversity(Sigma_pred):
    """Std of off-diagonal predictions across query instances.
    0 → all predictions identical; >0 → model differentiates queries."""
    d = Sigma_pred.shape[-1]
    ri, ci = torch.triu_indices(d, d, offset=1, device=Sigma_pred.device)
    off = Sigma_pred[0, :, ri, ci]   # (n_qry, n_pairs)
    return off.std(dim=0).mean().item()


def attn_h_norm(attn_w, n_sup):
    w = attn_w.clamp(min=1e-10)
    return (-(w * w.log()).sum(-1).mean() / math.log(max(n_sup, 2))).item()


def make_plots(run_name, steps_log, mse_log, div_log, R_pred_sorted,
               R_oracle_sorted, groups, n_test, n_weak, n_strong, success):
    out = OUTPUT_DIR / run_name
    out.mkdir(parents=True, exist_ok=True)

    # Training curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps_log, mse_log, color="steelblue", lw=1.5)
    axes[0].axhline(SUCCESS_THRESHOLD, color="green", ls="--", lw=1,
                    label=f"target ({SUCCESS_THRESHOLD})")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Off-diag MSE")
    axes[0].set_title("MSE (MSE-only loss — no NLL)")
    axes[0].legend(fontsize=8)

    axes[1].plot(steps_log, div_log, color="purple", lw=1.5)
    axes[1].axhline(0, color="red", ls="--", lw=0.8, alpha=0.5,
                    label="0 = all predictions identical")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Prediction diversity (std)")
    axes[1].set_title("Cross-query prediction diversity\n"
                       "(must be > 0 for model to distinguish groups)")
    axes[1].legend(fontsize=8)

    tag = "SUCCESS" if success else "FAILED"
    fig.suptitle(f"{run_name} — {tag}  final MSE={mse_log[-1]:.4f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "training_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Correlation grid — all test instances
    fig_g = plot_corr_grid(
        estimators={"Predicted": R_pred_sorted},
        oracle_R=R_oracle_sorted,
        n_instances=n_test,
        title=(f"{run_name} — all {n_test} test instances\n"
               f"{n_weak} weak (top) / {n_strong} strong (bottom)  [{tag}]"),
    )
    fig_g.savefig(out / "all_predictions.png", dpi=100, bbox_inches="tight")
    plt.close(fig_g)

    print(f"  → Plots saved to {out}/")


# ---------------------------------------------------------------------------
# Model A — SimpleCopulaNet with Woodbury output  (d_h configurable)
# ---------------------------------------------------------------------------

class CrossAttnCopulaNet(nn.Module):
    """Cross-attention copula net with either Woodbury or Cholesky output."""

    def __init__(self, p_max, d_max, rank_max, d_hidden, n_heads=4,
                 output="woodbury"):
        super().__init__()
        self.p_max    = p_max
        self.d_max    = d_max
        self.rank_max = rank_max
        self.d_hidden = d_hidden
        self.output   = output   # "woodbury" | "cholesky"

        d_vech   = d_max * (d_max + 1) // 2
        d_sup_in = p_max + d_max + d_vech

        self.enc_sup = nn.Sequential(
            nn.Linear(d_sup_in, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.LayerNorm(d_hidden),
        )
        self.enc_qry = nn.Sequential(
            nn.Linear(p_max,    d_hidden), nn.LayerNorm(d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.LayerNorm(d_hidden),
        )
        self.cross_attn    = nn.MultiheadAttention(d_hidden, n_heads,
                                                   batch_first=True)
        self.post_attn_norm = nn.LayerNorm(d_hidden)

        self.dim_emb = nn.Parameter(torch.randn(d_max, d_hidden // 4))

        d_head = d_hidden + d_hidden + d_hidden // 4
        if output == "woodbury":
            self.readout = nn.Sequential(
                nn.Linear(d_head, d_hidden), nn.GELU(),
                nn.Linear(d_hidden, rank_max),
            )
        else:  # cholesky: predict d*(d+1)/2 entries per query (not per dim)
            d_L = d_max * (d_max + 1) // 2
            self.readout_L = nn.Sequential(
                nn.Linear(d_hidden * 2, d_hidden), nn.GELU(),
                nn.Linear(d_hidden, d_L),
            )

    def _encode(self, X_all, Z_all, n_support):
        B, N, p = X_all.shape
        d = Z_all.shape[-1]

        if p < self.p_max:
            X_all = F.pad(X_all, (0, self.p_max - p))
        else:
            X_all = X_all[..., :self.p_max]

        Z_sup_raw = Z_all[:, :n_support, :]
        if d < self.d_max:
            Z_sup_pad = F.pad(Z_sup_raw, (0, self.d_max - d))
        else:
            Z_sup_pad = Z_sup_raw[:, :, :self.d_max]

        outer    = Z_sup_pad.unsqueeze(-1) * Z_sup_pad.unsqueeze(-2)
        ti, tj   = torch.tril_indices(self.d_max, self.d_max, device=X_all.device)
        vech     = outer[:, :, ti, tj]
        sup_in   = torch.cat([X_all[:, :n_support], Z_sup_pad, vech], -1)
        sup_emb  = self.enc_sup(sup_in)
        qry_emb  = self.enc_qry(X_all[:, n_support:])
        ctx, attn_w = self.cross_attn(qry_emb, sup_emb, sup_emb,
                                      need_weights=True,
                                      average_attn_weights=True)
        ctx = self.post_attn_norm(ctx + qry_emb)
        return ctx, qry_emb, attn_w, d

    def forward_woodbury(self, ctx, qry_emb, d):
        B, n_q, _ = ctx.shape
        ctx_exp = ctx.unsqueeze(2).expand(B, n_q, self.d_max, -1)
        qry_exp = qry_emb.unsqueeze(2).expand(B, n_q, self.d_max, -1)
        dim_exp = self.dim_emb.unsqueeze(0).unsqueeze(0).expand(B, n_q, -1, -1)
        U = self.readout(torch.cat([ctx_exp, qry_exp, dim_exp], -1))
        U = U[:, :, :d, :]
        U_sq = (U**2).sum(-1)
        C    = 1.0 / (1.0 + U_sq)
        W    = U  / (1.0 + U_sq.unsqueeze(-1)).sqrt()
        Sigma = torch.diag_embed(C) + W @ W.transpose(-2, -1)
        return Sigma

    def forward_cholesky(self, ctx, qry_emb, d):
        B, n_q, _ = ctx.shape
        head_in = torch.cat([ctx, qry_emb], -1)
        L_flat  = self.readout_L(head_in)       # (B, n_q, d*(d+1)/2)
        # Reconstruct lower-triangular L
        L = torch.zeros(B, n_q, self.d_max, self.d_max,
                        device=ctx.device, dtype=ctx.dtype)
        ti, tj = torch.tril_indices(self.d_max, self.d_max, device=ctx.device)
        L[:, :, ti, tj] = L_flat
        # Positive diagonal via softplus
        diag_idx = torch.arange(self.d_max, device=ctx.device)
        L[:, :, diag_idx, diag_idx] = F.softplus(L[:, :, diag_idx, diag_idx]) + 1e-4
        L = L[:, :, :d, :d]
        Sigma_raw = L @ L.transpose(-2, -1)
        std = Sigma_raw.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
        Sigma = Sigma_raw / (std.unsqueeze(-1) * std.unsqueeze(-2))
        return Sigma

    def forward(self, X_all, Z_all, n_support):
        ctx, qry_emb, attn_w, d = self._encode(X_all, Z_all, n_support)
        if self.output == "woodbury":
            Sigma = self.forward_woodbury(ctx, qry_emb, d)
        else:
            Sigma = self.forward_cholesky(ctx, qry_emb, d)
        return Sigma, attn_w


# ---------------------------------------------------------------------------
# Training loop — MSE ONLY
# ---------------------------------------------------------------------------

def train_run(run_name, model, X_fwd, Z_fwd, Z_test, R_ora, N, d, device,
              lr, n_steps):
    ri, ci    = torch.triu_indices(d, d, offset=1, device=device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    steps_log, mse_log, div_log, h_log = [], [], [], []

    print(f"\n{'═'*60}")
    print(f"  {run_name}")
    print(f"  params={sum(p.numel() for p in model.parameters()):,}  "
          f"output={model.output}  d_h={model.d_hidden}  lr={lr}")
    print(f"{'═'*60}")
    print(f"  {'step':>6}  {'MSE':>9}  {'diversity':>10}  {'H_norm':>8}")

    model.train()
    for step in range(n_steps):
        optimizer.zero_grad()
        Sigma_pred, attn_w = model(X_fwd, Z_fwd, n_support=N)
        loss = F.mse_loss(Sigma_pred[:, :, ri, ci], R_ora[:, :, ri, ci])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                mse  = off_diag_mse(Sigma_pred, R_ora)
                div  = prediction_diversity(Sigma_pred)
                h    = attn_h_norm(attn_w, N)
            steps_log.append(step);  mse_log.append(mse)
            div_log.append(div);     h_log.append(h)
            flag = " ← SUCCESS" if mse < SUCCESS_THRESHOLD else ""
            print(f"  {step:>6d}  {mse:>9.5f}  {div:>10.6f}  {h:>8.4f}{flag}")
            if mse < SUCCESS_THRESHOLD:
                break

    return steps_log, mse_log, div_log, h_log


# ---------------------------------------------------------------------------
# Main — iterate through runs
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load episode
    ep       = torch.load(EPISODE_FILE, weights_only=True)
    b        = BATCH_ELEM
    X_train  = ep["X_train"][[b]].float().to(device)
    Z_train  = ep["Z_train"][[b]].float().to(device)
    X_test   = ep["X_test"][[b]].float().to(device)
    Z_test   = ep["Z_test"][[b]].float().to(device)
    oracle_D = ep["oracle_D"][[b]].float().to(device)
    oracle_V = ep["oracle_V"][[b]].float().to(device)

    N, n_test, d = X_train.shape[1], X_test.shape[1], Z_train.shape[2]
    p            = X_train.shape[2]
    rank         = oracle_V.shape[-1]
    print(f"N={N}  n_test={n_test}  d={d}  p={p}  r={rank}")

    R_ora  = _cov_to_corr(oracle_D, oracle_V)   # (1, n_test, d, d)
    X_fwd  = torch.cat([X_train, X_test], dim=1)
    Z_fwd  = torch.cat([Z_train, Z_test], dim=1)

    v_norms = oracle_V[0].norm(dim=(-2, -1))
    groups  = (v_norms > v_norms.median()).long().cpu().numpy()
    n_weak  = int((groups == 0).sum())
    n_strong= int((groups == 1).sum())
    sort_order = np.argsort(groups)

    # Configurations to try in order
    configs = [
        dict(run="run1_simple128_woodbury",
             d_h=128, output="woodbury", lr=1e-3, steps=MAX_STEPS),
        dict(run="run2_tiny32_woodbury",
             d_h=32,  output="woodbury", lr=1e-3, steps=MAX_STEPS),
        dict(run="run3_tiny32_cholesky",
             d_h=32,  output="cholesky", lr=1e-3, steps=MAX_STEPS),
        dict(run="run4_tiny32_cholesky_lowlr",
             d_h=32,  output="cholesky", lr=3e-4, steps=MAX_STEPS),
    ]

    for cfg in configs:
        torch.manual_seed(42)
        model = CrossAttnCopulaNet(
            p_max=p, d_max=d, rank_max=rank,
            d_hidden=cfg["d_h"], output=cfg["output"],
        ).to(device)

        steps_log, mse_log, div_log, h_log = train_run(
            cfg["run"], model, X_fwd, Z_fwd, Z_test, R_ora, N, d, device,
            lr=cfg["lr"], n_steps=cfg["steps"],
        )

        # Final inference
        model.eval()
        with torch.no_grad():
            Sigma_pred, attn_w = model(X_fwd, Z_fwd, n_support=N)
        final_mse = off_diag_mse(Sigma_pred, R_ora)
        success   = final_mse < SUCCESS_THRESHOLD

        # Off-diagonal scatter stats
        ri_np, ci_np = np.triu_indices(d, k=1)
        R_pred_np   = Sigma_pred[0].cpu().numpy()
        R_oracle_np = R_ora[0].cpu().numpy()
        x_all = R_oracle_np[:, ri_np, ci_np].ravel()
        y_all = R_pred_np[:,  ri_np, ci_np].ravel()
        r_val = float(np.corrcoef(x_all, y_all)[0, 1]) if x_all.std()>1e-8 else float("nan")
        slope = float(np.polyfit(x_all, y_all, 1)[0]) if x_all.std()>1e-8 else 1.0

        print(f"\n  Final:  MSE={final_mse:.5f}  r={r_val:.3f}  slope={slope:.3f}"
              f"  {'SUCCESS ✓' if success else 'failed'}")

        # Plots
        R_pred_s = Sigma_pred[0, sort_order].cpu()
        R_ora_s  = R_ora[0, sort_order].cpu()
        make_plots(cfg["run"], steps_log, mse_log, div_log,
                   R_pred_s, R_ora_s, groups, n_test, n_weak, n_strong, success)

        if success:
            print(f"\n{'★'*60}")
            print(f"  SUCCESS — {cfg['run']} reached MSE={final_mse:.5f} < {SUCCESS_THRESHOLD}")
            print(f"  Pearson r={r_val:.3f}   OLS slope={slope:.3f}")
            print(f"{'★'*60}")
            break
        else:
            print(f"  MSE={final_mse:.5f} > {SUCCESS_THRESHOLD} — trying next config …")

    else:
        print("\nAll configurations failed to reach the success threshold.")
        print("Consider: more steps, different architecture, or checking data.")


if __name__ == "__main__":
    main()
