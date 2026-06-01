"""
nll_mse_balance.py — Find the MSE weight needed to overcome NLL dominance.

Findings so far:
  • MSE-only:       CopulaTabICLv2 → MSE=0.00000, r=1.000 in ~400 steps ✓
  • NLL + 1×MSE:    NLL dominates → MSE plateaus at 0.51, attention H_norm≈1.0 ✗

This script sweeps MSE weights [1, 10, 50, 100, 500] with NLL+w×MSE training
to find the minimum weight that keeps MSE converging toward zero.

Also runs the gradient-magnitude comparison at step 0 to understand the
NLL/MSE gradient ratio.

Usage:
    conda run -n multivariate-icl python debug/nll_mse_balance.py
"""

from __future__ import annotations

import math, os, sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import AdamW

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from loss import woodbury_nll              # noqa: E402
from model import build_copula_tabicl_v2  # noqa: E402

EPISODE_FILE = ROOT / "data" / "pit_hyperplane_debug" / "episode_000000.pt"
OUTPUT_DIR   = ROOT / "debug" / "overtrain_results" / "balance_sweep"
N_STEPS      = 5_000
LR           = 1e-3
GRAD_CLIP    = 1.0
LOG_EVERY    = 500
MSE_WEIGHTS  = [1, 10, 50, 100, 500, 0]   # 0 = MSE-only (control)
SUCCESS_MSE  = 0.02


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


def grad_magnitude(model):
    total = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().abs().mean().item()
            count += 1
    return total / max(count, 1)


def run_weight(w, X_train, Z_train, X_test, Z_test, oracle_D, oracle_V,
               device, N, d):
    R_ora  = _cov_to_corr(oracle_D, oracle_V)
    X_fwd  = torch.cat([X_train, X_test], dim=1)
    Z_fwd  = torch.cat([Z_train, Z_test], dim=1)
    ri, ci = torch.triu_indices(d, d, offset=1, device=device)
    D_ora_c, V_ora_c = _oracle_wby(oracle_D, oracle_V)

    torch.manual_seed(42)
    cfg_model = OmegaConf.load(ROOT / "conf" / "model" / "copula_tabicl_v2.yaml")
    model     = build_copula_tabicl_v2(SimpleNamespace(model=cfg_model)).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    label = "MSE-only" if w == 0 else f"NLL + {w}×MSE"
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"  {'step':>5}  {'NLL':>10}  {'MSE':>9}  {'div':>8}")

    steps_log, mse_log, nll_log = [], [], []
    grad_ratio_logged = False

    model.train()
    for step in range(N_STEPS):
        optimizer.zero_grad()
        mu_Z, d_Z, V_Z = model(X_fwd, Z_fwd, n_support=N)
        Sigma_pred = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)

        loss_mse = F.mse_loss(Sigma_pred[:, :, ri, ci], R_ora[:, :, ri, ci])

        if w == 0:
            loss = loss_mse
            loss_nll_val = float("nan")
        else:
            loss_nll  = woodbury_nll(Z_test, mu_Z, d_Z, V_Z)
            loss      = loss_nll + w * loss_mse
            loss_nll_val = loss_nll.item()

        loss.backward()

        # Log gradient magnitudes at first step
        if not grad_ratio_logged and w != 0:
            # Save NLL-only grad
            nll_grads = {n: p.grad.clone() for n, p in model.named_parameters()
                         if p.grad is not None}
            # Recompute MSE-only grad
            optimizer.zero_grad()
            mu_Z2, d_Z2, V_Z2 = model(X_fwd, Z_fwd, n_support=N)
            Sp2 = torch.diag_embed(d_Z2) + V_Z2 @ V_Z2.transpose(-2, -1)
            F.mse_loss(Sp2[:, :, ri, ci], R_ora[:, :, ri, ci]).backward()
            mse_grads = {n: p.grad.clone() for n, p in model.named_parameters()
                         if p.grad is not None}
            nll_mag = np.mean([g.abs().mean().item() for g in nll_grads.values()])
            mse_mag = np.mean([g.abs().mean().item() for g in mse_grads.values()])
            print(f"  [grad@0] NLL={nll_mag:.6f}  MSE={mse_mag:.6f}  "
                  f"ratio(NLL/MSE)={nll_mag/max(mse_mag,1e-10):.1f}×")
            # Redo the combined backward for the optimizer step
            optimizer.zero_grad()
            mu_Z3, d_Z3, V_Z3 = model(X_fwd, Z_fwd, n_support=N)
            Sp3 = torch.diag_embed(d_Z3) + V_Z3 @ V_Z3.transpose(-2, -1)
            lnll3 = woodbury_nll(Z_test, mu_Z3, d_Z3, V_Z3)
            lmse3 = F.mse_loss(Sp3[:, :, ri, ci], R_ora[:, :, ri, ci])
            (lnll3 + w * lmse3).backward()
            grad_ratio_logged = True

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                Sp4 = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
                mse_v = F.mse_loss(Sp4[:, :, ri, ci], R_ora[:, :, ri, ci]).item()
                off = Sp4[0, :, ri, ci]
                div_v = off.std(dim=0).mean().item()
            steps_log.append(step); mse_log.append(mse_v); nll_log.append(loss_nll_val)
            flag = " ✓" if mse_v < SUCCESS_MSE else ""
            print(f"  {step:>5d}  {loss_nll_val:>10.4f}  {mse_v:>9.5f}  {div_v:>8.4f}{flag}")

    return steps_log, mse_log


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    ep       = torch.load(EPISODE_FILE, weights_only=True)
    b        = 0
    X_train  = ep["X_train"][[b]].float().to(device)
    Z_train  = ep["Z_train"][[b]].float().to(device)
    X_test   = ep["X_test"][[b]].float().to(device)
    Z_test   = ep["Z_test"][[b]].float().to(device)
    oracle_D = ep["oracle_D"][[b]].float().to(device)
    oracle_V = ep["oracle_V"][[b]].float().to(device)
    N, n_test, d = X_train.shape[1], X_test.shape[1], Z_train.shape[2]
    print(f"N={N}  n_test={n_test}  d={d}")
    print(f"MSE weights to sweep: {MSE_WEIGHTS}  ({N_STEPS} steps each)")

    results = {}
    for w in MSE_WEIGHTS:
        steps, mse = run_weight(w, X_train, Z_train, X_test, Z_test,
                                oracle_D, oracle_V, device, N, d)
        results[w] = (steps, mse)

    # Summary table
    print(f"\n{'='*55}")
    print(f"{'Weight':>10}  {'Final MSE':>12}  {'Success?':>10}")
    print(f"{'─'*55}")
    for w in MSE_WEIGHTS:
        _, mse = results[w]
        label = "MSE-only" if w == 0 else f"NLL+{w}×MSE"
        success = mse[-1] < SUCCESS_MSE
        print(f"{label:>10}  {mse[-1]:>12.5f}  {'✓' if success else '✗'}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(MSE_WEIGHTS)))
    for (w, (steps, mse)), col in zip(results.items(), colors):
        label = "MSE-only (control)" if w == 0 else f"NLL + {w}×MSE"
        ax.plot(steps, mse, lw=1.5, color=col, label=label)
    ax.axhline(SUCCESS_MSE, color="green", ls="--", lw=1, label=f"target {SUCCESS_MSE}")
    ax.set_xlabel("Step"); ax.set_ylabel("Off-diag MSE")
    ax.set_title("Effect of MSE weight on CopulaTabICLv2 training\n"
                 "(NLL + w×MSE  vs  MSE-only)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = OUTPUT_DIR / "weight_sweep.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to: {path}")


if __name__ == "__main__":
    main()
