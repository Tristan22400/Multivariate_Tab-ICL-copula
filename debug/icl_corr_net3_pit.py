"""
icl_corr_net3_pit.py — ICLCorrNet3 trained on pre-generated PIT hyperplane episodes.

Uses data/pit_hyperplane_debug/*.pt (B=16, p=20, d=8, N_train=256, Z_tabicl input).
Logs training metrics and correlation-matrix images to Weights & Biases.

Usage
-----
  conda run -n multivariate-icl python debug/icl_corr_net3_pit.py
"""

from __future__ import annotations

import glob
import math
import os
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from scipy import stats as scipy_stats
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data_gen import generate_episode

DATA_DIR    = ROOT / "data" / "pit_hyperplane_debug"
OUTPUT_DIR  = ROOT / "debug" / "icl_test_results" / "pit"

# ── Data constants (from files) ────────────────────────────────────────────────
P, D        = 20, 8          # feature dim, output dim
N_TRAIN     = 256
N_TEST      = 16

# ── Training hyperparameters ───────────────────────────────────────────────────
D_HIDDEN    = 256
N_HEADS     = 8
N_LAYERS    = 2
N_STEPS     = 15_000
BATCH_SIZE  = 16             # batch elements per file = full episode
LR          = 3e-4
GRAD_CLIP   = 1.0

LOG_EVERY   = 100
VAL_EVERY   = 500
IMAGE_EVERY = 1_000
VAL_FILES   = 200            # number of val episodes to average

WANDB_PROJECT = "copula-icl"
WANDB_RUN     = "icl_corr_net3_pit_z_v2"


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

class PitEpisodeDataset(Dataset):
    def __init__(self, file_list: list[str], device: str = "cpu"):
        self.files  = file_list
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ep = torch.load(self.files[idx], weights_only=True)
        return (
            ep["X_train"].float(),    # (B, N, p)
            ep["Z_train"].float(),    # (B, N, d)  — TabICL PIT
            ep["X_test"].float(),     # (B, n_test, p)
            ep["oracle_D"].float(),   # (B, n_test, d)
            ep["oracle_V"].float(),   # (B, n_test, d, r)
        )


def collate_squeeze(batch):
    """DataLoader collate: batch of size 1 → squeeze outer dim."""
    X_tr, Z_tr, X_te, D_ora, V_ora = batch[0]
    return X_tr, Z_tr, X_te, D_ora, V_ora


def cov_to_corr(D: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    S   = torch.diag_embed(D) + V @ V.transpose(-2, -1)
    std = S.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    return S / (std.unsqueeze(-1) * std.unsqueeze(-2))


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class CrossAttnLayer(nn.Module):
    def __init__(self, d_h: int, n_heads: int):
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

        attn_w = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        ctx    = torch.matmul(attn_w, V).transpose(1, 2).reshape(B, n_q, -1)
        ctx    = self.norm1(Q_in + self.W_o(ctx))
        ctx    = self.norm2(ctx + self.ff(ctx))
        return ctx, attn_w.mean(dim=1)   # (B, n_q, d_h), (B, n_q, N)


class ICLCorrNetPIT(nn.Module):
    """
    ICLCorrNet3 adapted for PIT-episode data (p=20, d=8).

    Q = enc_qry(X_te)              — query position
    K = enc_key(X_tr)              — key position  (X similarity)
    V = enc_val(vech(Z_tr⊗Z_tr))  — value content (Z correlation evidence)
    Stacked n_layers cross-attention + Cholesky correlation readout.
    """

    def __init__(self, p: int, d: int, d_h: int = 256,
                 n_heads: int = 8, n_layers: int = 2):
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
        self.enc_val = mlp(d_vech, d_h)
        self.layers  = nn.ModuleList(
            [CrossAttnLayer(d_h, n_heads) for _ in range(n_layers)]
        )

        d_L = d * (d + 1) // 2
        self.readout_L = nn.Sequential(
            nn.Linear(d_h * 2, d_h), nn.GELU(),
            nn.Linear(d_h, d_L),
        )

        ti, tj = torch.tril_indices(d, d)
        self.register_buffer("ti", ti)
        self.register_buffer("tj", tj)
        self.register_buffer("diag_idx", torch.arange(d))

    def forward(self, X_tr, Z_tr, X_te):
        B, N, _ = X_tr.shape
        d = self.d

        outer = Z_tr.unsqueeze(-1) * Z_tr.unsqueeze(-2)
        vech  = outer[:, :, self.ti, self.tj]

        Q  = self.enc_qry(X_te)
        K  = self.enc_key(X_tr)
        V  = self.enc_val(vech)

        ctx, attn_last = Q, None
        for layer in self.layers:
            ctx, attn_last = layer(ctx, K, V)

        L_flat = self.readout_L(torch.cat([ctx, Q], dim=-1))
        L = torch.zeros(B, Q.shape[1], d, d, device=X_tr.device, dtype=X_tr.dtype)
        L[:, :, self.ti, self.tj] = L_flat
        L[:, :, self.diag_idx, self.diag_idx] = (
            F.softplus(L[:, :, self.diag_idx, self.diag_idx]) + 1e-4
        )
        Sigma = L @ L.transpose(-2, -1)
        std   = Sigma.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
        R_pred = Sigma / (std.unsqueeze(-1) * std.unsqueeze(-2))
        return R_pred, attn_last


# ──────────────────────────────────────────────────────────────────────────────
# Data comparison
# ──────────────────────────────────────────────────────────────────────────────

def data_comparison(file_path: str, device: str):
    ep = torch.load(file_path, weights_only=True)
    Z  = ep["Z_train"].float()   # (B, N, d)
    D, V = ep["oracle_D"].float(), ep["oracle_V"].float()
    R = cov_to_corr(D, V)
    ri, ci = torch.triu_indices(D.shape[-1], D.shape[-1], offset=1)
    off_file = R[:, :, ri, ci]

    X_tr_g, Y_tr_g, _, _, oracle_g = generate_episode(
        B=1, p=8, d=6, r=4, n_train=128, n_test=16,
        device="cpu", hyperplane_bimodal=True, return_oracle=True,
    )
    D_g, V_g = oracle_g["D"], oracle_g["V"]
    R_g = cov_to_corr(D_g, V_g)
    ri_g, ci_g = torch.triu_indices(6, 6, offset=1)
    off_gen = R_g[:, :, ri_g, ci_g]

    ks_z   = [scipy_stats.kstest(Z[0, :, j].numpy(), "norm").statistic for j in range(Z.shape[-1])]
    ks_y   = [scipy_stats.kstest(Y_tr_g[0, :, j].numpy(), "norm").statistic for j in range(6)]

    print(f"\n{'─'*65}")
    print(f"DATA COMPARISON")
    print(f"{'─'*65}")
    print(f"  {'':20s}  {'FILE (pit_hyperplane)':>22s}  {'GENERATED':>15s}")
    print(f"  {'p':20s}  {ep['p']:>22d}  {'8':>15s}")
    print(f"  {'d':20s}  {ep['d']:>22d}  {'6':>15s}")
    print(f"  {'r':20s}  {V.shape[-1]:>22d}  {'4':>15s}")
    print(f"  {'n_train':20s}  {ep['n_train']:>22d}  {'128':>15s}")
    print(f"  {'Input':20s}  {'Z_tabicl':>22s}  {'Y_raw':>15s}")
    print(f"  {'Input mean':20s}  {Z.mean().item():>22.3f}  {Y_tr_g.mean().item():>15.3f}")
    print(f"  {'Input std':20s}  {Z.std().item():>22.3f}  {Y_tr_g.std().item():>15.3f}")
    print(f"  {'KS vs N(0,1) mean':20s}  {np.mean(ks_z):>22.3f}  {np.mean(ks_y):>15.3f}")
    print(f"  {'Oracle off-diag std':20s}  {off_file.std().item():>22.3f}  {off_gen.std().item():>15.3f}")
    print(f"  {'Oracle off-diag max':20s}  {off_file.abs().max().item():>22.3f}  {off_gen.abs().max().item():>15.3f}")
    print(f"  {'Bimodal structure':20s}  {'YES (same generator)':>22s}  {'YES':>15s}")
    print(f"{'─'*65}")


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model: nn.Module, val_files: list[str], device: str,
             n_files: int = VAL_FILES) -> float:
    ri, ci = torch.triu_indices(D, D, offset=1, device=device)
    model.eval()
    mses = []
    files = random.sample(val_files, min(n_files, len(val_files)))
    for f in files:
        ep = torch.load(f, weights_only=True)
        X_tr = ep["X_train"].float().to(device)
        Z_tr = ep["Z_train"].float().to(device)
        X_te = ep["X_test"].float().to(device)
        R_ora = cov_to_corr(ep["oracle_D"].float().to(device),
                            ep["oracle_V"].float().to(device))
        R_pred, _ = model(X_tr, Z_tr, X_te)
        mses.append(F.mse_loss(R_pred[:, :, ri, ci], R_ora[:, :, ri, ci]).item())
    model.train()
    return float(np.mean(mses))


# ──────────────────────────────────────────────────────────────────────────────
# Wandb image helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def make_all_instances_plot(model: nn.Module, val_file: str, device: str,
                             ep_idx: int = 0) -> tuple:
    """
    Plot ALL n_test instances for one val episode.

    Layout: 4 rows × (n_test // 2) cols
      row 0: oracle,    instances 0 .. half-1
      row 1: predicted, instances 0 .. half-1
      row 2: oracle,    instances half .. n_test-1
      row 3: predicted, instances half .. n_test-1

    Returns (fig, mean_mse).
    """
    ep    = torch.load(val_file, weights_only=True)
    b     = 0
    X_tr  = ep["X_train"].float()[[b]].to(device)
    Z_tr  = ep["Z_train"].float()[[b]].to(device)
    X_te  = ep["X_test"].float()[[b]].to(device)
    R_ora = cov_to_corr(ep["oracle_D"].float()[[b]].to(device),
                        ep["oracle_V"].float()[[b]].to(device))
    R_pred, _ = model(X_tr, Z_tr, X_te)

    ri, ci  = torch.triu_indices(D, D, offset=1, device=device)
    n_test  = R_ora.shape[1]
    half    = n_test // 2

    mse_per  = [F.mse_loss(R_pred[0, i, ri, ci], R_ora[0, i, ri, ci]).item()
                for i in range(n_test)]
    mean_mse = float(np.mean(mse_per))

    fig, axes = plt.subplots(4, half, figsize=(2.5 * half, 10))

    im = None
    for i in range(half):
        for grp in range(2):          # grp 0 → instances 0..half-1, grp 1 → half..n_test-1
            inst    = i + grp * half
            for row_off, is_oracle in enumerate([True, False]):
                row  = grp * 2 + row_off
                ax   = axes[row, i]
                R    = R_ora[0, inst] if is_oracle else R_pred[0, inst]
                im   = ax.imshow(R.cpu().numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
                if is_oracle:
                    ax.set_title(f"Oracle #{inst}", fontsize=7)
                else:
                    ax.set_title(f"MSE={mse_per[inst]:.3f}", fontsize=7, color="darkred")
                ax.set_xticks([]); ax.set_yticks([])

    for row, label in enumerate(
        [f"Oracle  0–{half-1}", f"Pred  0–{half-1}",
         f"Oracle  {half}–{n_test-1}", f"Pred  {half}–{n_test-1}"]
    ):
        axes[row, 0].set_ylabel(label, fontsize=8)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.35, pad=0.02)
    fig.suptitle(
        f"Val episode {ep_idx} — all {n_test} test instances  "
        f"(mean MSE={mean_mse:.4f})",
        fontsize=10,
    )
    fig.tight_layout()
    return fig, mean_mse


@torch.no_grad()
def make_scatter_image(model: nn.Module, val_files: list[str], device: str,
                       n_files: int = 10) -> wandb.Image:
    """Off-diagonal scatter: oracle vs predicted."""
    ri, ci = torch.triu_indices(D, D, offset=1)
    all_ora, all_pred = [], []
    for f in random.sample(val_files, min(n_files, len(val_files))):
        ep = torch.load(f, weights_only=True)
        X_tr = ep["X_train"].float().to(device)
        Z_tr = ep["Z_train"].float().to(device)
        X_te = ep["X_test"].float().to(device)
        R_ora  = cov_to_corr(ep["oracle_D"].float().to(device),
                             ep["oracle_V"].float().to(device))
        R_pred, _ = model(X_tr, Z_tr, X_te)
        all_ora.append(R_ora[:, :, ri, ci].cpu().flatten())
        all_pred.append(R_pred[:, :, ri, ci].cpu().flatten())
    ora  = torch.cat(all_ora).numpy()
    pred = torch.cat(all_pred).numpy()
    r    = float(np.corrcoef(ora, pred)[0, 1])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ora, pred, alpha=0.15, s=4, color="steelblue")
    ax.plot([-1, 1], [-1, 1], "r--", lw=0.8)
    ax.set_xlabel("Oracle off-diag"); ax.set_ylabel("Predicted off-diag")
    ax.set_title(f"Off-diagonal scatter  r={r:.3f}")
    img = wandb.Image(fig)
    plt.close(fig)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# ICL test
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def icl_test(model: nn.Module, val_files: list[str], device: str) -> dict:
    ri, ci = torch.triu_indices(D, D, offset=1, device=device)
    model.eval()

    ep_a = torch.load(val_files[0], weights_only=True)
    ep_b = torch.load(val_files[1], weights_only=True)
    b = 0

    Xa_tr = ep_a["X_train"].float()[[b]].to(device)
    Za_tr = ep_a["Z_train"].float()[[b]].to(device)
    Xa_te = ep_a["X_test"].float()[[b]].to(device)
    Ra    = cov_to_corr(ep_a["oracle_D"].float()[[b]].to(device),
                        ep_a["oracle_V"].float()[[b]].to(device))

    Xb_tr = ep_b["X_train"].float()[[b]].to(device)
    Zb_tr = ep_b["Z_train"].float()[[b]].to(device)

    def mse(Rp, Ro): return F.mse_loss(Rp[:, :, ri, ci], Ro[:, :, ri, ci]).item()
    def div(Rp):     return Rp[0, :, ri, ci].std(dim=0).mean().item()
    def hn(w):
        w = w.clamp(min=1e-10)
        return (-(w * w.log()).sum(-1).mean() / math.log(max(N_TRAIN, 2))).item()

    Raa, waa = model(Xa_tr, Za_tr, Xa_te)
    Rba, wba = model(Xb_tr, Zb_tr, Xa_te)
    Rza, _   = model(torch.zeros_like(Xa_tr), torch.zeros_like(Za_tr), Xa_te)
    Ra0, _   = model(Xa_tr, Za_tr, torch.zeros_like(Xa_te))

    chg_swap = (Rba - Raa).abs().mean().item()
    chg_zero = (Rza - Raa).abs().mean().item()

    print(f"\n{'═'*65}")
    print(f"ICL TEST (PIT dataset)")
    print(f"{'═'*65}")
    print(f"  Baseline   MSE={mse(Raa,Ra):.5f}  div={div(Raa):.4f}  H={hn(waa):.4f}")
    print(f"  Supp swap  MSE={mse(Rba,Ra):.5f}  div={div(Rba):.4f}  "
          f"H={hn(wba):.4f}  chg={chg_swap:.5f}")
    print(f"  Zero supp  MSE={mse(Rza,Ra):.5f}  div={div(Rza):.4f}  chg={chg_zero:.5f}")
    print(f"  Zero Xte   MSE={mse(Ra0,Ra):.5f}  div={div(Ra0):.4f}")

    doing_icl = chg_swap > 0.02
    verdict   = "TRUE ICL" if doing_icl else "X REGRESSION"
    print(f"\n  VERDICT: {verdict}  chg_swap={chg_swap:.4f}")
    model.train()
    return {
        "mse_baseline": mse(Raa, Ra),
        "mse_swap":     mse(Rba, Ra),
        "chg_swap":     chg_swap,
        "chg_zero":     chg_zero,
        "verdict":      verdict,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(model: nn.Module, train_files: list[str], val_files: list[str], device: str):
    ri, ci    = torch.triu_indices(D, D, offset=1, device=device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_STEPS, eta_min=LR * 0.1)

    dataset    = PitEpisodeDataset(train_files)
    loader     = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4,
                            collate_fn=collate_squeeze, pin_memory=True)
    loader_it  = iter(loader)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nparams={n_params:,}  d_h={D_HIDDEN}  layers={N_LAYERS}  "
          f"B={BATCH_SIZE}  {N_STEPS} steps")
    print(f"train files={len(train_files)}  val files={len(val_files)}")
    print(f"{'─'*65}")
    print(f"  {'step':>6}  {'MSE':>8}  {'div':>8}  {'H_norm':>8}  {'LR':>10}")

    model.train()
    for step in range(N_STEPS):
        try:
            X_tr, Z_tr, X_te, D_ora, V_ora = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            X_tr, Z_tr, X_te, D_ora, V_ora = next(loader_it)

        X_tr   = X_tr.to(device)
        Z_tr   = Z_tr.to(device)
        X_te   = X_te.to(device)
        R_ora  = cov_to_corr(D_ora.to(device), V_ora.to(device))

        optimizer.zero_grad()
        R_pred, attn_w = model(X_tr, Z_tr, X_te)
        loss = F.mse_loss(R_pred[:, :, ri, ci], R_ora[:, :, ri, ci])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                mse  = loss.item()
                div  = R_pred[:, :, ri, ci].std(dim=1).mean().item()
                w    = attn_w.clamp(min=1e-10)
                h    = (-(w * w.log()).sum(-1).mean() / math.log(max(N_TRAIN, 2))).item()
                lr   = optimizer.param_groups[0]["lr"]
            print(f"  {step:>6d}  {mse:>8.5f}  {div:>8.5f}  {h:>8.4f}  {lr:>10.2e}")
            wandb.log({"train/mse": mse, "train/div": div,
                       "train/h_norm": h, "train/lr": lr}, step=step)

        if step % VAL_EVERY == 0 and step > 0:
            val_mse = validate(model, val_files, device)
            print(f"  {'val':>6s}  {val_mse:>8.5f}")
            wandb.log({"val/mse": val_mse}, step=step)

        if step % IMAGE_EVERY == 0:
            fig_mid, _ = make_all_instances_plot(model, val_files[2], device, ep_idx=0)
            img_mid     = wandb.Image(fig_mid)
            plt.close(fig_mid)
            img_scatter = make_scatter_image(model, val_files[:20], device)
            wandb.log({"images/corr_matrices": img_mid,
                       "images/scatter":       img_scatter}, step=step)

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  p={P} d={D}  N_train={N_TRAIN}")

    # ── File split ────────────────────────────────────────────────────────────
    all_files   = sorted(glob.glob(str(DATA_DIR / "episode_*.pt")))
    random.seed(42)
    random.shuffle(all_files)
    n_val       = min(2_000, len(all_files) // 10)
    val_files   = all_files[:n_val]
    train_files = all_files[n_val:]
    print(f"Episodes: {len(all_files)} total  |  train={len(train_files)}  val={len(val_files)}")

    # ── Data comparison ───────────────────────────────────────────────────────
    data_comparison(train_files[0], device)

    # ── Baseline MSE (independence assumption → identity matrix) ──────────────
    with torch.no_grad():
        ri_b, ci_b = torch.triu_indices(D, D, offset=1)
        mses = []
        for f in random.sample(val_files, 100):
            ep   = torch.load(f, weights_only=True)
            D_o  = ep["oracle_D"].float()
            V_o  = ep["oracle_V"].float()
            R    = cov_to_corr(D_o, V_o)               # (B, n_test, d, d)
            I    = torch.eye(D).unsqueeze(0).unsqueeze(0).expand_as(R)
            mses.append(F.mse_loss(I[:, :, ri_b, ci_b], R[:, :, ri_b, ci_b]).item())
        baseline_mse = float(np.mean(mses))
    print(f"\nIndependence baseline MSE: {baseline_mse:.5f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    torch.manual_seed(3)
    model = ICLCorrNetPIT(p=P, d=D, d_h=D_HIDDEN,
                         n_heads=N_HEADS, n_layers=N_LAYERS).to(device)

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN,
        config={
            "p": P, "d": D, "n_train": N_TRAIN, "n_test": N_TEST,
            "d_h": D_HIDDEN, "n_heads": N_HEADS, "n_layers": N_LAYERS,
            "n_steps": N_STEPS, "lr": LR, "batch_size": BATCH_SIZE,
            "input": "Z_tabicl",
            "baseline_mse": baseline_mse,
            "n_params": sum(p.numel() for p in model.parameters()),
            "train_files": len(train_files),
            "val_files": len(val_files),
        },
    )
    wandb.log({"baseline/mse": baseline_mse}, step=0)

    # ── Train ─────────────────────────────────────────────────────────────────
    train(model, train_files, val_files, device)

    # ── Final val ─────────────────────────────────────────────────────────────
    val_mse = validate(model, val_files, device, n_files=VAL_FILES)
    improvement = (1 - val_mse / baseline_mse) * 100
    print(f"\nFinal val MSE ({VAL_FILES} files): {val_mse:.5f}  "
          f"(baseline: {baseline_mse:.5f}  improvement: {improvement:.1f}%)")
    wandb.log({"val/mse_final": val_mse,
               "val/improvement_pct": improvement}, step=N_STEPS)

    # ── ICL test ──────────────────────────────────────────────────────────────
    icl_res = icl_test(model, val_files, device)
    wandb.log({
        "icl/mse_baseline": icl_res["mse_baseline"],
        "icl/mse_swap":     icl_res["mse_swap"],
        "icl/chg_swap":     icl_res["chg_swap"],
        "icl/chg_zero":     icl_res["chg_zero"],
        "icl/verdict":      icl_res["verdict"],
    }, step=N_STEPS)

    # ── Final images: all test instances for N_PLOT_EPISODES val episodes ────
    N_PLOT_EPISODES = 8
    print(f"\nGenerating final plots for {N_PLOT_EPISODES} val episodes...")
    wandb_episode_imgs = []
    for ep_idx in range(N_PLOT_EPISODES):
        fig, ep_mse = make_all_instances_plot(
            model, val_files[ep_idx], device, ep_idx=ep_idx
        )
        save_path = OUTPUT_DIR / f"val_ep{ep_idx:02d}_all_instances.png"
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        wandb_episode_imgs.append(
            wandb.Image(fig, caption=f"Val ep {ep_idx}  MSE={ep_mse:.4f}")
        )
        plt.close(fig)
        print(f"  Saved {save_path.name}  (MSE={ep_mse:.4f})")

    img_scatter = make_scatter_image(model, val_files[:50], device)
    wandb.log({
        "images/val_all_instances": wandb_episode_imgs,
        "images/final_scatter":     img_scatter,
    }, step=N_STEPS)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"SUMMARY")
    print(f"{'═'*65}")
    print(f"  v3/Y (generated, p=8, d=6):  val MSE = 0.02260  (75.9% over baseline)")
    print(f"  Z_emp (generated, p=8, d=6):  val MSE = 0.02447  (73.8% over baseline)")
    print(f"  PIT dataset (Z_tabicl, p=20, d=8): val MSE = {val_mse:.5f}  ({improvement:.1f}% over baseline)")
    print(f"  ICL: {icl_res['verdict']}  swap_chg={icl_res['chg_swap']:.4f}")
    print(f"Plots: {OUTPUT_DIR}/")

    wandb.finish()


if __name__ == "__main__":
    main()
