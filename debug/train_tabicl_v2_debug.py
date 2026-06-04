"""
train_tabicl_v2_debug.py — CopulaTabICLv2 debug training loop.

MSE-only on pit_hyperplane_debug episodes, N_STEPS steps.
Monitors per-block attention entropy, ICL gate, and prediction diversity.
Every VAL_EVERY steps runs the full validation suite from train.py:
  copula NLL, oracle NLL, OAS / kNN-5 / linear-factor baselines, energy score,
  off-diagonal scatter, and per-instance correlation grids — all saved to disk.

Usage (from project root):
    conda run -n multivariate-icl python debug/train_tabicl_v2_debug.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.covariance import OAS
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import infinite_episode_iter, make_episode_loader, split_episode_files
from loss import indep_normal_nll, woodbury_nll
from model import build_copula_tabicl_v2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR   = ROOT / "data" / "pit_hyperplane_debug"
OUTPUT_DIR = ROOT / "debug" / "tabicl_v2_train_debug_fixed"
N_STEPS    = 15_000
LR         = 3e-4
WD         = 1e-4
GRAD_CLIP  = 1.0
LOG_EVERY  = 100
ATTN_EVERY = 500
VAL_EVERY  = 2000   # full NLL validation (saves plots to OUTPUT_DIR/val/)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _cov_to_corr(D: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    S   = torch.diag_embed(D) + V @ V.transpose(-2, -1)
    std = S.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    return S / (std.unsqueeze(-1) * std.unsqueeze(-2))


def prediction_diversity(Sigma_pred: torch.Tensor) -> float:
    d  = Sigma_pred.shape[-1]
    ri, ci = torch.triu_indices(d, d, offset=1, device=Sigma_pred.device)
    off = Sigma_pred[..., ri, ci]
    return off.std(dim=-2).mean().item()


def get_s3_attn_entropy(model, X_fwd, Z_fwd, n_support):
    """Capture Stage-3 attention weights and return normalised entropy per block."""
    captured: dict[int, torch.Tensor] = {}
    orig_forwards = []
    for i, blk in enumerate(model.s3_blocks):
        orig_forwards.append(blk.forward)
        def _patch(idx, orig):
            def _f(x, ns, **kw):
                out, w = orig(x, ns, return_attn_weights=True, **kw)
                captured[idx] = w.detach().cpu()
                return out
            return _f
        blk.forward = _patch(i, blk.forward)

    was_train = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(X_fwd, Z_fwd, n_support=n_support)
    finally:
        for i, blk in enumerate(model.s3_blocks):
            blk.forward = orig_forwards[i]
        if was_train:
            model.train()

    log_ns = math.log(max(n_support, 2))
    h_norms = {}
    for idx, w in captured.items():
        wq = w[:, n_support:, :n_support].clamp(min=1e-12)
        H  = -(wq * wq.log()).sum(dim=-1).mean().item()
        h_norms[idx] = H / log_ns
    return h_norms


# ---------------------------------------------------------------------------
# Validation helpers (mirroring train.py's run_val_pit)
# ---------------------------------------------------------------------------

def _cov_to_woodbury_params(Sigma: torch.Tensor):
    """Decompose (N, d, d) covariance into (mu, D, V) Woodbury params."""
    diag_vals = Sigma.diagonal(dim1=-2, dim2=-1).clamp(min=1e-6)
    off_diag  = Sigma - torch.diag_embed(diag_vals)
    eigvals, eigvecs = torch.linalg.eigh(off_diag)
    eigvals = eigvals.clamp(min=0.0)
    V_out   = eigvecs * eigvals.sqrt().unsqueeze(-2)
    N, d    = diag_vals.shape
    mu_out  = torch.zeros(N, d, device=Sigma.device)
    return mu_out, diag_vals, V_out


def _energy_score(mu, D, V, y_ref, n_samples=50):
    B, T, d = mu.shape
    r = V.shape[-1]
    eps_d = torch.randn(B, T, n_samples, d, device=mu.device, dtype=mu.dtype)
    eps_r = torch.randn(B, T, n_samples, r, device=mu.device, dtype=mu.dtype)
    samples = mu.unsqueeze(2) + D.unsqueeze(2).sqrt() * eps_d + (eps_r @ V.transpose(-2, -1))
    term1 = (samples - y_ref.unsqueeze(2)).norm(dim=-1).mean(dim=-1)
    sf = samples.reshape(B * T, n_samples, d)
    term2 = torch.cdist(sf, sf).mean(dim=(-2, -1)).reshape(B, T)
    return (term1 - 0.5 * term2).mean().item()


def run_val(model, val_episodes, step, device, output_dir):
    """Full validation: NLL metrics + baselines + plots saved to output_dir."""
    model.eval()
    agg = {k: [] for k in [
        "copula_nll", "joint_y_nll", "train_nll",
        "oas_nll", "knn5_nll", "linear_nll",
        "oracle_nll_z", "oracle_nll_y",
        "vs_knn5", "oracle_frac", "energy_score",
    ]}
    all_off_pred, all_off_ora = [], []
    plot_eps = []   # store up to 2 episodes for correlation grid

    with torch.no_grad():
        for i_ep, ep in enumerate(val_episodes):
            X_tr      = ep["X_train"].to(device)
            Z_tr      = ep["Z_train"].to(device)
            X_te      = ep["X_test"].to(device)
            Z_te      = ep["Z_test"].to(device)
            log_p_te  = ep["log_p_test"].to(device)
            oracle_mu = ep["oracle_mu"].to(device)
            oracle_D  = ep["oracle_D"].to(device)
            oracle_V  = ep["oracle_V"].to(device)
            Y_test    = ep["Y_test"].to(device)

            B, n_train, _ = Z_tr.shape
            _, n_test, d  = Z_te.shape

            # 70/30 split: support from train, query = remaining train + all test
            n_sup    = max(1, int(0.7 * n_train))
            perm     = torch.randperm(n_train, device=device)
            X_sup    = X_tr[:, perm[:n_sup]]
            Z_sup    = Z_tr[:, perm[:n_sup]]
            X_tr_qry = X_tr[:, perm[n_sup:]]
            Z_tr_qry = Z_tr[:, perm[n_sup:]]
            n_tr_qry = X_tr_qry.shape[1]

            X_fwd = torch.cat([X_sup, X_tr_qry, X_te], dim=1)
            Z_fwd = torch.cat([Z_sup,
                                torch.zeros_like(Z_tr_qry),
                                torch.zeros_like(Z_te)], dim=1)

            mu_all, d_all, V_all = model(X_fwd, Z_fwd, n_support=n_sup)
            mu_all = mu_all.float(); d_all = d_all.float(); V_all = V_all.float()

            mu_tr, d_tr, V_tr = mu_all[:, :n_tr_qry], d_all[:, :n_tr_qry], V_all[:, :n_tr_qry]
            mu_Z,  d_Z,  V_Z  = mu_all[:, n_tr_qry:], d_all[:, n_tr_qry:], V_all[:, n_tr_qry:]

            indep_z = indep_normal_nll(Z_te).item()

            # train / test copula NLL
            train_cnll = woodbury_nll(Z_tr_qry, mu_tr, d_tr, V_tr).item() - indep_normal_nll(Z_tr_qry).item()
            wnll  = woodbury_nll(Z_te, mu_Z, d_Z, V_Z).item()
            cnll  = wnll - indep_z
            marginal_nll = -log_p_te.sum(-1).mean().item()
            agg["copula_nll"].append(cnll)
            agg["joint_y_nll"].append(cnll + marginal_nll)
            agg["train_nll"].append(train_cnll)
            agg["energy_score"].append(_energy_score(mu_Z, d_Z, V_Z, Z_te))

            # baselines: OAS, kNN-5, linear factor
            oas_nlls, knn5_nlls, lin_nlls = [], [], []
            for b in range(B):
                Z_tr_np = Z_tr[b].cpu().numpy()
                Z_tr_b, Z_te_b = Z_tr[b], Z_te[b]
                X_tr_b, X_te_b = X_tr[b], X_te[b]

                oas = OAS().fit(Z_tr_np)
                Sig_oas = torch.tensor(oas.covariance_, dtype=torch.float32, device=device)
                mu_p, d_p, V_p = _cov_to_woodbury_params(Sig_oas.unsqueeze(0))
                oas_nlls.append(woodbury_nll(
                    Z_te_b.unsqueeze(0),
                    torch.zeros(n_test, d, device=device).unsqueeze(0),
                    d_p.expand(n_test, -1).unsqueeze(0),
                    V_p.expand(n_test, -1, -1).unsqueeze(0),
                ).item())

                dists = torch.cdist(X_te_b, X_tr_b)
                k_eff = min(5, n_train)
                idx   = dists.topk(k_eff, largest=False).indices
                Z_nb  = Z_tr_b[idx]
                mu_nb = Z_nb.mean(dim=1)
                if k_eff > d:
                    Z_c = Z_nb - mu_nb.unsqueeze(1)
                    Sig_knn = Z_c.transpose(-2, -1) @ Z_c / max(k_eff - 1, 1)
                    Sig_knn = 0.5 * (Sig_knn + Sig_knn.transpose(-2, -1))
                else:
                    Sig_knn = torch.diag_embed(Z_nb.var(dim=1, unbiased=False).clamp(min=1e-6))
                mu_k, d_k, V_k = _cov_to_woodbury_params(Sig_knn, )
                knn5_nlls.append(woodbury_nll(
                    Z_te_b.unsqueeze(0),
                    mu_nb.unsqueeze(0),
                    d_k.unsqueeze(0),
                    V_k.unsqueeze(0),
                ).item())

                W       = torch.linalg.lstsq(X_tr_b, Z_tr_b).solution
                mu_lin  = X_te_b @ W
                resid   = Z_tr_b - X_tr_b @ W
                Sig_res = (torch.cov(resid.T) if n_train > d
                           else torch.diag(resid.var(0, unbiased=False).clamp(min=1e-6)))
                Sig_res = 0.5 * (Sig_res + Sig_res.T)
                _, d_l, V_l = _cov_to_woodbury_params(Sig_res.unsqueeze(0))
                lin_nlls.append(woodbury_nll(
                    Z_te_b.unsqueeze(0),
                    mu_lin.unsqueeze(0),
                    d_l.expand(n_test, -1).unsqueeze(0),
                    V_l.expand(n_test, -1, -1).unsqueeze(0),
                ).item())

            oas_c  = float(np.mean(oas_nlls))  - indep_z
            knn5_c = float(np.mean(knn5_nlls)) - indep_z
            lin_c  = float(np.mean(lin_nlls))  - indep_z
            agg["oas_nll"].append(oas_c)
            agg["knn5_nll"].append(knn5_c)
            agg["linear_nll"].append(lin_c)
            agg["vs_knn5"].append(cnll - knn5_c)

            # oracle copula NLL in Z-space
            Sig_Y = torch.diag_embed(oracle_D) + oracle_V @ oracle_V.transpose(-2, -1)
            std_Y = Sig_Y.diagonal(dim1=-2, dim2=-1).sqrt().clamp(min=1e-8)
            rho   = Sig_Y / (std_Y.unsqueeze(-1) * std_Y.unsqueeze(-2))
            lam, U_eig = torch.linalg.eigh(rho)
            _delta = 1e-4
            D_rho  = torch.full_like(oracle_D, _delta)
            V_rho  = U_eig * (lam - _delta).clamp(min=0).sqrt().unsqueeze(-2)
            oracle_z = woodbury_nll(Z_te, torch.zeros_like(oracle_mu), D_rho, V_rho).item() - indep_z
            oracle_y = woodbury_nll(Y_test, oracle_mu, oracle_D, oracle_V).item()
            agg["oracle_nll_z"].append(oracle_z)
            agg["oracle_nll_y"].append(oracle_y)
            denom = -oracle_z
            agg["oracle_frac"].append((cnll - oracle_z) / denom if abs(denom) > 1e-8 else float("nan"))

            # collect off-diagonal values for scatter
            Sig_pp = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
            std_pp = Sig_pp.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
            R_pp   = Sig_pp / (std_pp.unsqueeze(-1) * std_pp.unsqueeze(-2))
            Sig_oo = torch.diag_embed(oracle_D) + oracle_V @ oracle_V.transpose(-2, -1)
            std_oo = Sig_oo.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
            R_oo   = Sig_oo / (std_oo.unsqueeze(-1) * std_oo.unsqueeze(-2))
            ri_p, ci_p = torch.triu_indices(d, d, offset=1, device=device)
            all_off_pred.append(R_pp[..., ri_p, ci_p].float().cpu().numpy().flatten())
            all_off_ora.append(R_oo[..., ri_p, ci_p].float().cpu().numpy().flatten())

            if len(plot_eps) < 2:
                plot_eps.append({"d_Z": d_Z.clone(), "V_Z": V_Z.clone(),
                                 "oracle_D": oracle_D, "oracle_V": oracle_V,
                                 "key": f"ep{i_ep}"})

    metrics = {k: float(np.nanmean(v)) for k, v in agg.items()}

    print(
        f"[val step={step:>6d}]  "
        f"copula={metrics['copula_nll']:.4f}  "
        f"oracle_z={metrics['oracle_nll_z']:.4f}  "
        f"oracle_frac={metrics['oracle_frac']:.4f}  "
        f"oas={metrics['oas_nll']:.4f}  "
        f"knn5={metrics['knn5_nll']:.4f}  "
        f"vs_knn5={metrics['vs_knn5']:.4f}  "
        f"energy={metrics['energy_score']:.4f}"
    )

    # ---- Plots ----
    val_dir = Path(output_dir) / "val"
    val_dir.mkdir(parents=True, exist_ok=True)

    # Scatter: predicted vs oracle off-diagonal correlations
    if all_off_pred:
        off_p = np.concatenate(all_off_pred)
        off_o = np.concatenate(all_off_ora)
        r_sc    = float(np.corrcoef(off_o, off_p)[0, 1]) if off_p.std() > 1e-8 else float("nan")
        slope_sc = float(np.polyfit(off_o, off_p, 1)[0]) if off_o.std() > 1e-8 else float("nan")
        fig_sc, ax_sc = plt.subplots(figsize=(5, 5))
        ax_sc.scatter(off_o, off_p, s=2, alpha=0.3, rasterized=True)
        lo = min(float(off_o.min()), float(off_p.min()))
        hi = max(float(off_o.max()), float(off_p.max()))
        ax_sc.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax_sc.set_xlabel("Oracle off-diag corr")
        ax_sc.set_ylabel("Predicted off-diag corr")
        ax_sc.set_title(f"step {step} — r={r_sc:.3f}  slope={slope_sc:.3f}  n={len(off_p):,}")
        fig_sc.tight_layout()
        sc_path = val_dir / f"step{step:07d}_scatter.png"
        fig_sc.savefig(sc_path, dpi=100, bbox_inches="tight")
        plt.close(fig_sc)
        metrics["r"] = r_sc
        metrics["slope"] = slope_sc

    # Correlation grid: oracle vs predicted for all test instances (up to 2 episodes)
    for ep_data in plot_eps:
        d_Z_ep = ep_data["d_Z"]
        V_Z_ep = ep_data["V_Z"]
        oD_ep  = ep_data["oracle_D"]
        oV_ep  = ep_data["oracle_V"]
        key    = ep_data["key"]
        B_ep   = d_Z_ep.shape[0]
        for b_idx in range(min(2, B_ep)):
            Sig_pred = (torch.diag_embed(d_Z_ep[b_idx])
                        + V_Z_ep[b_idx] @ V_Z_ep[b_idx].transpose(-2, -1))
            std_p = Sig_pred.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
            R_pred = (Sig_pred / (std_p.unsqueeze(-1) * std_p.unsqueeze(-2))).float().cpu()

            Sig_ora = (torch.diag_embed(oD_ep[b_idx])
                       + oV_ep[b_idx] @ oV_ep[b_idx].transpose(-2, -1))
            std_o = Sig_ora.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
            R_ora = (Sig_ora / (std_o.unsqueeze(-1) * std_o.unsqueeze(-2))).float().cpu()

            ri_g, ci_g = torch.triu_indices(R_pred.shape[-1], R_pred.shape[-1], offset=1)
            mse_per = [F.mse_loss(R_pred[i, ri_g, ci_g], R_ora[i, ri_g, ci_g]).item()
                       for i in range(R_pred.shape[0])]
            n_test_ep = R_pred.shape[0]
            half = max(n_test_ep // 2, 1)
            n_cols = max(half, 1)

            fig_g, axes = plt.subplots(4, n_cols, figsize=(2.5 * n_cols, 10),
                                       constrained_layout=True)
            if n_cols == 1:
                axes = axes[:, np.newaxis]
            for i in range(half):
                for grp in range(2):
                    inst = i + grp * half
                    if inst >= n_test_ep:
                        continue
                    for row_off, is_ora in enumerate([True, False]):
                        row = grp * 2 + row_off
                        ax  = axes[row, i]
                        R   = R_ora[inst].numpy() if is_ora else R_pred[inst].numpy()
                        im  = ax.imshow(R, vmin=-1, vmax=1, cmap="RdBu_r")
                        ax.set_title(
                            f"Oracle #{inst}" if is_ora else f"MSE={mse_per[inst]:.3f}",
                            fontsize=7, color="black" if is_ora else "darkred"
                        )
                        ax.set_xticks([]); ax.set_yticks([])
            for row, lbl in enumerate([
                f"Oracle 0–{half-1}", f"Pred 0–{half-1}",
                f"Oracle {half}–{n_test_ep-1}", f"Pred {half}–{n_test_ep-1}",
            ]):
                axes[row, 0].set_ylabel(lbl, fontsize=8)
            fig_g.colorbar(im, ax=axes.ravel().tolist(), shrink=0.35, pad=0.02)
            fig_g.suptitle(
                f"{key} b={b_idx}  |  {n_test_ep} instances — mean MSE={np.mean(mse_per):.4f}",
                fontsize=10
            )
            grid_path = val_dir / f"step{step:07d}_corr_{key}_b{b_idx}.png"
            fig_g.savefig(grid_path, dpi=90, bbox_inches="tight")
            plt.close(fig_g)

    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Data: {DATA_DIR}")

    # ---- Model ----
    cfg_model = OmegaConf.load(ROOT / "conf" / "model" / "copula_tabicl_v2.yaml")
    model = build_copula_tabicl_v2(SimpleNamespace(model=cfg_model)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}  |  d_model={cfg_model.d_model}  "
          f"d_icl={model.d_icl}  n_s3={len(model.s3_blocks)}")

    # ---- Optimizer ----
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_STEPS, eta_min=1e-6)

    # ---- Data ----
    train_files, val_files = split_episode_files(str(DATA_DIR), val_n_episodes=50)
    print(f"Episodes: {len(train_files)} train / {len(val_files)} val")
    loader = make_episode_loader(files=train_files, shuffle=True, num_workers=2)
    ep_iter = infinite_episode_iter(loader)

    # Pre-load all val episodes for validation
    print(f"Pre-loading {len(val_files)} val episodes...")
    val_episodes = [torch.load(f, weights_only=True) for f in val_files]

    # Single fixed val episode for attention diagnostics
    val_ep = val_episodes[0]
    Xv_tr  = val_ep["X_train"][[0]].float().to(device)
    Zv_tr  = val_ep["Z_train"][[0]].float().to(device)
    Xv_te  = val_ep["X_test"][[0]].float().to(device)
    Zv_te  = val_ep["Z_test"][[0]].float().to(device)
    Nv     = Xv_tr.shape[1]
    Xv_fwd = torch.cat([Xv_tr, Xv_te], dim=1)
    Zv_fwd = torch.cat([Zv_tr, Zv_te], dim=1)

    n_s3 = len(model.s3_blocks)

    # ---- Logging buffers ----
    steps_log   = []
    mse_log     = []
    div_log     = []
    gate_log    = []
    attn_steps  = []
    h_log       = {i: [] for i in range(n_s3)}
    alpha_a_log = {i: [] for i in range(n_s3)}
    alpha_f_log = {i: [] for i in range(n_s3)}
    val_steps   = []
    val_copula  = []
    val_oracle  = []
    val_oas     = []
    val_knn5    = []

    # Diagnostic: check correlation matrix at step 0 to verify initialization
    model.eval()
    with torch.no_grad():
        _ep = val_episodes[0]
        _Xv = _ep["X_train"].to(device)
        _Zv = _ep["Z_train"].to(device)
        _Xq = _ep["X_test"].to(device)
        _Zq = _ep["Z_test"].to(device)
        _Nv = _Zv.shape[1]
        _mu, _dZ, _VZ = model(torch.cat([_Xv, _Xq], 1), torch.cat([_Zv, _Zq], 1), _Nv)
        _S = torch.diag_embed(_dZ) + _VZ @ _VZ.transpose(-2, -1)
        _nt = _S.shape[-1]
        _ri, _ci = torch.triu_indices(_nt, _nt, offset=1, device=device)
        print(
            f"[init]  C_diag mean={_dZ.mean():.3f}  "
            f"off_diag mean={_S[..., _ri, _ci].mean():.3f}  "
            f"off_diag std={_S[..., _ri, _ci].std():.3f}"
        )
    del _ep, _Xv, _Zv, _Xq, _Zq, _mu, _dZ, _VZ, _S

    t0 = time.perf_counter()
    model.train()

    for step in range(N_STEPS):
        ep      = next(ep_iter)
        X_train = ep["X_train"].to(device)
        Z_train = ep["Z_train"].to(device)
        X_test  = ep["X_test"].to(device)
        Z_test  = ep["Z_test"].to(device)
        oD      = ep["oracle_D"].to(device)
        oV      = ep["oracle_V"].to(device)

        B, N, d = Z_train.shape
        X_fwd   = torch.cat([X_train, X_test], dim=1)
        Z_fwd   = torch.cat([Z_train, Z_test], dim=1)

        optimizer.zero_grad()
        mu_Z, d_Z, V_Z = model(X_fwd, Z_fwd, n_support=N)

        # NLL loss on query Z values (copula training objective)
        loss_nll = woodbury_nll(Z_test, mu_Z, d_Z, V_Z)

        # Per-instance auxiliary loss: MSE between predicted R and per-instance oracle correlation.
        # Using oracle_D/oracle_V directly avoids the "mean over instances" that was training
        # toward a constant predictor.
        with torch.no_grad():
            S_oracle = torch.diag_embed(oD) + oV @ oV.transpose(-2, -1)
            std_o = S_oracle.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
            R_oracle = S_oracle / (std_o.unsqueeze(-1) * std_o.unsqueeze(-2))
            R_oracle = R_oracle.clamp(-1 + 1e-6, 1 - 1e-6)

        Sigma_pred = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
        ri_a, ci_a = torch.triu_indices(d, d, offset=1, device=device)
        loss_aux = F.mse_loss(Sigma_pred[..., ri_a, ci_a], R_oracle[..., ri_a, ci_a])

        LAMBDA_AUX  = 0.3
        loss        = loss_nll + LAMBDA_AUX * loss_aux

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                mse_val  = loss_nll.item() - indep_normal_nll(Z_test).item()  # copula NLL
                div_val  = prediction_diversity(Sigma_pred)
                gate_val = torch.sigmoid(model.icl_gate_sup).mean().item()
            steps_log.append(step)
            mse_log.append(mse_val)
            div_log.append(div_val)
            gate_log.append(gate_val)

        if step % ATTN_EVERY == 0:
            h_norms = get_s3_attn_entropy(model, Xv_fwd, Zv_fwd, Nv)
            attn_steps.append(step)
            h_vals = [h_norms.get(i, float("nan")) for i in range(n_s3)]
            has_rezero = hasattr(model.s3_blocks[0], "alpha_attn")
            aa = [model.s3_blocks[i].alpha_attn.item() if has_rezero else 0.0 for i in range(n_s3)]
            af = [model.s3_blocks[i].alpha_ffn.item()  if has_rezero else 0.0 for i in range(n_s3)]
            for i in range(n_s3):
                h_log[i].append(h_vals[i])
                alpha_a_log[i].append(aa[i])
                alpha_f_log[i].append(af[i])

            gate_v  = torch.sigmoid(model.icl_gate_sup).mean().item()
            elapsed = time.perf_counter() - t0
            h_str   = " ".join(f"b{i}={h_vals[i]:.3f}" for i in range(n_s3))
            print(
                f"[{step:>6d}]  nll={mse_log[-1] if mse_log else float('nan'):.5f}  "
                f"aux={loss_aux.item():.4f}  div={div_log[-1] if div_log else float('nan'):.4f}  "
                f"gate={gate_v:.3f}  H_norm=[{h_str}]  t={elapsed:.0f}s"
            )
            model.train()

        if step % VAL_EVERY == 0 and step > 0:
            vm = run_val(model, val_episodes, step, device, str(OUTPUT_DIR))
            val_steps.append(step)
            val_copula.append(vm["copula_nll"])
            val_oracle.append(vm["oracle_nll_z"])
            val_oas.append(vm["oas_nll"])
            val_knn5.append(vm["knn5_nll"])

    # ---------------------------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------------------------
    vm_final = run_val(model, val_episodes, N_STEPS, device, str(OUTPUT_DIR))
    val_steps.append(N_STEPS)
    val_copula.append(vm_final["copula_nll"])
    val_oracle.append(vm_final["oracle_nll_z"])
    val_oas.append(vm_final["oas_nll"])
    val_knn5.append(vm_final["knn5_nll"])

    # Final attention diagnostics on fixed val episode
    final_h = get_s3_attn_entropy(model, Xv_fwd, Zv_fwd, Nv)
    model.eval()
    with torch.no_grad():
        mu_f, d_f, V_f = model(Xv_fwd, Zv_fwd, n_support=Nv)
    Sigma_f  = torch.diag_embed(d_f) + V_f @ V_f.transpose(-2, -1)
    final_div = prediction_diversity(Sigma_f)

    print(f"\n{'='*70}")
    print(f"FINAL  r={vm_final.get('r', float('nan')):.3f}  "
          f"slope={vm_final.get('slope', float('nan')):.3f}  "
          f"copula_nll={vm_final['copula_nll']:.4f}  "
          f"oracle_frac={vm_final['oracle_frac']:.4f}  "
          f"div={final_div:.4f}")
    print("Stage-3 attention entropy (H_norm):")
    for i, h in final_h.items():
        status = "COLLAPSED" if h > 0.90 else ("ok" if h < 0.5 else "partial")
        print(f"  blk {i}: {h:.4f}  [{status}]")
    has_rezero = hasattr(model.s3_blocks[0], "alpha_attn")
    if not has_rezero:
        print("ReZero: removed (standard pre-norm residuals)")
    print(f"ICL gate: {torch.sigmoid(model.icl_gate_sup).mean().item():.4f}")
    print(f"{'='*70}")

    # ---------------------------------------------------------------------------
    # Training diagnostics plot
    # ---------------------------------------------------------------------------
    colors = plt.cm.viridis(np.linspace(0, 1, n_s3))
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    axes[0, 0].plot(steps_log, mse_log, lw=1.5, color="steelblue")
    axes[0, 0].axhline(0.02, color="green", ls="--", lw=1, label="target 0.02")
    axes[0, 0].set_xlabel("Step"); axes[0, 0].set_ylabel("Off-diag MSE")
    axes[0, 0].set_title("MSE (off-diagonal, oracle corr)"); axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(steps_log, div_log, lw=1.5, color="purple")
    axes[0, 1].axhline(0, color="red", ls="--", lw=0.8, alpha=0.5)
    axes[0, 1].set_xlabel("Step"); axes[0, 1].set_ylabel("Std of off-diag predictions")
    axes[0, 1].set_title("Prediction diversity (0 = all queries get same R)")

    axes[0, 2].plot(steps_log, gate_log, lw=1.5, color="orange")
    axes[0, 2].axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].set_xlabel("Step"); axes[0, 2].set_ylabel("sigmoid(icl_gate)")
    axes[0, 2].set_title("ICL gate value")

    for i in range(n_s3):
        axes[1, 0].plot(attn_steps, h_log[i], color=colors[i], lw=1.5,
                        marker="o", ms=3, label=f"blk {i}")
    axes[1, 0].axhline(1.0, color="red",   ls="--", lw=0.8, alpha=0.6, label="collapsed")
    axes[1, 0].axhline(0.0, color="green", ls="--", lw=0.8, alpha=0.6, label="peaked")
    axes[1, 0].set_ylim(-0.05, 1.1)
    axes[1, 0].set_xlabel("Step"); axes[1, 0].set_ylabel("H_norm")
    axes[1, 0].set_title("Stage-3 attention entropy"); axes[1, 0].legend(fontsize=7, ncol=2)

    # NLL validation curves
    if val_steps:
        axes[1, 1].plot(val_steps, val_copula, lw=1.5, color="steelblue",  marker="o", ms=4, label="model")
        axes[1, 1].plot(val_steps, val_oracle, lw=1.5, color="green",      marker="o", ms=4, label="oracle")
        axes[1, 1].plot(val_steps, val_oas,    lw=1.5, color="gray",       marker="o", ms=4, ls="--", label="OAS")
        axes[1, 1].plot(val_steps, val_knn5,   lw=1.5, color="darkorange", marker="o", ms=4, ls="--", label="kNN-5")
        axes[1, 1].set_xlabel("Step"); axes[1, 1].set_ylabel("Copula NLL")
        axes[1, 1].set_title("Validation copula NLL vs baselines")
        axes[1, 1].legend(fontsize=7)

    for i in range(n_s3):
        axes[1, 2].plot(attn_steps, alpha_a_log[i], color=colors[i], lw=1.5,
                        marker="o", ms=3, label=f"blk {i}")
    axes[1, 2].axhline(0, color="red", ls="--", lw=0.8, alpha=0.5)
    axes[1, 2].set_xlabel("Step"); axes[1, 2].set_ylabel("alpha_attn")
    axes[1, 2].set_title("ReZero alpha_attn per S3 block")
    axes[1, 2].legend(fontsize=7, ncol=2)

    fig.suptitle(
        f"CopulaTabICLv2 debug — {N_STEPS} steps MSE-only  |  "
        f"r={vm_final.get('r', float('nan')):.3f}  "
        f"copula_nll={vm_final['copula_nll']:.4f}  "
        f"oracle_frac={vm_final['oracle_frac']:.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = OUTPUT_DIR / "training_diagnostics.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")
    print(f"Val plots: {OUTPUT_DIR / 'val'}/")


if __name__ == "__main__":
    main()
