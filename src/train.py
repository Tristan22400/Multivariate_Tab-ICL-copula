"""
train.py — Phase 2 training loop for the CopulaTransformer.

Trains a CopulaTransformer to predict the parameters (mu, D, V) of a
low-rank Gaussian copula density in Z-space (after PIT), given the
context (X_train, Z_train) from pre-generated PIT episodes.

Each training step:
  1. Loads one pre-computed PIT episode from disk (X_train, Z_train, …).
  2. Applies a random 70/30 support/query split over the instance axis.
  3. Forwards the support through the model to predict query distribution params.
  4. Computes Woodbury NLL on the query instances.
  5. Backpropagates and logs to W&B.


Validation (every cfg.training.val_every steps):
  - Uses a fixed suite of synthetic generate_episode() episodes.
  - Y_test (z-normalised by generate_episode) serves as a proxy for Z_test,
    which is valid because generate_episode already z-normalises Y.

Usage (from project root):
    python src/train.py training.dataset_dir=./data/pit_episodes
    python src/train.py training.dataset_dir=./data/pit_episodes training.steps=50000
    python src/train.py training.dataset_dir=./data/pit_episodes training.lr=1e-4

Resume from checkpoint:
    Checkpoints are saved to cfg.training.ckpt_dir every cfg.training.save_every steps.
    Re-running the same command automatically resumes if the latest checkpoint is found.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# ---------------------------------------------------------------------------
# Path setup — must happen before local imports
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from sklearn.covariance import OAS

from dataset import infinite_episode_iter, make_episode_loader, split_episode_files
from loss import indep_normal_nll, woodbury_nll
from model import build_copula_tabicl_v2, build_icl_corr_net_v2

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set seeds for torch, numpy, random, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _cov_to_woodbury_params(
    Sigma: torch.Tensor,
    mu: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose (N, d, d) covariances into Woodbury params for woodbury_nll.

    Accepts a batch of N covariance matrices and returns (N, d), (N, d),
    (N, d, d) tensors ready to pass to woodbury_nll as a (1, N, ...) batch.
    All d eigenvectors are kept so the decomposition exactly recovers Sigma.
    """
    diag_vals = Sigma.diagonal(dim1=-2, dim2=-1).clamp(min=1e-6)  # (N, d)
    off_diag = Sigma - torch.diag_embed(diag_vals)  # (N, d, d)
    eigvals, eigvecs = torch.linalg.eigh(off_diag)  # (N, d), (N, d, d)
    eigvals = eigvals.clamp(min=0.0)
    V_out = eigvecs * eigvals.sqrt().unsqueeze(-2)  # (N, d, d)
    N, d = diag_vals.shape
    mu_out = mu if mu is not None else torch.zeros(N, d, device=Sigma.device)
    return mu_out, diag_vals, V_out


def _energy_score_batched(
    mu: torch.Tensor,
    D: torch.Tensor,
    V: torch.Tensor,
    y_ref: torch.Tensor,
    n_samples: int = 200,
) -> float:
    """Energy score for all (B, T) instances in a single GPU pass.

    Args:
        mu, D, V : (B, T, d) / (B, T, d) / (B, T, d, r)
        y_ref    : (B, T, d)
    """
    B, T, d = mu.shape
    r = V.shape[-1]
    eps_d = torch.randn(B, T, n_samples, d, device=mu.device, dtype=mu.dtype)
    eps_r = torch.randn(B, T, n_samples, r, device=mu.device, dtype=mu.dtype)
    samples = (
        mu.unsqueeze(2) + D.unsqueeze(2).sqrt() * eps_d + (eps_r @ V.transpose(-2, -1))
    )  # (B, T, M, d)
    term1 = (samples - y_ref.unsqueeze(2)).norm(dim=-1).mean(dim=-1)  # (B, T)
    samples_flat = samples.reshape(B * T, n_samples, d)
    term2 = torch.cdist(samples_flat, samples_flat).mean(dim=(-2, -1)).reshape(B, T)
    return (term1 - 0.5 * term2).mean().item()


# ---------------------------------------------------------------------------
# Validation plot helpers
# ---------------------------------------------------------------------------


def _corr_all_instances_fig(
    d_pred: torch.Tensor,   # (B, n_test, d)
    V_pred: torch.Tensor,   # (B, n_test, d, r)
    oracle_D: torch.Tensor, # (B, n_test, d)
    oracle_V: torch.Tensor, # (B, n_test, d, r)
    batch_idx: int = 0,
    label: str = "",
) -> tuple[plt.Figure, float]:
    """Grid of oracle vs predicted correlation matrices for all test instances.

    Layout mirrors the debug script's make_all_instances_plot:
      row 0: oracle    instances 0 .. half-1
      row 1: predicted instances 0 .. half-1
      row 2: oracle    instances half .. n_test-1
      row 3: predicted instances half .. n_test-1
    Returns (fig, mean_mse).
    """
    b = batch_idx

    Sigma_pred = torch.diag_embed(d_pred[b]) + V_pred[b] @ V_pred[b].transpose(-2, -1)
    std_p = Sigma_pred.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    R_pred = (Sigma_pred / (std_p.unsqueeze(-1) * std_p.unsqueeze(-2))).float().cpu()

    Sigma_ora = torch.diag_embed(oracle_D[b]) + oracle_V[b] @ oracle_V[b].transpose(-2, -1)
    std_o = Sigma_ora.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    R_ora = (Sigma_ora / (std_o.unsqueeze(-1) * std_o.unsqueeze(-2))).float().cpu()

    d = R_pred.shape[-1]
    ri, ci = torch.triu_indices(d, d, offset=1)
    n_test = R_pred.shape[0]
    half = max(n_test // 2, 1)

    mse_per = [F.mse_loss(R_pred[i, ri, ci], R_ora[i, ri, ci]).item() for i in range(n_test)]
    mean_mse = float(np.mean(mse_per))

    R_pred_np = R_pred.numpy()
    R_ora_np  = R_ora.numpy()

    n_cols = max(half, 1)
    fig, axes = plt.subplots(4, n_cols, figsize=(2.5 * n_cols, 10), layout="constrained")
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    im = None
    for i in range(half):
        for grp in range(2):
            inst = i + grp * half
            if inst >= n_test:
                continue
            for row_off, is_oracle in enumerate([True, False]):
                row = grp * 2 + row_off
                ax = axes[row, i]
                R = R_ora_np[inst] if is_oracle else R_pred_np[inst]
                im = ax.imshow(R, vmin=-1, vmax=1, cmap="RdBu_r")
                if is_oracle:
                    ax.set_title(f"Oracle #{inst}", fontsize=7)
                else:
                    ax.set_title(f"MSE={mse_per[inst]:.3f}", fontsize=7, color="darkred")
                ax.set_xticks([])
                ax.set_yticks([])

    for row, lbl in enumerate([
        f"Oracle  0–{half-1}", f"Pred  0–{half-1}",
        f"Oracle  {half}–{n_test-1}", f"Pred  {half}–{n_test-1}",
    ]):
        axes[row, 0].set_ylabel(lbl, fontsize=8)

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.35, pad=0.02)

    title = f"All {n_test} instances — mean MSE={mean_mse:.4f}"
    if label:
        title = f"{label}  |  {title}"
    fig.suptitle(title, fontsize=10)
    return fig, mean_mse


# ---------------------------------------------------------------------------
# PIT-episode validation (Z-space — same distribution as training)
# ---------------------------------------------------------------------------


def run_val_pit(
    model: nn.Module,
    val_episodes: list[dict],
    step: int,
    wandb_run,
    device: str,
    do_plot: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = False,
    n_think: int = 0,
) -> dict[str, float]:
    """Validate on held-out PIT episodes.  All NLL metrics are in copula-NLL units.

    Let I_z = indep_normal_nll(Z) = d/2 log(2π) + 1/2 ||z||² (Jacobian correction).
    Every metric is copula_nll = woodbury_nll - I_z  so that the Sklar decomposition
    holds exactly:   joint_y_nll = copula_nll + marginal_nll.

    Metrics:
        val/copula_nll        — model copula NLL on Z_test (70% of train as support)
        val/joint_y_nll       — copula_nll + marginal_nll = full joint NLL in Y-space
        val/train_nll         — model copula NLL on Z_train 70/30 split (overfitting check)
        val/marginal_nll      — always 0.0 (copula NLL of N(0,I); fixed by PIT construction)
        val/copula_gain       — -copula_nll = gain over N(0,I) fixed baseline (≥ 0 when model helps)
        val/oas_nll           — OAS baseline copula NLL (≈ woodbury_nll_oas - I_z)
        val/knn5_cov_nll      — kNN-5 baseline copula NLL
        val/linear_factor_nll — linear-factor baseline copula NLL
        val/oracle_nll_z      — oracle copula NLL in Z-space via N(0, corr(Σ_Y)) - I_z
        val/oracle_nll        — oracle full joint NLL in Y-space (direct reference for joint_y_nll)
        val/hetero_gain       — oas_nll - oracle_nll_z  (how much x_i structure matters)
        val/vs_knn5           — copula_nll - knn5_cov_nll  (< 0 means model beats kNN-5)
        val/oracle_frac       — (copula_nll - oracle_nll_z) / (0 - oracle_nll_z)
                                  0 = model at oracle,  1 = model at N(0,I) prior
        val/energy_score      — Energy Score (sample quality, independent of NLL units)
    """
    model.eval()
    agg: dict[str, list[float]] = {
        "val/copula_nll": [],
        "val/joint_y_nll": [],
        "val/train_nll": [],
        "val/marginal_nll": [],
        "val/copula_gain": [],
        "val/oas_nll": [],
        "val/knn5_cov_nll": [],
        "val/linear_factor_nll": [],
        "val/oracle_nll_z": [],
        "val/oracle_nll": [],
        "val/hetero_gain": [],
        "val/vs_knn5": [],
        "val/oracle_frac": [],
        "val/energy_score": [],
    }
    plot_episodes: list[dict] = []
    all_off_pred: list[np.ndarray] = []  # off-diag predicted values, for scatter
    all_off_ora:  list[np.ndarray] = []  # off-diag oracle values,    for scatter

    with torch.no_grad():
        for i_ep, ep in enumerate(val_episodes):
            X_tr = ep["X_train"].to(device)  # (B, n_train, p)
            Z_tr = ep["Z_train"].to(device)  # (B, n_train, d) — PIT
            Z_te = ep["Z_test"].to(device)  # (B, n_test,  d) — PIT
            X_te = ep["X_test"].to(device)
            log_p_te = ep["log_p_test"].to(
                device
            )  # (B, n_test,  d) — marginal log-densities
            oracle_mu = ep["oracle_mu"].to(device)
            oracle_D = ep["oracle_D"].to(device)
            oracle_V = ep["oracle_V"].to(device)
            Y_test = ep["Y_test"].to(device)

            B, n_train, _ = Z_tr.shape
            _, n_test, d = Z_te.shape

            # ---- Single forward pass: 70% of X_tr as support,
            #      query the remaining 30% of X_tr (overfitting check) and
            #      all X_te (test eval) together. This avoids a second model
            #      call which would overwrite the CUDAGraph output buffer.
            n_sup = max(1, int(0.7 * n_train))
            perm_tr = torch.randperm(n_train, device=device)
            X_sup = X_tr[:, perm_tr[:n_sup], :]
            Z_sup = Z_tr[:, perm_tr[:n_sup], :]
            X_tr_qry = X_tr[:, perm_tr[n_sup:], :]
            Z_tr_qry = Z_tr[:, perm_tr[n_sup:], :]
            n_tr_qry = X_tr_qry.shape[1]

            if n_think > 0:
                think_X = torch.zeros(B, n_think, X_sup.shape[-1], device=device, dtype=X_sup.dtype)
                think_Z = torch.zeros(B, n_think, Z_sup.shape[-1], device=device, dtype=Z_sup.dtype)
                X_sup = torch.cat([think_X, X_sup], dim=1)
                Z_sup = torch.cat([think_Z, Z_sup], dim=1)
                n_sup = n_sup + n_think

            X_fwd = torch.cat([X_sup, X_tr_qry, X_te], dim=1)
            Z_fwd = torch.cat(
                [Z_sup, torch.zeros_like(Z_tr_qry), torch.zeros_like(Z_te)], dim=1
            )

            torch.compiler.cudagraph_mark_step_begin()
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                mu_all, d_all, V_all = model(X_fwd, Z_fwd, n_support=n_sup)
            mu_all = mu_all.float()
            d_all = d_all.float()
            V_all = V_all.float()

            # Split outputs: first n_tr_qry slots → train check; rest → test
            mu_tr, d_tr, V_tr = (
                mu_all[:, :n_tr_qry],
                d_all[:, :n_tr_qry],
                V_all[:, :n_tr_qry],
            )
            mu_Z, d_Z, V_Z = (
                mu_all[:, n_tr_qry:],
                d_all[:, n_tr_qry:],
                V_all[:, n_tr_qry:],
            )

            # ---- Overfitting check (train copula NLL) ----
            train_copula_nll = (
                woodbury_nll(Z_tr_qry, mu_tr, d_tr, V_tr).item()
                - indep_normal_nll(Z_tr_qry).item()
            )
            agg["val/train_nll"].append(train_copula_nll)

            # ---- Test metrics ----
            # I_z = indep_normal_nll(Z_te) is computed once and reused for all
            # Jacobian corrections.  copula_nll = woodbury_nll - I_z for every metric.
            indep_z = indep_normal_nll(Z_te).item()

            wnll = woodbury_nll(Z_te, mu_Z, d_Z, V_Z).item()
            mnll = indep_z  # fixed: N(0,I) baseline (PIT guarantees z_j ~ N(0,1))
            cnll = wnll - indep_z  # model copula NLL
            c_marg = 0.0  # copula NLL of N(0,I) is always 0 by definition

            # Copula gain: improvement from V over diagonal; I_z cancels in the
            # difference so copula_gain = mnll - wnll = c_marg - cnll >= 0 when V helps.
            copula_gain = mnll - wnll

            # Full joint NLL in Y-space = copula_nll + sum of marginal log-densities
            marginal_nll_te = -log_p_te.sum(-1).mean().item()
            joint_y_nll = cnll + marginal_nll_te

            agg["val/copula_nll"].append(cnll)
            agg["val/joint_y_nll"].append(joint_y_nll)
            agg["val/marginal_nll"].append(c_marg)
            agg["val/copula_gain"].append(copula_gain)
            agg["val/energy_score"].append(
                _energy_score_batched(mu_Z, d_Z, V_Z, Z_te, n_samples=50)
            )

            # ---- OAS + kNN5 + linear baselines ----
            oas_nlls, knn5_nlls, linear_nlls = [], [], []
            for b in range(B):
                Z_tr_b_np = Z_tr[b].cpu().numpy()
                Z_tr_b = Z_tr[b]  # (n_train, d)
                Z_te_b = Z_te[b]  # (n_test,  d)
                X_tr_b = X_tr[b]  # (n_train, p)
                X_te_b = X_te[b]  # (n_test,  p)

                # OAS shrinkage — force mu=0 to match the copula assumption N(0,ρ).
                # Decompose the single (d,d) matrix once, then expand to (n_test,).
                oas = OAS().fit(Z_tr_b_np)
                Sigma_oas = torch.tensor(
                    oas.covariance_, dtype=torch.float32, device=device
                )  # (d, d)
                mu_p, d_p, V_p = _cov_to_woodbury_params(
                    Sigma_oas.unsqueeze(0), None
                )  # (1,d), (1,d), (1,d,d)
                d_p_exp = d_p.expand(n_test, -1)
                V_p_exp = V_p.expand(n_test, -1, -1)
                mu_p_exp = torch.zeros(n_test, d, device=device)
                oas_nlls.append(
                    woodbury_nll(
                        Z_te_b.unsqueeze(0),
                        mu_p_exp.unsqueeze(0),
                        d_p_exp.unsqueeze(0),
                        V_p_exp.unsqueeze(0),
                    ).item()
                )
                pass  # ep_sigma_oas_list no longer needed (replaced by corr grid)

                # kNN-5: gather all neighbors at once, batch covariance, one NLL call.
                dists = torch.cdist(X_te_b, X_tr_b)  # (n_test, n_train)
                k_eff = min(5, n_train)
                idx = dists.topk(k_eff, largest=False).indices  # (n_test, k_eff)
                Z_nb = Z_tr_b[idx]  # (n_test, k_eff, d)
                mu_nb = Z_nb.mean(dim=1)  # (n_test, d)
                if k_eff > d:
                    Z_c = Z_nb - mu_nb.unsqueeze(1)
                    Sigma_knn = Z_c.transpose(-2, -1) @ Z_c / max(k_eff - 1, 1)
                    Sigma_knn = 0.5 * (Sigma_knn + Sigma_knn.transpose(-2, -1))
                else:
                    Sigma_knn = torch.diag_embed(
                        Z_nb.var(dim=1, unbiased=False).clamp(min=1e-6)
                    )  # (n_test, d, d)
                mu_knn, d_knn, V_knn = _cov_to_woodbury_params(Sigma_knn, mu_nb)
                knn5_nlls.append(
                    woodbury_nll(
                        Z_te_b.unsqueeze(0),
                        mu_knn.unsqueeze(0),
                        d_knn.unsqueeze(0),
                        V_knn.unsqueeze(0),
                    ).item()
                )

                # Linear factor: Sigma_resid is shared across all n_test instances,
                # decompose once then expand — n_test NLL calls → one batched call.
                W = torch.linalg.lstsq(X_tr_b, Z_tr_b).solution
                mu_lin = X_te_b @ W  # (n_test, d)
                resid = Z_tr_b - X_tr_b @ W
                Sigma_resid = (
                    torch.cov(resid.T)
                    if n_train > d
                    else torch.diag(resid.var(0, unbiased=False).clamp(min=1e-6))
                )
                Sigma_resid = 0.5 * (Sigma_resid + Sigma_resid.T)
                _, d_lin, V_lin = _cov_to_woodbury_params(
                    Sigma_resid.unsqueeze(0), None
                )  # (1,d), (1,d,d)
                d_lin_exp = d_lin.expand(n_test, -1)
                V_lin_exp = V_lin.expand(n_test, -1, -1)
                linear_nlls.append(
                    woodbury_nll(
                        Z_te_b.unsqueeze(0),
                        mu_lin.unsqueeze(0),
                        d_lin_exp.unsqueeze(0),
                        V_lin_exp.unsqueeze(0),
                    ).item()
                )

            # Convert baselines from Gaussian NLL to copula NLL units by subtracting I_z.
            # These baselines fit non-zero mean and free variance, so they are not strict
            # copula NLLs, but the Jacobian correction makes them comparable to val/copula_nll
            # under the reasonable approximation that Z marginals are approximately N(0,1).
            oas_copula_ep = float(np.mean(oas_nlls)) - indep_z
            knn5_copula_ep = float(np.mean(knn5_nlls)) - indep_z
            linear_copula_ep = float(np.mean(linear_nlls)) - indep_z
            agg["val/oas_nll"].append(oas_copula_ep)
            agg["val/knn5_cov_nll"].append(knn5_copula_ep)
            agg["val/linear_factor_nll"].append(linear_copula_ep)

            # ---- Oracle copula NLL in Z-space: 1/2 log|ρ| + 1/2 z^T(ρ^{-1}-I)z ----
            # ρ = corr(Σ_Y) is the correlation matrix derived from the oracle Y-space covariance.
            # woodbury_nll(z; 0, ρ) = oracle_copula_nll_z + I_z, so we subtract I_z.
            Sigma_Y = torch.diag_embed(oracle_D) + oracle_V @ oracle_V.transpose(-2, -1)
            std_Y = Sigma_Y.diagonal(dim1=-2, dim2=-1).sqrt().clamp(min=1e-8)
            rho = Sigma_Y / (std_Y.unsqueeze(-1) * std_Y.unsqueeze(-2))
            lam, U_eig = torch.linalg.eigh(rho)
            _delta = 1e-4
            D_rho = torch.full_like(oracle_D, _delta)
            V_rho = U_eig * (lam - _delta).clamp(min=0).sqrt().unsqueeze(-2)
            oracle_copula_z = (
                woodbury_nll(Z_te, torch.zeros_like(oracle_mu), D_rho, V_rho).item()
                - indep_z
            )
            agg["val/oracle_nll_z"].append(oracle_copula_z)

            # ---- Oracle full joint NLL in Y-space ----
            oracle_nll_y = woodbury_nll(Y_test, oracle_mu, oracle_D, oracle_V).item()
            agg["val/oracle_nll"].append(oracle_nll_y)

            # ---- Diagnostic gaps (all in copula NLL units) ----
            # hetero_gain: I_z cancels in this difference — same numerical value as before
            agg["val/hetero_gain"].append(oas_copula_ep - oracle_copula_z)
            # vs_knn5: I_z cancels — same numerical value as before
            agg["val/vs_knn5"].append(cnll - knn5_copula_ep)
            # oracle_frac: 0 = model at oracle copula NLL,  1 = model at N(0,I) prior (copula NLL = 0)
            # prior copula NLL = 0 since R=I gives 1/2 log|I| + 1/2 z^T(I-I)z = 0
            denom = (
                0.0 - oracle_copula_z
            )  # = -oracle_copula_z  (oracle < 0 for structured data)
            agg["val/oracle_frac"].append(
                (cnll - oracle_copula_z) / denom if abs(denom) > 1e-8 else float("nan")
            )

            # ---- Collect data for plots ----
            if do_plot:
                # Off-diagonal correlation values for the global scatter
                Sigma_pp = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-2, -1)
                std_pp = Sigma_pp.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
                R_pp = Sigma_pp / (std_pp.unsqueeze(-1) * std_pp.unsqueeze(-2))
                Sigma_oo = torch.diag_embed(oracle_D) + oracle_V @ oracle_V.transpose(-2, -1)
                std_oo = Sigma_oo.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
                R_oo = Sigma_oo / (std_oo.unsqueeze(-1) * std_oo.unsqueeze(-2))
                ri_p, ci_p = torch.triu_indices(d, d, offset=1, device=d_Z.device)
                all_off_pred.append(R_pp[..., ri_p, ci_p].float().cpu().numpy().flatten())
                all_off_ora.append(R_oo[..., ri_p, ci_p].float().cpu().numpy().flatten())

                # Store up to 2 episodes for the all-instances grid plot
                if len(plot_episodes) < 2:
                    plot_episodes.append(
                        {
                            "key": f"pit_ep{i_ep}",
                            "d_pred": d_Z.clone(),
                            "V_pred": V_Z.clone(),
                            "oracle_D": oracle_D,
                            "oracle_V": oracle_V,
                            "n_test": n_test,
                        }
                    )

    metrics = {k: float(np.mean(v)) for k, v in agg.items()}
    metrics["step"] = step

    print(
        f"[val step={step:>6d}]  "
        f"copula={metrics['val/copula_nll']:.4f}  "
        f"joint_y={metrics['val/joint_y_nll']:.4f}  "
        f"oracle_y={metrics['val/oracle_nll']:.4f}  "
        f"oracle_z={metrics['val/oracle_nll_z']:.4f}  "
        f"train={metrics['val/train_nll']:.4f}  "
        f"gain={metrics['val/copula_gain']:.4f}  "
        f"oas={metrics['val/oas_nll']:.4f}  "
        f"knn5={metrics['val/knn5_cov_nll']:.4f}  "
        f"vs_knn5={metrics['val/vs_knn5']:.4f}  "
        f"oracle_frac={metrics['val/oracle_frac']:.4f}"
    )

    plot_figs = []
    if do_plot and wandb_run is not None:
        # — All-instances correlation grids (one per stored episode, up to 2 batch elems each) —
        for ep in plot_episodes:
            B_ep = ep["d_pred"].shape[0]
            for b_idx in range(min(2, B_ep)):
                fig, _ = _corr_all_instances_fig(
                    ep["d_pred"], ep["V_pred"],
                    ep["oracle_D"], ep["oracle_V"],
                    batch_idx=b_idx,
                    label=f"{ep['key']} b={b_idx}",
                )
                plot_figs.append(fig)

        # — Off-diagonal scatter: predicted vs oracle across all episodes —
        if all_off_pred and all_off_ora:
            off_p = np.concatenate(all_off_pred)
            off_o = np.concatenate(all_off_ora)
            lo = min(float(off_o.min()), float(off_p.min()))
            hi = max(float(off_o.max()), float(off_p.max()))

            # — 2D histogram (density) of off-diagonal correlations —
            fig_den, ax_den = plt.subplots(figsize=(5, 5))
            hb = ax_den.hexbin(off_o, off_p, gridsize=60, cmap="YlOrRd", mincnt=1, bins="log")
            fig_den.colorbar(hb, ax=ax_den, label="log10(count)")
            ax_den.plot([lo, hi], [lo, hi], "b--", lw=1)
            ax_den.set_xlabel("Oracle off-diag corr")
            ax_den.set_ylabel("Predicted off-diag corr")
            ax_den.set_title(f"step {step} — density ({len(off_p):,} values)")
            fig_den.tight_layout()
            plot_figs.append(fig_den)

    if wandb_run is not None:
        if plot_figs:
            import io
            import wandb as _wandb
            from PIL import Image as PILImage

            def _fig_to_pil(fig, dpi=100):
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
                buf.seek(0)
                return PILImage.open(buf).copy()

            TARGET_W = 1200
            pil_imgs = [_fig_to_pil(f) for f in plot_figs]
            resized = [
                im.resize(
                    (TARGET_W, max(1, int(im.height * TARGET_W / im.width))),
                    PILImage.LANCZOS,
                )
                for im in pil_imgs
            ]
            target_h = max(im.height for im in resized)
            padded = []
            for im in resized:
                canvas = PILImage.new("RGB", (TARGET_W, target_h), (255, 255, 255))
                canvas.paste(im, (0, 0))
                padded.append(canvas)
            metrics["val/plot"] = [_wandb.Image(im) for im in padded]
        wandb_run.log(metrics, step=step)
        for f in plot_figs:
            plt.close(f)

    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _latest_ckpt(ckpt_dir: str | None) -> str | None:
    """Return the path of the most recent checkpoint in ckpt_dir, or None."""
    if ckpt_dir is None:
        return None
    ckpt_dir_path = Path(ckpt_dir)
    if not ckpt_dir_path.is_dir():
        return None
    candidates = sorted(ckpt_dir_path.glob("step_*.pt"))
    return str(candidates[-1]) if candidates else None


def _save_checkpoint(
    path: str,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: DictConfig,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Unwrap compiled model so checkpoints are always plain state dicts
    raw_model = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "step": step,
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "hyperparameters": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # ---- Seed ----
    set_seed(int(cfg.seed))

    # ---- Device ----
    device: str = cfg.training.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(OmegaConf.to_yaml(cfg))

    # ---- Mixed precision setup ----
    use_amp: bool = device == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        torch.backends.cudnn.benchmark = True
        print(f"AMP enabled  dtype={amp_dtype}  cudnn.benchmark=True")

    # ---- Validate required config ----
    if cfg.training.dataset_dir is None:
        raise ValueError(
            "cfg.training.dataset_dir must be set to the pre-computed PIT episode directory.\n"
            "Run  python src/generate_pit_dataset.py  first, then pass "
            "training.dataset_dir=./data/pit_episodes"
        )

    # ---- Model ----
    _model_name = getattr(cfg.model, "name", "copula_tabicl_v2")
    if _model_name == "copula_tabicl_v2":
        model: nn.Module = build_copula_tabicl_v2(cfg).to(device)
    elif _model_name == "icl_corr_net_v2":
        model: nn.Module = build_icl_corr_net_v2(cfg).to(device)
    else:
        raise ValueError(
            f"Unknown model.name={_model_name!r}. "
            "Use 'copula_tabicl_v2' or 'icl_corr_net_v2'."
        )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {n_params:,}")

    # ---- Optimizer & scheduler ----
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    warmup_steps = int(cfg.training.get("warmup_steps", 0))
    cosine_steps = max(int(cfg.training.steps) - warmup_steps, 1)
    _cosine = CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=float(cfg.training.lr_min)
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

    # ---- W&B (optional) ----
    wandb_run = None
    try:
        import wandb

        _data_tag = Path(cfg.training.dataset_dir).name
        _lr = cfg.training.lr
        _lr_str = f"{_lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
        # n_layers key differs across model configs
        _n_layers = getattr(cfg.model, "n_layers",
                            f"{getattr(cfg.model, 'n_layers_s1', '?')}/"
                            f"{getattr(cfg.model, 'n_layers_s2', '?')}/"
                            f"{getattr(cfg.model, 'n_layers_s3', '?')}")
        _d_hidden = getattr(cfg.model, "d_hidden", getattr(cfg.model, "d_model", "?"))
        _rank = getattr(cfg.model, "rank", "?")
        _aux_w = float(cfg.training.get("aux_mse_weight", 0.0))
        _nll_w = float(cfg.training.get("nll_weight", 1.0))
        _run_name = (
            f"{_model_name}"
            f"_lr={_lr_str}"
            f"_steps={cfg.training.steps}"
            f"_dh={_d_hidden}"
            f"_L={_n_layers}"
            f"_H={cfg.model.n_heads}"
            f"_r={_rank}"
            f"_nllw={_nll_w}"
            f"_auxw={_aux_w}"
            f"_data={_data_tag}"
        )
        wandb_run = wandb.init(
            project="multivariate-tab-icl",
            name=_run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
        print(f"W&B run : {wandb_run.url}")
    except Exception as e:
        print(f"W&B unavailable ({e}), logging to console only.")

    # ---- Resume from checkpoint ----
    start_step = 0
    ckpt_dir = cfg.training.ckpt_dir
    resume_from = cfg.training.get("resume_from", None)
    if resume_from is not None:
        if not Path(resume_from).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
        print(f"Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        if missing:
            print(f"  New parameters (default init): {missing}")
        if unexpected:
            print(f"  Unexpected keys ignored: {unexpected}")
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_step = int(ckpt["step"]) + 1
        print(f"Resumed at step {start_step}.")
    else:
        print("No resume_from specified, training from scratch.")

    # ---- torch.compile (optional, ~60 s one-time compilation cost) ----
    if cfg.training.get("compile", False) and device == "cuda":
        model = torch.compile(model, mode="default", dynamic=False)
        print(
            "torch.compile enabled (mode=default) — first forward will trigger JIT compilation."
        )

    # ---- Data loader (training episodes only — val episodes held out) ----
    val_n_episodes = int(cfg.training.get("val_n_episodes", 50))
    train_files, val_files = split_episode_files(
        cfg.training.dataset_dir, val_n_episodes
    )
    print(f"Dataset split: {len(train_files)} train / {len(val_files)} val episodes")

    loader = make_episode_loader(
        files=train_files,
        shuffle=True,
        num_workers=int(cfg.dataset.num_workers),
    )
    episode_iter = infinite_episode_iter(loader)

    # ---- Validation episodes (pre-loaded, held-out PIT episodes) ----
    print(f"Loading {len(val_files)} validation episodes …")
    val_episodes = [torch.load(f, weights_only=True) for f in val_files]
    print("Validation episodes loaded.")

    # ---- Training loop ----
    model.train()
    t0 = time.perf_counter()
    accum_steps = int(cfg.training.get("gradient_accumulation_steps", 1))
    optimizer.zero_grad()

    for step in range(start_step, int(cfg.training.steps)):
        episode = next(episode_iter)

        X_train = episode["X_train"].to(device)  # (B, N, p)
        Z_train = episode["Z_train"].to(device)  # (B, N, d)
        X_test_ep = episode["X_test"].to(device)  # (B, n_test, p)
        Z_test_ep = episode["Z_test"].to(device)  # (B, n_test, d)
        oracle_D_ep = episode["oracle_D"].to(device)  # (B, n_test, d)
        oracle_V_ep = episode["oracle_V"].to(device)  # (B, n_test, d, r)

        B, N, d = Z_train.shape

        n_think = int(cfg.training.get("n_think", 0))
        if n_think > 0:
            think_X = torch.zeros(B, n_think, X_train.shape[-1], device=device, dtype=X_train.dtype)
            think_Z = torch.zeros(B, n_think, Z_train.shape[-1], device=device, dtype=Z_train.dtype)
            X_train = torch.cat([think_X, X_train], dim=1)
            Z_train = torch.cat([think_Z, Z_train], dim=1)
            N += n_think

        # ---- Single forward: full X_train as support, X_test as queries ----
        X_fwd = torch.cat([X_train, X_test_ep], dim=1)  # (B, N+n_test, p)
        Z_fwd = torch.cat([Z_train, Z_test_ep], dim=1)  # (B, N+n_test, d)

        torch.compiler.cudagraph_mark_step_begin()
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            mu_Z, d_Z, V_Z = model(X_fwd, Z_fwd, n_support=N)

            mu_Z, d_Z, V_Z = mu_Z.clone(), d_Z.clone(), V_Z.clone()
            # mu_Z: (B, n_test, d)
            # d_Z:  (B, n_test, d)
            # V_Z:  (B, n_test, d, r)

            # ---- Loss: Woodbury NLL on test queries ----
            Z_query = Z_test_ep  # (B, n_test, d)
            nll_weight = float(cfg.training.get("nll_weight", 1.0))
            # Guard: only run woodbury_nll when it contributes to the loss.
            # With bfloat16 the Cholesky can produce NaN; 0.0 * NaN = NaN which
            # would corrupt the MSE auxiliary loss even when nll_weight=0.
            if nll_weight > 0.0:
                loss_nll = woodbury_nll(Z_query, mu_Z, d_Z, V_Z)
                loss = nll_weight * loss_nll
            else:
                loss_nll = torch.zeros(1, device=device)
                loss = torch.zeros(1, device=device)

            # ---- Auxiliary off-diagonal MSE loss ----
            aux_weight = float(cfg.training.get("aux_mse_weight", 0.0))
            anneal_frac = float(cfg.training.get("aux_mse_anneal_frac", 0.7))
            progress = min(1.0, step / max(1, anneal_frac * int(cfg.training.steps)))
            alpha = aux_weight * (1.0 - progress)

            # Oracle correlation matrix from ground-truth low-rank factors
            Sigma_ora = torch.diag_embed(
                oracle_D_ep
            ) + oracle_V_ep @ oracle_V_ep.transpose(-1, -2)  # (B, n_test, d, d)
            std_ora = Sigma_ora.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
            R_ora = Sigma_ora / (std_ora.unsqueeze(-1) * std_ora.unsqueeze(-2))

            # Predicted correlation matrix (diag=1 by Woodbury construction)
            Sigma_pred = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-1, -2)

            ri, ci = torch.triu_indices(d, d, offset=1, device=device)
            loss_mse = F.mse_loss(Sigma_pred[..., ri, ci], R_ora[..., ri, ci])
            if alpha > 0.0:
                loss = loss + alpha * loss_mse

        # ---- Backward ----
        scaler.scale(loss / accum_steps).backward()
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg.training.clip_grad_norm)
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # ---- Logging ----
        if step % int(cfg.training.log_every) == 0:
            with torch.no_grad():
                wnll = loss_nll.item()
                indep_z_train = indep_normal_nll(Z_query).item()
                cnll_train = wnll - indep_z_train  # copula NLL
                copula_gain = (
                    indep_z_train - wnll
                )  # = -cnll; gain over N(0,I) fixed baseline

                oracle_mu = episode["oracle_mu"].to(device)
                Y_test = episode["Y_test"].to(device)
                oracle_nll_y = woodbury_nll(
                    Y_test, oracle_mu, oracle_D_ep, oracle_V_ep
                ).item()

                Sigma_pred = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-1, -2)
                d_dim = Sigma_pred.shape[-1]
                ri, ci = torch.triu_indices(d_dim, d_dim, offset=1, device=device)
                off_diag_pred = Sigma_pred[..., ri, ci]
                pred_off_diag_var = off_diag_pred.var(dim=1).mean().item()

                Sigma_oracle = torch.diag_embed(
                    oracle_D_ep
                ) + oracle_V_ep @ oracle_V_ep.transpose(-1, -2)
                off_diag_oracle = Sigma_oracle[..., ri, ci]
                oracle_off_diag_var = off_diag_oracle.var(dim=1).mean().item()

            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.perf_counter() - t0

            loss_mse_val = loss_mse.item()
            div = pred_off_diag_var / max(oracle_off_diag_var, 1e-8)
            print(
                f"[step {step:>6d}]  "
                f"mse={loss_mse_val:.5f}  "
                f"div={div:.4f}  "
                f"copula={cnll_train:.4f}  "
                f"gain={copula_gain:.4f}  "
                f"alpha={alpha:.3f}  "
                f"lr={lr_now:.2e}  "
                f"elapsed={elapsed:.1f}s"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/mse": loss_mse_val,
                        "train/div": div,
                        "train/copula_nll": cnll_train,
                        "train/copula_gain": copula_gain,
                        "train/oracle_nll_y": oracle_nll_y,
                        "train/pred_off_diag_var": pred_off_diag_var,
                        "train/oracle_off_diag_var": oracle_off_diag_var,
                        "train/alpha": alpha,
                        "train/nll_weight": nll_weight,
                        "train/lr": lr_now,
                        "step": step,
                    },
                    step=step,
                )

        # ---- Validation & Plotting ----
        do_val = step % int(cfg.training.val_every) == 0
        do_plot = step % int(cfg.training.plot_every) == 0
        if do_val or do_plot:
            run_val_pit(
                model,
                val_episodes,
                step,
                wandb_run,
                device,
                do_plot=do_plot,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
                n_think=int(cfg.training.get("n_think", 0)),
            )
            model.train()

        # ---- Checkpoint ----
        if (
            ckpt_dir is not None
            and step % int(cfg.training.save_every) == 0
            and step > start_step
        ):
            ckpt_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
            _save_checkpoint(ckpt_path, step, model, optimizer, scheduler, cfg)
            print(f"Saved checkpoint → {ckpt_path}")

    # ---- Final checkpoint & validation ----
    final_step = int(cfg.training.steps) - 1
    if ckpt_dir is not None:
        ckpt_path = os.path.join(ckpt_dir, f"step_{final_step:07d}_final.pt")
        _save_checkpoint(ckpt_path, final_step, model, optimizer, scheduler, cfg)
        print(f"Training complete. Final checkpoint → {ckpt_path}")
    else:
        print("Training complete. (No checkpoint saved)")

    run_val_pit(
        model,
        val_episodes,
        final_step,
        wandb_run,
        device,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        n_think=int(cfg.training.get("n_think", 0)),
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
