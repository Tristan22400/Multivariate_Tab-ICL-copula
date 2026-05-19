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
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------------------------------------------------------------------
# Path setup — must happen before local imports
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from sklearn.covariance import OAS

from dataset import infinite_episode_iter, make_episode_loader, split_episode_files
from loss import energy_score, marginal_nll, woodbury_nll
from model import build_copula_tabicl_v2
from viz import plot_prediction_comparison

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
    n_test: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose a dense (d, d) covariance into (mu, D, V) for woodbury_nll.

    Keeps ALL d eigenvectors so the decomposition exactly recovers Sigma
    (modulo clamping of numerically negative eigenvalues on the off-diagonal
    part). V has shape (d, d), giving the true full-covariance NLL rather
    than a low-rank approximation.
    """
    d = Sigma.shape[0]
    diag_vals = Sigma.diagonal().clamp(min=1e-6)
    off_diag = Sigma - torch.diag(diag_vals)
    eigvals, eigvecs = torch.linalg.eigh(off_diag)
    eigvals = eigvals.clamp(min=0.0)
    V_flat = eigvecs * eigvals.sqrt()  # (d, d)
    mu_out = mu if mu is not None else torch.zeros(d, device=device)
    return (
        mu_out.unsqueeze(0).expand(n_test, -1),
        diag_vals.unsqueeze(0).expand(n_test, -1),
        V_flat.unsqueeze(0).expand(n_test, -1, -1),
    )


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
) -> dict[str, float]:
    """Validate on held-out PIT episodes in Z-space (PIT), same as training.

    All core metrics are in Z-space so val/woodbury_nll and val/train_nll are
    directly comparable to train/woodbury_nll logged during training steps.

    Metrics:
        val/woodbury_nll     — model NLL on Z_test with full context (100% support)
        val/train_nll        — model NLL on Z_train with 70/30 split (mirrors training loss)
        val/marginal_nll     — model NLL ignoring V (diagonal only)
        val/nll_ratio        — woodbury / marginal
        val/prior_nll        — N(0,I) baseline (floor for Z-space NLL)
        val/oas_nll          — OAS shrinkage covariance baseline
        val/knn5_cov_nll     — k=5 nearest-neighbour covariance baseline
        val/linear_factor_nll — linear x→mean + global residual covariance
        val/oracle_nll_z     — oracle lower bound in Z-space via N(0, corr(Σ_Y))
        val/oracle_nll       — oracle lower bound in Y-space (reference)
        val/hetero_gain      — oas_nll - oracle_nll_z (how much x_i matters, Z-space)
        val/vs_knn5          — woodbury_nll - knn5_cov_nll (< 0 means model wins)
        val/oracle_frac      — (woodbury - oracle_z) / (prior - oracle_z)
    """
    model.eval()
    agg: dict[str, list[float]] = {
        "val/woodbury_nll": [],
        "val/train_nll": [],
        "val/marginal_nll": [],
        "val/nll_ratio": [],
        "val/prior_nll": [],
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

    with torch.no_grad():
        for i_ep, ep in enumerate(val_episodes):
            X_tr = ep["X_train"].to(device)   # (B, n_train, p)
            Z_tr = ep["Z_train"].to(device)   # (B, n_train, d) — PIT
            Z_te = ep["Z_test"].to(device)    # (B, n_test,  d) — PIT
            X_te = ep["X_test"].to(device)
            oracle_mu = ep["oracle_mu"].to(device)
            oracle_D  = ep["oracle_D"].to(device)
            oracle_V  = ep["oracle_V"].to(device)
            Y_test    = ep["Y_test"].to(device)

            B, n_train, _ = Z_tr.shape
            _, n_test, d  = Z_te.shape

            # ---- Model: full context → Z_test ----
            X_all = torch.cat([X_tr, X_te], dim=1)
            Z_all = torch.cat([Z_tr, torch.zeros_like(Z_te)], dim=1)
            mu_Z, d_Z, V_Z = model(X_all, Z_all, n_support=n_train)

            wnll = woodbury_nll(Z_te, mu_Z, d_Z, V_Z).item()
            mnll = marginal_nll(Z_te, mu_Z, d_Z).item()
            agg["val/woodbury_nll"].append(wnll)
            agg["val/marginal_nll"].append(mnll)
            agg["val/nll_ratio"].append(wnll / (mnll + 1e-8))
            agg["val/energy_score"].append(float(np.mean([
                energy_score(
                    mu=mu_Z[b, i], D=d_Z[b, i], V=V_Z[b, i],
                    y_ref=Z_te[b, i], n_samples=200,
                ).item()
                for b in range(B) for i in range(n_test)
            ])))

            # ---- Train NLL: 70/30 split on Z_train (mirrors training loss) ----
            n_sup = max(1, int(0.7 * n_train))
            perm_tr = torch.randperm(n_train, device=device)
            X_tr_perm = X_tr[:, perm_tr, :]
            Z_tr_perm = Z_tr[:, perm_tr, :]
            mu_tr, d_tr, V_tr = model(X_tr_perm, Z_tr_perm, n_support=n_sup)
            Z_query_tr = Z_tr_perm[:, n_sup:, :]
            agg["val/train_nll"].append(woodbury_nll(Z_query_tr, mu_tr, d_tr, V_tr).item())

            # ---- Prior: N(0, I) ----
            prior_nll = marginal_nll(
                Z_te, torch.zeros_like(mu_Z), torch.ones_like(d_Z)
            ).item()
            agg["val/prior_nll"].append(prior_nll)

            # ---- OAS + kNN5 + linear baselines ----
            oas_nlls, knn5_nlls, linear_nlls = [], [], []
            ep_sigma_oas_list: list[np.ndarray] = []
            for b in range(B):
                Z_tr_b_np = Z_tr[b].cpu().numpy()
                Z_tr_b = Z_tr[b]
                Z_te_b = Z_te[b]
                X_tr_b = X_tr[b]
                X_te_b = X_te[b]

                # OAS shrinkage
                oas = OAS().fit(Z_tr_b_np)
                Sigma_oas = torch.tensor(oas.covariance_, dtype=torch.float32, device=device)
                mu_oas = torch.tensor(oas.location_, dtype=torch.float32, device=device)
                mu_p, d_p, V_p = _cov_to_woodbury_params(Sigma_oas, mu_oas, n_test, device)
                oas_nlls.append(woodbury_nll(Z_te_b, mu_p, d_p, V_p).item())
                if do_plot:
                    ep_sigma_oas_list.append(oas.covariance_.copy())

                # kNN-5
                dists = torch.cdist(X_te_b, X_tr_b)
                k_eff = min(5, n_train)
                nlls_i = []
                for i in range(n_test):
                    idx = dists[i].topk(k_eff, largest=False).indices
                    Z_nb = Z_tr_b[idx]
                    Sigma_i = (
                        torch.cov(Z_nb.T)
                        if k_eff > d
                        else torch.diag(Z_nb.var(0, unbiased=False).clamp(min=1e-6))
                    )
                    Sigma_i = 0.5 * (Sigma_i + Sigma_i.T)
                    mu_nb = Z_nb.mean(0)
                    mu_p, d_p, V_p = _cov_to_woodbury_params(Sigma_i, mu_nb, 1, device)
                    nlls_i.append(woodbury_nll(Z_te_b[i : i + 1], mu_p, d_p, V_p).item())
                knn5_nlls.append(float(np.mean(nlls_i)))

                # Linear factor
                W = torch.linalg.lstsq(X_tr_b, Z_tr_b).solution
                mu_lin = X_te_b @ W
                resid = Z_tr_b - X_tr_b @ W
                Sigma_resid = (
                    torch.cov(resid.T)
                    if n_train > d
                    else torch.diag(resid.var(0, unbiased=False).clamp(min=1e-6))
                )
                Sigma_resid = 0.5 * (Sigma_resid + Sigma_resid.T)
                lin_nlls_i = []
                for i in range(n_test):
                    mu_p, d_p, V_p = _cov_to_woodbury_params(Sigma_resid, mu_lin[i], 1, device)
                    lin_nlls_i.append(
                        woodbury_nll(Z_te_b[i : i + 1], mu_p, d_p, V_p).item()
                    )
                linear_nlls.append(float(np.mean(lin_nlls_i)))

            oas_nll_ep = float(np.mean(oas_nlls))
            agg["val/oas_nll"].append(oas_nll_ep)
            agg["val/knn5_cov_nll"].append(float(np.mean(knn5_nlls)))
            agg["val/linear_factor_nll"].append(float(np.mean(linear_nlls)))

            # ---- Oracle NLL in Z-space: N(0, ρ) where ρ = corr(Σ_Y) ----
            Sigma_Y = torch.diag_embed(oracle_D) + oracle_V @ oracle_V.transpose(-2, -1)
            std_Y = Sigma_Y.diagonal(dim1=-2, dim2=-1).sqrt().clamp(min=1e-8)
            rho = Sigma_Y / (std_Y.unsqueeze(-1) * std_Y.unsqueeze(-2))
            lam, U = torch.linalg.eigh(rho)
            _delta = 1e-4
            D_rho = torch.full_like(oracle_D, _delta)
            V_rho = U * (lam - _delta).clamp(min=0).sqrt().unsqueeze(-2)
            oracle_nll_z = woodbury_nll(
                Z_te, torch.zeros_like(oracle_mu), D_rho, V_rho
            ).item()
            agg["val/oracle_nll_z"].append(oracle_nll_z)

            # ---- Oracle NLL in Y-space (reference) ----
            oracle_nll_y = woodbury_nll(Y_test, oracle_mu, oracle_D, oracle_V).item()
            agg["val/oracle_nll"].append(oracle_nll_y)

            # ---- Diagnostic gaps (Z-space) ----
            agg["val/hetero_gain"].append(oas_nll_ep - oracle_nll_z)
            agg["val/vs_knn5"].append(wnll - float(np.mean(knn5_nlls)))
            denom = prior_nll - oracle_nll_z
            agg["val/oracle_frac"].append(
                (wnll - oracle_nll_z) / denom if abs(denom) > 1e-8 else float("nan")
            )

            # ---- Collect up to 2 episodes for the comparison plot ----
            if do_plot and len(plot_episodes) < 2:
                plot_episodes.append(
                    {
                        "key": f"pit_ep{i_ep}",
                        "mu_pred": mu_Z,
                        "D_pred": d_Z,
                        "V_pred": V_Z,
                        "mu_true": oracle_mu,
                        "D_true": oracle_D,
                        "V_true": oracle_V,
                        "sigma_oas_list": ep_sigma_oas_list,
                        "n_test": n_test,
                    }
                )

    metrics = {k: float(np.mean(v)) for k, v in agg.items()}
    metrics["step"] = step

    print(
        f"[val step={step:>6d}]  "
        f"woodbury={metrics['val/woodbury_nll']:.4f}  "
        f"train_nll={metrics['val/train_nll']:.4f}  "
        f"oracle_z={metrics['val/oracle_nll_z']:.4f}  "
        f"oracle_y={metrics['val/oracle_nll']:.4f}  "
        f"knn5={metrics['val/knn5_cov_nll']:.4f}  "
        f"linear={metrics['val/linear_factor_nll']:.4f}  "
        f"oas={metrics['val/oas_nll']:.4f}  "
        f"vs_knn5={metrics['val/vs_knn5']:.4f}  "
        f"oracle_frac={metrics['val/oracle_frac']:.4f}"
    )

    plot_figs = []
    if do_plot and plot_episodes and wandb_run is not None:
        n_inst = min(3, plot_episodes[0]["n_test"])
        ep0 = plot_episodes[0]
        B0 = ep0["mu_pred"].shape[0]
        if len(plot_episodes) == 1 and B0 >= 2:
            # Single episode: show two different batch elements side-by-side
            for b_idx in range(2):
                sigma_oas_b = (
                    ep0["sigma_oas_list"][b_idx]
                    if b_idx < len(ep0["sigma_oas_list"])
                    else None
                )
                fig = plot_prediction_comparison(
                    mu_pred=ep0["mu_pred"],
                    D_pred=ep0["D_pred"],
                    V_pred=ep0["V_pred"],
                    mu_true=ep0["mu_true"],
                    D_true=ep0["D_true"],
                    V_true=ep0["V_true"],
                    batch_idx=b_idx,
                    n_instances=n_inst,
                    sigma_oas=sigma_oas_b,
                    dataset_label=f"{ep0['key']} — batch {b_idx}",
                )
                plot_figs.append(fig)
        else:
            # Multiple episodes: cycle batch index across episodes for diversity
            for ep_idx, ep_data in enumerate(plot_episodes):
                b_idx = ep_idx % ep_data["mu_pred"].shape[0]
                sigma_oas_b = (
                    ep_data["sigma_oas_list"][b_idx]
                    if b_idx < len(ep_data["sigma_oas_list"])
                    else None
                )
                fig = plot_prediction_comparison(
                    mu_pred=ep_data["mu_pred"],
                    D_pred=ep_data["D_pred"],
                    V_pred=ep_data["V_pred"],
                    mu_true=ep_data["mu_true"],
                    D_true=ep_data["D_true"],
                    V_true=ep_data["V_true"],
                    batch_idx=b_idx,
                    n_instances=n_inst,
                    sigma_oas=sigma_oas_b,
                    dataset_label=f"Dataset: {ep_data['key']} — batch {b_idx}",
                )
                plot_figs.append(fig)

    if wandb_run is not None:
        if plot_figs:
            import wandb as _wandb
            metrics["val/plot"] = [_wandb.Image(f) for f in plot_figs]
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
    scheduler: CosineAnnealingLR,
    cfg: DictConfig,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
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

    # ---- Validate required config ----
    if cfg.training.dataset_dir is None:
        raise ValueError(
            "cfg.training.dataset_dir must be set to the pre-computed PIT episode directory.\n"
            "Run  python src/generate_pit_dataset.py  first, then pass "
            "training.dataset_dir=./data/pit_episodes"
        )

    # ---- Model ----
    model: nn.Module = build_copula_tabicl_v2(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {n_params:,}")

    # ---- Optimizer & scheduler ----
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.training.steps),
        eta_min=float(cfg.training.lr_min),
    )

    # ---- W&B (optional) ----
    wandb_run = None
    try:
        import wandb

        _data_tag = Path(cfg.training.dataset_dir).name
        _lr = cfg.training.lr
        _lr_str = f"{_lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
        _n_layers = getattr(
            cfg.model,
            "n_layers",
            f"s1={getattr(cfg.model, 'n_layers_s1', '?')}"
            f"s2={getattr(cfg.model, 'n_layers_s2', '?')}"
            f"s3={getattr(cfg.model, 'n_layers_s3', '?')}",
        )
        _run_name = (
            f"lr={_lr_str}"
            f"_steps={cfg.training.steps}"
            f"_d={cfg.model.d_model}"
            f"_L={_n_layers}"
            f"_H={cfg.model.n_heads}"
            f"_r={cfg.model.rank}"
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
    latest = _latest_ckpt(ckpt_dir)
    if latest is not None:
        print(f"Resuming from checkpoint: {latest}")
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_step = int(ckpt["step"]) + 1
        print(f"Resumed at step {start_step}.")
    else:
        print("No checkpoint found, training from scratch.")

    # ---- Data loader (training episodes only — val episodes held out) ----
    val_n_episodes = int(cfg.training.get("val_n_episodes", 50))
    train_files, val_files = split_episode_files(cfg.training.dataset_dir, val_n_episodes)
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

    for step in range(start_step, int(cfg.training.steps)):
        episode = next(episode_iter)

        X_train = episode["X_train"].to(device)  # (B, n_train, p)
        Z_train = episode["Z_train"].to(device)  # (B, n_train, d)

        B, N, d = Z_train.shape

        # ---- Random 70/30 support/query split ----
        perm = torch.randperm(N, device=device)
        n_support = max(1, int(0.7 * N))
        # n_query  = N - n_support  (implicit)

        X_perm = X_train[:, perm, :]  # (B, N, p)
        Z_perm = Z_train[:, perm, :]  # (B, N, d)

        # ---- Forward pass ----
        model.train()
        mu_Z, d_Z, V_Z = model(X_perm, Z_perm, n_support)
        # mu_Z: (B, n_query, d)
        # d_Z:  (B, n_query, d)  — diagonal variance (must be > 0; model applies softplus)
        # V_Z:  (B, n_query, d, r)

        # ---- Loss: Woodbury NLL on query instances ----
        Z_query = Z_perm[:, n_support:, :]  # (B, n_query, d)
        loss = woodbury_nll(Z_query, mu_Z, d_Z, V_Z)

        # ---- Backward ----
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg.training.clip_grad_norm)
        )
        optimizer.step()
        scheduler.step()

        # ---- Logging ----
        if step % int(cfg.training.log_every) == 0:
            with torch.no_grad():
                mnll = marginal_nll(Z_query, mu_Z, d_Z).item()

                # Oracle NLL on test instances: lower bound for model NLL.
                # oracle_mu/D/V: (B, n_test, d) / (B, n_test, d) / (B, n_test, d, r)
                oracle_mu = episode["oracle_mu"].to(device)
                oracle_D = episode["oracle_D"].to(device)
                oracle_V = episode["oracle_V"].to(device)
                Y_test = episode["Y_test"].to(device)
                oracle_nll = woodbury_nll(Y_test, oracle_mu, oracle_D, oracle_V).item()

                # Mean absolute off-diagonal covariance variance across query instances.
                # Computed for both model predictions and oracle.
                Sigma_pred = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-1, -2)
                d_dim = Sigma_pred.shape[-1]
                ri, ci = torch.triu_indices(d_dim, d_dim, offset=1, device=device)
                off_diag_pred = Sigma_pred[..., ri, ci]

                pred_off_diag_var = off_diag_pred.var(dim=1).mean().item()

                Sigma_oracle = torch.diag_embed(
                    oracle_D
                ) + oracle_V @ oracle_V.transpose(-1, -2)
                off_diag_oracle = Sigma_oracle[..., ri, ci]
                oracle_off_diag_var = off_diag_oracle.var(dim=1).mean().item()

            wnll = loss.item()
            ratio = wnll / (mnll + 1e-8)
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.perf_counter() - t0

            print(
                f"[step {step:>6d}]  loss={wnll:.4f}  marginal_nll={mnll:.4f}  "
                f"oracle_nll={oracle_nll:.4f}  nll_ratio={ratio:.4f}  "
                f"pred_var={pred_off_diag_var:.4e}  oracle_var={oracle_off_diag_var:.4e}  "
                f"lr={lr_now:.2e}  elapsed={elapsed:.1f}s"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/woodbury_nll": wnll,
                        "train/marginal_nll": mnll,
                        "train/oracle_nll": oracle_nll,
                        "train/nll_ratio": ratio,
                        "train/pred_off_diag_var": pred_off_diag_var,
                        "train/oracle_off_diag_var": oracle_off_diag_var,
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
                model, val_episodes, step, wandb_run, device, do_plot=do_plot
            )

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

    run_val_pit(model, val_episodes, final_step, wandb_run, device)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
