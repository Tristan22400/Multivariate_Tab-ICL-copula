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

from data_gen import GlobalAnchorCovGen, GlobalFixedNets, build_val_suite
from dataset import infinite_episode_iter, make_episode_loader
from loss import energy_score, marginal_nll, woodbury_nll
from model import build_copula_transformer
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
# Validation
# ---------------------------------------------------------------------------


def run_validation(
    model: nn.Module,
    val_suite: dict[str, dict],
    step: int,
    cfg: DictConfig,
    wandb_run,
    device: str,
    do_plot: bool = False,
) -> dict[str, float]:
    """Run the model on the fixed validation suite and return a metrics dict.

    Each entry in val_suite is a synthetic episode from generate_episode().
    Y_test is z-normalised by generate_episode and acts as a proxy for Z_test,
    since the model operates in the same normalised unit-variance space.

    For each (key, episode), the model is conditioned on the full X_train /
    Z_train context (100% support), and evaluated on X_test / Y_test (as Z_test).

    Metrics are grouped into three categories:

    Model:
        woodbury_nll        — primary loss (full covariance, per-instance)
        marginal_nll        — model diagonal only (V ignored)
        nll_ratio           — woodbury_nll / marginal_nll (< 1 means correlation helps)
        energy_score        — MC energy score on the first val instance

    Global baselines (misspecified — do not condition on x_i, floor references):
        prior_nll           — standard normal, zero in-context learning
        mean_only_nll       — empirical mean + identity covariance
        independent_nll     — zero mean + per-dim variance from Z_train
        independent_mean_nll — empirical mean + per-dim variance (no correlation)
        full_mle_nll        — sample covariance MLE from Z_train
        oas_nll             — OAS shrinkage covariance from Z_train

    Heteroskedastic baselines (condition on x_i, genuine competitors):
        knn5_cov_nll        — k=5 nearest-neighbor covariance
        knn20_cov_nll       — k=20 nearest-neighbor covariance
        kernel_cov_nll      — kernel-weighted covariance (median bandwidth)
        linear_factor_nll   — linear x→mean + global residual covariance

    Oracle:
        oracle_nll          — ground-truth per-instance parameters (lower bound)

    Diagnostic gaps:
        heteroskedastic_gain — oas_nll - oracle_nll (how much x_i matters)
        model_vs_knn5_gap    — woodbury_nll - knn5_cov_nll (< 0 means model wins)
        oracle_gap_fraction  — (model - oracle) / (prior - oracle), 0=oracle, 1=prior

    Args:
        model     : CopulaTransformer in eval mode.
        val_suite : dict from build_val_suite(); each value has keys
                    X_train, Y_train, X_test, Y_test, oracle_mu, oracle_D, oracle_V.
        step      : current training step (for logging).
        cfg       : Hydra DictConfig.
        wandb_run : active W&B run (may be None if W&B is unavailable).
        device    : torch device string.

    Returns:
        Flat dict of averaged scalar metrics.
    """
    model.eval()
    agg: dict[str, list[float]] = {
        # model
        "val/woodbury_nll": [],
        "val/marginal_nll": [],
        "val/nll_ratio": [],
        "val/energy_score": [],
        # global baselines
        "val/prior_nll": [],
        "val/mean_only_nll": [],
        "val/independent_nll": [],
        "val/independent_mean_nll": [],
        "val/full_mle_nll": [],
        "val/oas_nll": [],
        # heteroskedastic baselines
        "val/knn5_cov_nll": [],
        "val/knn20_cov_nll": [],
        "val/kernel_cov_nll": [],
        "val/linear_factor_nll": [],
        # oracle
        "val/oracle_nll": [],
        # diagnostic gaps
        "val/heteroskedastic_gain": [],
        "val/model_vs_knn5_gap": [],
        "val/oracle_gap_fraction": [],
    }
    plot_fig = None

    with torch.no_grad():
        for key, ep in val_suite.items():
            X_tr = ep["X_train"].to(device)  # (B, n_train, p)
            Z_tr = ep["Y_train"].to(device)  # (B, n_train, d) — Y_train as Z proxy
            Z_te = ep["Y_test"].to(device)  # (B, n_test,  d)

            B, n_train, _ = Z_tr.shape
            _, n_test, d = Z_te.shape

            X_te = ep["X_test"].to(device)  # (B, n_test, p)
            X_all = torch.cat([X_tr, X_te], dim=1)
            Z_all = torch.cat([Z_tr, torch.zeros_like(Z_te)], dim=1)

            # ---- Model forward ----
            mu_Z, d_Z, V_Z = model(X_all, Z_all, n_support=n_train)

            wnll = woodbury_nll(Z_te, mu_Z, d_Z, V_Z).item()
            mnll = marginal_nll(Z_te, mu_Z, d_Z).item()
            agg["val/woodbury_nll"].append(wnll)
            agg["val/marginal_nll"].append(mnll)
            agg["val/nll_ratio"].append(wnll / (mnll + 1e-8))

            es = energy_score(
                mu=mu_Z[0, 0], D=d_Z[0, 0], V=V_Z[0, 0], y_ref=Z_te[0, 0], n_samples=200
            ).item()
            agg["val/energy_score"].append(es)

            # ---- Global baselines ----
            # Prior: N(0, I)
            prior_nll = marginal_nll(
                Z_te, torch.zeros_like(mu_Z), torch.ones_like(d_Z)
            ).item()
            agg["val/prior_nll"].append(prior_nll)

            # Empirical mean + identity covariance (isolates mean estimation)
            mu_emp = Z_tr.mean(dim=1, keepdim=True).expand(
                -1, n_test, -1
            )  # (B, n_test, d)
            mean_only_nll = marginal_nll(Z_te, mu_emp, torch.ones_like(d_Z)).item()
            agg["val/mean_only_nll"].append(mean_only_nll)

            # Independent: zero mean + per-dim variance
            var_train = Z_tr.var(dim=1, unbiased=True).clamp(min=1e-6)  # (B, d)
            d_ind = var_train.unsqueeze(1).expand(-1, n_test, -1)
            independent_nll = marginal_nll(Z_te, torch.zeros_like(mu_Z), d_ind).item()
            agg["val/independent_nll"].append(independent_nll)

            # Independent with empirical mean (separates mean calibration from covariance)
            independent_mean_nll = marginal_nll(Z_te, mu_emp, d_ind).item()
            agg["val/independent_mean_nll"].append(independent_mean_nll)

            # Full MLE covariance + OAS — loop over batch elements (sklearn works on 2-D)
            full_mle_nlls, oas_nlls = [], []
            for b in range(B):
                Z_tr_b_np = Z_tr[b].cpu().numpy()  # (n_train, d)
                Z_te_b = Z_te[b]  # (n_test, d)

                # MLE covariance
                mu_mle = Z_tr[b].mean(0)
                diff = Z_tr[b] - mu_mle
                Sigma_mle = (diff.T @ diff) / max(n_train - 1, 1)
                Sigma_mle = 0.5 * (Sigma_mle + Sigma_mle.T)
                mu_p, d_p, V_p = _cov_to_woodbury_params(
                    Sigma_mle, mu_mle, n_test, device
                )
                full_mle_nlls.append(woodbury_nll(Z_te_b, mu_p, d_p, V_p).item())

                # OAS shrinkage covariance
                oas = OAS().fit(Z_tr_b_np)
                Sigma_oas = torch.tensor(
                    oas.covariance_, dtype=torch.float32, device=device
                )
                mu_oas = torch.tensor(oas.location_, dtype=torch.float32, device=device)
                mu_p, d_p, V_p = _cov_to_woodbury_params(
                    Sigma_oas, mu_oas, n_test, device
                )
                oas_nlls.append(woodbury_nll(Z_te_b, mu_p, d_p, V_p).item())

            agg["val/full_mle_nll"].append(float(np.mean(full_mle_nlls)))
            agg["val/oas_nll"].append(float(np.mean(oas_nlls)))

            # ---- Heteroskedastic baselines ----
            knn5_nlls, knn20_nlls, kernel_nlls, linear_nlls = [], [], [], []
            for b in range(B):
                Z_tr_b = Z_tr[b]  # (n_train, d)
                Z_te_b = Z_te[b]  # (n_test, d)
                X_tr_b = X_tr[b]  # (n_train, p)
                X_te_b = X_te[b]  # (n_test, p)

                dists = torch.cdist(X_te_b, X_tr_b)  # (n_test, n_train)

                # kNN covariance (k=5 and k=20)
                for k, nlls_list in ((5, knn5_nlls), (20, knn20_nlls)):
                    k_eff = min(k, n_train)
                    nlls_i = []
                    for i in range(n_test):
                        idx = dists[i].topk(k_eff, largest=False).indices
                        Z_nb = Z_tr_b[idx]  # (k_eff, d)
                        if k_eff > d:
                            Sigma_i = torch.cov(Z_nb.T)
                        else:
                            # Too few neighbors for full cov — fall back to diagonal
                            Sigma_i = torch.diag(
                                Z_nb.var(0, unbiased=False).clamp(min=1e-6)
                            )
                        Sigma_i = 0.5 * (Sigma_i + Sigma_i.T)
                        mu_nb = Z_nb.mean(0)
                        mu_p, d_p, V_p = _cov_to_woodbury_params(
                            Sigma_i, mu_nb, 1, device
                        )
                        nlls_i.append(
                            woodbury_nll(Z_te_b[i : i + 1], mu_p, d_p, V_p).item()
                        )
                    nlls_list.append(float(np.mean(nlls_i)))

                # Kernel-weighted covariance (median bandwidth)
                dists_sq = dists**2  # (n_test, n_train)
                tau = torch.median(torch.cdist(X_tr_b, X_tr_b)).clamp(min=1e-6)
                kernel_nlls_i = []
                for i in range(n_test):
                    w = torch.softmax(-dists_sq[i] / tau, dim=0)  # (n_train,)
                    mu_w = (w[:, None] * Z_tr_b).sum(0)
                    Z_c = Z_tr_b - mu_w
                    Sigma_w = (w[:, None] * Z_c).T @ Z_c
                    Sigma_w = 0.5 * (Sigma_w + Sigma_w.T)
                    mu_p, d_p, V_p = _cov_to_woodbury_params(Sigma_w, mu_w, 1, device)
                    kernel_nlls_i.append(
                        woodbury_nll(Z_te_b[i : i + 1], mu_p, d_p, V_p).item()
                    )
                kernel_nlls.append(float(np.mean(kernel_nlls_i)))

                # Linear factor: OLS x → Z mean + global residual covariance
                W = torch.linalg.lstsq(X_tr_b, Z_tr_b).solution  # (p, d)
                mu_lin = X_te_b @ W  # (n_test, d)
                resid = Z_tr_b - X_tr_b @ W  # (n_train, d)
                Sigma_resid = (
                    torch.cov(resid.T)
                    if n_train > d
                    else torch.diag(resid.var(0, unbiased=False).clamp(min=1e-6))
                )
                Sigma_resid = 0.5 * (Sigma_resid + Sigma_resid.T)
                lin_nlls_i = []
                for i in range(n_test):
                    mu_p, d_p, V_p = _cov_to_woodbury_params(
                        Sigma_resid, mu_lin[i], 1, device
                    )
                    lin_nlls_i.append(
                        woodbury_nll(Z_te_b[i : i + 1], mu_p, d_p, V_p).item()
                    )
                linear_nlls.append(float(np.mean(lin_nlls_i)))

            agg["val/knn5_cov_nll"].append(float(np.mean(knn5_nlls)))
            agg["val/knn20_cov_nll"].append(float(np.mean(knn20_nlls)))
            agg["val/kernel_cov_nll"].append(float(np.mean(kernel_nlls)))
            agg["val/linear_factor_nll"].append(float(np.mean(linear_nlls)))

            # ---- Oracle ----
            oracle_nll = woodbury_nll(
                Z_te,
                ep["oracle_mu"].to(device),
                ep["oracle_D"].to(device),
                ep["oracle_V"].to(device),
            ).item()
            agg["val/oracle_nll"].append(oracle_nll)

            # ---- Diagnostic gaps ----
            oas_nll_ep = float(np.mean(oas_nlls))
            agg["val/heteroskedastic_gain"].append(oas_nll_ep - oracle_nll)
            agg["val/model_vs_knn5_gap"].append(wnll - float(np.mean(knn5_nlls)))
            denom = prior_nll - oracle_nll
            agg["val/oracle_gap_fraction"].append(
                (wnll - oracle_nll) / denom if abs(denom) > 1e-8 else float("nan")
            )

            # --- Optional Plotting (first episode only) ---
            if do_plot and plot_fig is None and wandb_run is not None:
                plot_fig = plot_prediction_comparison(
                    mu_pred=mu_Z,
                    D_pred=d_Z,
                    V_pred=V_Z,
                    mu_true=ep["oracle_mu"].to(device),
                    D_true=ep["oracle_D"].to(device),
                    V_true=ep["oracle_V"].to(device),
                    n_instances=min(3, n_test),
                )

    # Average across val suite entries
    metrics = {k: float(np.mean(v)) for k, v in agg.items()}
    metrics["step"] = step

    # Console output
    print(
        f"[val step={step:>6d}]  "
        f"woodbury={metrics['val/woodbury_nll']:.4f}  "
        f"oracle={metrics['val/oracle_nll']:.4f}  "
        f"knn5={metrics['val/knn5_cov_nll']:.4f}  "
        f"linear={metrics['val/linear_factor_nll']:.4f}  "
        f"oas={metrics['val/oas_nll']:.4f}  "
        f"prior={metrics['val/prior_nll']:.4f}  "
        f"hetero_gain={metrics['val/heteroskedastic_gain']:.4f}  "
        f"vs_knn5={metrics['val/model_vs_knn5_gap']:.4f}  "
        f"oracle_frac={metrics['val/oracle_gap_fraction']:.4f}"
    )

    if wandb_run is not None:
        if plot_fig is not None:
            import wandb

            metrics["val/plot"] = wandb.Image(plot_fig)
        wandb_run.log(metrics, step=step)
        if plot_fig is not None:
            plt.close(plot_fig)

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
    model: nn.Module = build_copula_transformer(cfg).to(device)
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
        _run_name = (
            f"lr={_lr_str}"
            f"_steps={cfg.training.steps}"
            f"_d={cfg.model.d_model}"
            f"_L={cfg.model.n_layers}"
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

    # ---- Data loader ----
    loader = make_episode_loader(
        cfg.training.dataset_dir,
        shuffle=True,
        num_workers=int(cfg.dataset.num_workers),
    )
    episode_iter = infinite_episode_iter(loader)

    # ---- Validation suite (synthetic, fixed) ----
    # Build covariance generator consistent with the training data
    fixed_cov = bool(cfg.data.get("fixed_cov", False))
    r_data = int(cfg.data.r_data)
    mlp_hidden = int(cfg.data.mlp_hidden)
    fixed_nets: GlobalFixedNets | None = None
    anchor_gen: GlobalAnchorCovGen | None = None
    if not fixed_cov:
        cov_type = str(cfg.data.get("cov_type", "mlp"))
        if cov_type == "anchor":
            anchor_gen = GlobalAnchorCovGen(
                K=int(cfg.data.get("num_anchors", 8)),
                r=r_data,
                tau=float(cfg.data.get("anchor_temp", 1.0)),
                device=device,
            )
        else:
            fixed_nets = GlobalFixedNets(r=r_data, hidden=mlp_hidden, device=device)

    print("Building validation suite …")
    val_suite = build_val_suite(
        cfg,
        device,
        fixed_nets=fixed_nets,
        anchor_gen=anchor_gen,
    )
    print(f"Validation suite: {len(val_suite)} episodes  ({list(val_suite.keys())})")

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
            run_validation(
                model, val_suite, step, cfg, wandb_run, device, do_plot=do_plot
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

    run_validation(model, val_suite, final_step, cfg, wandb_run, device)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
