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

    Metrics reported per episode and averaged across the val suite:
        woodbury_nll   — primary loss (full covariance)
        marginal_nll   — diagonal-only baseline
        nll_ratio      — woodbury_nll / marginal_nll  (< 1 means model helps)
        energy_score   — MC energy score on the first val instance

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
        "val/woodbury_nll": [],
        "val/marginal_nll": [],
        "val/nll_ratio": [],
        "val/energy_score": [],
    }
    plot_fig = None

    with torch.no_grad():
        for key, ep in val_suite.items():
            X_tr = ep["X_train"].to(device)  # (B, n_train, p)
            Z_tr = ep["Y_train"].to(device)  # (B, n_train, d) — Y_train as Z proxy
            Z_te = ep["Y_test"].to(device)  # (B, n_test,  d) — Y_test  as Z proxy

            B, n_train, _ = Z_tr.shape
            _, n_test, d = Z_te.shape

            # Full context: n_support = n_train, query = n_test
            # We concatenate X/Z so the model sees [support | query] along T dim,
            # with n_support pointing at the split boundary.
            X_te = ep["X_test"].to(device)  # (B, n_test, p)
            X_all = torch.cat([X_tr, X_te], dim=1)  # (B, n_train+n_test, p)
            Z_all = torch.cat([Z_tr, torch.zeros_like(Z_te)], dim=1)  # mask query Z

            # Forward: use full train set as context, predict over test positions
            mu_Z, d_Z, V_Z = model(X_all, Z_all, n_support=n_train)

            # Woodbury NLL on query (test) portion
            wnll = woodbury_nll(Z_te, mu_Z, d_Z, V_Z).item()
            mnll = marginal_nll(Z_te, mu_Z, d_Z).item()
            ratio = wnll / (mnll + 1e-8)

            agg["val/woodbury_nll"].append(wnll)
            agg["val/marginal_nll"].append(mnll)
            agg["val/nll_ratio"].append(ratio)

            # Energy score: evaluate on the first instance of the first batch element
            es = energy_score(
                mu=mu_Z[0, 0],  # (d,)
                D=d_Z[0, 0],  # (d,)
                V=V_Z[0, 0],  # (d, r)
                y_ref=Z_te[0, 0],  # (d,)
                n_samples=200,
            ).item()
            agg["val/energy_score"].append(es)

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
        f"woodbury_nll={metrics['val/woodbury_nll']:.4f}  "
        f"marginal_nll={metrics['val/marginal_nll']:.4f}  "
        f"nll_ratio={metrics['val/nll_ratio']:.4f}  "
        f"energy_score={metrics['val/energy_score']:.4f}"
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

        wandb_run = wandb.init(
            project="multivariate-tab-icl",
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
                Z_test = episode["Z_test"].to(device)
                oracle_nll = woodbury_nll(Z_test, oracle_mu, oracle_D, oracle_V).item()

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
