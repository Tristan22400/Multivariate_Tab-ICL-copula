"""
copula_train.py — Training entry point for the In-Context Attentional Copula Model.

Usage (from the project root):
    python src/copula_train.py
    python src/copula_train.py training.steps=500            # quick smoke-test
    python src/copula_train.py model.n_layers=4 model.n_bins=50

Training loop:
  1. Sample p, d, n_train, n_test from configured ranges.
  2. Generate one episode of B synthetic datasets.
  3. Compute U_train = empirical_pit(Y_train) inside the model.
  4. Compute U_test  = smooth_context_pit(Y_test, Y_train) inside the model.
  5. Teacher-forced AR forward over a random dimension permutation.
  6. Cross-entropy loss over discretized bins.
  7. Backprop + gradient clip + cosine annealing step.
  8. Periodic validation and W&B logging.

No frozen base model is loaded — the copula model is fully trainable.
"""

from __future__ import annotations

import os
import sys
import math
import random

import numpy as np
import torch
import wandb
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from data_gen import generate_episode, build_val_suite, GlobalFixedNets, GlobalAnchorCovGen
from copula_loss import (
    copula_ce_loss,
    copula_energy_score,
    marginal_calibration_hist,
    smooth_context_pit,
    empirical_pit,
)
from models.attentional_copula import build_copula_model


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(
    model,
    val_suite: dict,
    step:      int,
    cfg:       DictConfig,
    wandb_run,
    device:    str,
) -> dict[str, float]:
    """Evaluate on the fixed validation suite and log to W&B.

    Metrics per grid key (d, n_train):
      • CE loss    : teacher-forced cross-entropy (lower is better)
      • ES         : energy score in uniform copula space (lower is better)
      • cal_mae    : mean absolute deviation of U histogram from Uniform[0,1]
                     (calibration; 0 = perfectly uniform marginals)
    """
    model.eval()
    metrics: dict[str, float] = {}
    n_bins    = cfg.model.n_bins
    n_perm    = cfg.model.get("n_perm_avg", 5)
    es_S      = cfg.training.get("es_samples", 50)
    fig_logged = False

    for key, batch in val_suite.items():
        d       = batch["d"]
        X_tr    = batch["X_train"].to(device)
        Y_tr    = batch["Y_train"].to(device)
        X_te    = batch["X_test"].to(device)
        Y_te    = batch["Y_test"].to(device)
        B       = X_tr.shape[0]
        N       = X_te.shape[1]

        # --- CE loss (teacher forcing, single permutation) ---
        perm = torch.randperm(d, device=device)
        logits, U_test = model(X_tr, X_te, Y_tr, Y_te, perm=perm)
        ce = copula_ce_loss(logits, U_test, n_bins).item()

        # --- Energy score: collect es_S samples across n_perm permutations ---
        all_U_samples = []
        for _ in range(max(1, es_S // n_perm)):
            p_i = torch.randperm(d, device=device)
            all_U_samples.append(model(X_tr, X_te, Y_tr, perm=p_i))  # (B, N, d)
        # Stack into (B, N, d, S)
        U_samples_stack = torch.stack(all_U_samples, dim=-1)           # (B, N, d, S)
        es = copula_energy_score(U_samples_stack, U_test)

        # --- Marginal calibration: mean |hist - 1/n_bins| across dims ---
        U_flat = torch.cat(all_U_samples, dim=0).reshape(-1, N, d)    # (B*S, N, d)
        hist   = marginal_calibration_hist(U_flat.reshape(-1, N, d), n_bins=20)
        cal_mae = (hist - 1.0 / 20).abs().mean().item()

        tag = f"val/{key}"
        metrics[f"{tag}/CE"]      = ce
        metrics[f"{tag}/ES"]      = es
        metrics[f"{tag}/cal_mae"] = cal_mae

        # --- Calibration plot (one per validation call) ---
        if wandb_run is not None and not fig_logged:
            try:
                fig_cal = _plot_calibration(hist, key)
                wandb_run.log({"val/calibration": wandb.Image(fig_cal)}, step=step)
                plt.close(fig_cal)

                fig_cop = _plot_copula_scatter(U_test, U_samples_stack, key)
                if fig_cop is not None:
                    wandb_run.log({"val/copula_scatter": wandb.Image(fig_cop)}, step=step)
                    plt.close(fig_cop)

                fig_logged = True
            except Exception as exc:
                print(f"  [val] figure logging failed: {exc}")

    if wandb_run is not None:
        wandb_run.log(metrics, step=step)

    model.train()
    return metrics


def _plot_calibration(hist: torch.Tensor, title: str) -> plt.Figure:
    """Per-dimension U histogram vs Uniform[0,1] reference."""
    d, n_bins = hist.shape
    uniform   = 1.0 / n_bins
    fig, axes = plt.subplots(1, d, figsize=(3 * d, 3), tight_layout=True)
    if d == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        x = np.linspace(0, 1, n_bins)
        ax.bar(x, hist[j].cpu().numpy(), width=1.0 / n_bins, align="edge", alpha=0.7)
        ax.axhline(uniform, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"dim {j}")
        ax.set_ylim(0, max(hist[j].max().item() * 1.2, 2 * uniform))
    fig.suptitle(f"Marginal calibration — {title}", fontsize=9)
    return fig


def _plot_copula_scatter(
    U_true:    torch.Tensor,   # (B, N, d)
    U_samples: torch.Tensor,   # (B, N, d, S)
    title:     str,
) -> plt.Figure | None:
    if U_true.shape[-1] < 2:
        return None
    d1, d2 = 0, 1
    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(6, 3), tight_layout=True)
    u_t = U_true[..., d1].reshape(-1).cpu().numpy()
    v_t = U_true[..., d2].reshape(-1).cpu().numpy()
    ax_t.scatter(u_t, v_t, s=4, alpha=0.4)
    ax_t.set_title("True U (test)")

    u_s = U_samples[..., d1, :].reshape(-1).cpu().numpy()
    v_s = U_samples[..., d2, :].reshape(-1).cpu().numpy()
    ax_s.scatter(u_s, v_s, s=2, alpha=0.2)
    ax_s.set_title("Sampled U")

    for ax in (ax_t, ax_s):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(f"u_{d1}"); ax.set_ylabel(f"u_{d2}")
    fig.suptitle(f"Copula scatter — {title}", fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../conf", config_name="config_copula")
def main(cfg: DictConfig):
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = cfg.training.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device  : {device}")
    print(f"Config  :\n{OmegaConf.to_yaml(cfg)}")

    # --- Model ---
    print("Building AttentionalCopulaModel …")
    model = build_copula_model(cfg, device)
    model.train()

    # --- Optimiser + scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.steps, eta_min=cfg.training.lr_min
    )

    # --- W&B ---
    m = cfg.model
    _run_name = (
        f"copula_h{m.d_model}_nh{m.n_heads}_nl{m.n_layers}"
        f"_bins{m.n_bins}_lr{cfg.training.lr}"
    )
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(
        project = cfg.experiment.wandb_project,
        name    = _run_name,
        dir     = _ROOT,
        config  = wandb_cfg,
    )

    # --- Data generation helpers ---
    p_lo,  p_hi  = int(cfg.data.p_range[0]),       int(cfg.data.p_range[1])
    d_lo,  d_hi  = int(cfg.data.d_range[0]),       int(cfg.data.d_range[1])
    pt_lo, pt_hi = int(cfg.data.n_train_range[0]), int(cfg.data.n_train_range[1])
    nt_lo, nt_hi = int(cfg.data.n_test_range[0]),  int(cfg.data.n_test_range[1])
    r_data       = int(cfg.data.r_data)
    mlp_hidden   = int(cfg.data.mlp_hidden)
    B            = int(cfg.training.batch_size)
    n_bins       = int(cfg.model.n_bins)
    diag_alpha   = float(cfg.data.get("diag_alpha", 0.0))
    fixed_cov    = bool(cfg.data.get("fixed_cov", False))

    cov_type   = cfg.data.get("cov_type", "mlp")
    fixed_nets: GlobalFixedNets    | None = None
    anchor_gen: GlobalAnchorCovGen | None = None
    if not fixed_cov:
        if cov_type == "anchor":
            anchor_gen = GlobalAnchorCovGen(
                K   = int(cfg.data.get("num_anchors", 4)),
                r   = r_data,
                tau = float(cfg.data.get("anchor_temp", 1.0)),
                device = device,
            )
            print(f"GlobalAnchorCovGen — K={anchor_gen.K}, τ={anchor_gen.tau}")
        else:
            fixed_nets = GlobalFixedNets(r=r_data, hidden=mlp_hidden, device=device)
            print("GlobalFixedNets created.")

    # --- Fixed validation suite ---
    print("Building validation suite …")
    val_suite = build_val_suite(cfg, device, fixed_nets=fixed_nets, anchor_gen=anchor_gen)
    print(f"  {len(val_suite)} grid points: {list(val_suite.keys())}")

    # --- Training ---
    print(f"\nStarting training for {cfg.training.steps} steps …\n")
    running_loss = 0.0

    for step in range(cfg.training.steps):

        p       = int(torch.randint(p_lo,  p_hi  + 1, ()).item())
        d       = int(torch.randint(d_lo,  d_hi  + 1, ()).item())
        n_train = int(torch.randint(pt_lo, pt_hi + 1, ()).item())
        n_test  = int(torch.randint(nt_lo, nt_hi + 1, ()).item())

        X_tr, Y_tr, X_te, Y_te = generate_episode(
            B, p, d, r_data, n_train, n_test, device,
            mlp_hidden  = mlp_hidden,
            fixed_cov   = fixed_cov,
            fixed_nets  = fixed_nets,
            anchor_gen  = anchor_gen,
            diag_alpha  = diag_alpha,
            return_oracle = False,
        )

        perm = torch.randperm(d, device=device)
        logits, U_test = model(X_tr, X_te, Y_tr, Y_te, perm=perm)
        loss = copula_ce_loss(logits, U_test, n_bins)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        # --- Logging ---
        if step % cfg.training.log_every == 0:
            avg_loss = running_loss / max(1, cfg.training.log_every)
            lr_now   = scheduler.get_last_lr()[0]
            running_loss = 0.0
            print(
                f"step {step:5d} | p={p:3d} d={d} P={n_train:4d} N={n_test:3d} "
                f"| CE={avg_loss:.4f}  (ref log({n_bins})={math.log(n_bins):.2f})  lr={lr_now:.2e}"
            )
            wandb.log({
                "train/CE":         avg_loss,
                "train/lr":         lr_now,
                "train/p":          p,
                "train/d":          d,
                "train/n_train":    n_train,
                "train/n_test":     n_test,
            }, step=step)

        # --- Validation ---
        if step % cfg.training.val_every == 0 or step == cfg.training.steps - 1:
            val_metrics = run_validation(model, val_suite, step, cfg, run, device)
            ce_vals = [v for k, v in val_metrics.items() if k.endswith("/CE")]
            es_vals = [v for k, v in val_metrics.items() if k.endswith("/ES")]
            cal_vals = [v for k, v in val_metrics.items() if k.endswith("/cal_mae")]
            n = len(ce_vals)
            print(
                f"  [val] step {step} | "
                f"CE={sum(ce_vals)/n:.4f}  "
                f"ES={sum(es_vals)/n:.4f}  "
                f"cal_mae={sum(cal_vals)/n:.4f}"
            )
            model.train()

        # --- Checkpoint ---
        if (step + 1) % cfg.training.save_every == 0 or step == cfg.training.steps - 1:
            os.makedirs(cfg.training.ckpt_dir, exist_ok=True)
            fname = f"copula_step{step + 1}.pt"
            ckpt_path = os.path.join(cfg.training.ckpt_dir, fname)
            torch.save({
                "step":       step + 1,
                "seed":       seed,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "config":     OmegaConf.to_container(cfg, resolve=True),
            }, ckpt_path)
            print(f"  [ckpt] saved → {ckpt_path}")

    run.finish()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
