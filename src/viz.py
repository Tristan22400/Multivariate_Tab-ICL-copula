"""
viz.py — Visualisation utilities for CopulaTransformer predictions.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_prediction_comparison(
    mu_pred: torch.Tensor,
    D_pred: torch.Tensor,
    V_pred: torch.Tensor,
    mu_true: torch.Tensor,
    D_true: torch.Tensor,
    V_true: torch.Tensor,
    batch_idx: int = 0,
    n_instances: int = 3,
    mu_baseline: torch.Tensor | None = None,
    baseline_label: str = "Baseline",
) -> plt.Figure:
    """Compare predicted vs oracle mean and covariance for multiple query instances.

    One row per instance, five columns:
      0 — Oracle  Sigma*    heatmap
      1 — Predicted Sigma   heatmap
      2 — |Sigma* - Sigma|  heatmap
      3 — mu* vs mu hat (+ optional baseline) bar chart
      4 — |mu* - mu hat| vs |mu* - baseline| bar chart

    Args:
        mu_pred      : (B, N, d)    — predicted conditional means
        D_pred       : (B, N, d)    — predicted diagonal variances
        V_pred       : (B, N, d, r) — predicted low-rank factors
        mu_true      : (B, N, d)    — oracle conditional means
        D_true       : (B, N, d)    — oracle diagonal variances
        V_true       : (B, N, d, r) — oracle low-rank factors
        batch_idx    : which batch element to visualise
        n_instances  : number of query instances (rows) to plot
        mu_baseline  : (B, N, d) optional — baseline scalar predictions
                       (e.g. marginal-only or independent-TabICL means)
        baseline_label : legend label for the baseline bars

    Returns:
        matplotlib Figure with n_instances × 5 subplots.
    """
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    N = D_pred.shape[1]
    n_instances = min(n_instances, N)
    indices = np.linspace(0, N - 1, n_instances, dtype=int)

    fig, axes = plt.subplots(n_instances, 5, figsize=(26, 5 * n_instances))
    if n_instances == 1:
        axes = axes[np.newaxis, :]

    for row, inst_idx in enumerate(indices):
        # ------------------------------------------------------------------ #
        # Covariance matrices                                                  #
        # ------------------------------------------------------------------ #
        Sp_V = V_pred[batch_idx, inst_idx]                        # (d, r)
        Sp_D = torch.diag(D_pred[batch_idx, inst_idx])            # (d, d)
        Sigma_pred = (Sp_D + Sp_V @ Sp_V.T).detach().cpu().numpy()

        St_V = V_true[batch_idx, inst_idx]
        St_D = torch.diag(D_true[batch_idx, inst_idx])
        Sigma_true = (St_D + St_V @ St_V.T).detach().cpu().numpy()

        cov_max = max(np.abs(Sigma_true).max(), np.abs(Sigma_pred).max())
        heatmap_kw = dict(cmap="coolwarm", center=0, vmin=-cov_max, vmax=cov_max, square=True)

        if sns is not None:
            sns.heatmap(Sigma_true, ax=axes[row, 0], **heatmap_kw)
            sns.heatmap(Sigma_pred, ax=axes[row, 1], **heatmap_kw)
            sns.heatmap(np.abs(Sigma_true - Sigma_pred), ax=axes[row, 2], cmap="Reds", square=True)
        else:
            axes[row, 0].imshow(Sigma_true, cmap="coolwarm", vmin=-cov_max, vmax=cov_max)
            axes[row, 1].imshow(Sigma_pred, cmap="coolwarm", vmin=-cov_max, vmax=cov_max)
            axes[row, 2].imshow(np.abs(Sigma_true - Sigma_pred), cmap="Reds")

        axes[row, 0].set_title(rf"Oracle $\Sigma^*$ (inst {inst_idx})")
        axes[row, 1].set_title(rf"Predicted $\hat{{\Sigma}}$ (inst {inst_idx})")
        axes[row, 2].set_title(rf"$|\Sigma^* - \hat{{\Sigma}}|$ (inst {inst_idx})")

        # ------------------------------------------------------------------ #
        # Mean vectors                                                         #
        # ------------------------------------------------------------------ #
        mu_t = mu_true[batch_idx, inst_idx].detach().cpu().numpy()   # (d,)
        mu_p = mu_pred[batch_idx, inst_idx].detach().cpu().numpy()   # (d,)
        d = len(mu_t)
        dims = np.arange(d)

        ax = axes[row, 3]
        if mu_baseline is not None:
            mu_b = mu_baseline[batch_idx, inst_idx].detach().cpu().numpy()
            width = 0.25
            ax.bar(dims - width, mu_t, width, label=r"Oracle $\mu^*$",   color="#2563EB", alpha=0.8)
            ax.bar(dims,          mu_p, width, label=r"Predicted $\hat{\mu}$", color="#EA580C", alpha=0.8)
            ax.bar(dims + width,  mu_b, width, label=baseline_label,     color="#16A34A", alpha=0.8)
        else:
            width = 0.35
            ax.bar(dims - width / 2, mu_t, width, label=r"Oracle $\mu^*$",   color="#2563EB", alpha=0.8)
            ax.bar(dims + width / 2, mu_p, width, label=r"Predicted $\hat{\mu}$", color="#EA580C", alpha=0.8)

        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xticks(dims)
        ax.set_xlabel("dim")
        ax.set_title(rf"Mean (inst {inst_idx})")
        ax.legend(fontsize=8)

        # ------------------------------------------------------------------ #
        # Absolute mean error                                                  #
        # ------------------------------------------------------------------ #
        ax = axes[row, 4]
        if mu_baseline is not None:
            mu_b = mu_baseline[batch_idx, inst_idx].detach().cpu().numpy()
            width = 0.35
            ax.bar(dims - width / 2, np.abs(mu_t - mu_p), width,
                   label=r"$|\mu^*-\hat{\mu}|$", color="#7C3AED", alpha=0.8)
            ax.bar(dims + width / 2, np.abs(mu_t - mu_b), width,
                   label=f"|μ*−{baseline_label}|", color="#16A34A", alpha=0.8)
            ax.legend(fontsize=8)
        else:
            ax.bar(dims, np.abs(mu_t - mu_p), color="#7C3AED", alpha=0.8)

        ax.set_xticks(dims)
        ax.set_xlabel("dim")
        ax.set_title(rf"$|\mu^* - \hat{{\mu}}|$ (inst {inst_idx})")

    plt.tight_layout()
    return fig
