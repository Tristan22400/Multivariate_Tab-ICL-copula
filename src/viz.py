"""
viz.py — Visualisation utilities for CopulaTransformer predictions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_corr_grid(
    estimators: dict[str, torch.Tensor],
    oracle_R: torch.Tensor,
    n_instances: int = 3,
    title: str = "",
) -> plt.Figure:
    """Compare correlation matrices from multiple estimators against the oracle.

    Layout: one column per estimator (plus one oracle column), n_instances rows.
    Each cell shows the (d×d) correlation heatmap.
    A second block of rows shows the absolute error |R* - R_hat| per estimator.

    Args:
        estimators : dict mapping estimator name → R tensor of shape (d, d) or
                     (n_qry, d, d).  A (d, d) tensor is broadcast to all instances.
        oracle_R   : (n_qry, d, d) or (d, d) — oracle correlation matrices.
        n_instances: number of query instances (rows) to plot.
        title      : figure suptitle.

    Returns:
        matplotlib Figure.
    """
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    # Resolve oracle shape
    if oracle_R.dim() == 2:
        oracle_R = oracle_R.unsqueeze(0)
    n_qry = oracle_R.shape[0]
    n_instances = min(n_instances, n_qry)
    d = oracle_R.shape[-1]
    indices = np.linspace(0, n_qry - 1, n_instances, dtype=int)

    # Normalise each estimator to (n_qry, d, d)
    est_tensors: dict[str, np.ndarray] = {}
    for name, R in estimators.items():
        if R.dim() == 2:
            R = R.unsqueeze(0).expand(n_qry, -1, -1)
        est_tensors[name] = R.detach().cpu().numpy()

    oracle_np = oracle_R.detach().cpu().numpy()
    names = list(est_tensors.keys())
    n_est = len(names)

    # Layout: 2 * n_instances rows (predicted | error), n_est+1 columns (oracle + estimators)
    n_cols = n_est + 1
    n_rows = 2 * n_instances
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    def _heatmap(ax, data, vmin, vmax, cmap="coolwarm"):
        if sns is not None:
            sns.heatmap(
                data,
                ax=ax,
                cmap=cmap,
                center=0 if cmap == "coolwarm" else None,
                vmin=vmin,
                vmax=vmax,
                square=True,
                xticklabels=False,
                yticklabels=False,
                cbar=False,
            )
        else:
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])

    for row_i, inst_idx in enumerate(indices):
        R_oracle = oracle_np[inst_idx]  # (d, d)
        cov_max = max(abs(R_oracle).max(), 1.0)

        # --- Top half: predicted correlation matrices ---
        pred_row = row_i * 2

        # Oracle column (col 0)
        ax = axes[pred_row, 0]
        _heatmap(ax, R_oracle, -cov_max, cov_max)
        ax.set_title(f"Oracle R* (i={inst_idx})", fontsize=8)
        if row_i == 0:
            axes[pred_row, 0].set_ylabel("Predicted", fontsize=8)

        # Estimator columns
        for col_i, name in enumerate(names):
            R_est = est_tensors[name][inst_idx]
            ax = axes[pred_row, col_i + 1]
            _heatmap(ax, R_est, -cov_max, cov_max)
            ax.set_title(name, fontsize=8)

        # --- Bottom half: absolute error |R* - R_hat| ---
        err_row = row_i * 2 + 1

        # Empty oracle column for the error row (just blank)
        axes[err_row, 0].axis("off")
        if row_i == 0:
            axes[err_row, 0].set_ylabel("|R* − R̂|", fontsize=8)

        for col_i, name in enumerate(names):
            R_est = est_tensors[name][inst_idx]
            err = np.abs(R_oracle - R_est)
            ax = axes[err_row, col_i + 1]
            _heatmap(ax, err, 0, err.max().clip(min=1e-6), cmap="Reds")
            ax.set_title(f"|R*−{name}|", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97] if title else None)
    return fig


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
    sigma_oas: np.ndarray | None = None,
    fig: plt.Figure | None = None,
    dataset_label: str = "",
) -> plt.Figure:
    """Compare predicted vs oracle covariance for multiple query instances.

    One row per instance, four columns:
      0 — Oracle  Sigma*    heatmap
      1 — Predicted Sigma   heatmap
      2 — |Sigma* - Sigma|  heatmap
      3 — OAS Sigma heatmap (if sigma_oas provided, first row only)

    Args:
        mu_pred      : (B, N, d)    — unused, kept for API compatibility
        D_pred       : (B, N, d)    — predicted diagonal variances
        V_pred       : (B, N, d, r) — predicted low-rank factors
        mu_true      : (B, N, d)    — unused, kept for API compatibility
        D_true       : (B, N, d)    — oracle diagonal variances
        V_true       : (B, N, d, r) — oracle low-rank factors
        batch_idx    : which batch element to visualise
        n_instances  : number of query instances (rows) to plot
        mu_baseline  : unused, kept for API compatibility
        baseline_label : unused, kept for API compatibility
        sigma_oas    : (d, d) optional — OAS shrinkage covariance matrix
        fig          : optional existing Figure (or SubFigure) to draw into;
                       if None a new Figure is created.
        dataset_label: if non-empty, added as a suptitle to identify the dataset.

    Returns:
        matplotlib Figure with n_instances × 4 subplots.
    """

    try:
        import seaborn as sns
    except ImportError:
        sns = None

    N = D_pred.shape[1]
    n_instances = min(n_instances, N)
    indices = np.linspace(0, N - 1, n_instances, dtype=int)

    created_fig = fig is None
    if created_fig:
        fig = plt.figure(figsize=(20, 5 * n_instances))
    axes = fig.subplots(n_instances, 4)
    if dataset_label:
        fig.suptitle(dataset_label, fontsize=13, fontweight="bold", y=1.01)
    if n_instances == 1:
        axes = axes[np.newaxis, :]

    for row, inst_idx in enumerate(indices):
        # ------------------------------------------------------------------ #
        # Covariance matrices                                                  #
        # ------------------------------------------------------------------ #
        Sp_V = V_pred[batch_idx, inst_idx]  # (d, r)
        Sp_D = torch.diag(D_pred[batch_idx, inst_idx])  # (d, d)
        Sigma_pred = (Sp_D + Sp_V @ Sp_V.T).detach().cpu().numpy()

        St_V = V_true[batch_idx, inst_idx]
        St_D = torch.diag(D_true[batch_idx, inst_idx])
        Sigma_true = (St_D + St_V @ St_V.T).detach().cpu().numpy()

        cov_max = max(np.abs(Sigma_true).max(), np.abs(Sigma_pred).max())
        heatmap_kw = dict(
            cmap="coolwarm", center=0, vmin=-cov_max, vmax=cov_max, square=True
        )

        if sns is not None:
            sns.heatmap(Sigma_true, ax=axes[row, 0], **heatmap_kw)
            sns.heatmap(Sigma_pred, ax=axes[row, 1], **heatmap_kw)
            sns.heatmap(
                np.abs(Sigma_true - Sigma_pred),
                ax=axes[row, 2],
                cmap="Reds",
                square=True,
            )
        else:
            axes[row, 0].imshow(
                Sigma_true, cmap="coolwarm", vmin=-cov_max, vmax=cov_max
            )
            axes[row, 1].imshow(
                Sigma_pred, cmap="coolwarm", vmin=-cov_max, vmax=cov_max
            )
            axes[row, 2].imshow(np.abs(Sigma_true - Sigma_pred), cmap="Reds")

        axes[row, 0].set_title(rf"Oracle $\Sigma^*$ (inst {inst_idx})")
        axes[row, 1].set_title(rf"Predicted $\hat{{\Sigma}}$ (inst {inst_idx})")
        axes[row, 2].set_title(rf"$|\Sigma^* - \hat{{\Sigma}}|$ (inst {inst_idx})")

        # ------------------------------------------------------------------ #
        # Column 3: OAS covariance heatmap (first row only)                  #
        # ------------------------------------------------------------------ #
        ax = axes[row, 3]
        if sigma_oas is not None and row == 0:
            oas_max = max(np.abs(sigma_oas).max(), cov_max)
            oas_heatmap_kw = dict(
                cmap="coolwarm", center=0, vmin=-oas_max, vmax=oas_max, square=True
            )
            if sns is not None:
                sns.heatmap(sigma_oas, ax=ax, **oas_heatmap_kw)
            else:
                ax.imshow(sigma_oas, cmap="coolwarm", vmin=-oas_max, vmax=oas_max)
            ax.set_title(r"OAS $\hat{\Sigma}_{OAS}$ (global baseline)")
        else:
            ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.97] if dataset_label else None)
    return fig
