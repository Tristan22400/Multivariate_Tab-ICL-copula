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
    n_instances: int | None = None,
    title: str = "",
) -> plt.Figure:
    """Compare correlation matrices from multiple estimators against the oracle.

    Layout: one column per estimator (plus one oracle column), one row per test
    instance.  Each cell shows the (d×d) correlation heatmap.

    Args:
        estimators : dict mapping estimator name → R tensor of shape (d, d) or
                     (n_qry, d, d).  A (d, d) tensor is broadcast to all instances.
        oracle_R   : (n_qry, d, d) or (d, d) — oracle correlation matrices.
        n_instances: number of query instances (rows) to plot.  Defaults to all.
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
    if n_instances is None:
        n_instances = n_qry
    else:
        n_instances = min(n_instances, n_qry)
    indices = list(range(n_instances))

    # Normalise each estimator to (n_qry, d, d)
    est_tensors: dict[str, np.ndarray] = {}
    for name, R in estimators.items():
        if R.dim() == 2:
            R = R.unsqueeze(0).expand(n_qry, -1, -1)
        est_tensors[name] = R.detach().cpu().numpy()

    oracle_np = oracle_R.detach().cpu().numpy()
    names = list(est_tensors.keys())
    n_est = len(names)

    n_cols = n_est + 1
    n_rows = n_instances
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

        # Oracle column (col 0)
        ax = axes[row_i, 0]
        _heatmap(ax, R_oracle, -cov_max, cov_max)
        ax.set_title(f"Oracle R* (i={inst_idx})", fontsize=8)

        # Estimator columns
        for col_i, name in enumerate(names):
            R_est = est_tensors[name][inst_idx]
            ax = axes[row_i, col_i + 1]
            _heatmap(ax, R_est, -cov_max, cov_max)
            ax.set_title(name, fontsize=8)

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
      3 — Row 0: OAS Sigma heatmap (if provided); rows 1+: off-diagonal
          scatter (predicted vs oracle) with OLS regression line, annotated
          with Pearson r, slope, and intercept.  A slope < 1 indicates the
          model under-disperses predictions; a non-zero intercept is a
          constant offset.  hexbin is used when n_pairs > 30.

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

    # ------------------------------------------------------------------ #
    # Pre-collect correlation matrices for the off-diagonal scatter.
    # The model enforces unit diagonal by construction so Sigma_pred is
    # already a correlation matrix.  The oracle D/V are in Y-space where
    # diag(D) + ||V||^2 = Var(Y_test)/Var(Y_train) != 1, so we normalize
    # to the copula correlation matrix R[i,j] = Sigma[i,j]/sqrt(Sigma[ii]*Sigma[jj]).
    # ------------------------------------------------------------------ #
    def _cov_to_corr(sigma: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.diag(sigma).clip(min=1e-8))
        return sigma / np.outer(std, std)

    all_sigma_pred: list[np.ndarray] = []
    all_sigma_true: list[np.ndarray] = []
    for inst_idx in indices:
        Sp_V = V_pred[batch_idx, inst_idx]
        Sp_D = torch.diag(D_pred[batch_idx, inst_idx])
        all_sigma_pred.append((Sp_D + Sp_V @ Sp_V.T).detach().cpu().numpy())
        St_V = V_true[batch_idx, inst_idx]
        St_D = torch.diag(D_true[batch_idx, inst_idx])
        all_sigma_true.append(
            _cov_to_corr((St_D + St_V @ St_V.T).detach().cpu().numpy())
        )

    d = all_sigma_pred[0].shape[0]
    tri_r, tri_c = np.triu_indices(d, k=1)  # upper-triangle off-diagonal indices
    # Shapes: (n_instances, n_pairs)
    pred_off = np.stack([s[tri_r, tri_c] for s in all_sigma_pred])
    true_off = np.stack([s[tri_r, tri_c] for s in all_sigma_true])

    created_fig = fig is None
    if created_fig:
        fig = plt.figure(figsize=(20, 5 * n_instances))
    axes = fig.subplots(n_instances, 4)
    if dataset_label:
        fig.suptitle(dataset_label, fontsize=13, fontweight="bold", y=1.01)
    if n_instances == 1:
        axes = axes[np.newaxis, :]

    for row, inst_idx in enumerate(indices):
        Sigma_pred = all_sigma_pred[row]
        Sigma_true = all_sigma_true[row]

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

        axes[row, 0].set_title(rf"Oracle $R^*$ (inst {inst_idx})")
        axes[row, 1].set_title(rf"Predicted $\hat{{R}}$ (inst {inst_idx})")
        axes[row, 2].set_title(rf"$|R^* - \hat{{R}}|$ (inst {inst_idx})")

        # ------------------------------------------------------------------ #
        # Column 3: OAS (row 0) or off-diagonal scatter with OLS (rows 1+)   #
        # ------------------------------------------------------------------ #
        ax = axes[row, 3]
        if row == 0:
            if sigma_oas is not None:
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
        else:
            # Off-diagonal scatter: predicted vs oracle, with OLS regression.
            # slope < 1  → model under-disperses (regresses to mean)
            # intercept ≠ 0 → constant offset bias
            x = true_off[row]  # oracle off-diagonal entries
            y = pred_off[row]  # predicted off-diagonal entries
            n_pairs = len(x)

            if n_pairs > 30:
                ax.hexbin(x, y, gridsize=20, cmap="Blues", mincnt=1)
            else:
                ax.scatter(x, y, alpha=0.6, s=20, color="steelblue", linewidths=0)

            # OLS regression line
            slope, intercept = np.polyfit(x, y, 1) if x.std() > 1e-8 else (1.0, 0.0)
            x_lo, x_hi = x.min(), x.max()
            x_line = np.array([x_lo, x_hi])
            ax.plot(x_line, slope * x_line + intercept, "r-", lw=1.5, label="OLS fit")

            # Identity reference
            lim = max(np.abs(x).max(), np.abs(y).max(), 1e-8) * 1.15
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.35, label="y=x")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.axhline(0, color="gray", lw=0.4, ls=":")
            ax.axvline(0, color="gray", lw=0.4, ls=":")

            r = (
                float(np.corrcoef(x, y)[0, 1])
                if x.std() > 1e-8 and y.std() > 1e-8
                else float("nan")
            )
            ax.set_title(
                rf"Off-diag $\hat{{R}}$ vs $R^*$ (inst {inst_idx})"
                + f"\n$r={r:.2f}$  slope$={slope:.2f}$  $b={intercept:.3f}$",
                fontsize=7,
            )
            ax.set_xlabel(r"Oracle $R^*_{ij}$", fontsize=7)
            ax.set_ylabel(r"Predicted $\hat{R}_{ij}$", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.tight_layout(rect=[0, 0, 1, 0.97] if dataset_label else None)
    return fig
