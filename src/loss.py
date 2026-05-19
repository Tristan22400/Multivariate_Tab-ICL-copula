"""
loss.py — Woodbury-identity Negative Log-Likelihood for low-rank Gaussians.

For a LowRankMultivariateNormal with covariance

    Sigma = diag(D) + V V^T       (D > 0, V ∈ R^{d×r})

the NLL can be computed in O(d r^2 + r^3) using the Woodbury identity
instead of the naive O(d^3) Cholesky on the full d×d covariance.

The formula (per §4 of the research spec) is:

    NLL_i = 0.5 * [ d·log(2π) + log|D_i| + log|M_i|
                    + r_i^T D_i^{-1} r_i
                    - (V_i^T D_i^{-1} r_i)^T M_i^{-1} (V_i^T D_i^{-1} r_i) ]

where M_i = I_r + V_i^T D_i^{-1} V_i  is the r×r capacitance matrix,
and  r_i = y_i - μ_i  is the residual.

The log-determinant uses the Sylvester/Matrix Determinant Lemma:

    log|diag(D)+VV^T| = log|D| + log|M|

All operations are fully batched and differentiable w.r.t. μ, D, V.

Additionally, utility functions `energy_score` and `kl_gaussian` are provided
for evaluation (not used during training).
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Numerically stable Cholesky
# ---------------------------------------------------------------------------


def _safe_cholesky(K: torch.Tensor, max_attempts: int = 8) -> torch.Tensor:
    """Cholesky decomposition with adaptive jitter.

    Handles two failure modes:
      1. Non-finite entries (NaN/Inf) — jitter is useless here; fall back to
         identity so training does not crash, and print a warning.
      2. Slightly non-PSD due to floating-point asymmetry — symmetrize first,
         then retry with progressively larger jitter (1e-6 → 1e-1).

    Args:
        K            : (..., n, n) symmetric PSD matrix (before jitter).
        max_attempts : number of jitter doublings before giving up.

    Returns:
        L : (..., n, n) lower-triangular Cholesky factor of K + jitter * I.
    """
    n = K.shape[-1]
    eye = torch.eye(n, dtype=K.dtype, device=K.device)

    # Non-finite guard: NaN/Inf + jitter = NaN/Inf — no point trying.
    if not torch.isfinite(K).all():
        import warnings

        warnings.warn(
            f"_safe_cholesky: non-finite values in K "
            f"(shape={tuple(K.shape)}, "
            f"nan={K.isnan().sum().item()}, inf={K.isinf().sum().item()}). "
            "Falling back to identity Cholesky — check D/V for NaN/Inf.",
            RuntimeWarning,
            stacklevel=2,
        )
        return eye.expand_as(K).clone()

    # Symmetrize to eliminate floating-point asymmetry from batched matmuls.
    K = 0.5 * (K + K.transpose(-2, -1))

    jitter = 1e-6
    for _ in range(max_attempts):
        try:
            return torch.linalg.cholesky(K + jitter * eye)
        except torch.linalg.LinAlgError:
            jitter *= 10
    raise RuntimeError(
        f"Cholesky failed after {max_attempts} attempts "
        f"(final jitter={jitter:.0e}, shape={K.shape[-2:]}, "
        f"min_eig≈{torch.linalg.eigvalsh(K).min().item():.3e})."
    )


# ---------------------------------------------------------------------------
# Woodbury NLL
# ---------------------------------------------------------------------------


def woodbury_nll(
    y: torch.Tensor,
    mu: torch.Tensor,
    D: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood for LowRankMultivariateNormal via Woodbury identity.

    Computes the mean NLL over all (batch, instance) pairs.

    Args:
        y  : (B, N, d)     — observed targets
        mu : (B, N, d)     — predicted mean
        D  : (B, N, d)     — diagonal variances  (must be strictly positive)
        V  : (B, N, d, r)  — low-rank factor

    Returns:
        Scalar NLL averaged over B*N instances.
    """
    r_vec = y - mu  # (B, N, d)
    r = V.shape[-1]  # rank

    # Early NaN/Inf detection — if D or V are non-finite, M will inherit the
    # corruption and _safe_cholesky's fallback will hide the root cause.
    if not (torch.isfinite(D).all() and torch.isfinite(V).all()):
        import warnings

        warnings.warn(
            f"woodbury_nll: non-finite values in D or V "
            f"(D nan={D.isnan().sum().item()} inf={D.isinf().sum().item()}, "
            f"V nan={V.isnan().sum().item()} inf={V.isinf().sum().item()}). "
            "This typically indicates a gradient explosion — check your LR / clip_grad_norm.",
            RuntimeWarning,
            stacklevel=2,
        )

    # D^{-1} r  — reused in both the quadratic term and the V^T D^{-1} r product
    D_inv_r = r_vec / D  # (B, N, d)

    # D^{-1} V  — reused in the capacitance matrix
    D_inv_V = V / D.unsqueeze(-1)  # (B, N, d, r)

    # Capacitance matrix M = I_r + V^T D^{-1} V            # (B, N, r, r)
    # Symmetrize explicitly: batched matmuls can introduce tiny asymmetry that
    # breaks Cholesky even when M is mathematically PSD.
    M_raw = torch.matmul(V.transpose(-2, -1), D_inv_V)  # (B, N, r, r)
    M = torch.eye(r, dtype=V.dtype, device=V.device) + 0.5 * (
        M_raw + M_raw.transpose(-2, -1)
    )
    # Cholesky of M for stable solve and log-det
    L_M = _safe_cholesky(M)  # (B, N, r, r)

    # V^T D^{-1} r = V^T (D_inv_r)                         # (B, N, r)
    VT_Dinv_r = torch.matmul(V.transpose(-2, -1), D_inv_r.unsqueeze(-1)).squeeze(-1)

    # M^{-1} (V^T D^{-1} r) via two triangular solves
    # torch.cholesky_solve is missing sm_75 kernels in PyTorch 2.11+cu130
    rhs = VT_Dinv_r.unsqueeze(-1)  # (B, N, r, 1)
    tmp = torch.linalg.solve_triangular(L_M, rhs, upper=False)
    Minv_VT_Dinv_r = torch.linalg.solve_triangular(
        L_M.transpose(-2, -1), tmp, upper=True
    ).squeeze(-1)  # (B, N, r)

    # Quadratic form  r^T D^{-1} r - (V^T D^{-1} r)^T M^{-1} (V^T D^{-1} r)
    quad = (
        (r_vec * D_inv_r).sum(-1)  # (B, N)
        - (VT_Dinv_r * Minv_VT_Dinv_r).sum(-1)  # (B, N)
    )

    # Log-determinant  log|D| + log|M|  (Sylvester/Matrix Determinant Lemma)
    log_det_D = D.log().sum(-1)  # (B, N)
    log_det_M = (
        2.0 * L_M.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (B, N)
    )
    log_det = log_det_D + log_det_M

    # NLL = 0.5 * (d log 2π + log|Σ| + quadratic)
    d_size = y.shape[-1]
    nll = 0.5 * (d_size * math.log(2.0 * math.pi) + log_det + quad)

    return nll.mean()


# ---------------------------------------------------------------------------
# Independent standard-normal NLL — Jacobian correction for copula reporting
# ---------------------------------------------------------------------------


def indep_normal_nll(z: torch.Tensor) -> torch.Tensor:
    """Mean NLL of i.i.d. N(0,1) at z.  Subtract from woodbury_nll to get the
    true Gaussian copula NLL as derived from Sklar's theorem:

        copula_nll = woodbury_nll(z; mu=0, R) - indep_normal_nll(z)

    Derivation: the Gaussian copula density is

        c(u) = phi_R(z) / prod_j phi(z_j)

    so  -log c(u) = -log phi_R(z) + sum_j log phi(z_j)
                  = woodbury_nll(z; 0, R) - d/2 log(2pi) - 1/2 ||z||^2.

    The subtracted term equals indep_normal_nll(z) = d/2 log(2pi) + 1/2 ||z||^2.
    It is constant w.r.t. model parameters (z is fixed after the PIT), so
    woodbury_nll and copula_nll yield identical gradients — this function is
    for reporting only.

    Args:
        z : (B, N, d) or (B, d) — Z-space observations (probit-PIT outputs).

    Returns:
        Scalar averaged over all instances.
    """
    d_size = z.shape[-1]
    return 0.5 * (d_size * math.log(2.0 * math.pi) + (z**2).sum(-1)).mean()


# ---------------------------------------------------------------------------
# Marginal NLL  (diagonal-only baseline, V ignored)
# ---------------------------------------------------------------------------


def marginal_nll(
    y: torch.Tensor,
    mu: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """NLL assuming independent marginals — ignores V entirely.

    Equivalent to woodbury_nll with V = 0, i.e. the covariance is purely
    diagonal diag(D).  Used as a baseline to quantify how much the low-rank
    component improves predictive performance.

    Args:
        y  : (B, N, d) — observed targets
        mu : (B, N, d) — predicted mean
        D  : (B, N, d) — diagonal variances (must be strictly positive)

    Returns:
        Scalar NLL averaged over B*N instances.
    """
    d_size = y.shape[-1]
    log_det_D = D.log().sum(dim=-1)  # (B, N)
    quad_form = ((y - mu) ** 2 / D).sum(dim=-1)  # (B, N)
    nll = 0.5 * (d_size * math.log(2.0 * math.pi) + log_det_D + quad_form)
    return nll.mean()


# ---------------------------------------------------------------------------
# Covariance comparison plot  (evaluation utility)
# ---------------------------------------------------------------------------


def plot_prediction_comparison(
    mu_pred: torch.Tensor,
    D_pred: torch.Tensor,
    V_pred: torch.Tensor,
    mu_true: torch.Tensor,
    D_true: torch.Tensor,
    V_true: torch.Tensor,
    batch_idx: int = 0,
    n_instances: int = 3,
    mu_tabicl: torch.Tensor | None = None,
):
    """Compare predicted vs oracle mean and covariance for multiple instances.

    One row per instance, five columns:
      0 — Oracle  Sigma* heatmap
      1 — Predicted Sigma hat heatmap
      2 — |Sigma* - Sigma hat| heatmap
      3 — mu* vs mu hat (+ TabICL base) bar chart
      4 — |mu* - mu hat| bar chart

    Args:
        mu_pred, mu_true : (B, N, d)    — conditional means
        D_pred, D_true   : (B, N, d)    — diagonal variances
        V_pred, V_true   : (B, N, d, r) — low-rank factors
        batch_idx        : which batch element to visualise
        n_instances      : number of instances (rows) to plot
        mu_tabicl        : (B, N, d) optional — base TabICL scalar predictions

    Returns:
        matplotlib Figure with n_instances × 5 subplots.
    """
    import seaborn as sns

    print(mu_pred.shape, D_pred.shape, V_pred.shape, "pred")
    print(mu_true.shape, D_true.shape, V_true.shape, "true")
    print(mu_tabicl.shape if mu_tabicl is not None else None, "tabicl")

    N = D_pred.shape[1]
    n_instances = min(n_instances, N)
    indices = np.linspace(0, N - 1, n_instances, dtype=int)

    fig, axes = plt.subplots(n_instances, 5, figsize=(26, 5 * n_instances))
    if n_instances == 1:
        axes = axes[np.newaxis, :]

    for row, inst_idx in enumerate(indices):
        # ---- Covariance ----
        Sp_V = V_pred[batch_idx, inst_idx]
        Sp_D = torch.diag(D_pred[batch_idx, inst_idx])
        Sigma_pred = (Sp_D + Sp_V @ Sp_V.T).detach().cpu().numpy()

        St_V = V_true[batch_idx, inst_idx]
        St_D = torch.diag(D_true[batch_idx, inst_idx])
        Sigma_true = (St_D + St_V @ St_V.T).detach().cpu().numpy()

        cov_max = max(np.abs(Sigma_true).max(), np.abs(Sigma_pred).max())

        sns.heatmap(
            Sigma_true,
            ax=axes[row, 0],
            cmap="coolwarm",
            center=0,
            vmin=-cov_max,
            vmax=cov_max,
            square=True,
        )
        axes[row, 0].set_title(rf"Oracle $\Sigma^*$ (inst {inst_idx})")

        sns.heatmap(
            Sigma_pred,
            ax=axes[row, 1],
            cmap="coolwarm",
            center=0,
            vmin=-cov_max,
            vmax=cov_max,
            square=True,
        )
        axes[row, 1].set_title(rf"Predicted $\hat{{\Sigma}}$ (inst {inst_idx})")

        sns.heatmap(
            np.abs(Sigma_true - Sigma_pred), ax=axes[row, 2], cmap="Reds", square=True
        )
        axes[row, 2].set_title(rf"$|\Sigma^* - \hat{{\Sigma}}|$ (inst {inst_idx})")

        # ---- Mean ----
        mu_t = mu_true[batch_idx, inst_idx].detach().cpu().numpy()  # (d,)
        mu_p = mu_pred[batch_idx, inst_idx].detach().cpu().numpy()  # (d,)
        d = len(mu_t)
        dims = np.arange(d)

        ax = axes[row, 3]
        if mu_tabicl is not None:
            mu_b = mu_tabicl[batch_idx, inst_idx].detach().cpu().numpy()  # (d,)
            width = 0.25
            ax.bar(
                dims - width,
                mu_t,
                width,
                label=r"Oracle $\mu^*$",
                color="#2563EB",
                alpha=0.8,
            )
            ax.bar(
                dims,
                mu_p,
                width,
                label=r"Predicted $\hat{\mu}$",
                color="#EA580C",
                alpha=0.8,
            )
            ax.bar(
                dims + width,
                mu_b,
                width,
                label=r"TabICL $\mu_{\rm base}$",
                color="#16A34A",
                alpha=0.8,
            )
        else:
            width = 0.35
            ax.bar(
                dims - width / 2,
                mu_t,
                width,
                label=r"Oracle $\mu^*$",
                color="#2563EB",
                alpha=0.8,
            )
            ax.bar(
                dims + width / 2,
                mu_p,
                width,
                label=r"Predicted $\hat{\mu}$",
                color="#EA580C",
                alpha=0.8,
            )
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xticks(dims)
        ax.set_xlabel("dim")
        ax.set_title(rf"Mean (inst {inst_idx})")
        ax.legend(fontsize=8)

        ax = axes[row, 4]
        if mu_tabicl is not None:
            width = 0.35
            ax.bar(
                dims - width / 2,
                np.abs(mu_t - mu_p),
                width,
                label=r"$|\mu^*-\hat{\mu}|$",
                color="#7C3AED",
                alpha=0.8,
            )
            ax.bar(
                dims + width / 2,
                np.abs(mu_t - mu_b),
                width,
                label=r"$|\mu^*-\mu_{\rm base}|$",
                color="#16A34A",
                alpha=0.8,
            )
            ax.legend(fontsize=8)
        else:
            ax.bar(dims, np.abs(mu_t - mu_p), color="#7C3AED", alpha=0.8)
        ax.set_xticks(dims)
        ax.set_xlabel("dim")
        ax.set_title(rf"$|\mu^* - \hat{{\mu}}|$ (inst {inst_idx})")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Energy Score  (evaluation metric, not used for training)
# ---------------------------------------------------------------------------


def energy_score(
    mu: torch.Tensor,
    D: torch.Tensor,
    V: torch.Tensor,
    y_ref: torch.Tensor,
    n_samples: int = 100,
) -> torch.Tensor:
    """Compute the Energy Score for a predicted LowRankMultivariateNormal.

    ES(P, y) = E_P[||Y - y||] - 0.5 * E_P[||Y - Y'||]

    where Y, Y' ~ P (the predicted distribution) and y is a reference
    sample from the true distribution.  Lower is better.

    Args:
        mu       : (d,)   — predicted mean
        D        : (d,)   — diagonal variances
        V        : (d, r) — low-rank factor
        y_ref    : (d,)   — single reference / ground-truth sample
        n_samples: M      — number of MC samples from predicted distribution

    Returns:
        Scalar Energy Score.
    """
    d = mu.shape[-1]
    r = V.shape[-1]

    # Sample via reparameterisation: Y = mu + D^{1/2} eps_diag + V eps_low
    eps_diag = torch.randn(n_samples, d, device=mu.device)  # (M, d)
    eps_low = torch.randn(n_samples, r, device=mu.device)  # (M, r)

    # samples: (M, d)
    samples = mu + D.sqrt() * eps_diag + (eps_low @ V.T)

    # E[||Y - y_ref||]  — distance from each sample to the reference
    diff_ref = (samples - y_ref.unsqueeze(0)).norm(dim=-1)  # (M,)
    term1 = diff_ref.mean()

    # E[||Y - Y'||]  — mean pairwise distance over all M^2 pairs
    term2 = torch.cdist(samples, samples).mean()  # scalar

    return term1 - 0.5 * term2


# ---------------------------------------------------------------------------
# KL divergence between two multivariate Gaussians (Gaussian approx of ref)
# ---------------------------------------------------------------------------


def kl_gaussian(
    mu_q: torch.Tensor,
    D_q: torch.Tensor,
    V_q: torch.Tensor,
    mu_p: torch.Tensor,
    Sigma_p: torch.Tensor,
) -> torch.Tensor:
    """KL( Q || P ) where Q is LowRankMVN and P is a dense MVN.

    Used during evaluation to compare the predicted distribution Q against
    a reference posterior P (approximated from MCMC / reference samples).

    KL(Q||P) = 0.5 * [ log|Sigma_p| - log|Sigma_q| - d
                        + tr(Sigma_p^{-1} Sigma_q)
                        + (mu_p - mu_q)^T Sigma_p^{-1} (mu_p - mu_q) ]

    The trace term is computed as  ||L_p^{-1} L_q||_F^2  where L_p, L_q are
    the Cholesky factors of Sigma_p and Sigma_q respectively.

    Args:
        mu_q    : (d,)    — Q mean
        D_q     : (d,)    — Q diagonal variances
        V_q     : (d, r)  — Q low-rank factor
        mu_p    : (d,)    — P mean
        Sigma_p : (d, d)  — P covariance (dense, estimated from reference samples)

    Returns:
        Scalar KL divergence (nats).
    """
    d = mu_q.shape[0]

    # Build Q covariance (dense) for Cholesky
    Sigma_q = torch.diag(D_q) + V_q @ V_q.T  # (d, d)

    # Cholesky factors
    L_p = _safe_cholesky(Sigma_p)  # (d, d)
    L_q = _safe_cholesky(Sigma_q)  # (d, d)

    # log|Sigma_p| = 2 * sum log diag(L_p)
    log_det_p = 2.0 * L_p.diagonal().log().sum()

    # log|Sigma_q| = 2 * sum log diag(L_q)
    log_det_q = 2.0 * L_q.diagonal().log().sum()

    # tr(Sigma_p^{-1} Sigma_q) = ||L_p^{-1} L_q||_F^2
    # Solve L_p A = L_q  →  A = L_p^{-1} L_q
    A = torch.linalg.solve_triangular(L_p, L_q, upper=False)  # (d, d)
    trace_term = (A * A).sum()

    # (mu_p - mu_q)^T Sigma_p^{-1} (mu_p - mu_q)
    diff = (mu_p - mu_q).unsqueeze(-1)  # (d, 1)
    v = torch.linalg.solve_triangular(L_p, diff, upper=False)  # (d, 1)
    quad_term = (v * v).sum()

    kl = 0.5 * (log_det_p - log_det_q - d + trace_term + quad_term)
    return kl
