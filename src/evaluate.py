"""
evaluate.py — UCI Evaluation Suite for MultivariateTabICL.

Runs Phase 1 PIT on three UCI datasets, evaluates the pre-trained
CopulaTransformer against five baselines, and prints a comparison table.

Datasets:
    ENB          — Energy Efficiency (UCI ID 242), d=2
    StudentMat   — Student Performance (UCI ID 320), d=2 (G1, G2 only)
    CommsCrime   — Communities & Crime Unnormalized (UCI ID 211), d=5

Usage:
    python src/evaluate.py --ckpt checkpoints/copula/copula_step60.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — must run before local imports
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_TABICL_SRC = os.path.join(_HERE, "..", "tabicl_upstream", "src")
sys.path.insert(0, _HERE)
sys.path.insert(0, _TABICL_SRC)

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from loss import energy_score, marginal_nll, woodbury_nll  # noqa: E402
from model import build_copula_transformer  # noqa: E402
from pit import load_tabicl, run_pit  # noqa: E402

# ---------------------------------------------------------------------------
# Optional dependency — XGBoost for baseline 3
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBRegressor  # noqa: F401

    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

# ---------------------------------------------------------------------------
# Reproducibility helper
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
# Dataset loaders
# ===========================================================================


def _to_tensor(arr, device: str, dtype=torch.float32) -> torch.Tensor:
    """Convert numpy array or DataFrame to float tensor on device."""
    if hasattr(arr, "values"):
        arr = arr.values
    return torch.tensor(arr, dtype=dtype, device=device)


def _normalize_X(
    X_train: np.ndarray,
    X_test: np.ndarray,
    cat_cols: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-normalize continuous columns using training mean/std.

    One-hot / binary columns listed in *cat_cols* are left unchanged.
    """
    n_cols = X_train.shape[1]
    cont_cols = [j for j in range(n_cols) if cat_cols is None or j not in cat_cols]

    X_train_out = X_train.copy().astype(np.float32)
    X_test_out = X_test.copy().astype(np.float32)

    for j in cont_cols:
        mu = X_train[:, j].mean()
        sigma = X_train[:, j].std()
        if sigma < 1e-9:
            sigma = 1.0
        X_train_out[:, j] = (X_train[:, j] - mu) / sigma
        X_test_out[:, j] = (X_test[:, j] - mu) / sigma

    return X_train_out, X_test_out


def load_enb(device: str):
    """Energy Efficiency — UCI ID 242.

    d=2 targets: Y1 (Heating Load), Y2 (Cooling Load).
    """
    from sklearn.model_selection import train_test_split
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=242)
    X = ds.data.features.values.astype(np.float32)
    y = ds.data.targets.values.astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_te = _normalize_X(X_tr, X_te)

    return (
        _to_tensor(X_tr, device),
        _to_tensor(y_tr, device),
        _to_tensor(X_te, device),
        _to_tensor(y_te, device),
        False,  # dequantize
    )


def load_student(device: str):
    """Student Performance — UCI ID 320.

    d=2 targets: G1, G2 (period 1 & 2 grades). G3 is excluded.
    Categorical X features are one-hot encoded before normalization.
    One-hot columns (already {0,1}) are not z-normalized.
    dequantize=True (integer grades 0-20).
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=320)
    X_df = ds.data.features.copy()
    y_df = ds.data.targets[["G1", "G2"]].copy()

    # Drop rows with any NaN in X or y
    valid = ~(X_df.isnull().any(axis=1) | y_df.isnull().any(axis=1))
    X_df = X_df[valid].reset_index(drop=True)
    y_df = y_df[valid].reset_index(drop=True)

    # One-hot encode categorical columns
    cat_col_names = X_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    X_df = pd.get_dummies(X_df, columns=cat_col_names, drop_first=False)
    X_df = X_df.astype(np.float32)

    X = X_df.values
    y = y_df.values.astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify one-hot (binary) columns — values always {0, 1}
    ohe_mask = []
    for j in range(X_tr.shape[1]):
        unique_vals = np.unique(X_tr[:, j])
        is_binary = set(unique_vals).issubset({0.0, 1.0})
        ohe_mask.append(j if is_binary else None)
    cat_idx = [j for j in ohe_mask if j is not None]

    X_tr, X_te = _normalize_X(X_tr, X_te, cat_cols=cat_idx)

    return (
        _to_tensor(X_tr, device),
        _to_tensor(y_tr, device),
        _to_tensor(X_te, device),
        _to_tensor(y_te, device),
        True,  # dequantize
    )


def load_comms_crime(device: str):
    """Communities & Crime Unnormalized — UCI ID 211.

    d=5 targets: murders, rapes, robberies, assaults, burglaries.
    Missing X values → column-wise median imputation, then z-normalize.
    """
    from sklearn.model_selection import train_test_split
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=211)
    X_df = ds.data.features.fillna(ds.data.features.median())

    target_cols = ["murders", "rapes", "robberies", "assaults", "burglaries"]
    y_df = ds.data.targets[target_cols].dropna()
    X_df = X_df.loc[y_df.index]

    X = X_df.values.astype(np.float32)
    y = y_df.values.astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_te = _normalize_X(X_tr, X_te)

    return (
        _to_tensor(X_tr, device),
        _to_tensor(y_tr, device),
        _to_tensor(X_te, device),
        _to_tensor(y_te, device),
        False,  # dequantize
    )


# ===========================================================================
# Metric helpers
# ===========================================================================


def compute_crps_sum(
    tabicl: torch.nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
) -> float:
    """CRPS-sum: per-dimension CRPS summed over all target dimensions.

    For each target dimension j, runs TabICL forward to get a QuantileDistribution
    over all N_test instances, then calls .crps(Y_test[:,j]).
    """
    N_tr = X_train.shape[0]
    X_concat = torch.cat([X_train, X_test], dim=0).unsqueeze(0)  # (1, N_tr+N_te, p)
    crps_total = 0.0
    with torch.no_grad():
        for j in range(Y_train.shape[1]):
            y_context = Y_train[:, j].unsqueeze(0)  # (1, N_tr)
            logits = tabicl(X_concat, y_context)  # (1, N_te, Q)
            dist = tabicl.quantile_dist(logits.squeeze(0))  # batch_shape=(N_te,)
            crps_j = dist.crps(Y_test[:, j]).mean().item()
            crps_total += crps_j
    return crps_total


def compute_rmse(
    tabicl: torch.nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    mu_Z: torch.Tensor,
    d_Z: torch.Tensor,
) -> list[float]:
    """Per-target RMSE via back-transform: mu_Z → Φ(mu_Z) → icdf → ŷ.

    icdf(alpha) with alpha shape (n_alpha,) returns (N_test, n_alpha), so we
    call it once per test instance using a scalar alpha.
    """
    X_concat = torch.cat([X_train, X_test], dim=0).unsqueeze(0)  # (1, N_tr+N_te, p)
    N_test = X_test.shape[0]
    rmse_list = []

    with torch.no_grad():
        for j in range(Y_train.shape[1]):
            y_context = Y_train[:, j].unsqueeze(0)  # (1, N_tr)
            logits = tabicl(X_concat, y_context)  # (1, N_te, Q)
            dist = tabicl.quantile_dist(logits.squeeze(0))  # batch_shape=(N_te,)

            # mu_Z[:, j]: (N_test,) → CDF of standard normal
            mu_j = mu_Z[:, j]  # (N_test,)
            u_j = 0.5 * (1.0 + torch.erf(mu_j / math.sqrt(2.0)))
            u_j = u_j.clamp(1e-6, 1.0 - 1e-6)

            # dist.icdf takes alpha (n,) and returns (N_test, n)
            # Pass all per-instance alphas at once: shape (N_test,)
            # Then take the diagonal element for each instance.
            # API: alpha (n,) → (N_test, n). Pass a single shared grid is wrong
            # for per-instance alpha. Use per-instance calls instead.
            y_hat_j = torch.empty(N_test, device=mu_Z.device, dtype=mu_Z.dtype)
            for i in range(N_test):
                alpha_i = u_j[i].unsqueeze(0)  # (1,)
                # dist.icdf with (1,) returns (N_test, 1) — slice [i, 0]
                y_hat_j[i] = dist.icdf(alpha_i)[i, 0]

            rmse_j = ((y_hat_j - Y_test[:, j]) ** 2).mean().sqrt().item()
            rmse_list.append(rmse_j)

    return rmse_list


def compute_frobenius_error(
    Z_test: torch.Tensor,
    mu_Z: torch.Tensor,
    d_Z: torch.Tensor,
    V_Z: torch.Tensor,
) -> float:
    """Frobenius distance between mean predicted correlation and empirical correlation."""
    R_empirical = torch.corrcoef(Z_test.T)  # (d, d)

    # Sigma_pred[i] = diag(d_Z[i]) + V_Z[i] @ V_Z[i]^T  =>  (N_test, d, d)
    Sigma_pred = torch.diag_embed(d_Z) + V_Z @ V_Z.transpose(-1, -2)

    stds = Sigma_pred.diagonal(dim1=-2, dim2=-1).clamp(min=1e-9).sqrt()  # (N_test, d)
    R_pred = Sigma_pred / (stds.unsqueeze(-1) * stds.unsqueeze(-2))  # (N_test, d, d)
    R_pred_mean = R_pred.mean(0)  # (d, d)

    return (R_pred_mean - R_empirical).norm(p="fro").item()


# ===========================================================================
# Main metrics aggregator
# ===========================================================================


def compute_metrics(
    Z_test: torch.Tensor,
    mu_Z: torch.Tensor,
    d_Z: torch.Tensor,
    V_Z: torch.Tensor,
    log_p_test: torch.Tensor,
    Y_test: torch.Tensor,
    tabicl: torch.nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    label: str = "CopulaTransformer",
) -> dict:
    """Compute all evaluation metrics for a given set of predicted parameters.

    Args:
        Z_test      : (N_test, d)    — PIT-transformed test targets
        mu_Z        : (N_test, d)    — predicted means in Z-space
        d_Z         : (N_test, d)    — predicted diagonal variances
        V_Z         : (N_test, d, r) — predicted low-rank factors
        log_p_test  : (N_test, d)    — per-dimension log-marginal from PIT
        Y_test      : (N_test, d)    — original test targets
        tabicl      : the TabICL model (for CRPS / RMSE helpers)
        X_train/Y_train/X_test: training and test tensors

    Returns:
        dict with metric names → scalar values.
    """
    # --- Joint NLL in Z-space (mean over instances) ---
    joint_nll_z = woodbury_nll(
        Z_test.unsqueeze(0),
        mu_Z.unsqueeze(0),
        d_Z.unsqueeze(0),
        V_Z.unsqueeze(0),
    )

    # --- Joint NLL in Y-space via Jacobian correction ---
    # log P(Y) = log P_copula(Z) + sum_j log p_j(y_j)
    # woodbury_nll returns the *mean* NLL over instances, so:
    #   joint_nll_y = joint_nll_z - mean_j_log_p
    jacobian = log_p_test.sum(
        dim=-1
    ).mean()  # mean over test instances of sum over dims
    joint_nll_y = joint_nll_z - jacobian

    # --- Energy Score (per instance, MC with 200 samples) ---
    es_list = []
    with torch.no_grad():
        for i in range(len(Z_test)):
            es_i = energy_score(mu_Z[i], d_Z[i], V_Z[i], Z_test[i], n_samples=200)
            es_list.append(es_i.item())
    energy_score_mean = float(np.mean(es_list))

    # --- CRPS-sum ---
    crps_total = compute_crps_sum(tabicl, X_train, Y_train, X_test, Y_test)

    # --- RMSE per target ---
    rmse_per_target = compute_rmse(tabicl, X_train, Y_train, X_test, Y_test, mu_Z, d_Z)

    # --- Frobenius correlation error ---
    frob_err = compute_frobenius_error(Z_test, mu_Z, d_Z, V_Z)

    return {
        "label": label,
        "joint_nll_z": joint_nll_z.item(),
        "joint_nll_y": joint_nll_y.item(),
        "energy_score": energy_score_mean,
        "crps_sum": crps_total,
        "rmse_per_target": rmse_per_target,
        "frobenius_error": frob_err,
    }


# ===========================================================================
# Baselines
# ===========================================================================


def _safe_cholesky_dense(K: torch.Tensor) -> torch.Tensor:
    """Cholesky with progressive jitter for dense matrices (d, d)."""
    n = K.shape[-1]
    eye = torch.eye(n, dtype=K.dtype, device=K.device)
    K = 0.5 * (K + K.T)
    jitter = 1e-6
    for _ in range(8):
        try:
            return torch.linalg.cholesky(K + jitter * eye)
        except torch.linalg.LinAlgError:
            jitter *= 10
    raise RuntimeError("Cholesky failed after 8 attempts.")


def _mvn_nll_cholesky(
    y: torch.Tensor,  # (N, d)
    mu: torch.Tensor,  # (d,) or (N, d)
    L: torch.Tensor,  # (d, d) lower triangular
) -> float:
    """Mean NLL for MVN with Cholesky factor L (shared across all instances)."""
    d = y.shape[-1]
    r = y - mu  # (N, d)
    # solve L z = r^T  → z shape (d, N)
    z = torch.linalg.solve_triangular(L, r.T, upper=False)  # (d, N)
    quad = (z * z).sum(0)  # (N,)
    log_det = 2.0 * L.diagonal().log().sum()
    nll_per_instance = 0.5 * (d * math.log(2.0 * math.pi) + log_det + quad)
    return nll_per_instance.mean().item()


def baseline_independent_tabicl(
    Z_test: torch.Tensor,
    Z_train: torch.Tensor,
    d: int,
    r: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Baseline 1: Independent TabICL (diagonal Σ, V=0).

    mu = 0 (Z is approximately standard normal), D = per-dim variance of Z_train.
    """
    N_test = Z_test.shape[0]
    mu_Z = torch.zeros(N_test, d, device=device)
    d_Z = Z_train.var(0, unbiased=True).clamp(min=1e-6).unsqueeze(0).expand(N_test, -1)
    V_Z = torch.zeros(N_test, d, r, device=device)
    return mu_Z, d_Z, V_Z


def baseline_static_gaussian_copula(
    Z_test: torch.Tensor,
    Z_train: torch.Tensor,
    d: int,
    r: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Baseline 2: Static Gaussian Copula (unconditional training correlation).

    All test instances share the same correlation structure R from Z_train.
    Returns NLL computed via dense Cholesky.
    """
    N_test = Z_test.shape[0]
    R = torch.corrcoef(Z_train.T)  # (d, d)
    R = 0.5 * (R + R.T)
    # Add small jitter to ensure PD
    R = R + 1e-5 * torch.eye(d, device=device)

    mu = torch.zeros(d, device=device)
    L = _safe_cholesky_dense(R)
    nll = _mvn_nll_cholesky(Z_test, mu, L)

    # Decompose R for the Woodbury/metric helpers: R ≈ diag(1) + 0·V·V^T
    # We represent the full R via its low-rank approximation using eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(R)
    eigvals = eigvals.clamp(min=1e-6)

    # d_Z = diagonal of R (all ones for a correlation matrix)
    d_Z = torch.ones(N_test, d, device=device)
    # Represent off-diagonal as low-rank factor: R - I = V V^T
    # Pick top-r eigenvectors weighted by (lambda_i - 1).clamp(0)
    excess = (eigvals - 1.0).clamp(min=0.0)
    V_flat = eigvecs * excess.sqrt().unsqueeze(0)  # (d, d)
    V_flat = V_flat[:, :r]  # (d, r)
    V_Z = V_flat.unsqueeze(0).expand(N_test, -1, -1)  # (N_test, d, r)

    mu_Z = torch.zeros(N_test, d, device=device)
    return mu_Z, d_Z, V_Z, nll


def baseline_gbdt_residual(
    Z_test: torch.Tensor,
    Z_train: torch.Tensor,
    Y_train: torch.Tensor,
    Y_test: torch.Tensor,
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    d: int,
    r: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Baseline 3: GBDT + Residual Covariance.

    Fits one XGBoost regressor per target, computes training residuals,
    applies marginal normalization (Z-score), fits shared covariance,
    returns predicted parameters.

    Returns None if xgboost is unavailable.
    """
    if not _XGBOOST_AVAILABLE:
        warnings.warn(
            "xgboost not installed — skipping GBDT baseline. "
            "Install with: pip install xgboost",
            RuntimeWarning,
        )
        return None

    from xgboost import XGBRegressor

    X_tr_np = X_train.cpu().numpy()
    X_te_np = X_test.cpu().numpy()
    Y_tr_np = Y_train.cpu().numpy()
    Y_te_np = Y_test.cpu().numpy()
    N_test = X_te_np.shape[0]

    models = []
    resid_train_list = []
    resid_test_list = []

    for j in range(d):
        reg = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        reg.fit(X_tr_np, Y_tr_np[:, j])
        models.append(reg)

        resid_train_list.append(Y_tr_np[:, j] - reg.predict(X_tr_np))
        resid_test_list.append(Y_te_np[:, j] - reg.predict(X_te_np))

    R_train = np.stack(resid_train_list, axis=1)  # (N_train, d)
    R_test = np.stack(resid_test_list, axis=1)  # (N_test, d)

    # Z-score residuals using training stats
    mu_r = R_train.mean(0)
    std_r = R_train.std(0).clip(1e-9)
    Z_resid_train = (R_train - mu_r) / std_r
    Z_resid_test = (R_test - mu_r) / std_r

    # Fit covariance on Z-space residuals
    Z_r_t = torch.tensor(Z_resid_train, dtype=torch.float32, device=device)
    Z_r_te = torch.tensor(Z_resid_test, dtype=torch.float32, device=device)

    N_tr = Z_r_t.shape[0]
    Cov = (Z_r_t.T @ Z_r_t) / max(N_tr - 1, 1)  # (d, d)
    Cov = 0.5 * (Cov + Cov.T)

    # Represent as diag + low-rank
    diag_vals = Cov.diagonal().clamp(min=1e-6)
    off_diag = Cov - torch.diag(diag_vals)
    eigvals, eigvecs = torch.linalg.eigh(off_diag)
    eigvals = eigvals.clamp(min=0.0)
    top_r = eigvals.topk(r).indices
    V_flat = eigvecs[:, top_r] * eigvals[top_r].sqrt().unsqueeze(0)  # (d, r)

    mu_Z = Z_r_te  # predicted mean in Z-space = GBDT residual in Z-space
    d_Z = diag_vals.unsqueeze(0).expand(N_test, -1)
    V_Z = V_flat.unsqueeze(0).expand(N_test, -1, -1)

    return mu_Z, d_Z, V_Z


def baseline_full_covariance(
    Z_test: torch.Tensor,
    Z_train: torch.Tensor,
    d: int,
    r: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float] | None:
    """Baseline 4: Full Covariance (unconstrained d×d MLE).

    Only valid for d ≤ 5. Returns None otherwise.
    """
    if d > 5:
        warnings.warn(
            f"Full covariance baseline skipped: d={d} > 5 (too expensive).",
            RuntimeWarning,
        )
        return None

    N_test = Z_test.shape[0]
    N_tr = Z_train.shape[0]
    mu_mle = Z_train.mean(0)  # (d,)
    diff = Z_train - mu_mle  # (N_tr, d)
    Sigma_mle = (diff.T @ diff) / max(N_tr - 1, 1)  # (d, d)
    Sigma_mle = 0.5 * (Sigma_mle + Sigma_mle.T)

    L = _safe_cholesky_dense(Sigma_mle)
    nll = _mvn_nll_cholesky(Z_test, mu_mle, L)

    # Decompose for helpers
    diag_vals = Sigma_mle.diagonal().clamp(min=1e-6)
    off_diag = Sigma_mle - torch.diag(diag_vals)
    eigvals, eigvecs = torch.linalg.eigh(off_diag)
    eigvals = eigvals.clamp(min=0.0)
    top_r = eigvals.topk(r).indices
    V_flat = eigvecs[:, top_r] * eigvals[top_r].sqrt().unsqueeze(0)  # (d, r)

    mu_Z = mu_mle.unsqueeze(0).expand(N_test, -1)
    d_Z = diag_vals.unsqueeze(0).expand(N_test, -1)
    V_Z = V_flat.unsqueeze(0).expand(N_test, -1, -1)

    return mu_Z, d_Z, V_Z, nll


def baseline_mean_prediction(
    Z_test: torch.Tensor,
    Z_train: torch.Tensor,
    d: int,
    r: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Baseline 5: Mean prediction (constant predictor, identity covariance)."""
    N_test = Z_test.shape[0]
    mu_Z = Z_train.mean(0).unsqueeze(0).expand(N_test, -1)
    d_Z = torch.ones(N_test, d, device=device)
    V_Z = torch.zeros(N_test, d, r, device=device)
    return mu_Z, d_Z, V_Z


# ===========================================================================
# Dataset evaluation driver
# ===========================================================================


def evaluate_dataset(
    dataset_name: str,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    model: torch.nn.Module,
    tabicl: torch.nn.Module,
    device: str,
    dequantize: bool = False,
    pit_batch_size: int = 64,
    pit_eps: float = 1e-6,
) -> list[dict]:
    """Run Phase 1 PIT, model inference, metrics, and all 5 baselines.

    Returns a list of result dicts (one per method).
    """
    print(f"\n{'=' * 60}")
    print(
        f"Dataset: {dataset_name}  "
        f"(N_train={X_train.shape[0]}, N_test={X_test.shape[0]}, "
        f"d={Y_train.shape[1]}, p={X_train.shape[1]})"
    )
    print("=" * 60)

    d = Y_train.shape[1]

    # -----------------------------------------------------------------------
    # Phase 1: PIT to map Y → Z
    # -----------------------------------------------------------------------
    print("  Running Phase 1 PIT...")
    with torch.no_grad():
        Z_train, Z_test, log_p_test = run_pit(
            tabicl,
            X_train,
            Y_train,
            X_test,
            Y_test,
            pit_batch_size=pit_batch_size,
            eps=pit_eps,
            dequantize=dequantize,
        )

    print(f"  Z_train: {Z_train.shape}, Z_test: {Z_test.shape}")

    # -----------------------------------------------------------------------
    # Phase 2: CopulaTransformer forward pass
    # -----------------------------------------------------------------------
    print("  Running CopulaTransformer inference...")
    model.eval()
    with torch.no_grad():
        # model expects (B, N_context, p) and (B, N_context, d)
        X_ctx = X_train.unsqueeze(0)  # (1, N_tr, p)
        Z_ctx = Z_train.unsqueeze(0)  # (1, N_tr, d)
        X_q = X_test.unsqueeze(0)  # (1, N_te, p)

        out = model(X_ctx, Z_ctx, X_q)  # dict or (mu, D, V) depending on model API
        # model.py is expected to return (mu_Z, d_Z, V_Z) each (1, N_te, ...)
        if isinstance(out, dict):
            mu_Z = out["mu"]  # (1, N_te, d)
            d_Z = out["D"]  # (1, N_te, d)
            V_Z = out["V"]  # (1, N_te, d, r)
        elif isinstance(out, (list, tuple)) and len(out) == 3:
            mu_Z, d_Z, V_Z = out
        else:
            raise ValueError(
                f"Unexpected CopulaTransformer output type: {type(out)}. "
                "Expected dict with keys ('mu','D','V') or 3-tuple."
            )

        mu_Z = mu_Z.squeeze(0)  # (N_te, d)
        d_Z = d_Z.squeeze(0)  # (N_te, d)
        V_Z = V_Z.squeeze(0)  # (N_te, d, r)
        d_Z = d_Z.clamp(min=1e-6)

    r = V_Z.shape[-1]

    # -----------------------------------------------------------------------
    # CopulaTransformer metrics
    # -----------------------------------------------------------------------
    print("  Computing CopulaTransformer metrics...")
    ct_metrics = compute_metrics(
        Z_test,
        mu_Z,
        d_Z,
        V_Z,
        log_p_test,
        Y_test,
        tabicl,
        X_train,
        Y_train,
        X_test,
        label="CopulaTransformer",
    )
    all_results = [ct_metrics]

    # -----------------------------------------------------------------------
    # Baselines
    # -----------------------------------------------------------------------

    # --- Baseline 1: Independent TabICL ---
    print("  [Baseline 1] Independent TabICL...")
    b1_mu, b1_d, b1_V = baseline_independent_tabicl(Z_test, Z_train, d, r, device)
    b1_nll_z = marginal_nll(
        Z_test.unsqueeze(0), b1_mu.unsqueeze(0), b1_d.unsqueeze(0)
    ).item()
    b1_metrics = {
        "label": "Independent-TabICL",
        "joint_nll_z": b1_nll_z,
        "joint_nll_y": b1_nll_z - log_p_test.sum(dim=-1).mean().item(),
        "energy_score": _mc_energy_score_batch(b1_mu, b1_d, b1_V, Z_test),
        "crps_sum": compute_crps_sum(tabicl, X_train, Y_train, X_test, Y_test),
        "rmse_per_target": compute_rmse(
            tabicl, X_train, Y_train, X_test, Y_test, b1_mu, b1_d
        ),
        "frobenius_error": compute_frobenius_error(Z_test, b1_mu, b1_d, b1_V),
    }
    all_results.append(b1_metrics)

    # --- Baseline 2: Static Gaussian Copula ---
    print("  [Baseline 2] Static Gaussian Copula...")
    b2_mu, b2_d, b2_V, b2_nll_dense = baseline_static_gaussian_copula(
        Z_test, Z_train, d, r, device
    )
    b2_metrics = {
        "label": "Static-GaussianCopula",
        "joint_nll_z": b2_nll_dense,
        "joint_nll_y": b2_nll_dense - log_p_test.sum(dim=-1).mean().item(),
        "energy_score": _mc_energy_score_batch(b2_mu, b2_d, b2_V, Z_test),
        "crps_sum": compute_crps_sum(tabicl, X_train, Y_train, X_test, Y_test),
        "rmse_per_target": compute_rmse(
            tabicl, X_train, Y_train, X_test, Y_test, b2_mu, b2_d
        ),
        "frobenius_error": compute_frobenius_error(Z_test, b2_mu, b2_d, b2_V),
    }
    all_results.append(b2_metrics)

    # --- Baseline 3: GBDT + Residual Covariance ---
    print("  [Baseline 3] GBDT + Residual Covariance...")
    b3_out = baseline_gbdt_residual(
        Z_test, Z_train, Y_train, Y_test, X_train, X_test, d, r, device
    )
    if b3_out is not None:
        b3_mu, b3_d, b3_V = b3_out
        b3_nll_z = woodbury_nll(
            Z_test.unsqueeze(0),
            b3_mu.unsqueeze(0),
            b3_d.unsqueeze(0),
            b3_V.unsqueeze(0),
        ).item()
        b3_metrics = {
            "label": "GBDT-ResidCov",
            "joint_nll_z": b3_nll_z,
            "joint_nll_y": b3_nll_z - log_p_test.sum(dim=-1).mean().item(),
            "energy_score": _mc_energy_score_batch(b3_mu, b3_d, b3_V, Z_test),
            "crps_sum": compute_crps_sum(tabicl, X_train, Y_train, X_test, Y_test),
            "rmse_per_target": compute_rmse(
                tabicl, X_train, Y_train, X_test, Y_test, b3_mu, b3_d
            ),
            "frobenius_error": compute_frobenius_error(Z_test, b3_mu, b3_d, b3_V),
        }
        all_results.append(b3_metrics)
    else:
        print("    Skipped (xgboost unavailable).")

    # --- Baseline 4: Full Covariance (d ≤ 5 only) ---
    print("  [Baseline 4] Full Covariance MLE...")
    b4_out = baseline_full_covariance(Z_test, Z_train, d, r, device)
    if b4_out is not None:
        b4_mu, b4_d, b4_V, b4_nll_dense = b4_out
        b4_metrics = {
            "label": "FullCov-MLE",
            "joint_nll_z": b4_nll_dense,
            "joint_nll_y": b4_nll_dense - log_p_test.sum(dim=-1).mean().item(),
            "energy_score": _mc_energy_score_batch(b4_mu, b4_d, b4_V, Z_test),
            "crps_sum": compute_crps_sum(tabicl, X_train, Y_train, X_test, Y_test),
            "rmse_per_target": compute_rmse(
                tabicl, X_train, Y_train, X_test, Y_test, b4_mu, b4_d
            ),
            "frobenius_error": compute_frobenius_error(Z_test, b4_mu, b4_d, b4_V),
        }
        all_results.append(b4_metrics)
    else:
        print("    Skipped (d > 5).")

    # --- Baseline 5: Mean Prediction ---
    print("  [Baseline 5] Mean Prediction...")
    b5_mu, b5_d, b5_V = baseline_mean_prediction(Z_test, Z_train, d, r, device)
    b5_nll_z = woodbury_nll(
        Z_test.unsqueeze(0),
        b5_mu.unsqueeze(0),
        b5_d.unsqueeze(0),
        b5_V.unsqueeze(0),
    ).item()
    b5_metrics = {
        "label": "MeanPrediction",
        "joint_nll_z": b5_nll_z,
        "joint_nll_y": b5_nll_z - log_p_test.sum(dim=-1).mean().item(),
        "energy_score": _mc_energy_score_batch(b5_mu, b5_d, b5_V, Z_test),
        "crps_sum": compute_crps_sum(tabicl, X_train, Y_train, X_test, Y_test),
        "rmse_per_target": compute_rmse(
            tabicl, X_train, Y_train, X_test, Y_test, b5_mu, b5_d
        ),
        "frobenius_error": compute_frobenius_error(Z_test, b5_mu, b5_d, b5_V),
    }
    all_results.append(b5_metrics)

    return all_results


# ---------------------------------------------------------------------------
# MC energy score helper for baselines (batched over instances)
# ---------------------------------------------------------------------------


def _mc_energy_score_batch(
    mu_Z: torch.Tensor,  # (N, d)
    d_Z: torch.Tensor,  # (N, d)
    V_Z: torch.Tensor,  # (N, d, r)
    Z_test: torch.Tensor,  # (N, d)
    n_samples: int = 200,
) -> float:
    es_vals = []
    with torch.no_grad():
        for i in range(len(Z_test)):
            es_i = energy_score(mu_Z[i], d_Z[i], V_Z[i], Z_test[i], n_samples=n_samples)
            es_vals.append(es_i.item())
    return float(np.mean(es_vals))


# ===========================================================================
# Pretty printer
# ===========================================================================


def print_results_table(dataset_name: str, results: list[dict]) -> None:
    """Print a formatted comparison table for all methods on one dataset."""
    print(f"\n{'─' * 90}")
    print(f"  Results for: {dataset_name}")
    print(f"{'─' * 90}")

    header = (
        f"  {'Method':<28} {'NLL-Z':>9} {'NLL-Y':>9} "
        f"{'ES':>9} {'CRPS-sum':>10} {'Frob-err':>10}  RMSE/target"
    )
    print(header)
    print(f"  {'─' * 28} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 10} {'─' * 10}")

    for r in results:
        rmse_str = "  ".join(f"{v:.4f}" for v in r["rmse_per_target"])
        row = (
            f"  {r['label']:<28} "
            f"{r['joint_nll_z']:>9.4f} "
            f"{r['joint_nll_y']:>9.4f} "
            f"{r['energy_score']:>9.4f} "
            f"{r['crps_sum']:>10.4f} "
            f"{r['frobenius_error']:>10.4f}  "
            f"{rmse_str}"
        )
        print(row)

    print(f"{'─' * 90}\n")


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UCI Evaluation Suite for MultivariateTabICL"
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to CopulaTransformer checkpoint (.pt file).",
    )
    parser.add_argument(
        "--tabicl_ckpt",
        default="tabicl-regressor-v2-20260212.ckpt",
        help="TabICL checkpoint name (passed to load_tabicl).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: 'auto', 'cpu', or 'cuda'.",
    )
    parser.add_argument(
        "--pit_batch_size",
        type=int,
        default=64,
        help="LOO chunk size for Phase 1 PIT.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ENB", "StudentMat", "CommsCrime"],
        choices=["ENB", "StudentMat", "CommsCrime"],
        help="Which datasets to evaluate.",
    )
    args = parser.parse_args()

    set_seed(42)

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # Load TabICL
    # -----------------------------------------------------------------------
    print(f"Loading TabICL from: {args.tabicl_ckpt}")
    tabicl = load_tabicl(args.tabicl_ckpt, device)
    tabicl.eval()

    # -----------------------------------------------------------------------
    # Load CopulaTransformer
    # -----------------------------------------------------------------------
    print(f"Loading CopulaTransformer from: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)

    # Reconstruct model from saved config
    cfg_dict = ckpt.get("cfg", {})

    # build_copula_transformer accepts an OmegaConf DictConfig or plain dict
    try:
        from omegaconf import OmegaConf

        if not isinstance(cfg_dict, dict):
            cfg_obj = cfg_dict  # already a DictConfig
        else:
            cfg_obj = OmegaConf.create(cfg_dict)
    except ImportError:
        cfg_obj = cfg_dict  # plain dict fallback

    model = build_copula_transformer(cfg_obj)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -----------------------------------------------------------------------
    # Dataset loaders registry
    # -----------------------------------------------------------------------
    loader_map = {
        "ENB": load_enb,
        "StudentMat": load_student,
        "CommsCrime": load_comms_crime,
    }

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    all_dataset_results = {}
    for name in args.datasets:
        print(f"\nLoading dataset: {name}")
        loader_fn = loader_map[name]
        X_train, Y_train, X_test, Y_test, dequantize = loader_fn(device)

        results = evaluate_dataset(
            name,
            X_train,
            Y_train,
            X_test,
            Y_test,
            model,
            tabicl,
            device,
            dequantize=dequantize,
            pit_batch_size=args.pit_batch_size,
        )
        all_dataset_results[name] = results
        print_results_table(name, results)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
