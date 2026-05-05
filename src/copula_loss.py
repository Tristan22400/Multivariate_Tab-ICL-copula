"""
copula_loss.py — PIT utilities, discretized CE loss, and evaluation metrics
for the In-Context Attentional Copula Model.

Two distinct PIT functions:
  empirical_pit(Y_train)            — self-ranks for the encoder context input
  smooth_context_pit(Y_test, Y_train) — smoothed context-conditional CDF for
                                        teacher-forcing targets (no leakage)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Probability Integral Transforms
# ---------------------------------------------------------------------------

def empirical_pit(Y: Tensor) -> Tensor:
    """Normalized ranks of Y within itself (Hazen plotting positions).

    Used to convert Y_train into the encoder's uniform context representation.
    Ranking within the context against itself is not leakage — these are the
    very points that define the empirical copula of the context set.

    Args:
        Y: (B, n, d)

    Returns:
        U: (B, n, d)  values in [1/(n+1), n/(n+1)]
    """
    n = Y.shape[1]
    # argsort twice gives rank (0-indexed); +1 for 1-indexed Hazen positions
    ranks = Y.argsort(dim=1).argsort(dim=1).float() + 1.0
    return ranks / (n + 1.0)


def smooth_context_pit(
    Y_test: Tensor,
    Y_train: Tensor,
    eps: float = 1e-6,
    quartile_frac: float = 0.25,
) -> Tensor:
    """Project Y_test into uniform space via a smoothed ECDF built from Y_train.

    Causally correct: each test point is mapped independently through the
    *training* distribution — Y_test values are never ranked against each other,
    avoiding data leakage between query instances.

    Method (per batch element b, dimension j):
      1. Sort Y_train[b,:,j] → y_(1) < ... < y_(n).  Assign anchors u_(i)=i/(n+1).
      2. Interior (y_(1) ≤ y_test ≤ y_(n)):
           Linearly interpolate between bracketing anchors via torch.searchsorted.
      3. Left tail (y_test < y_(1)):
           sigma_lo = population std of lowest quartile_frac of Y_train (≥ 1e-8).
           u = Phi((y_test - y_(1)) / sigma_lo) * u_(1)
      4. Right tail (y_test > y_(n)):
           sigma_hi = population std of highest quartile_frac of Y_train (≥ 1e-8).
           u = u_(n) + (1 - u_(n)) * Phi((y_test - y_(n)) / sigma_hi)
      5. Clamp all outputs to (eps, 1-eps).

    Population std (unbiased=False) is used throughout so that a tail of size 1
    returns std=0 (instead of NaN from Bessel's N-1 correction), which is then
    safely clamped to 1e-8.

    Args:
        Y_test:       (B, n_test,  d)
        Y_train:      (B, n_train, d)
        eps:          safety clamp on output (default 1e-6)
        quartile_frac: fraction of each tail used to estimate tail sigma

    Returns:
        U_test: (B, n_test, d)  strictly in (eps, 1-eps)
    """
    B, n_train, d = Y_train.shape
    _, n_test, _  = Y_test.shape
    device        = Y_train.device

    # Sort training values for each (batch, dim)
    Y_sorted, _ = Y_train.sort(dim=1)            # (B, n_train, d)

    # Hazen anchors for context points
    i_idx   = torch.arange(1, n_train + 1, dtype=torch.float32, device=device)
    anchors = i_idx / (n_train + 1.0)            # (n_train,)

    # Reshape to (B*d, *) for batched searchsorted
    Y_sorted_bd = Y_sorted.permute(0, 2, 1).reshape(B * d, n_train)   # (B*d, n_train)
    Y_test_bd   = Y_test.permute(0, 2, 1).reshape(B * d, n_test)      # (B*d, n_test)

    # Insertion positions: k ∈ [0, n_train]
    #   k == 0       → y_test < y_(1)       (left tail)
    #   k == n_train → y_test >= y_(n)      (right tail)
    #   otherwise    → interior
    k = torch.searchsorted(Y_sorted_bd.contiguous(), Y_test_bd.contiguous())

    # --- Tail sigma: population std to avoid NaN when tail has 1 element ---
    n_tail = max(1, int(n_train * quartile_frac))

    # unbiased=False → divides by N, returns 0 for n=1 instead of NaN
    raw_lo = Y_sorted_bd[:, :n_tail].std(dim=1, unbiased=False, keepdim=True)
    raw_hi = Y_sorted_bd[:, -n_tail:].std(dim=1, unbiased=False, keepdim=True)
    # nan_to_num handles any residual NaN (e.g. all-identical values) before clamp
    sigma_lo = torch.nan_to_num(raw_lo, nan=0.0).clamp(min=1e-8)      # (B*d, 1)
    sigma_hi = torch.nan_to_num(raw_hi, nan=0.0).clamp(min=1e-8)      # (B*d, 1)

    y_lo = Y_sorted_bd[:, 0:1]    # (B*d, 1) minimum context value
    y_hi = Y_sorted_bd[:, -1:]    # (B*d, 1) maximum context value
    u_lo = anchors[0]              # scalar = 1/(n+1)
    u_hi = anchors[-1]             # scalar = n/(n+1)

    std_normal = torch.distributions.Normal(
        torch.zeros(1, device=device), torch.ones(1, device=device)
    )

    # --- Left tail ---
    left_u = std_normal.cdf((Y_test_bd - y_lo) / sigma_lo) * u_lo     # (B*d, n_test)

    # --- Right tail ---
    right_u = u_hi + (1.0 - u_hi) * std_normal.cdf(
        (Y_test_bd - y_hi) / sigma_hi
    )                                                                   # (B*d, n_test)

    # --- Interior: linear interpolation between sorted anchors ---
    k_lo  = k.clamp(1, n_train - 1) - 1    # left bracket index
    k_hi  = k.clamp(1, n_train - 1)        # right bracket index

    y_lo_i = Y_sorted_bd.gather(1, k_lo)
    y_hi_i = Y_sorted_bd.gather(1, k_hi)
    u_lo_i = anchors[k_lo]
    u_hi_i = anchors[k_hi]

    dy = (y_hi_i - y_lo_i).clamp(min=1e-10)
    t  = ((Y_test_bd - y_lo_i) / dy).clamp(0.0, 1.0)
    interior_u = u_lo_i + t * (u_hi_i - u_lo_i)                       # (B*d, n_test)

    # --- Select by region ---
    is_left  = k == 0
    is_right = k >= n_train
    U_bd = torch.where(is_left, left_u,
           torch.where(is_right, right_u, interior_u))

    U_bd = torch.nan_to_num(U_bd, nan=0.5).clamp(eps, 1.0 - eps)

    return U_bd.reshape(B, d, n_test).permute(0, 2, 1).contiguous()   # (B, n_test, d)


# ---------------------------------------------------------------------------
# Discretization and CE loss
# ---------------------------------------------------------------------------

def quantize_to_bins(U: Tensor, n_bins: int) -> Tensor:
    """Map uniform values in (0,1) to integer bin indices in [0, n_bins-1]."""
    return (U * n_bins).long().clamp(0, n_bins - 1)


def copula_ce_loss(logits: Tensor, U_true: Tensor, n_bins: int) -> Tensor:
    """Cross-entropy loss between predicted bin logits and true discretized uniforms.

    Args:
        logits: (B, n_test, d, n_bins)   raw (pre-softmax) bin logits
        U_true: (B, n_test, d)           smooth context-conditional uniform values
        n_bins: number of discrete bins

    Returns:
        Scalar mean CE loss (nats per dimension per instance).
    """
    bin_targets = quantize_to_bins(U_true, n_bins)    # (B, n_test, d)
    return F.cross_entropy(
        logits.reshape(-1, n_bins),
        bin_targets.reshape(-1),
    )


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def copula_energy_score(U_samples: Tensor, U_true: Tensor) -> float:
    """Energy score in uniform copula space: E||U-y|| - 0.5*E||U-U'||.

    Args:
        U_samples: (B, n_test, d, S)   S independent samples from the model
        U_true:    (B, n_test, d)      ground-truth uniform values

    Returns:
        Scalar mean energy score (lower is better).
    """
    if U_samples.dim() == 3:
        U_samples = U_samples.unsqueeze(-1)   # treat as single sample

    B, N, d, S = U_samples.shape
    y = U_true.unsqueeze(-1)                           # (B, N, d, 1)

    term1 = (U_samples - y).norm(dim=2).mean(dim=-1)  # (B, N)

    i_idx = torch.randint(S, (S,), device=U_samples.device)
    j_idx = torch.randint(S, (S,), device=U_samples.device)
    term2 = 0.5 * (
        U_samples[..., i_idx] - U_samples[..., j_idx]
    ).norm(dim=2).mean(dim=-1)                         # (B, N)

    return (term1 - term2).mean().item()


@torch.no_grad()
def marginal_calibration_hist(U_samples: Tensor, n_bins: int = 20) -> Tensor:
    """Per-dimension histogram of sampled U values; should approach Uniform[0,1].

    Args:
        U_samples: (B, n_test, d)
        n_bins:    number of histogram bins

    Returns:
        hist: (d, n_bins) normalized counts
    """
    B, N, d = U_samples.shape
    hist = torch.zeros(d, n_bins, device=U_samples.device)
    for j in range(d):
        counts = torch.histc(
            U_samples[:, :, j].reshape(-1).float(), bins=n_bins, min=0.0, max=1.0
        )
        total = counts.sum().clamp(min=1.0)
        hist[j] = counts / total
    return hist
