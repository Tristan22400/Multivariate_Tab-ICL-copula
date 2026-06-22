"""
data_gen.py — Episodic data generator for TFM-Multivariate training.

Implements the simulation protocol from §2 of the research specification:

    y_i | x_i ~ N( f(x_i),  Sigma(x_i) )

where the per-instance covariance uses a rank-r low-rank + diagonal structure:

    Sigma(x_i) = diag( d(x_i)^2 ) + V(x_i) @ V(x_i)^T

The three functions f, d, V are parameterised by independently and randomly
initialised MLPs that are frozen (and unique) for each training episode.
This gives every episode a distinct ground-truth conditional distribution for
the in-context learner to discover.

Sampling path (avoids instantiating a d×d matrix per point):

    eps_diag ~ N(0, I_d)      (diagonal noise)
    eps_low  ~ N(0, I_r)      (low-rank noise)
    y = f(x) + diag(x)^{1/2} * eps_diag + V(x) @ eps_low

Both X and Y are z-normalised feature-by-feature after sampling.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Batched random MLP — fixed weights, one independent MLP per batch element
# ---------------------------------------------------------------------------


class BatchedRandomMLP(nn.Module):
    """Fixed (non-trainable) random MLPs applied in parallel across a batch.

    Each batch element receives an independently sampled set of weight matrices,
    drawn once at construction time and never updated.  Weights are stored as
    ``register_buffer`` so they move with the module to the target device.

    Architecture: x → Linear(p_in, H) → ReLU → Linear(H, p_out)

    He/Kaiming initialisation is used throughout:
        std = sqrt(2 / fan_in)

    Args:
        batch_size : B — number of independent MLPs
        p_in       : input dimension
        p_out      : output dimension
        hidden     : hidden layer width H
        device     : torch device

    Usage::

        mlp = BatchedRandomMLP(B, p_in=10, p_out=5, hidden=64, device=device)
        out = mlp(X)   # X: (B, T, p_in) → out: (B, T, p_out)
    """

    def __init__(
        self,
        batch_size: int,
        p_in: int,
        p_out: int,
        hidden: int,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        H = hidden
        # Layer 1: p_in → H   (He init: std = sqrt(2 / p_in))
        self.register_buffer(
            "W1", torch.randn(batch_size, H, p_in) * math.sqrt(2.0 / p_in)
        )
        self.register_buffer("b1", torch.zeros(batch_size, H))
        # Layer 2: H → p_out  (He init: std = sqrt(2 / H))
        self.register_buffer(
            "W2", torch.randn(batch_size, p_out, H) * math.sqrt(2.0 / H)
        )
        self.register_buffer("b2", torch.zeros(batch_size, p_out))
        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            X : (B, T, p_in)

        Returns:
            out : (B, T, p_out)
        """
        # (B, T, p_in) × (B, p_in, H)^T → (B, T, H)
        h = F.relu(torch.bmm(X, self.W1.transpose(-2, -1)) + self.b1.unsqueeze(1))
        return torch.bmm(h, self.W2.transpose(-2, -1)) + self.b2.unsqueeze(1)


# ---------------------------------------------------------------------------
# Tabular feature prior sampler
# ---------------------------------------------------------------------------


def sample_tabular_x(
    B: int,
    T: int,
    p: int,
    device: torch.device | str,
    n_train: int | None = None,
) -> torch.Tensor:
    """Sample x from a tabular-like mixture prior.

    Each of the p feature dimensions is independently drawn from one of three
    marginal families, chosen uniformly at random for that (episode, feature):

        - Standard normal N(0, 1)
        - Uniform on [-3, 3]
        - Log-normal (exp of N(0, 0.5)) — positive-valued, skewed

    After sampling, every feature dimension is z-normalised using stats
    computed from the first n_train instances only (train split), then applied
    to all T instances.  This prevents test-set statistics from leaking into
    the normalisation.  When n_train is None the full T instances are used
    (backwards-compatible behaviour).

    Args:
        B       : batch size (number of independent datasets)
        T       : total number of instances (n_train + n_test)
        p       : feature dimension
        device  : torch device
        n_train : number of training instances used to compute norm stats;
                  if None, normalise over all T instances.

    Returns:
        X : (B, T, p) — z-normalised tabular features
    """
    # Choose family per (B, p) pair
    family = torch.randint(
        0, 3, (B, p), device=device
    )  # 0=normal, 1=uniform, 2=lognormal

    # Sample all three families up-front, select per feature
    x_normal = torch.randn(B, T, p, device=device)
    x_uniform = torch.rand(B, T, p, device=device) * 6.0 - 3.0
    x_lognorm = torch.exp(torch.randn(B, T, p, device=device) * 0.5)
    # family: (B, p) → (B, 1, p) for broadcasting
    f = family.unsqueeze(1)
    X = torch.where(f == 0, x_normal, torch.where(f == 1, x_uniform, x_lognorm))

    # Z-normalise using train-split stats to avoid test leakage
    ref = X[:, :n_train, :] if n_train is not None else X
    mu = ref.mean(dim=1, keepdim=True)
    std = ref.std(dim=1, keepdim=True) + 1e-8
    return (X - mu) / std


# ---------------------------------------------------------------------------
# Kernel-based x-dependent covariance generator
# ---------------------------------------------------------------------------

_KERNEL_NAMES = ["rbf", "exponential", "matern32", "rational_quadratic", "periodic"]


# ---------------------------------------------------------------------------
# Stationary isotropic kernel functions  (for IsotropicModulatedKernel)
# ---------------------------------------------------------------------------
# Signature: (r: Tensor[...,d,d], l: Tensor[...,1,1]) -> Tensor[...,d,d]
# All satisfy k(0, l) = 1, so C has unit diagonal by construction.


def _rbf_kernel(r: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """Matern-∞ / RBF: exp(−r²/(2l²))."""
    return torch.exp(-(r**2) / (2 * l**2))


def _matern12_kernel(r: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """Matern-½ (Ornstein-Uhlenbeck): exp(−r/l)."""
    return torch.exp(-r / l)


def _matern32_kernel(r: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """Matern-³⁄₂: (1 + √3 r/l) exp(−√3 r/l)."""
    rs = math.sqrt(3) * r / l
    return (1.0 + rs) * torch.exp(-rs)


def _matern52_kernel(r: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """Matern-⁵⁄₂: (1 + √5 r/l + 5r²/(3l²)) exp(−√5 r/l)."""
    rs = math.sqrt(5) * r / l
    return (1.0 + rs + rs**2 / 3.0) * torch.exp(-rs)


_STATIONARY_KERNEL_FNS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "rbf": _rbf_kernel,
    "matern12": _matern12_kernel,
    "matern32": _matern32_kernel,
    "matern52": _matern52_kernel,
}
_ISO_KERNEL_NAMES: list[str] = list(_STATIONARY_KERNEL_FNS)


# ---------------------------------------------------------------------------
# Dataset 1 — Isotropic Modulated Kernel  →  Elliptope
# ---------------------------------------------------------------------------


class IsotropicModulatedKernel:
    """Dataset 1: hierarchical isotropic kernel mapping x_i to the Elliptope.

    Produces per-instance **correlation** matrices C(x_i) ∈ R^{d×d} (unit
    diagonal, strict PD) and returns their Cholesky factors.  Shares the same
    call interface as KernelCovGen so it can be passed as `kernel_cov_gen` to
    `generate_episode`.

    Class attribute `is_copula_gen = True` signals `generate_episode` to
    suppress the extra diagonal noise term (which would break the unit-diagonal
    / copula structure).

    Generative process (all priors sampled fresh each episode):

      Step 1 — Spatial geometry (static per episode)
        k      ~ U{1,...,d}                — latent rank
        σ_E    ~ LogU(1e-2, 1e1)          — embedding scale
        E      ~ N(0, σ_E² I)  ∈ R^{d×k} — target latent positions
        D_{mn} = ‖E_m − E_n‖²             — pairwise squared distance

      Step 2 — Covariate projection (1-D bottleneck)
        α_w    ~ LogU(1e-2, 1e1)
        w      ~ N(0, α_w I)  ∈ R^p
        δ      ~ U(0.1, 1.0)
        l(x_i) = softplus(wᵀ x_i) + δ    — strictly positive scalar

      Step 3 — Kernel pushforward + nugget regularization
        K_{mn}(x_i) = kernel_fn(√D_{mn}, l(x_i))
        C(x_i)      = (1−ε) K(x_i) + ε I_d      (diagonal = 1 exactly)

    Supported kernels (see _STATIONARY_KERNEL_FNS):
        rbf      — Matern-∞:  exp(−r²/(2l²))
        matern12 — Matern-½:  exp(−r/l)
        matern32 — Matern-³⁄₂: (1 + √3 r/l) exp(−√3 r/l)
        matern52 — Matern-⁵⁄₂: (1 + √5 r/l + 5r²/(3l²)) exp(−√5 r/l)
        random   — one of the above, chosen uniformly per episode
    """

    KERNELS: list[str] = _ISO_KERNEL_NAMES
    is_copula_gen: bool = True  # tells generate_episode to zero diag noise

    def __init__(self, kernel_type: str = "rbf", nugget: float = 1e-4) -> None:
        if kernel_type not in self.KERNELS + ["random"]:
            raise ValueError(
                f"Unknown kernel_type {kernel_type!r}. "
                f"Choose from {self.KERNELS + ['random']}"
            )
        self.kernel_type = kernel_type
        self.nugget = nugget

    def __call__(self, X: torch.Tensor, d: int) -> torch.Tensor:
        """Build per-instance Cholesky factors L(x_i) of the correlation matrix C(x_i).

        Args:
            X : (B, T, p) — z-normalised input features
            d : target dimension

        Returns:
            L : (B, T, d, d) — lower-triangular Cholesky factor of C(x_i)
        """
        B, T, p = X.shape
        device = X.device
        eps = self.nugget

        kernel_name = (
            _ISO_KERNEL_NAMES[torch.randint(len(_ISO_KERNEL_NAMES), (1,)).item()]
            if self.kernel_type == "random"
            else self.kernel_type
        )
        kernel_fn = _STATIONARY_KERNEL_FNS[kernel_name]

        # Step 1: PER-BATCH spatial geometry — each dataset gets independent priors.
        # k_b ~ U{1,...,d}: variable latent rank per dataset.
        k_each = torch.randint(1, d + 1, (B,), device=device)           # (B,)
        # E_b ∈ R^{d × d}; columns beyond k_b are zeroed so the effective rank is k_b.
        sigma_E = torch.empty(B, device=device).uniform_(
            math.log(1e-2), math.log(1e1)
        ).exp().view(B, 1, 1)                                            # (B, 1, 1)
        col_mask = (
            torch.arange(d, device=device).unsqueeze(0) < k_each.unsqueeze(1)
        ).float()                                                         # (B, d)
        E = torch.randn(B, d, d, device=device) * sigma_E               # (B, d, d)
        E = E * col_mask.unsqueeze(1)                                    # zero cols > k_b
        diff_E = E.unsqueeze(2) - E.unsqueeze(1)                        # (B, d, d, d)
        r_dist = (diff_E**2).sum(-1).clamp(min=0).sqrt()               # (B, d, d)

        # Step 2: PER-BATCH covariate projection → per-instance lengthscale.
        alpha_w = torch.empty(B, device=device).uniform_(
            math.log(1e-2), math.log(1e1)
        ).exp()                                                           # (B,)
        w = torch.randn(B, p, device=device) * alpha_w.sqrt().unsqueeze(1)  # (B, p)
        delta = torch.empty(B, device=device).uniform_(0.1, 1.0)        # (B,)

        proj = torch.einsum("btp,bp->bt", X, w)                         # (B, T)
        l_x = F.softplus(proj) + delta.unsqueeze(1)                     # (B, T), > 0
        l = l_x.unsqueeze(-1).unsqueeze(-1)                             # (B, T, 1, 1)

        # Step 3: kernel evaluation and nugget regularization → correlation matrix
        r_d = r_dist.unsqueeze(1)                                        # (B, 1, d, d)
        K = kernel_fn(r_d, l)                                            # (B, T, d, d)
        C = (1.0 - eps) * K + eps * torch.eye(d, device=device).view(1, 1, d, d)

        # Cholesky — C is strictly PD by construction (nugget ε > 0, K PSD)
        L = torch.linalg.cholesky(C.reshape(B * T, d, d)).reshape(B, T, d, d)
        return L


# ---------------------------------------------------------------------------
# TabICL feature-kernel covariance generator
# ---------------------------------------------------------------------------

_TABICL_FUNC_NAMES = [
    "linear",
    "mlp",
    "quadratic",
    "product",
    "nn_discretize",
    "tree_ensemble",
    "rff_gp",
    "plateau",
]


def _tabicl_apply_func(
    func_name: str,
    Xin: torch.Tensor,       # (B, T, c)
    q: int,
    device: torch.device,
) -> torch.Tensor:
    """Apply one random function from the TabICLv2-inspired pool.

    All weight tensors are sampled here (per batch element, per output-dim /
    feature-slot call) with a leading B dim and NO T dim, enforcing the
    construction-fixed-per-dataset invariant.

    Args:
        func_name : one of _TABICL_FUNC_NAMES
        Xin       : (B, T, c) input
        q         : output embedding dim

    Returns:
        out : (B, T, q)
    """
    B, T, c = Xin.shape

    if func_name == "linear":
        # A:(B,q,c), b:(B,q)
        A = torch.randn(B, q, c, device=device) * math.sqrt(2.0 / max(c, 1))
        b = torch.zeros(B, q, device=device)
        return torch.bmm(Xin, A.transpose(-2, -1)) + b.unsqueeze(1)

    elif func_name == "mlp":
        H = max(q * 2, 8)
        W1 = torch.randn(B, H, c, device=device) * math.sqrt(2.0 / max(c, 1))
        b1 = torch.zeros(B, H, device=device)
        W2 = torch.randn(B, q, H, device=device) * math.sqrt(2.0 / H)
        b2 = torch.zeros(B, q, device=device)
        h = F.relu(torch.bmm(Xin, W1.transpose(-2, -1)) + b1.unsqueeze(1))
        # random activation drawn once per call (not per T)
        act_choice = torch.randint(3, (1,)).item()
        if act_choice == 0:
            h = torch.tanh(h)
        elif act_choice == 1:
            h = F.gelu(h)
        # else keep relu output
        return torch.bmm(h, W2.transpose(-2, -1)) + b2.unsqueeze(1)

    elif func_name == "quadratic":
        # linear proj then element-wise square + another linear mix
        A1 = torch.randn(B, q, c, device=device) * math.sqrt(2.0 / max(c, 1))
        A2 = torch.randn(B, q, c, device=device) * math.sqrt(2.0 / max(c, 1))
        lin = torch.bmm(Xin, A1.transpose(-2, -1))
        sq = torch.bmm(Xin, A2.transpose(-2, -1)) ** 2
        mix = torch.randn(B, q, device=device) * 0.5
        return lin + mix.unsqueeze(1) * sq

    elif func_name == "product":
        # product of 2 random linear projections, each (B,T,q)
        n_factors = 2
        out = torch.ones(B, T, q, device=device)
        for _ in range(n_factors):
            A = torch.randn(B, q, c, device=device) * math.sqrt(1.0 / max(c, 1))
            out = out * torch.bmm(Xin, A.transpose(-2, -1))
        return out

    elif func_name == "nn_discretize":
        # K random centroids in c-space; nearest-centroid → per-centroid embedding
        K = max(q, 4)
        centroids = torch.randn(B, K, c, device=device)   # (B, K, c)
        embs = torch.randn(B, K, q, device=device) * 0.5  # (B, K, q)
        # pairwise distances: (B, T, K)
        diff = Xin.unsqueeze(2) - centroids.unsqueeze(1)   # (B, T, K, c)
        dists = (diff ** 2).sum(-1)                         # (B, T, K)
        idx = dists.argmin(dim=-1)                          # (B, T)
        # gather: for each (b,t) pick embs[b, idx[b,t], :]
        idx_exp = idx.unsqueeze(-1).expand(B, T, q)
        embs_exp = embs.unsqueeze(1).expand(B, T, K, q)
        return embs_exp.gather(2, idx_exp.unsqueeze(2)).squeeze(2)

    elif func_name == "tree_ensemble":
        # sum of random axis-aligned soft step functions (piecewise-constant surrogate)
        n_trees = 4
        feat_idx = torch.randint(c, (B, n_trees), device=device)   # (B, n_trees)
        thresholds = torch.randn(B, n_trees, device=device)         # (B, n_trees)
        out_embs = torch.randn(B, n_trees, q, device=device)        # (B, n_trees, q)
        # extract selected feature for each tree: (B, T, n_trees)
        feat_vals = Xin.gather(
            2, feat_idx.unsqueeze(1).expand(B, T, n_trees)
        )                                                             # (B, T, n_trees)
        # soft step: sigmoid(50 * (feat - threshold))
        steps = torch.sigmoid(50.0 * (feat_vals - thresholds.unsqueeze(1)))  # (B, T, n_trees)
        # weighted sum over trees: (B, T, q)
        return torch.einsum("btn,bnq->btq", steps, out_embs)

    elif func_name == "rff_gp":
        # random Fourier features: Σ_k a_k cos(Ω_k · x + φ_k)
        n_rff = max(q * 4, 16)
        Omega = torch.randn(B, n_rff, c, device=device)       # (B, n_rff, c)
        phi = torch.rand(B, n_rff, device=device) * 2 * math.pi
        proj = torch.bmm(Xin, Omega.transpose(-2, -1)) + phi.unsqueeze(1)  # (B, T, n_rff)
        feats = torch.cos(proj) * math.sqrt(2.0 / n_rff)                   # (B, T, n_rff)
        A_out = torch.randn(B, q, n_rff, device=device) * math.sqrt(2.0 / n_rff)
        return torch.bmm(feats, A_out.transpose(-2, -1))

    elif func_name == "plateau":
        # sum of a few soft logistic plateaus: a·σ(k·(x−lo)) − a·σ(k·(x−hi))
        n_plateaus = 3
        feat_idx = torch.randint(c, (B, n_plateaus), device=device)
        lo = torch.randn(B, n_plateaus, device=device) - 0.5
        hi = lo + torch.rand(B, n_plateaus, device=device).clamp(min=0.2)
        steepness = torch.rand(B, n_plateaus, device=device) * 8 + 2
        out_embs = torch.randn(B, n_plateaus, q, device=device)
        feat_vals = Xin.gather(
            2, feat_idx.unsqueeze(1).expand(B, T, n_plateaus)
        )                                                              # (B, T, n_plateaus)
        lo_u = lo.unsqueeze(1)
        hi_u = hi.unsqueeze(1)
        k_u = steepness.unsqueeze(1)
        activations = (
            torch.sigmoid(k_u * (feat_vals - lo_u))
            - torch.sigmoid(k_u * (feat_vals - hi_u))
        )                                                              # (B, T, n_plateaus)
        return torch.einsum("btn,bnq->btq", activations, out_embs)

    else:
        raise ValueError(f"Unknown func_name: {func_name!r}")


class TabICLFeatureKernel:
    """Per-instance d×d correlation matrices via instance-dependent output-dim embeddings.

    Generalises IsotropicModulatedKernel: instead of a single scalar lengthscale
    modulating a fixed per-dataset geometry, each output dimension m gets a
    q-dim embedding W_m(x) built from a random TabICLv2-style program applied to
    sampled input features.  The per-instance correlation matrix is then

        Σ_mn(x) = k( ‖W_m(x) − W_n(x)‖₂ ,  l_b )

    where l_b is a per-dataset (per batch element) lengthscale and k is a
    stationary isotropic kernel.

    **Construction-fixed-per-dataset invariant**: all random structure (feature
    subsets, function weights, aggregation ops, kernel/lengthscale) is sampled
    once per __call__ with a leading B dim and no T dim.  W_m(x) varies across
    instances only because x does, not because the program is re-sampled.

    Generative process (per __call__, all priors fresh per episode):

      For each output dim m (independently):
        1. Sample n_feat ~ U{1 .. max_feats}, draw feature indices S_m ∈ {0..p-1}^n_feat.
        2. Sample aggregation mode:
             concat   — apply ONE function g:(B,T,n_feat)→(B,T,q)
             separate — apply n_feat functions g_j:(B,T,1)→(B,T,q) then
                        aggregate element-wise (sum | product | max | logsumexp)
        3. W[:,  :, m, :] ← result ∈ (B, T, q)

      Standardise W over the T axis per (b,m,q) → W normalised.

      Per-batch lengthscale: l_b ~ LogU(lengthscale_lo, lengthscale_hi).

      Pairwise distance:  r_mn = ‖W_m − W_n‖₂  over q  →  (B, T, d, d).

      Kernel:  K_mn = kernel_fn(r_mn, l_b).

      Correlation + nugget:  C = (1−ε)K + ε I   (unit diagonal by construction).

      Cholesky:  L  s.t.  L Lᵀ = C.

    Supported kernels (same as IsotropicModulatedKernel):
        rbf      — exp(−r²/(2l²))
        matern12 — exp(−r/l)
        matern32 — (1 + √3 r/l) exp(−√3 r/l)
        matern52 — (1 + √5 r/l + 5r²/(3l²)) exp(−√5 r/l)
        random   — one of the above chosen uniformly per episode

    Function pool (set func_pool to a subset of _TABICL_FUNC_NAMES to restrict):
        linear, mlp, quadratic, product, nn_discretize,
        tree_ensemble, rff_gp, plateau
    """

    KERNELS: list[str] = _ISO_KERNEL_NAMES
    is_copula_gen: bool = True

    def __init__(
        self,
        kernel_type: str = "random",
        nugget: float = 1e-4,
        embed_dim: int = 4,
        max_feats: int = 3,
        lengthscale_lo: float = 0.1,
        lengthscale_hi: float = 10.0,
        func_pool: list[str] | None = None,
    ) -> None:
        if kernel_type not in self.KERNELS + ["random"]:
            raise ValueError(
                f"Unknown kernel_type {kernel_type!r}. "
                f"Choose from {self.KERNELS + ['random']}"
            )
        self.kernel_type = kernel_type
        self.nugget = nugget
        self.embed_dim = embed_dim
        self.max_feats = max(1, max_feats)
        self.lengthscale_lo = lengthscale_lo
        self.lengthscale_hi = lengthscale_hi
        self.func_pool = func_pool if func_pool is not None else _TABICL_FUNC_NAMES

    def __call__(self, X: torch.Tensor, d: int) -> torch.Tensor:
        """Build per-instance Cholesky factors L(x_i) of the correlation matrix C(x_i).

        Args:
            X : (B, T, p) — z-normalised input features
            d : target dimension

        Returns:
            L : (B, T, d, d) — lower-triangular Cholesky factor of C(x_i)
        """
        B, T, p = X.shape
        device = X.device
        q = self.embed_dim
        eps = self.nugget

        kernel_name = (
            _ISO_KERNEL_NAMES[torch.randint(len(_ISO_KERNEL_NAMES), (1,)).item()]
            if self.kernel_type == "random"
            else self.kernel_type
        )
        kernel_fn = _STATIONARY_KERNEL_FNS[kernel_name]

        # Per-batch lengthscale: l_b ~ LogU(lo, hi),  shape (B, 1, 1, 1)
        log_lo = math.log(self.lengthscale_lo)
        log_hi = math.log(self.lengthscale_hi)
        l_b = torch.empty(B, device=device).uniform_(log_lo, log_hi).exp()
        l = l_b.view(B, 1, 1, 1)   # broadcast over (T, d, d)

        _AGG_OPS = ["sum", "product", "max", "logsumexp"]
        pool = self.func_pool

        # Build W ∈ (B, T, d, q) — one embedding per output dim per instance
        W = torch.zeros(B, T, d, q, device=device)
        for m in range(d):
            # Sample number of features and which features (per dataset)
            n_feat = int(torch.randint(1, self.max_feats + 1, (1,)).item())
            # Per-batch feature indices: (B, n_feat) — fixed for this dim m
            feat_idx = torch.stack(
                [torch.randperm(p, device=device)[:n_feat] for _ in range(B)]
            )  # (B, n_feat)

            # Gather selected features: (B, T, n_feat)
            Xsel = X.gather(2, feat_idx.unsqueeze(1).expand(B, T, n_feat))

            agg_mode = "concat" if torch.rand(1).item() < 0.5 else "separate"

            if agg_mode == "concat" or n_feat == 1:
                func_name = pool[torch.randint(len(pool), (1,)).item()]
                w_m = _tabicl_apply_func(func_name, Xsel, q, device)  # (B, T, q)
            else:
                agg_op = _AGG_OPS[torch.randint(len(_AGG_OPS), (1,)).item()]
                parts = []
                for j in range(n_feat):
                    func_name = pool[torch.randint(len(pool), (1,)).item()]
                    xj = Xsel[:, :, j : j + 1]   # (B, T, 1)
                    parts.append(_tabicl_apply_func(func_name, xj, q, device))
                stacked = torch.stack(parts, dim=0)  # (n_feat, B, T, q)
                if agg_op == "sum":
                    w_m = stacked.sum(0)
                elif agg_op == "product":
                    w_m = stacked.prod(0)
                elif agg_op == "max":
                    w_m = stacked.max(0).values
                else:  # logsumexp
                    w_m = stacked.logsumexp(0)

            W[:, :, m, :] = w_m

        # Standardize W over the T axis per (b, m, q) to keep distances O(1)
        W_mean = W.mean(dim=1, keepdim=True)            # (B, 1, d, q)
        W_std = W.std(dim=1, keepdim=True).clamp(min=1e-6)
        W = (W - W_mean) / W_std                        # (B, T, d, q)

        # Pairwise Euclidean distance between output-dim embeddings: (B, T, d, d)
        # W_m ∈ (B, T, q) for each m; diff[b,t,m,n] = W[b,t,m,:] - W[b,t,n,:]
        diff = W.unsqueeze(3) - W.unsqueeze(2)          # (B, T, d, d, q)
        r = (diff ** 2).sum(-1).clamp(min=0).sqrt()     # (B, T, d, d)

        # Kernel evaluation — l broadcasts over (T, d, d)
        K = kernel_fn(r, l)                              # (B, T, d, d)

        # Nugget → exact unit diagonal (k(0)=1 by construction of stationary kernels)
        I_d = torch.eye(d, device=device).view(1, 1, d, d)
        C = (1.0 - eps) * K + eps * I_d                 # (B, T, d, d)

        L = torch.linalg.cholesky(C.reshape(B * T, d, d)).reshape(B, T, d, d)
        return L


_SIMPLE_AGG_OPS: list[str] = ["sum", "product", "max", "min", "mean", "logsumexp"]
_SIMPLE_AGG_KERNEL_NAMES: list[str] = _ISO_KERNEL_NAMES + ["cosine"]


class SimpleAggKernel:
    """Per-instance d×d covariance via direct feature aggregation and PSD kernels.

    Simplified version of TabICLFeatureKernel: replaces all random function
    transforms (MLP, RFF-GP, trees, …) with direct element-wise aggregations.

    W construction (per output dim m, per embedding component j ∈ {0..q-1}):
      1. Sample n_feat ~ U{1..max_feats} feature indices (per batch element).
      2. Gather selected features: Xsel ∈ (B, T, n_feat).
      3. Aggregate element-wise: w = agg_op(Xsel, dim=-1) → (B, T).
         agg_op drawn from {sum, product, max, min, mean, logsumexp}.
         If n_feat == 1, w is just that feature directly.
      4. W[:,:,m,j] = w.  Result: W ∈ (B, T, d, q).

    Kernel types:
      "cosine"  — K[b,t,m,n] = (W_m · W_n) / (‖W_m‖ ‖W_n‖).
                  Gram matrix of unit-norm vectors → always PSD, values in [−1, 1].
      Distance-based (rbf, matern12, matern32, matern52) — same path as
                  TabICLFeatureKernel; values in [0, 1].
      "random"  — drawn uniformly from all available kernel names.
    """

    KERNELS: list[str] = _SIMPLE_AGG_KERNEL_NAMES
    is_copula_gen: bool = True

    def __init__(
        self,
        kernel_type: str = "cosine",
        nugget: float = 1e-4,
        embed_dim: int = 4,
        max_feats: int = 3,
        lengthscale_lo: float = 0.1,
        lengthscale_hi: float = 10.0,
    ) -> None:
        if kernel_type not in self.KERNELS + ["random"]:
            raise ValueError(
                f"Unknown kernel_type {kernel_type!r}. "
                f"Choose from {self.KERNELS + ['random']}"
            )
        self.kernel_type = kernel_type
        self.nugget = nugget
        self.embed_dim = max(1, embed_dim)
        self.max_feats = max(1, max_feats)
        self.lengthscale_lo = lengthscale_lo
        self.lengthscale_hi = lengthscale_hi

    def __call__(self, X: torch.Tensor, d: int) -> torch.Tensor:
        """Build per-instance Cholesky factors L(x_i) of the covariance matrix.

        Args:
            X: (B, T, p) — z-normalised input features
            d: target output dimension
        Returns:
            L: (B, T, d, d) — lower-triangular Cholesky factor
        """
        B, T, p = X.shape
        device = X.device
        q = self.embed_dim
        eps = self.nugget

        kernel_name = (
            _SIMPLE_AGG_KERNEL_NAMES[
                torch.randint(len(_SIMPLE_AGG_KERNEL_NAMES), (1,)).item()
            ]
            if self.kernel_type == "random"
            else self.kernel_type
        )

        # Build W ∈ (B, T, d, q): q independent scalar aggregations per output dim
        W = torch.zeros(B, T, d, q, device=device)
        for m in range(d):
            for j in range(q):
                n_feat = int(torch.randint(1, self.max_feats + 1, (1,)).item())
                feat_idx = torch.stack(
                    [torch.randperm(p, device=device)[:n_feat] for _ in range(B)]
                )  # (B, n_feat)
                Xsel = X.gather(2, feat_idx.unsqueeze(1).expand(B, T, n_feat))

                if n_feat == 1:
                    w = Xsel.squeeze(-1)
                else:
                    agg_op = _SIMPLE_AGG_OPS[
                        torch.randint(len(_SIMPLE_AGG_OPS), (1,)).item()
                    ]
                    if agg_op == "sum":
                        w = Xsel.sum(-1)
                    elif agg_op == "product":
                        w = Xsel.prod(-1)
                    elif agg_op == "max":
                        w = Xsel.max(-1).values
                    elif agg_op == "min":
                        w = Xsel.min(-1).values
                    elif agg_op == "mean":
                        w = Xsel.mean(-1)
                    else:  # logsumexp
                        w = Xsel.logsumexp(-1)
                W[:, :, m, j] = w

        # Standardize W over T per (b, m, j)
        W_mean = W.mean(dim=1, keepdim=True)
        W_std = W.std(dim=1, keepdim=True).clamp(min=1e-6)
        W = (W - W_mean) / W_std  # (B, T, d, q)

        if kernel_name == "cosine":
            # Gram matrix of unit-norm vectors — always PSD, diagonal = 1, values in [-1,1]
            W_normed = W / W.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            K = torch.einsum("btmq,btnq->btmn", W_normed, W_normed)  # (B, T, d, d)
        else:
            # Distance-based stationary kernel (same path as TabICLFeatureKernel)
            log_lo = math.log(self.lengthscale_lo)
            log_hi = math.log(self.lengthscale_hi)
            l_b = torch.empty(B, device=device).uniform_(log_lo, log_hi).exp()
            l = l_b.view(B, 1, 1, 1)
            kernel_fn = _STATIONARY_KERNEL_FNS[kernel_name]
            diff = W.unsqueeze(3) - W.unsqueeze(2)  # (B, T, d, d, q)
            r = (diff ** 2).sum(-1).clamp(min=0).sqrt()
            K = kernel_fn(r, l)

        I_d = torch.eye(d, device=device).view(1, 1, d, d)
        C = (1.0 - eps) * K + eps * I_d  # (B, T, d, d)

        L = torch.linalg.cholesky(C.reshape(B * T, d, d)).reshape(B, T, d, d)
        return L


class KernelCovGen:
    """Per-instance x-dependent covariance via MLP-parameterized GP kernels.

    Each call samples fresh latent embeddings E ∈ R^(d × latent_dim) for the d
    output dimensions plus two frozen random MLPs that map each input x_i to
    scalar kernel hyperparameters:

        l(x_i)  = Softplus(MLP_l(x_i))   — lengthscale
        σ²(x_i) = Softplus(MLP_σ(x_i))   — signal variance

    The d×d covariance at x_i is:

        Σ(x_i)_jk = σ²(x_i) · kernel(e_j, e_k ; l(x_i)) + nugget · δ_jk

    Cholesky decomposition gives L(x_i) so Y|x_i ~ N(0, L L^T) exactly.
    Everything is re-sampled per call → every episode has a distinct x→Σ mapping.

    Supported kernels
    -----------------
    rbf               : σ² exp(−‖eⱼ−eₖ‖²/(2l²))
    exponential       : σ² exp(−‖eⱼ−eₖ‖/l)
    matern32          : σ²(1+√3 r/l) exp(−√3 r/l)
    rational_quadratic: σ²(1+r²/(2αl²))^(−α),  α ~ LogU(0.1, 10)
    periodic          : σ² exp(−2 sin²(πr/p)/l²), p ~ LogU(0.5, 5)
    random            : one of the above, chosen uniformly per call
    """

    KERNELS = _KERNEL_NAMES

    def __init__(
        self,
        kernel_type: str,
        latent_dim: int,
        mlp_hidden: int,
        nugget: float,
    ) -> None:
        if kernel_type not in self.KERNELS + ["random"]:
            raise ValueError(
                f"Unknown kernel_type {kernel_type!r}. "
                f"Choose from {self.KERNELS + ['random']}"
            )
        self.kernel_type = kernel_type
        self.latent_dim = latent_dim
        self.mlp_hidden = mlp_hidden
        self.nugget = nugget

    def __call__(self, X: torch.Tensor, d: int) -> torch.Tensor:
        """Build per-instance Cholesky factors L(x_i) for one episode.

        Args:
            X : (B, T, p) — input features
            d : target dimension

        Returns:
            L_x : (B, T, d, d) — lower-triangular Cholesky factor of Σ(x_i)
        """
        B, T, p = X.shape
        device = X.device

        kernel = (
            _KERNEL_NAMES[torch.randint(len(_KERNEL_NAMES), (1,)).item()]
            if self.kernel_type == "random"
            else self.kernel_type
        )

        # Fixed latent embeddings for output dims — fresh per episode
        E = torch.randn(d, self.latent_dim, device=device)  # (d, k)
        diff_E = E.unsqueeze(1) - E.unsqueeze(0)  # (d, d, k)
        sq_dist = (diff_E**2).sum(-1)  # (d, d)
        dist = sq_dist.clamp(min=0).sqrt()  # (d, d)

        # Per-instance kernel hyperparameters from frozen random MLPs
        l_net = BatchedRandomMLP(
            B, p_in=p, p_out=1, hidden=self.mlp_hidden, device=device
        )
        s_net = BatchedRandomMLP(
            B, p_in=p, p_out=1, hidden=self.mlp_hidden, device=device
        )
        l_x = F.softplus(l_net(X)) + 1e-3  # (B, T, 1)  — lengthscale
        s_x = F.softplus(s_net(X)) + 1e-3  # (B, T, 1)  — signal variance

        # Broadcast: (d,d) → (1,1,d,d) and (B,T,1) → (B,T,1,1)
        sq_d = sq_dist.view(1, 1, d, d)
        r_d = dist.view(1, 1, d, d)
        l = l_x.unsqueeze(-1)  # (B, T, 1, 1)
        s = s_x.unsqueeze(-1)  # (B, T, 1, 1)

        if kernel == "rbf":
            K = s * torch.exp(-sq_d / (2 * l**2))
        elif kernel == "exponential":
            K = s * torch.exp(-r_d / l)
        elif kernel == "matern32":
            rs = math.sqrt(3) * r_d / l
            K = s * (1.0 + rs) * torch.exp(-rs)
        elif kernel == "rational_quadratic":
            alpha = math.exp(
                math.log(0.1) + (math.log(10.0) - math.log(0.1)) * torch.rand(1, device=device).item()
            )
            K = s * (1.0 + sq_d / (2 * alpha * l**2)).pow(-alpha)
        elif kernel == "periodic":
            p_period = math.exp(
                math.log(0.5) + (math.log(5.0) - math.log(0.5)) * torch.rand(1, device=device).item()
            )
            K = s * torch.exp(-2.0 * torch.sin(math.pi * r_d / p_period).pow(2) / l**2)
        else:
            raise ValueError(kernel)

        # Nugget for strict positive-definiteness
        K = K + self.nugget * torch.eye(d, device=device).view(
            1, 1, d, d
        )  # (B, T, d, d)

        # Batched Cholesky over all instances
        L_x = torch.linalg.cholesky(K.reshape(B * T, d, d)).reshape(B, T, d, d)
        return L_x


# ---------------------------------------------------------------------------
# Anchor-based covariance generator
# ---------------------------------------------------------------------------


class AnchorCovarianceGen:
    """Anchor keys + V prototypes for structured x-dependent covariance generation.

    Two modes:
      1) shared_across_batch=False (default): each batch element b has its own
         K anchor keys c_{b,k} ∈ R^p and anchor V matrices A_{b,k} ∈ R^{d×r}.
      2) shared_across_batch=True: a single set of anchors (c_k, A_k) is shared
         across all batch elements, reducing per-dataset variation.

    For instance x_i in batch b, the low-rank factor is:

        w_k  = softmax( x_i · c_{b,k} / τ )       k = 1 … K
        V(x_i) = Σ_k w_k * A_{b,k}                 ∈ R^{d×r}

    This creates smooth, x-dependent covariances as a convex combination of K
    prototypes per batch element, with a richer structure than a single random MLP.

    Provides the same call interface as BatchedRandomMLP:
        X : (B, T, p) → (B, T, d*r)
    """

    def __init__(
        self,
        K: int,
        B: int,
        p: int,
        d: int,
        r: int,
        tau: float,
        device: torch.device | str,
        use_mean: bool = False,
        shared_across_batch: bool = False,
    ) -> None:
        self.tau = tau
        self.shared_across_batch = shared_across_batch

        if shared_across_batch:
            # Shared anchors: (K, p), (K, d, r), (K, d)
            C = torch.randn(K, p, device=device)
            self.C = F.normalize(C, dim=-1)
            self.A = torch.randn(K, d, r, device=device) * math.sqrt(2.0 / r)
            self.M = torch.randn(K, d, device=device) if use_mean else None
        else:
            # Per-batch anchors: (B, K, p), (B, K, d, r), (B, K, d)
            C = torch.randn(B, K, p, device=device)
            self.C = F.normalize(C, dim=-1)
            self.A = torch.randn(B, K, d, r, device=device) * math.sqrt(2.0 / r)
            self.M = torch.randn(B, K, d, device=device) if use_mean else None

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X : (B, T, p)
        Returns:
            (B, T, d*r) — same shape as BatchedRandomMLP output (reshape externally)
        """
        if self.shared_across_batch:
            # C: (K, p), A: (K, d, r)
            S = torch.einsum("btp,kp->btk", X, self.C) / self.tau  # (B, T, K)
            W = F.softmax(S, dim=-1)  # (B, T, K)
            V_x = torch.einsum("btk,kdr->btdr", W, self.A)  # (B, T, d, r)
        else:
            # C: (B, K, p), A: (B, K, d, r)
            S = torch.einsum("btp,bkp->btk", X, self.C) / self.tau  # (B, T, K)
            W = F.softmax(S, dim=-1)  # (B, T, K)
            V_x = torch.einsum("btk,bkdr->btdr", W, self.A)  # (B, T, d, r)
        B, T, d, r = V_x.shape
        return V_x.reshape(B, T, d * r)

    def get_mean(self, X: torch.Tensor) -> torch.Tensor:
        """Piecewise constant mean via hard nearest-anchor assignment.

        Each x_i is assigned to the closest anchor by cosine similarity (argmax over
        dot products; C is unit-normalised so this equals nearest Voronoi region).
        All points in the same region share the same constant mean vector M_{b,k*}.

        Args:
            X : (B, T, p)
        Returns:
            mu : (B, T, d)
        """
        if self.M is None:
            raise RuntimeError("get_mean() called but use_mean=False at construction.")
        if self.shared_across_batch:
            # M: (K, d) shared across batch
            dots = torch.einsum("btp,kp->btk", X, self.C)  # (B, T, K)
            k_star = dots.argmax(dim=-1)  # (B, T)
            return self.M[k_star]  # (B, T, d)
        else:
            B_x = X.shape[0]
            dots = torch.einsum("btp,bkp->btk", X, self.C)  # (B, T, K)
            k_star = dots.argmax(dim=-1)  # (B, T)
            return self.M[
                torch.arange(B_x, device=X.device).unsqueeze(1), k_star, :
            ]  # (B, T, d)


class GlobalAnchorCovGen:
    """Cache of AnchorCovarianceGen instances keyed by (p, d) or (B, p, d).

    Drop-in replacement for GlobalFixedNets: created once at training start,
    returns the same AnchorCovarianceGen for a given (B, p, d) combination so
    the x→covariance mapping is fixed across all episodes.
    """

    def __init__(
        self,
        K: int,
        r: int,
        tau: float,
        device: torch.device | str,
        use_mean: bool = False,
        shared_across_batch: bool = False,
    ) -> None:
        self.K = K
        self.r = r
        self.tau = tau
        self.device = device
        self.use_mean = use_mean
        self.shared_across_batch = shared_across_batch
        self._cache: dict[tuple, AnchorCovarianceGen] = {}

    def get(self, B: int, p: int, d: int) -> AnchorCovarianceGen:
        key = (p, d) if self.shared_across_batch else (B, p, d)
        if key not in self._cache:
            self._cache[key] = AnchorCovarianceGen(
                K=self.K,
                B=B,
                p=p,
                d=d,
                r=self.r,
                tau=self.tau,
                device=self.device,
                use_mean=self.use_mean,
                shared_across_batch=self.shared_across_batch,
            )
        return self._cache[key]


# ---------------------------------------------------------------------------
# Global fixed nets cache
# ---------------------------------------------------------------------------


class GlobalFixedNets:
    """Cache of fixed random MLPs keyed by (B, p, d), created once and reused.

    When passed to generate_episode, the same x→covariance function is used
    for every episode throughout training instead of re-sampling new MLPs each
    call.  Different (p, d) combinations each get their own persistent MLP set.
    """

    def __init__(self, r: int, hidden: int, device: torch.device | str) -> None:
        self.r = r
        self.hidden = hidden
        self.device = device
        self._cache: dict[tuple, BatchedRandomMLP] = {}

    def get(self, B: int, p: int, d: int) -> BatchedRandomMLP:
        key = (B, p, d)
        if key not in self._cache:
            v_net = BatchedRandomMLP(
                B, p_in=p, p_out=d * self.r, hidden=self.hidden, device=self.device
            )
            self._cache[key] = v_net
        return self._cache[key]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def select_group_representative_indices(
    groups_b: "torch.Tensor | None",
    max_n: int,
    n_total: int | None = None,
) -> list[int]:
    """Return up to max_n indices that cover all unique groups.

    One index per unique group (first occurrence) is selected first, then
    remaining slots are filled in order. Falls back to list(range(min(max_n,
    n_total))) when groups_b is None (non-multimodal episodes).
    """
    if groups_b is None:
        return list(range(min(max_n, n_total or max_n)))
    n_test = groups_b.shape[0]
    groups_np = groups_b.cpu().numpy() if hasattr(groups_b, "cpu") else groups_b
    seen: set[int] = set()
    reps: list[int] = []
    for idx in range(n_test):
        g = int(groups_np[idx])
        if g not in seen:
            seen.add(g)
            reps.append(idx)
        if len(reps) >= max_n:
            break
    if len(reps) < max_n:
        rep_set = set(reps)
        for idx in range(n_test):
            if idx not in rep_set:
                reps.append(idx)
            if len(reps) >= max_n:
                break
    return reps[:max_n]


# ---------------------------------------------------------------------------
# Episode generator
# ---------------------------------------------------------------------------


def generate_episode(
    B: int,
    p: int,
    d: int,
    r: int,
    n_train: int,
    n_test: int,
    device: torch.device | str,
    mlp_hidden: int = 64,
    return_oracle: bool = False,
    fixed_nets: GlobalFixedNets | None = None,
    anchor_gen: GlobalAnchorCovGen | None = None,
    kernel_cov_gen: KernelCovGen | None = None,
    diag_alpha: float | torch.Tensor = 0.0,
    return_norm_stats: bool = False,
    hyperplane_multimodal: bool = False,
    hyperplane_multimodal_scale_lo: float = 0.1,
    hyperplane_multimodal_scale_hi: float = 6.0,
    hyperplane_multimodal_n_groups: int | None = None,
    hyperplane_multimodal_use_mean: bool = False,
    fixed_cov: bool = False,
    fixed_cov_n_anchors: int = 4,
) -> tuple:
    """Generate one training episode (one gradient step worth of data).

    Creates B independent datasets.  Each dataset has its own randomly
    initialised (and frozen) V_net network that defines the ground-truth
    conditional distribution for that dataset.

    The target covariance is low-rank plus diagonal noise:

        Sigma(x_i) = diag(diag_alpha) + V_net(x_i) @ V_net(x_i)^T

    Y is sampled via the reparameterisation:

        eps_low ~ N(0, I_r)
        eps_diag ~ N(0, I_d)
        y_i = diag_alpha^{1/2} * eps_diag_i + V_net(x_i) @ eps_low_i

    Both X (feature-wise) and Y (dimension-wise) are z-normalised across
    all T = n_train + n_test instances within each batch element.

    Args:
        B          : number of independent datasets per episode (batch size)
        p          : feature dimension  (x_i ∈ R^p)
        d          : target dimension   (y_i ∈ R^d)
        r          : low-rank factor    (V_i ∈ R^{d×r})
        n_train    : context (training) set size
        n_test     : query (test) set size
        device     : torch device
        mlp_hidden : hidden layer width for the three random MLPs

    Returns:
        X_train : (B, n_train, p) — z-normalised context features
        Y_train : (B, n_train, d) — z-normalised context targets
        X_test  : (B, n_test, p)  — z-normalised query features
        Y_test  : (B, n_test, d)  — z-normalised query targets

        If return_oracle=True, also returns a dict with keys:
            "mu" : (B, n_test, d) — ground-truth conditional mean (normalised space)
            "D"  : (B, n_test, d) — ground-truth diagonal variance (normalised space)
            "V"  : (B, n_test, d, r) — ground-truth low-rank factor (normalised space)

    """
    T = n_train + n_test

    # 1. Sample x from the tabular prior (z-normalised using train-split stats)
    X = sample_tabular_x(B, T, p, device, n_train=n_train)  # (B, T, p)
    diag_alpha_t = torch.as_tensor(diag_alpha, device=device, dtype=X.dtype)
    if diag_alpha_t.ndim == 1:
        if diag_alpha_t.numel() != B:
            raise ValueError(
                f"diag_alpha has {diag_alpha_t.numel()} values, "
                f"expected one per batch element ({B})"
            )
        diag_alpha_t = diag_alpha_t.view(B, 1, 1)

    groups: torch.Tensor | None = None
    with torch.no_grad():
        if hyperplane_multimodal:
            # K-group multimodal covariance.
            # K is fixed when hyperplane_multimodal_n_groups is set, otherwise random in {2..6}.
            K = (
                int(hyperplane_multimodal_n_groups)
                if hyperplane_multimodal_n_groups is not None
                else int(torch.randint(2, 7, (1,)).item())
            )

            # 6-value log-spaced pool — endpoints controlled by scale_lo/scale_hi
            scale_pool = torch.logspace(
                math.log10(hyperplane_multimodal_scale_lo),
                math.log10(hyperplane_multimodal_scale_hi),
                6,
                device=device,
            )  # (6,)

            # Select K evenly-spread indices: always keeps lo (idx 0) and hi (idx 5)
            pool_idx = torch.linspace(0, 5, K).round().long()  # (K,)
            scales = scale_pool[pool_idx] / math.sqrt(r)       # (K,)

            # Build K covariance structures (D_k, V_k) each with its own scale
            Ds, Vs = [], []
            for k in range(K):
                Ds.append(F.softplus(torch.randn(B, d, device=device)) + 1e-6)    # (B, d)
                Vs.append(torch.randn(B, d, r, device=device) * scales[k].item()) # (B, d, r)

            D_all = torch.stack(Ds, dim=1)  # (B, K, d)
            V_all = torch.stack(Vs, dim=1)  # (B, K, d, r)

            # K random unit-norm normals; assign each instance to its argmax group
            W = F.normalize(torch.randn(B, K, p, device=device), dim=-1)  # (B, K, p)
            scores = torch.einsum("btp,bkp->btk", X, W)                   # (B, T, K)
            groups = scores.argmax(dim=-1)                                  # (B, T)

            # Advanced indexing: for each (b, t), gather covariance of group groups[b, t]
            b_idx = torch.arange(B, device=device)           # (B,)
            diag_x = D_all[b_idx.unsqueeze(1), groups]       # (B, T, d)
            V_x    = V_all[b_idx.unsqueeze(1), groups]       # (B, T, d, r)
            _r = r

            if hyperplane_multimodal_use_mean:
                # Piecewise-constant mean: each group k gets its own random mean vector.
                mu_all = torch.randn(B, K, d, device=device)  # (B, K, d)
                mu_x = mu_all[b_idx.unsqueeze(1), groups]     # (B, T, d)
            else:
                mu_x = torch.zeros(B, T, d, device=device)

        elif fixed_cov:
            # Fixed-per-dataset covariance: D_b and V_b are sampled once per
            # dataset and held constant across all T instances (not x-dependent).
            # Mean is piecewise-constant via Voronoi assignment over anchors.
            # Broadcast diag_alpha correctly: scalar or (B, 1, 1) → (B, 1).
            da = diag_alpha_t.view(B, 1) if diag_alpha_t.numel() == B else diag_alpha_t
            D_b = F.softplus(torch.randn(B, d, device=device)) * da + 1e-6   # (B, d)
            V_b = torch.randn(B, d, r, device=device) / math.sqrt(r)          # (B, d, r)
            diag_x = D_b.unsqueeze(1).expand(B, T, d)     # (B, T, d)
            V_x    = V_b.unsqueeze(1).expand(B, T, d, r)   # (B, T, d, r)
            _r = r
            if fixed_cov_n_anchors > 0:
                n_anch  = fixed_cov_n_anchors
                C       = F.normalize(torch.randn(B, n_anch, p, device=device), dim=-1)  # (B, K, p)
                mu_anch = torch.randn(B, n_anch, d, device=device)                        # (B, K, d)
                dots    = torch.einsum("btp,bkp->btk", X, C)                             # (B, T, K)
                k_star  = dots.argmax(dim=-1)                                              # (B, T)
                b_idx   = torch.arange(B, device=device)
                mu_x    = mu_anch[b_idx.unsqueeze(1), k_star]                            # (B, T, d)
            else:
                mu_x = torch.zeros(B, T, d, device=device)

        elif kernel_cov_gen is not None:
            # Kernel-based: full x-dependent distribution.
            # IsotropicModulatedKernel sets is_copula_gen=True, meaning V_x is
            # the Cholesky of a correlation matrix (diagonal = 1) — no extra
            # diagonal noise should be added so the copula structure is preserved.
            mu_net = BatchedRandomMLP(
                B, p_in=p, p_out=d, hidden=mlp_hidden, device=device
            )
            mu_x = mu_net(X)  # (B, T, d)
            if getattr(kernel_cov_gen, "is_copula_gen", False):
                diag_x = torch.zeros(B, T, d, device=device, dtype=X.dtype)
            else:
                diag_net = BatchedRandomMLP(
                    B, p_in=p, p_out=d, hidden=mlp_hidden, device=device
                )
                diag_x = F.softplus(diag_net(X)) * diag_alpha_t + 1e-6  # (B, T, d)
            V_x = kernel_cov_gen(X, d)  # (B, T, d, d)
            _r = d
        else:
            # 2. Frozen covariance function — anchor-based, globally-fixed MLP, or fresh MLP.
            #    Priority: anchor_gen > fixed_nets > fresh BatchedRandomMLP per episode.
            if anchor_gen is not None:
                v_net = anchor_gen.get(B, p, d)
            elif fixed_nets is not None:
                v_net = fixed_nets.get(B, p, d)
            else:
                v_net = BatchedRandomMLP(
                    B, p_in=p, p_out=d * r, hidden=mlp_hidden, device=device
                )

            # Piecewise constant mean: hard nearest-anchor assignment. Falls back to zero
            # when anchor_gen does not carry mean vectors.
            if anchor_gen is not None and anchor_gen.use_mean:
                mu_x = v_net.get_mean(X)  # (B, T, d)
            else:
                mu_x = torch.zeros(B, T, d, device=device)  # (B, T, d)
            diag_x = (
                torch.ones((B, T, d), device=device, dtype=X.dtype) * diag_alpha_t
            )
            V_x = v_net(X).reshape(B, T, d, r)  # (B, T, d, r)
            _r = r

        # 3. Sample Y via reparameterisation
        eps_diag = torch.randn(B, T, d, device=device)
        eps_low = torch.randn(B, T, _r, device=device)

        Y = (
            mu_x
            + diag_x.sqrt() * eps_diag
            + torch.einsum("btdr,btr->btd", V_x, eps_low)
        )  # (B, T, d)

    # 4. Z-normalise Y dimension-wise across context only — prevents leakage
    Y_train_raw = Y[:, :n_train, :]
    mu_y = Y_train_raw.mean(dim=1, keepdim=True)  # (B, 1, d)
    std_y = Y_train_raw.std(dim=1, keepdim=True) + 1e-8
    Y = (Y - mu_y) / std_y

    # 5. Split into train / test
    X_train, X_test = X[:, :n_train], X[:, n_train:]
    Y_train, Y_test = Y[:, :n_train], Y[:, n_train:]

    if return_oracle:
        # Ground-truth parameters in the normalised Y space.
        # y_norm = (y - mu_y) / std_y  transforms the conditional:
        #   mu*  →  (mu_x  - mu_y) / std_y
        #   D*   →  diag_x / std_y**2
        #   V*   →  V_x   / std_y.unsqueeze(-1)
        mu_oracle = (mu_x - mu_y) / std_y
        D_oracle = (diag_x / (std_y**2)).clamp(min=1e-6)  # Woodbury requires D > 0
        V_oracle = V_x / std_y.unsqueeze(-1)
        oracle = {
            "mu": mu_oracle[:, n_train:].detach(),
            "D": D_oracle[:, n_train:].detach(),
            "V": V_oracle[:, n_train:].detach(),
            "mu_train": mu_oracle[:, :n_train].detach(),
            "D_train": D_oracle[:, :n_train].detach(),
            "V_train": V_oracle[:, :n_train].detach(),
        }
        if groups is not None:
            oracle["groups"] = groups[:, n_train:].detach()  # (B, n_test)
        if return_norm_stats:
            oracle["mu_y"] = mu_y.detach()  # (B, 1, d) in raw Y space (pre-norm)
            oracle["std_y"] = std_y.detach()  # (B, 1, d) in raw Y space (pre-norm)
        return X_train, Y_train, X_test, Y_test, oracle

    if return_norm_stats:
        norm_stats = {"mu_y": mu_y.detach(), "std_y": std_y.detach()}
        return X_train, Y_train, X_test, Y_test, norm_stats

    return X_train, Y_train, X_test, Y_test


# ---------------------------------------------------------------------------
# Validation suite builder
# ---------------------------------------------------------------------------


def build_val_suite(
    cfg,
    device: torch.device | str,
    fixed_nets: GlobalFixedNets | None = None,
    anchor_gen: GlobalAnchorCovGen | None = None,
    kernel_cov_gen: KernelCovGen | None = None,
) -> dict[str, dict]:
    """Pre-generate a fixed validation suite.

    Creates one episode per (d, n_train) grid point.  Each entry stores the
    episode tensors together with the episode metadata.

    Args:
        cfg    : Hydra DictConfig with cfg.data and cfg.training fields
        device : torch device

    Returns:
        suite : dict keyed by "d{d}_p{n_train}" whose values are dicts with
                keys X_train, Y_train, X_test, Y_test, d, p, n_train, n_test.
    """
    suite: dict[str, dict] = {}
    B = int(cfg.data.val_batch)
    n_test = int(cfg.data.val_n_test)
    r = int(cfg.data.r_data)
    hidden = int(cfg.data.mlp_hidden)
    diag_alpha_range = cfg.data.get("diag_alpha_range", [0.05, 2.0])
    diag_alpha_lo = float(diag_alpha_range[0])
    diag_alpha_hi = float(diag_alpha_range[1])

    for d in cfg.data.val_d_list:
        for n_train in cfg.data.val_n_train_list:
            # Feature dimension p: sample randomly in p_range for each grid point
            p_lo, p_hi = cfg.data.p_range
            p = int(torch.randint(int(p_lo), int(p_hi) + 1, ()).item())
            diag_alpha = float(
                torch.empty((), device=device)
                .uniform_(diag_alpha_lo, diag_alpha_hi)
                .item()
            )

            key = f"d{d}_p{n_train}"
            X_tr, Y_tr, X_te, Y_te, oracle = generate_episode(
                B,
                p,
                d,
                r,
                n_train,
                n_test,
                device,
                mlp_hidden=hidden,
                return_oracle=True,
                fixed_nets=fixed_nets,
                anchor_gen=anchor_gen,
                kernel_cov_gen=kernel_cov_gen,
                diag_alpha=diag_alpha,
            )
            suite[key] = {
                "X_train": X_tr,
                "Y_train": Y_tr,
                "X_test": X_te,
                "Y_test": Y_te,
                "oracle_mu": oracle["mu"],
                "oracle_D": oracle["D"],
                "oracle_V": oracle["V"],
                "d": d,
                "p": p,
                "n_train": n_train,
                "n_test": n_test,
            }
    return suite
