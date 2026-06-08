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
                math.log(0.1) + (math.log(10.0) - math.log(0.1)) * torch.rand(1).item()
            )
            K = s * (1.0 + sq_d / (2 * alpha * l**2)).pow(-alpha)
        elif kernel == "periodic":
            p_period = math.exp(
                math.log(0.5) + (math.log(5.0) - math.log(0.5)) * torch.rand(1).item()
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

    with torch.no_grad():
        if hyperplane_multimodal:
            # K-group multimodal covariance: K ~ Uniform{2,...,6}.
            # A pool of 6 log-spaced scales between scale_lo and scale_hi is built;
            # K evenly-spread scales are selected so the extremes are always included.
            # Each instance is assigned to the group whose random hyperplane normal
            # it projects onto most strongly (argmax over K projections).
            K = int(torch.randint(2, 7, (1,)).item())  # K in {2, 3, 4, 5, 6}

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
        }
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
