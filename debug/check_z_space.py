"""
check_z_space.py — Diagnose whether Z values in pit_hyperplane_debug are N(0,1).

If TabICL correctly estimates marginal CDFs on the synthetic data, then:
  u_{i,j} = F̂_j(y_{i,j} | context) ~ Uniform(0,1)
  z_{i,j} = Φ⁻¹(u_{i,j})           ~ N(0,1)

Deviations from N(0,1) indicate TabICL is failing on the synthetic marginals,
making the Gaussian copula training mathematically invalid.

Output: debug/z_space_diagnosis.png + printed stats table.
"""

import glob
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.special as special
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "pit_hyperplane_debug")
OUT_PATH = os.path.join(REPO_ROOT, "debug", "z_space_diagnosis.png")
N_EPISODES = 50
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Step 1: Load Z values from a sample of episodes
# ---------------------------------------------------------------------------

random.seed(RANDOM_SEED)
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "episode_*.pt")))
if not all_files:
    sys.exit(f"No episode files found in {DATA_DIR}")

sample_files = random.sample(all_files, min(N_EPISODES, len(all_files)))
print(f"Loading {len(sample_files)} episodes from {DATA_DIR} ...")

z_train_list, z_test_list = [], []
for f in sample_files:
    ep = torch.load(f, map_location="cpu", weights_only=False)
    z_train_list.append(ep["Z_train"].float())  # (B, N_train, d)
    z_test_list.append(ep["Z_test"].float())    # (B, N_test,  d)

Z_train_nd = torch.cat(z_train_list, dim=0)   # (B*N_eps, N_train, d)
Z_test_nd  = torch.cat(z_test_list,  dim=0)   # (B*N_eps, N_test,  d)
d = Z_train_nd.shape[-1]

Z_train_flat = Z_train_nd.reshape(-1).numpy()
Z_test_flat  = Z_test_nd.reshape(-1).numpy()
Z_train_2d   = Z_train_nd.reshape(-1, d).numpy()   # (N_total, d)

print(f"Z_train total values : {Z_train_flat.size:,}")
print(f"Z_test  total values : {Z_test_flat.size:,}")

# ---------------------------------------------------------------------------
# Step 5: Statistical tests (done early so we can annotate plots)
# ---------------------------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

def ks_test(z_flat, n=5000, label="Z"):
    sub = rng.choice(z_flat, size=min(n, len(z_flat)), replace=False)
    ks_stat, ks_p = stats.kstest(sub, "norm")
    sk = stats.skew(sub)
    ku = stats.kurtosis(sub)  # excess kurtosis; N(0,1) → 0
    mu, sigma = sub.mean(), sub.std()
    print(f"\n{label} stats (n={len(sub):,}):")
    print(f"  mean={mu:.4f}  std={sigma:.4f}  skew={sk:.4f}  excess_kurt={ku:.4f}")
    print(f"  KS stat={ks_stat:.4f}  p={ks_p:.4e}  {'PASS' if ks_p>0.05 else 'FAIL'}")
    return mu, sigma, sk, ku, ks_stat, ks_p

mu_tr, std_tr, sk_tr, ku_tr, ks_s_tr, ks_p_tr = ks_test(Z_train_flat, label="Z_train")
mu_te, std_te, sk_te, ku_te, ks_s_te, ks_p_te = ks_test(Z_test_flat,  label="Z_test")

# Per-dimension stats
print("\nPer-dimension stats (Z_train):")
print(f"{'dim':>4}  {'mean':>8}  {'std':>8}  {'skew':>8}  {'kurt':>8}  {'KS_p':>10}")
for j in range(d):
    col = Z_train_2d[:, j]
    sub = rng.choice(col, size=min(5000, len(col)), replace=False)
    m, s = sub.mean(), sub.std()
    sk = stats.skew(sub)
    ku = stats.kurtosis(sub)
    _, p = stats.kstest(sub, "norm")
    flag = "" if p > 0.05 else "  <-- FAIL"
    print(f"{j:>4}  {m:>8.4f}  {s:>8.4f}  {sk:>8.4f}  {ku:>8.4f}  {p:>10.2e}{flag}")

# ---------------------------------------------------------------------------
# Build figure: 4 rows × (d//2 + extras) columns
# Layout:
#   Row 0: global histogram (train) + QQ-plot (train)
#   Row 1: global histogram (test)  + u-space histogram (train)
#   Row 2: per-dim histograms (d=8 → 8 subplots across 2 rows)
#   Row 3: (continued)
# ---------------------------------------------------------------------------

x_ref = np.linspace(-5, 5, 400)
pdf_ref = stats.norm.pdf(x_ref)

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Z-space diagnosis — pit_hyperplane_debug", fontsize=14, fontweight="bold")

gs = fig.add_gridspec(4, 4, hspace=0.45, wspace=0.35)

# --- Row 0 left: global Z_train histogram ---
ax0 = fig.add_subplot(gs[0, :2])
ax0.hist(Z_train_flat, bins=200, density=True, alpha=0.6, color="steelblue", label="Z_train")
ax0.plot(x_ref, pdf_ref, "r-", lw=2, label="N(0,1)")
ax0.set_title(
    f"Z_train global  (n={Z_train_flat.size:,})\n"
    f"μ={mu_tr:.3f}  σ={std_tr:.3f}  skew={sk_tr:.3f}  kurt={ku_tr:.3f}  "
    f"KS p={ks_p_tr:.2e}  {'✓' if ks_p_tr>0.05 else '✗ FAIL'}"
)
ax0.legend(fontsize=8)
ax0.set_xlim(-6, 6)

# --- Row 0 right: QQ-plot Z_train ---
ax1 = fig.add_subplot(gs[0, 2:])
sub_qq = rng.choice(Z_train_flat, size=min(5000, len(Z_train_flat)), replace=False)
(osm, osr), (slope, intercept, r) = stats.probplot(sub_qq, dist="norm")
ax1.scatter(osm, osr, s=2, alpha=0.4, color="steelblue")
ax1.plot(osm, slope * np.array(osm) + intercept, "r-", lw=1.5, label=f"r={r:.4f}")
ax1.set_title("QQ-plot Z_train vs N(0,1)")
ax1.set_xlabel("Theoretical quantiles")
ax1.set_ylabel("Sample quantiles")
ax1.legend(fontsize=8)

# --- Row 1 left: global Z_test histogram ---
ax2 = fig.add_subplot(gs[1, :2])
ax2.hist(Z_test_flat, bins=200, density=True, alpha=0.6, color="darkorange", label="Z_test")
ax2.plot(x_ref, pdf_ref, "r-", lw=2, label="N(0,1)")
ax2.set_title(
    f"Z_test global  (n={Z_test_flat.size:,})\n"
    f"μ={mu_te:.3f}  σ={std_te:.3f}  skew={sk_te:.3f}  kurt={ku_te:.3f}  "
    f"KS p={ks_p_te:.2e}  {'✓' if ks_p_te>0.05 else '✗ FAIL'}"
)
ax2.legend(fontsize=8)
ax2.set_xlim(-6, 6)

# --- Row 1 right: u-space (back-transform Z_train via Φ) ---
ax3 = fig.add_subplot(gs[1, 2:])
u_vals = special.ndtr(Z_train_flat)  # Φ(z) maps N(0,1) → Uniform(0,1)
ax3.hist(u_vals, bins=50, density=True, alpha=0.6, color="steelblue")
ax3.axhline(1.0, color="r", lw=2, label="Uniform(0,1)")
ax3.set_title("u = Φ(Z_train) — should be Uniform(0,1)")
ax3.set_xlabel("u")
ax3.set_ylabel("density")
ax3.legend(fontsize=8)

# --- Rows 2-3: per-dimension histograms ---
for j in range(d):
    row = 2 + j // 4
    col = j % 4
    ax = fig.add_subplot(gs[row, col])
    col_data = Z_train_2d[:, j]
    ax.hist(col_data, bins=80, density=True, alpha=0.6, color="steelblue")
    ax.plot(x_ref, pdf_ref, "r-", lw=1.2)
    sub = rng.choice(col_data, size=min(5000, len(col_data)), replace=False)
    _, p = stats.kstest(sub, "norm")
    ax.set_title(
        f"dim {j}  μ={col_data.mean():.2f} σ={col_data.std():.2f}\n"
        f"KS p={p:.2e} {'✓' if p>0.05 else '✗'}",
        fontsize=8,
    )
    ax.set_xlim(-6, 6)
    ax.tick_params(labelsize=7)

plt.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
print(f"\nFigure saved to {OUT_PATH}")
