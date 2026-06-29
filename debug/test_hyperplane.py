"""Quick correctness check for the hyperplane multimodal data-generation mode."""
import sys
sys.path.insert(0, 'src')
import torch
from data_gen import generate_episode

torch.manual_seed(42)

B, p, d, r = 4, 20, 8, 4
n_train, n_test = 128, 32

# generate_episode with hyperplane_multimodal=True now produces K groups per episode
# (K ~ Uniform{2,...,6}).  Each instance gets one of K covariance structures,
# so oracle_D has between 2 and 6 distinct row-vectors per dataset.

X_tr, Y_tr, X_te, Y_te, oracle = generate_episode(
    B=B, p=p, d=d, r=r,
    n_train=n_train, n_test=n_test,
    device='cpu',
    return_oracle=True,
    hyperplane_multimodal=True,
    hyperplane_multimodal_scale_lo=0.1,
    hyperplane_multimodal_scale_hi=6.0,
)

print("=== Shape checks ===")
for name, t in [("X_train", X_tr), ("Y_train", Y_tr),
                ("X_test",  X_te), ("Y_test",  Y_te),
                ("oracle_D", oracle['D']), ("oracle_V", oracle['V'])]:
    print(f"  {name}: {tuple(t.shape)}")

# ---------- V-norm multimodality (key correctness signal) ----------
print("\n=== oracle_V Frobenius-norm per test instance ===")
print("    (should show between 2 and 6 distinct values per dataset)")
for b in range(B):
    V_norms = oracle['V'][b].norm(dim=(-2, -1))   # (n_test,)
    unique = V_norms.unique().tolist()
    counts = [(V_norms == u).sum().item() for u in unique]
    n_unique = len(unique)
    ok = 2 <= n_unique <= 6
    print(f"  dataset {b}: {n_unique} distinct norms = {[f'{u:.3f}' for u in unique]}  "
          f"counts = {counts}  {'OK' if ok else 'FAIL'}")
    assert ok, f"Expected 2-6 distinct groups, got {n_unique}"

# ---------- Correlation magnitude per group ----------
print("\n=== Mean |off-diagonal| correlation per group ===")
print("    (groups with higher V-norm should show larger off-diagonal correlations)")
for b in range(B):
    D_b = oracle['D'][b]        # (n_test, d)
    V_b = oracle['V'][b]        # (n_test, d, r)

    Sigma = torch.diag_embed(D_b) + V_b @ V_b.transpose(-1, -2)
    std = Sigma.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    R = Sigma / (std.unsqueeze(-1) * std.unsqueeze(-2))

    V_norms = V_b.norm(dim=(-2, -1))
    unique_norms = V_norms.unique().tolist()

    ri, ci = torch.triu_indices(d, d, offset=1)
    for k, u in enumerate(unique_norms):
        mask = V_norms == u
        if mask.any():
            mean_r = R[mask][:, ri, ci].abs().mean().item()
            print(f"  dataset {b}  group {k} (V-norm={u:.3f}): mean|r_ij| = {mean_r:.3f}")

# ---------- Y normalisation ----------
print("\n=== Y_train z-normalisation (mean≈0, std≈1) ===")
for b in range(B):
    print(f"  dataset {b}: mean={Y_tr[b].mean():.4f}  std={Y_tr[b].std():.4f}")

# ---------- Oracle D diagonal sanity ----------
print("\n=== oracle_D: distinct row-vectors per dataset ===")
print("    (D positive, between 2 and 6 unique row-vectors)")
for b in range(B):
    D_b = oracle['D'][b]
    unique_rows = D_b.unique(dim=0)
    n_unique = unique_rows.shape[0]
    ok = 2 <= n_unique <= 6
    print(f"  dataset {b}: {n_unique} unique D vectors  "
          f"(min={D_b.min():.4f}, max={D_b.max():.4f})  {'OK' if ok else 'FAIL'}")
    assert ok, f"Expected 2-6 unique D rows, got {n_unique}"

print("\nAll checks complete.")
