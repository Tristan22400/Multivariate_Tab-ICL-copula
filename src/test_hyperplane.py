"""Quick correctness check for the hyperplane_bimodal data-generation mode."""
import sys
sys.path.insert(0, 'src')
import torch
from data_gen import generate_episode

torch.manual_seed(42)

B, p, d, r = 4, 20, 8, 4
n_train, n_test = 128, 32

# Need to re-implement the hyperplane assignment to check training balance.
# We call generate_episode with return_norm_stats=False (default), which
# only returns oracle for test instances.  To verify training balance we
# reconstruct group membership from the per-instance oracle_D values:
# every instance in a dataset takes exactly one of two D vectors (D1 or D2),
# so we can cluster by rounding oracle_D to identify the two groups.

X_tr, Y_tr, X_te, Y_te, oracle = generate_episode(
    B=B, p=p, d=d, r=r,
    n_train=n_train, n_test=n_test,
    device='cpu',
    return_oracle=True,
    hyperplane_bimodal=True,
    hyperplane_bimodal_scale_lo=0.3,
    hyperplane_bimodal_scale_hi=3.0,
)

print("=== Shape checks ===")
for name, t in [("X_train", X_tr), ("Y_train", Y_tr),
                ("X_test",  X_te), ("Y_test",  Y_te),
                ("oracle_D", oracle['D']), ("oracle_V", oracle['V'])]:
    print(f"  {name}: {tuple(t.shape)}")

# ---------- V-norm bimodality (key correctness signal) ----------
print("\n=== oracle_V Frobenius-norm per test instance ===")
print("    (should show exactly 2 distinct values per dataset)")
for b in range(B):
    V_norms = oracle['V'][b].norm(dim=(-2, -1))   # (n_test,)
    unique = V_norms.unique().tolist()
    counts = [(V_norms == u).sum().item() for u in unique]
    print(f"  dataset {b}: unique norms = {[f'{u:.3f}' for u in unique]}  "
          f"counts = {counts}  (total {n_test})")

# ---------- Balance: fraction in each group for test set ----------
print("\n=== Test-set group balance ===")
print("    (balance is guaranteed in train; test may deviate slightly)")
for b in range(B):
    V_norms = oracle['V'][b].norm(dim=(-2, -1))
    lo_val = V_norms.unique().min()
    frac_weak = (V_norms == lo_val).float().mean().item()
    print(f"  dataset {b}: weak={frac_weak:.2f}  strong={1-frac_weak:.2f}")

# ---------- Correlation magnitude per group ----------
print("\n=== Mean |off-diagonal| correlation per group ===")
print("    (group2/strong should be clearly larger than group1/weak)")
for b in range(B):
    D_b = oracle['D'][b]        # (n_test, d)
    V_b = oracle['V'][b]        # (n_test, d, r)

    Sigma = torch.diag_embed(D_b) + V_b @ V_b.transpose(-1, -2)
    std = Sigma.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
    R = Sigma / (std.unsqueeze(-1) * std.unsqueeze(-2))

    V_norms = V_b.norm(dim=(-2, -1))
    lo_val = V_norms.unique().min()
    g1 = V_norms == lo_val      # weak
    g2 = ~g1                     # strong

    ri, ci = torch.triu_indices(d, d, offset=1)
    def mean_offdiag(mask):
        if not mask.any():
            return float('nan')
        return R[mask][:, ri, ci].abs().mean().item()

    print(f"  dataset {b}: group1 (weak)   mean|r_ij| = {mean_offdiag(g1):.3f}")
    print(f"  dataset {b}: group2 (strong) mean|r_ij| = {mean_offdiag(g2):.3f}")

# ---------- Y normalisation ----------
print("\n=== Y_train z-normalisation (mean≈0, std≈1) ===")
for b in range(B):
    print(f"  dataset {b}: mean={Y_tr[b].mean():.4f}  std={Y_tr[b].std():.4f}")

# ---------- Oracle D diagonal sanity ----------
print("\n=== oracle_D: min and max across test instances ===")
print("    (D should be positive and take exactly 2 distinct row-vectors per dataset)")
for b in range(B):
    D_b = oracle['D'][b]
    unique_rows = D_b.unique(dim=0)
    print(f"  dataset {b}: {unique_rows.shape[0]} unique D vectors  "
          f"(min={D_b.min():.4f}, max={D_b.max():.4f})")

print("\nAll checks complete.")
