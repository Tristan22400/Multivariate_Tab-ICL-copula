import torch
import sys
import os

# Path setup
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from data_gen import KernelCovarianceGen

def test_kernel_gen():
    B, d = 2, 4
    kernel_type = "gaussian"
    ls_range = (0.2, 0.2)
    device = "cpu"
    
    gen = KernelCovarianceGen(B, d, kernel_type, ls_range, device)
    
    # Check V shape
    print(f"V shape: {gen.V.shape}") # Expected (B, d, d)
    assert gen.V.shape == (B, d, d)
    
    # Check if Sigma = V @ V^T has the right properties
    Sigma = gen.V @ gen.V.transpose(-1, -2)
    print(f"Sigma[0]:\n{Sigma[0]}")
    
    # For a gaussian kernel with ls=0.2 and d=4, points at [0, 0.33, 0.66, 1.0]
    # distance between adjacent points is 0.33.
    # K(0, 0.33) = exp(-0.5 * (0.33/0.2)^2) = exp(-0.5 * 1.65^2) = exp(-1.36) approx 0.25
    
    # Diagonal should be approx 1 (plus jitter)
    assert torch.allclose(Sigma.diagonal(dim1=-1, dim2=-2), torch.ones(B, d), atol=1e-5)
    
    # Off-diagonal should be positive
    assert (Sigma[0, 0, 1] > 0).item()
    print(f"Sigma[0, 0, 1]: {Sigma[0, 0, 1].item():.4f}")

    # Test __call__
    X = torch.randn(B, 10, 20)
    out = gen(X)
    print(f"Call output shape: {out.shape}") # Expected (B, 10, d*d)
    assert out.shape == (B, 10, d*d)

if __name__ == "__main__":
    test_kernel_gen()
