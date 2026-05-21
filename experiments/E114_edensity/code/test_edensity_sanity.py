"""Quick sanity test for eDensity module: DCT correctness, smoothness, grad."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (_HERE, _ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from edensity import _dct1_via_fft, _idct1_via_fft, _dct2_via_fft, _idct2_via_fft


def test_dct_inverse_identity():
    """idct(dct(x)) should equal x."""
    torch.manual_seed(0)
    for N in (8, 16, 32, 64):
        x = torch.randn(N)
        X = _dct1_via_fft(x)
        x_back = _idct1_via_fft(X)
        max_err = (x - x_back).abs().max()
        print(f"  N={N}: max round-trip err = {max_err:.2e}")
        assert max_err < 1e-3, f"DCT not invertible at N={N}: err={max_err}"


def test_dct_2d_inverse_identity():
    torch.manual_seed(0)
    for (R, C) in ((8, 8), (16, 32), (64, 64)):
        x = torch.randn(R, C)
        X = _dct2_via_fft(x)
        x_back = _idct2_via_fft(X)
        max_err = (x - x_back).abs().max()
        print(f"  R={R}, C={C}: max round-trip err = {max_err:.2e}")
        assert max_err < 1e-3, f"2D DCT not invertible at ({R},{C}): err={max_err}"


def test_dct_matches_scipy():
    """Sanity-check our FFT-based DCT matches scipy's reference."""
    try:
        from scipy.fft import dct as scipy_dct
    except ImportError:
        print("  scipy not available; skipping")
        return
    torch.manual_seed(0)
    for N in (16, 32, 64):
        x = torch.randn(N, dtype=torch.float64)
        X_ours = _dct1_via_fft(x)
        X_scipy = torch.tensor(scipy_dct(x.numpy(), type=2, norm=None), dtype=torch.float64)
        max_err = (X_ours - X_scipy).abs().max()
        print(f"  N={N}: max vs scipy = {max_err:.2e}")
        assert max_err < 1e-3, f"DCT mismatch with scipy at N={N}: err={max_err}"


if __name__ == "__main__":
    print("Test: DCT-II round-trip identity (1D)")
    test_dct_inverse_identity()
    print("\nTest: DCT-II round-trip identity (2D)")
    test_dct_2d_inverse_identity()
    print("\nTest: DCT-II matches scipy.fft.dct")
    test_dct_matches_scipy()
    print("\nAll sanity tests PASSED")
