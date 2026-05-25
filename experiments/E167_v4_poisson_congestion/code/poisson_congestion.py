"""E146 — Poisson-FFT congestion proxy (DCGP-lite).

Inspired by DCGP (DAC'25): solve Poisson equation ∇²φ = -ρ on net-demand
grid via FFT. Returns scalar congestion estimate = top-5% mean of |φ|.

Differentiable (PyTorch rfft2/irfft2 support autograd). Uses even-symmetric
reflection padding for approximate Neumann boundary conditions (gradient = 0
at canvas edges).

Mathematically: combined V+H route demand acts as charge density; Poisson
potential φ gives "pressure field". High-pressure regions concentrate
routing demand. Different signal than ABU-5% which is purely local.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def poisson_potential(demand: torch.Tensor) -> torch.Tensor:
    """Solve ∇²φ = -ρ on a 2D grid with approximate Neumann BC.

    Args:
        demand: [gr, gc] non-negative charge density (route demand).

    Returns:
        [gr, gc] potential field φ. Differentiable w.r.t. demand.
    """
    gr, gc = demand.shape

    # Even-symmetric reflection padding → cosine transform via FFT.
    # Reflects demand across both axes to enforce zero normal gradient at
    # original boundaries (Neumann BC), which is more physically right than
    # zero-pad's Dirichlet BC.
    pad_v = torch.flip(demand, dims=[0])
    bot = torch.cat([demand, pad_v], dim=0)  # [2gr, gc]
    pad_h = torch.flip(bot, dims=[1])
    full = torch.cat([bot, pad_h], dim=1)  # [2gr, 2gc]

    rho_hat = torch.fft.rfft2(full)  # complex [2gr, gc+1]

    # Spectral Laplacian eigenvalues (in units where Δx=Δy=1)
    ky = torch.fft.fftfreq(2 * gr, d=1.0).to(demand.device) * 2 * math.pi
    kx = torch.fft.rfftfreq(2 * gc, d=1.0).to(demand.device) * 2 * math.pi
    k_sq = ky[:, None] ** 2 + kx[None, :] ** 2  # [2gr, gc+1] real

    # Avoid div by zero at DC component
    k_sq_safe = torch.where(k_sq > 1e-20, k_sq, torch.ones_like(k_sq))
    phi_hat = -rho_hat / k_sq_safe

    # Zero out DC component (gauge — potential defined up to constant)
    phi_hat = phi_hat.clone()
    phi_hat[0, 0] = 0.0

    phi_full = torch.fft.irfft2(phi_hat, s=(2 * gr, 2 * gc))
    phi = phi_full[:gr, :gc]
    return phi


def poisson_congestion_scalar(
    V_total: torch.Tensor,
    H_total: torch.Tensor,
    top_k_frac: float = 0.05,
) -> torch.Tensor:
    """Scalar Poisson-congestion proxy from V/H route demand grids.

    Args:
        V_total: [gr, gc] vertical route demand (normalized by capacity).
        H_total: [gr, gc] horizontal route demand (normalized by capacity).
        top_k_frac: fraction for top-k mean.

    Returns:
        Scalar tensor. Differentiable w.r.t. V_total, H_total.
    """
    demand = V_total + H_total
    phi = poisson_potential(demand)
    flat = phi.abs().flatten()
    k = max(1, int(top_k_frac * flat.numel()))
    top_k, _ = torch.topk(flat, k)
    return top_k.mean()


if __name__ == "__main__":
    # Sanity test: known-demand pattern → expected potential shape
    torch.manual_seed(42)
    demand = torch.zeros(32, 32)
    demand[15:17, 15:17] = 1.0  # point source at center
    phi = poisson_potential(demand)
    print(f"Potential shape: {phi.shape}")
    print(f"Potential range: [{phi.min().item():.4e}, {phi.max().item():.4e}]")
    print(f"Center potential: {phi[16, 16].item():.4e}")
    print(f"Corner potential: {phi[0, 0].item():.4e}")
    # Differentiability check
    d = demand.clone().requires_grad_(True)
    s = poisson_congestion_scalar(d, torch.zeros_like(d))
    s.backward()
    print(f"Grad max: {d.grad.abs().max().item():.4e}")
    print(f"Grad nonzero: {(d.grad.abs() > 0).sum().item()} of {d.numel()}")
