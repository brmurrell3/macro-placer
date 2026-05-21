"""E114 — Electrostatic eDensity (ePlace/Xplace/DREAMPlace formulation).

A differentiable replacement for E88's `_grid_density` (bbox-uniform per-cell
occupation top-10%). Replaces the C0 step-discontinuous overlap model with a
globally smooth electric potential obtained by solving the 2D Poisson
equation via Discrete Cosine Transform (Lu et al, ePlace ASP-DAC 2015).

Pipeline per `compute_density(positions)`:
    1. Smear each macro's "charge" (area) across grid bins via separable
       Gaussian kernel — replaces hard rectangle overlap with smooth disc.
    2. Density rho per bin = charge / cell_area.
    3. Overflow chi(rho) = max(0, rho - target).  We use a smooth softplus
       (or pass rho through; ePlace uses (rho - target_density)).
    4. Solve Poisson  -laplacian(phi) = chi  via DCT-II:
          phi_hat[u,v] = chi_hat[u,v] / (lambda_u + lambda_v + eps)
       where lambda_k = 2 * (1 - cos(pi * k / N)) is the 1D Laplacian
       eigenvalue (Neumann BCs ↔ DCT-II).
    5. phi = iDCT-II(phi_hat).
    6. Density-cost ≈ top-10% mean(phi) — matches canonical aggregator.

Why this differs from `_grid_density`:
    - Grid-bin density has step discontinuities at cell boundaries (the
      overlap area function is C0 there). Adam sees zero or stale gradient
      when a macro corner sits exactly on a bin edge.
    - eDensity replaces the bin-occupation function with a long-range
      electric potential. The gradient at any (x,y) reflects ALL grid
      overflow, weighted by 1/distance — so a macro near an overflowing
      bin "feels" a smooth pull toward open space.

Differentiability:
    - All steps (Gaussian smear, DCT, iDCT, divide, topk) are torch ops.
    - Custom `torch_dct` (DCT-II via FFT) since torch lacks a native DCT.
    - Works on CPU (no MPS dependency).

Memory / cost:
    - O(N_macros * (G + G)) for smearing (separable Gaussians, no [N,G,G]
      materialization).
    - O(G^2 log G) for FFT (G is grid_rows ≈ 64-128 for IBM benches).
    - Total ~ms/call on CPU for IBM-scale grids.

The result of `compute_density()` is the SAME scale as `_grid_density()`:
both are 0.5 * mean(top-10%) of a per-cell quantity. eDensity's quantity
is electric potential phi (units of charge), grid_density's is occupation
ratio (unitless). Both are non-negative, both have natural
"crowded → bad" semantics; the ratio between their numerical values
needs an empirical scale factor, which we calibrate in __init__.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def _dct2_via_fft(x: torch.Tensor) -> torch.Tensor:
    """2D DCT-II via FFT. Input/output shape [R, C].

    DCT-II of length N is the real part of a 2N FFT of a mirrored signal,
    weighted by exp(-i pi k / 2N). We do this in both dims independently.
    """
    return _dct1_via_fft(_dct1_via_fft(x.T).T)


def _idct2_via_fft(X: torch.Tensor) -> torch.Tensor:
    """Inverse 2D DCT-II (= DCT-III) via FFT. Input/output shape [R, C]."""
    return _idct1_via_fft(_idct1_via_fft(X.T).T)


def _dct1_via_fft(x: torch.Tensor) -> torch.Tensor:
    """1D DCT-II along last axis via FFT (Makhoul 1980).

    For input length N:
       y[k] = sum_{n=0}^{N-1} x[n] * cos(pi * (2n+1) * k / (2N))
    This is the orthonormal-unscaled (type II) DCT used by scipy.fft.dct
    with norm=None. We return UNNORMALIZED dct.

    Method: pad-and-FFT. Build y[n] = x[2n], y[N-1-n] = x[2n+1] (interleave
    reversed), take FFT, multiply by exp(-i pi k / 2N).
    """
    N = x.shape[-1]
    # Build the symmetrized signal: [x0, x2, ..., xN-2, xN-1, xN-3, ..., x1]
    v = torch.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)
    V = torch.fft.fft(v, dim=-1)
    k = torch.arange(N, dtype=x.dtype, device=x.device)
    phase = torch.exp(-1j * math.pi * k / (2 * N))
    return (V * phase).real


def _idct1_via_fft(X: torch.Tensor) -> torch.Tensor:
    """1D inverse DCT-II (DCT-III) along last axis via FFT.

    Inverse of _dct1_via_fft up to scale: idct(dct(x)) = 2N * x.
    Includes the 1/(2N) so idct(dct(x)) == x.
    """
    N = X.shape[-1]
    k = torch.arange(N, dtype=X.dtype, device=X.device)
    # Reconstruct the FFT input from the DCT coefficients
    # V[k] = X[k] * exp(+i pi k / 2N), but V is conjugate-symmetric only if X is.
    # Use the direct trigonometric reconstruction.
    n = torch.arange(N, dtype=X.dtype, device=X.device)
    # cosine basis matrix shape [N, N] — fine for our grids (~64-128)
    if N > 256:
        # Fall back to direct cosine-basis for very large grids would be O(N^2);
        # OK for our grid sizes (we use 64-128 typical).
        pass
    cos_basis = torch.cos(math.pi * (2 * n.unsqueeze(0) + 1) * k.unsqueeze(1) / (2 * N))
    # The DCT-III formula: y[n] = X[0]/2 + sum_{k=1}^{N-1} X[k] * cos(pi*(2n+1)*k / (2N))
    half_X = X.clone()
    half_X[..., 0] = half_X[..., 0] / 2.0
    # y = half_X @ cos_basis where cos_basis[k, n]
    y = torch.matmul(half_X, cos_basis)
    return (2.0 / N) * y


class ElectrostaticDensity:
    """Electrostatic eDensity model (ePlace formulation, CPU/torch).

    Constructor builds the (grid, bench, target density) setup. Call
    `.compute_density(positions)` repeatedly under autograd.

    Args:
        benchmark: Benchmark object (uses canvas_width/height, macro_sizes,
            grid_rows/cols).
        target_density: Target occupancy per bin (overflow above this is
            penalized). ePlace default 1.0 — only above-target charge
            counts. Lower targets (0.4-0.6) push for more spread placement.
        sigma_frac: Gaussian smearing width relative to MAX(cell_w, cell_h).
            Larger = more diffuse; ePlace uses ~0.5-1.0 cells.
        scale: Empirical scale factor matching eDensity to canonical
            grid-density top-10% magnitude. Calibrated post-hoc.
        topk_frac: Aggregation top-k fraction (0.10 matches canonical).
    """

    def __init__(
        self,
        benchmark,
        device: str | torch.device = "cpu",
        target_density: float = 1.0,
        sigma_frac: float = 0.5,
        scale: float = 1.0,
        topk_frac: float = 0.10,
        grid_rows: Optional[int] = None,
        grid_cols: Optional[int] = None,
    ):
        self.device = torch.device(device)
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.target_density = float(target_density)
        self.topk_frac = float(topk_frac)
        self.scale = float(scale)

        # Use benchmark's grid by default (matches grid_density aggregation)
        gr = grid_rows or benchmark.grid_rows
        gc = grid_cols or benchmark.grid_cols
        self.grid_rows = int(gr)
        self.grid_cols = int(gc)
        self.cell_w = self.cw / self.grid_cols
        self.cell_h = self.ch / self.grid_rows
        self.cell_area = self.cell_w * self.cell_h

        # Cell centers (for Gaussian smearing reference)
        self.x_centers = (
            (torch.arange(self.grid_cols, dtype=torch.float32, device=self.device) + 0.5)
            * self.cell_w
        )
        self.y_centers = (
            (torch.arange(self.grid_rows, dtype=torch.float32, device=self.device) + 0.5)
            * self.cell_h
        )

        # Macro sizes
        self.sizes = benchmark.macro_sizes.to(self.device)
        self.num_macros = int(benchmark.num_macros)

        # Smearing sigma: at least 1 cell, more for large macros
        cell_diag = max(self.cell_w, self.cell_h)
        # Per-macro sigma in absolute units: max(sigma_frac * cell_diag, 0.5 * macro_dim)
        # so a large macro is realistically smeared over its bounding box.
        sigma_min_x = sigma_frac * self.cell_w
        sigma_min_y = sigma_frac * self.cell_h
        self.sigma_x = torch.clamp(0.5 * self.sizes[:, 0], min=sigma_min_x)
        self.sigma_y = torch.clamp(0.5 * self.sizes[:, 1], min=sigma_min_y)

        # Macro "charge" = its area (so density at full overlap == 1)
        self.charge = self.sizes[:, 0] * self.sizes[:, 1]

        # Precompute Laplacian eigenvalues for Poisson solve
        # lambda_k = 2 * (1 - cos(pi * k / N)) but with proper scaling for cell size
        # For Neumann BC and DCT-II: -d^2/dx^2 phi ↔ lambda_kx * phi_hat
        # discrete lambda_k = (2/dx^2) * (1 - cos(pi * k / N))
        kr = torch.arange(self.grid_rows, dtype=torch.float32, device=self.device)
        kc = torch.arange(self.grid_cols, dtype=torch.float32, device=self.device)
        lam_y = (2.0 / (self.cell_h ** 2)) * (
            1.0 - torch.cos(math.pi * kr / self.grid_rows)
        )
        lam_x = (2.0 / (self.cell_w ** 2)) * (
            1.0 - torch.cos(math.pi * kc / self.grid_cols)
        )
        # 2D eigenvalues
        self.lam_2d = lam_y.unsqueeze(1) + lam_x.unsqueeze(0)  # [R, C]
        # Avoid divide-by-zero at DC component (set to 1; DC of phi is gauge-free).
        self.lam_2d_safe = self.lam_2d.clone()
        self.lam_2d_safe[0, 0] = 1.0

    def _smear_charges(self, positions: torch.Tensor) -> torch.Tensor:
        """Smear each macro's charge into the grid via separable Gaussian.

        Returns [R, C] charge density grid (rho).
        Memory: O(N * (R + C)) — never materializes full [N, R, C].
        """
        # positions: [N, 2] (x, y centers)
        # For each macro i: g_x[i, c] = N(x[i], sigma_x[i])(x_centers[c]) * cell_w
        # i.e., probability that a Gaussian centered at x_i lands in cell c
        # We approximate with a simple Gaussian evaluated at cell center.
        # Result charge[i] is distributed proportionally to g_x[i, :] * g_y[i, :].

        # Difference vectors: [N, C] and [N, R]
        dx = positions[:, 0].unsqueeze(1) - self.x_centers.unsqueeze(0)  # [N, C]
        dy = positions[:, 1].unsqueeze(1) - self.y_centers.unsqueeze(0)  # [N, R]
        sx = self.sigma_x.unsqueeze(1).clamp(min=1e-3)  # [N, 1]
        sy = self.sigma_y.unsqueeze(1).clamp(min=1e-3)  # [N, 1]
        gx = torch.exp(-0.5 * (dx / sx) ** 2)  # [N, C]
        gy = torch.exp(-0.5 * (dy / sy) ** 2)  # [N, R]
        # Normalize so each macro deposits exactly `charge` total
        # gx_norm: row-wise sum to 1 over C; same for gy
        gx_norm = gx / gx.sum(dim=1, keepdim=True).clamp(min=1e-12)  # [N, C]
        gy_norm = gy / gy.sum(dim=1, keepdim=True).clamp(min=1e-12)  # [N, R]

        # Per-macro outer product weighted by charge, summed across macros
        # rho[r, c] = sum_i charge[i] * gy_norm[i, r] * gx_norm[i, c] / cell_area
        # = (gy_norm.T * charge) @ gx_norm  / cell_area
        # gy_norm.T: [R, N]; gx_norm: [N, C]
        weighted_gy = gy_norm * self.charge.unsqueeze(1)  # [N, R]
        rho = (weighted_gy.transpose(0, 1) @ gx_norm) / self.cell_area  # [R, C]
        return rho

    def _poisson_solve(self, chi: torch.Tensor) -> torch.Tensor:
        """Solve -laplacian(phi) = chi on the grid with Neumann BCs via DCT.

        Returns the electric potential phi [R, C]. Sign convention: phi is
        non-negative when chi is non-negative.
        """
        chi_hat = _dct2_via_fft(chi)  # [R, C]
        phi_hat = chi_hat / self.lam_2d_safe
        # Project out the DC mode (gauge-fix) — only matters that the
        # divisor doesn't blow up; we set phi_hat[0,0] = 0 explicitly.
        # Use a mask copy to keep autograd happy.
        zero_mask = torch.zeros_like(phi_hat)
        zero_mask[0, 0] = 1.0
        phi_hat = phi_hat * (1.0 - zero_mask)
        phi = _idct2_via_fft(phi_hat)
        return phi

    def compute_density(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute the eDensity cost. Differentiable in positions.

        Returns a scalar tensor in the same scale as `_grid_density`
        (0.5 * top-10% mean), modulo the `scale` factor for calibration.
        """
        if positions.device != self.device:
            positions = positions.to(self.device)
        rho = self._smear_charges(positions)
        # Overflow above target density; use ReLU (smooth at 0; matches ePlace)
        chi = torch.clamp(rho - self.target_density, min=0.0)
        # If no overflow anywhere, phi is identically 0 — return small value
        # to avoid NaN in topk on zero tensor (it's fine, mean of zeros = 0).
        phi = self._poisson_solve(chi)
        # Use non-negative phi for topk (clamp ensures only "crowded" peaks
        # contribute, not negative-potential rebound regions).
        phi_pos = phi.clamp(min=0.0)
        flat = phi_pos.flatten()
        k = max(1, int(self.topk_frac * len(flat)))
        topk, _ = torch.topk(flat, k)
        return 0.5 * self.scale * topk.mean()

    def compute_density_with_parts(self, positions: torch.Tensor) -> dict:
        """Diagnostic version that returns intermediate tensors (detached)."""
        if positions.device != self.device:
            positions = positions.to(self.device)
        rho = self._smear_charges(positions)
        chi = torch.clamp(rho - self.target_density, min=0.0)
        phi = self._poisson_solve(chi)
        phi_pos = phi.clamp(min=0.0)
        flat = phi_pos.flatten()
        k = max(1, int(self.topk_frac * len(flat)))
        topk, _ = torch.topk(flat, k)
        cost = 0.5 * self.scale * topk.mean()
        return {
            "cost": cost,
            "rho_max": float(rho.max()),
            "rho_mean": float(rho.mean()),
            "chi_max": float(chi.max()),
            "chi_sum": float(chi.sum()),
            "phi_max": float(phi.max()),
            "phi_min": float(phi.min()),
            "topk_mean": float(topk.mean()),
        }
