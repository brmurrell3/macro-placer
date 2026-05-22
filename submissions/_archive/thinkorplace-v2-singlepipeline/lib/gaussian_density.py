"""E117 — Gaussian-smoothed density model.

A differentiable replacement for `_grid_density` (bbox-uniform per-cell
overlap area top-10%). Replaces each macro's hard rectangle with a 2D
Gaussian, then computes per-cell density by analytic integration of the
Gaussian over the cell rectangle. The resulting density field is C^inf
smooth in macro positions — eliminating the step discontinuity at cell
boundaries that troubles Adam descent.

This is a SIMPLER variant of E114's electrostatic eDensity: same idea of
smearing macros into the grid, but without the Poisson FFT solve. The
density is just the per-cell area-normalized integral of the 2D Gaussian.

Per macro i centered at (x_i, y_i) with size (w_i, h_i):
    sigma_x_i = w_i / 2     (half-macro-size)
    sigma_y_i = h_i / 2

Per-cell density (analytic):
    p(cell) = charge_i * 1/(cell_area) *
              [Phi((cx_max - x_i)/sigma_x) - Phi((cx_min - x_i)/sigma_x)] *
              [Phi((cy_max - y_i)/sigma_y) - Phi((cy_min - y_i)/sigma_y)]
where Phi is the standard normal CDF (computed via torch.erf).

Aggregator matches canonical:
    density_cost = 0.5 * mean(top_10% of grid)

Why this differs from `_grid_density`:
    - `_grid_density` uses clamp(min=0) on per-axis overlap → C0 at cell
      edges (gradient jumps when macro corner crosses a bin boundary).
    - Gaussian smearing distributes a macro's "charge" across many cells.
      The derivative w.r.t. macro position is smooth everywhere because
      erf is C^inf.
    - Total mass per macro is conserved (`charge_i = w_i * h_i`) regardless
      of how close to a cell boundary it sits.

Drop-in compatibility: same shape grid as `_grid_density`, same 0.5 *
top-10% mean output. Top-10% selection itself has a kink (sorting), but
that's identical to canonical and is rarely a gradient bottleneck.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


# Standard normal CDF using erf
# Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
_SQRT2 = math.sqrt(2.0)


def gaussian_grid_density(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    cell_x_min: torch.Tensor,
    cell_x_max: torch.Tensor,
    cell_y_min: torch.Tensor,
    cell_y_max: torch.Tensor,
    cell_area: float,
    grid_rows: int,
    grid_cols: int,
    sigma_floor_x: float = 0.0,
    sigma_floor_y: float = 0.0,
    sigma_scale: float = 1.0,
    topk_frac: float = 0.10,
) -> torch.Tensor:
    """Differentiable Gaussian-smeared grid density, top-10% aggregator.

    Each macro contributes a 2D Gaussian with sigma = half-macro-size.
    Per-cell mass is the integral of the Gaussian over the cell area
    (computed analytically via erf). Output is 0.5 * mean(top-10%) to
    match `_grid_density`.

    Args:
        positions: [N, 2] macro centers (x, y).
        sizes: [N, 2] macro sizes (w, h).
        cell_x_min, cell_x_max: [C] x-bounds of grid columns.
        cell_y_min, cell_y_max: [R] y-bounds of grid rows.
        cell_area: scalar grid cell area (cell_w * cell_h).
        grid_rows, grid_cols: integer grid dimensions.
        sigma_floor_x, sigma_floor_y: minimum sigma per axis (to keep
            Gaussians wide enough to span >=1 cell). Recommend = cell_w/2,
            cell_h/2 so tiny macros are not delta-spikes.
        sigma_scale: multiplier on (half-size) used as sigma. 1.0 = exact
            half-size; >1 = broader smear; <1 = sharper.
        topk_frac: fraction of cells for top-k aggregation. 0.10 = top-10%.

    Returns:
        scalar tensor, 0.5 * mean of top-k cells of the density grid.
    """
    N = positions.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=positions.device)

    # Sigma per macro per axis: max(half_size * sigma_scale, sigma_floor)
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5
    sigma_x = torch.clamp(sigma_scale * half_w, min=sigma_floor_x).unsqueeze(1)  # [N, 1]
    sigma_y = torch.clamp(sigma_scale * half_h, min=sigma_floor_y).unsqueeze(1)  # [N, 1]

    # "Charge" per macro = its true area, so that for a sigma → 0 limit
    # (large macro), the integrated mass equals the macro's footprint.
    charge = (sizes[:, 0] * sizes[:, 1]).unsqueeze(1).unsqueeze(2)  # [N, 1, 1]

    # X-CDF differences: [N, C]
    # Phi((cx_max - x_i)/sigma_x) - Phi((cx_min - x_i)/sigma_x)
    # = 0.5 * (erf(z_max / sqrt(2)) - erf(z_min / sqrt(2)))
    dx_max = (cell_x_max.unsqueeze(0) - positions[:, 0:1])  # [N, C]
    dx_min = (cell_x_min.unsqueeze(0) - positions[:, 0:1])
    erf_x_max = torch.erf(dx_max / (sigma_x * _SQRT2))
    erf_x_min = torch.erf(dx_min / (sigma_x * _SQRT2))
    cdf_diff_x = 0.5 * (erf_x_max - erf_x_min)  # [N, C], integrated mass per col

    # Y-CDF differences: [N, R]
    dy_max = (cell_y_max.unsqueeze(0) - positions[:, 1:2])
    dy_min = (cell_y_min.unsqueeze(0) - positions[:, 1:2])
    erf_y_max = torch.erf(dy_max / (sigma_y * _SQRT2))
    erf_y_min = torch.erf(dy_min / (sigma_y * _SQRT2))
    cdf_diff_y = 0.5 * (erf_y_max - erf_y_min)  # [N, R]

    # Per-macro, per-cell mass: charge * cdf_y * cdf_x  → [N, R, C]
    # Then sum across macros and normalize by cell_area to get density.
    # Memory: [N, R, C] — same as `_grid_density`, fine for IBM scale.
    # For large N or large grids, batch in chunks over N if needed.
    mass = charge * cdf_diff_y.unsqueeze(2) * cdf_diff_x.unsqueeze(1)  # [N, R, C]
    density_grid = mass.sum(dim=0) / cell_area  # [R, C]

    # Top-10% aggregator (matches canonical `_grid_density`)
    flat = density_grid.flatten()
    k = max(1, int(topk_frac * len(flat)))
    top_k, _ = torch.topk(flat, k)
    return 0.5 * top_k.mean()


def gaussian_grid_density_chunked(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    cell_x_min: torch.Tensor,
    cell_x_max: torch.Tensor,
    cell_y_min: torch.Tensor,
    cell_y_max: torch.Tensor,
    cell_area: float,
    grid_rows: int,
    grid_cols: int,
    sigma_floor_x: float = 0.0,
    sigma_floor_y: float = 0.0,
    sigma_scale: float = 1.0,
    topk_frac: float = 0.10,
    chunk_macros: int = 256,
) -> torch.Tensor:
    """Chunked version that accumulates density grid in chunks of macros.

    Avoids materializing [N, R, C] when N * R * C is large (>1M float32
    is borderline on CPU). For IBM scale (~500 macros, 64x64 grid =
    ~2M elements) the direct version is fine; this exists for safety on
    NG45 / unforeseen large grids.
    """
    N = positions.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=positions.device)

    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5
    sigma_x = torch.clamp(sigma_scale * half_w, min=sigma_floor_x)
    sigma_y = torch.clamp(sigma_scale * half_h, min=sigma_floor_y)
    charge = sizes[:, 0] * sizes[:, 1]

    R, C = grid_rows, grid_cols
    density_grid = torch.zeros(R, C, device=positions.device, dtype=positions.dtype)

    for start in range(0, N, chunk_macros):
        end = min(start + chunk_macros, N)
        chunk_pos = positions[start:end]
        chunk_sx = sigma_x[start:end].unsqueeze(1)
        chunk_sy = sigma_y[start:end].unsqueeze(1)
        chunk_q = charge[start:end].unsqueeze(1).unsqueeze(2)

        dx_max = cell_x_max.unsqueeze(0) - chunk_pos[:, 0:1]
        dx_min = cell_x_min.unsqueeze(0) - chunk_pos[:, 0:1]
        erf_x_max = torch.erf(dx_max / (chunk_sx * _SQRT2))
        erf_x_min = torch.erf(dx_min / (chunk_sx * _SQRT2))
        cdf_diff_x = 0.5 * (erf_x_max - erf_x_min)

        dy_max = cell_y_max.unsqueeze(0) - chunk_pos[:, 1:2]
        dy_min = cell_y_min.unsqueeze(0) - chunk_pos[:, 1:2]
        erf_y_max = torch.erf(dy_max / (chunk_sy * _SQRT2))
        erf_y_min = torch.erf(dy_min / (chunk_sy * _SQRT2))
        cdf_diff_y = 0.5 * (erf_y_max - erf_y_min)

        mass = chunk_q * cdf_diff_y.unsqueeze(2) * cdf_diff_x.unsqueeze(1)
        density_grid = density_grid + mass.sum(dim=0)

    density_grid = density_grid / cell_area

    flat = density_grid.flatten()
    k = max(1, int(topk_frac * len(flat)))
    top_k, _ = torch.topk(flat, k)
    return 0.5 * top_k.mean()


class GaussianGridDensity:
    """Object wrapper carrying grid/cell tensors so we don't re-allocate per call.

    Use this inside DiffProxyV3GaussianDensity to avoid rebuilding cell
    bounds every gradient step.
    """

    def __init__(
        self,
        benchmark,
        device: str | torch.device = "cpu",
        sigma_scale: float = 1.0,
        sigma_floor_frac: float = 0.5,
        topk_frac: float = 0.10,
        grid_rows: Optional[int] = None,
        grid_cols: Optional[int] = None,
        chunk_macros: Optional[int] = None,
    ):
        self.device = torch.device(device)
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        gr = grid_rows or benchmark.grid_rows
        gc = grid_cols or benchmark.grid_cols
        self.grid_rows = int(gr)
        self.grid_cols = int(gc)
        self.cell_w = self.cw / self.grid_cols
        self.cell_h = self.ch / self.grid_rows
        self.cell_area = self.cell_w * self.cell_h

        self.cell_x_min = (
            torch.arange(self.grid_cols, dtype=torch.float32, device=self.device)
            * self.cell_w
        )
        self.cell_x_max = self.cell_x_min + self.cell_w
        self.cell_y_min = (
            torch.arange(self.grid_rows, dtype=torch.float32, device=self.device)
            * self.cell_h
        )
        self.cell_y_max = self.cell_y_min + self.cell_h

        self.sigma_scale = float(sigma_scale)
        self.sigma_floor_x = float(sigma_floor_frac) * self.cell_w
        self.sigma_floor_y = float(sigma_floor_frac) * self.cell_h
        self.topk_frac = float(topk_frac)
        self.chunk_macros = chunk_macros

        self.sizes = benchmark.macro_sizes.to(self.device)
        self.num_macros = int(benchmark.num_macros)

    def compute_density(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute density cost. Differentiable in positions."""
        if positions.device != self.device:
            positions = positions.to(self.device)
        if self.chunk_macros is None or self.num_macros <= self.chunk_macros:
            return gaussian_grid_density(
                positions, self.sizes,
                self.cell_x_min, self.cell_x_max,
                self.cell_y_min, self.cell_y_max,
                self.cell_area, self.grid_rows, self.grid_cols,
                sigma_floor_x=self.sigma_floor_x,
                sigma_floor_y=self.sigma_floor_y,
                sigma_scale=self.sigma_scale,
                topk_frac=self.topk_frac,
            )
        else:
            return gaussian_grid_density_chunked(
                positions, self.sizes,
                self.cell_x_min, self.cell_x_max,
                self.cell_y_min, self.cell_y_max,
                self.cell_area, self.grid_rows, self.grid_cols,
                sigma_floor_x=self.sigma_floor_x,
                sigma_floor_y=self.sigma_floor_y,
                sigma_scale=self.sigma_scale,
                topk_frac=self.topk_frac,
                chunk_macros=self.chunk_macros,
            )

    def compute_density_with_parts(self, positions: torch.Tensor) -> dict:
        """Diagnostic — returns intermediate tensors."""
        if positions.device != self.device:
            positions = positions.to(self.device)
        N = positions.shape[0]
        half_w = self.sizes[:, 0] * 0.5
        half_h = self.sizes[:, 1] * 0.5
        sx = torch.clamp(self.sigma_scale * half_w, min=self.sigma_floor_x).unsqueeze(1)
        sy = torch.clamp(self.sigma_scale * half_h, min=self.sigma_floor_y).unsqueeze(1)
        q = (self.sizes[:, 0] * self.sizes[:, 1]).unsqueeze(1).unsqueeze(2)

        dx_max = self.cell_x_max.unsqueeze(0) - positions[:, 0:1]
        dx_min = self.cell_x_min.unsqueeze(0) - positions[:, 0:1]
        cdf_x = 0.5 * (torch.erf(dx_max / (sx * _SQRT2))
                       - torch.erf(dx_min / (sx * _SQRT2)))

        dy_max = self.cell_y_max.unsqueeze(0) - positions[:, 1:2]
        dy_min = self.cell_y_min.unsqueeze(0) - positions[:, 1:2]
        cdf_y = 0.5 * (torch.erf(dy_max / (sy * _SQRT2))
                       - torch.erf(dy_min / (sy * _SQRT2)))

        mass = q * cdf_y.unsqueeze(2) * cdf_x.unsqueeze(1)
        density_grid = mass.sum(dim=0) / self.cell_area
        flat = density_grid.flatten()
        k = max(1, int(self.topk_frac * len(flat)))
        topk, _ = torch.topk(flat, k)
        return {
            "cost": 0.5 * topk.mean(),
            "density_grid_max": float(density_grid.max()),
            "density_grid_mean": float(density_grid.mean()),
            "density_grid_sum": float(density_grid.sum()),
            "topk_mean": float(topk.mean()),
            "num_macros": N,
            "grid_cells": self.grid_rows * self.grid_cols,
        }
