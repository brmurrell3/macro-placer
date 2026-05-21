"""E131 — fp64-aware copy of gaussian_grid_density (from E117).

Identical math. The functions already use `positions.dtype` for the
intermediate accumulator (line 170 of E117 gaussian_density.py uses
`dtype=positions.dtype`), so the kernel itself was already dtype-safe.
This copy exists for symmetry / clean E131 isolation.

The GaussianGridDensity class — which DOES hardcode float32 cell tensors
— is NOT reproduced here because V4Gaussian builds its own cell tensors
on the FastDiffProxy side; that path is fp64-corrected in fast_proxy_fp64.py.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


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
    N = positions.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=positions.device, dtype=positions.dtype)

    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5
    sigma_x = torch.clamp(sigma_scale * half_w, min=sigma_floor_x).unsqueeze(1)
    sigma_y = torch.clamp(sigma_scale * half_h, min=sigma_floor_y).unsqueeze(1)

    charge = (sizes[:, 0] * sizes[:, 1]).unsqueeze(1).unsqueeze(2)

    dx_max = (cell_x_max.unsqueeze(0) - positions[:, 0:1])
    dx_min = (cell_x_min.unsqueeze(0) - positions[:, 0:1])
    erf_x_max = torch.erf(dx_max / (sigma_x * _SQRT2))
    erf_x_min = torch.erf(dx_min / (sigma_x * _SQRT2))
    cdf_diff_x = 0.5 * (erf_x_max - erf_x_min)

    dy_max = (cell_y_max.unsqueeze(0) - positions[:, 1:2])
    dy_min = (cell_y_min.unsqueeze(0) - positions[:, 1:2])
    erf_y_max = torch.erf(dy_max / (sigma_y * _SQRT2))
    erf_y_min = torch.erf(dy_min / (sigma_y * _SQRT2))
    cdf_diff_y = 0.5 * (erf_y_max - erf_y_min)

    mass = charge * cdf_diff_y.unsqueeze(2) * cdf_diff_x.unsqueeze(1)
    density_grid = mass.sum(dim=0) / cell_area

    flat = density_grid.flatten()
    k = max(1, int(topk_frac * len(flat)))
    top_k, _ = torch.topk(flat, k)
    return 0.5 * top_k.mean()
