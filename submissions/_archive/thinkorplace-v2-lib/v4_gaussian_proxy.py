"""E127 — FastDiffProxy + Gaussian-smoothed density.

Composition: E115 V4 backbone (fast WL + per-net-trace congestion via
index_select + dropped chunk loop) with E117 Gaussian density
(erf-based per-cell integration).

If composition works, it gets:
  - V4's 1.4-2.8x per-step speedup
  - E117's -2.07% quality lift (verified --all = 0.982)

E125 (Gaussian + Hessian saddle) failed to compose, so composition risk
is real. But V4 changes are WL/congestion ONLY; Gaussian changes density
ONLY. They are mathematically orthogonal — no shared autograd path.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (_HERE, _ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from fast_proxy import FastDiffProxy, fast_lse_hpwl  # noqa: E402
from gaussian_density import gaussian_grid_density  # noqa: E402


class FastDiffProxyGaussian(FastDiffProxy):
    """V4 + Gaussian density. Drop-in for FastDiffProxy."""

    def __init__(self, *args, sigma_scale: float = 1.0, topk_frac: float = 0.10, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_scale = sigma_scale
        self.topk_frac = topk_frac
        cell_w = float(self.cw / self.grid_cols)
        cell_h = float(self.ch / self.grid_rows)
        self.sigma_floor_x = cell_w * 0.5
        self.sigma_floor_y = cell_h * 0.5

    def cost(self, positions: torch.Tensor, *, include_congestion: bool = True):
        if positions.device != self.device:
            positions = positions.to(self.device)
        clamped = self._clamp_to_canvas(positions)
        wl = fast_lse_hpwl(clamped, self.net_data, self.port_base, self.gamma) / self.wl_norm
        density = gaussian_grid_density(
            clamped, self.sizes,
            self.cell_x_min, self.cell_x_max,
            self.cell_y_min, self.cell_y_max,
            self.cell_area, self.grid_rows, self.grid_cols,
            sigma_floor_x=self.sigma_floor_x,
            sigma_floor_y=self.sigma_floor_y,
            sigma_scale=self.sigma_scale,
            topk_frac=self.topk_frac,
        )
        if include_congestion:
            cong = self.trace.compute_congestion(clamped)
        else:
            cong = torch.tensor(0.0, device=self.device)
        return wl + 0.5 * density + 0.5 * cong, {
            "wl": wl.detach(),
            "density": density.detach(),
            "cong": cong.detach(),
        }
