"""E131 — fp64-aware copy of FastDiffProxyGaussian (from E127).

Identical math; subclasses the fp64-aware FastDiffProxyFp64 and uses the
fp64-aware gaussian_grid_density. All cell tensors and pin tensors
inherit dtype from the parent FastDiffProxyFp64 constructor.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from fast_proxy_fp64 import FastDiffProxyFp64, fast_lse_hpwl  # noqa: E402
from gaussian_density_fp64 import gaussian_grid_density  # noqa: E402


class FastDiffProxyGaussianFp64(FastDiffProxyFp64):
    """dtype-aware V4 + Gaussian density. Drop-in for FastDiffProxyGaussian."""

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
            cong = torch.tensor(0.0, device=self.device, dtype=positions.dtype)
        return wl + 0.5 * density + 0.5 * cong, {
            "wl": wl.detach(),
            "density": density.detach(),
            "cong": cong.detach(),
        }
