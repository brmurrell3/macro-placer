"""E88 — standalone differentiable proxy for the C1 spike.

Autograd-compatible approximation of the canonical PlacementCost objective:
HPWL + 0.5*density + 0.5*congestion. The spike defers TILOS-RUDY and ships
LSE-HPWL + Gaussian-style grid density + RUDY-area congestion.

Imports DPO primitives directly from writeup/archive/.../ablation_v2_steps.py;
does NOT depend on experiments/E87_gpu_cd/ — that path is owned by another
session.

Also includes a soft-overlap Lagrangian penalty for the descent step
(quadratic in unsigned x*y overlap area per hard-macro pair).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load DPO primitives without exec'ing the whole submission as a sibling import.
_DPO_PATH = _ROOT / "writeup" / "archive" / "submissions" / "dpo" / "ablation_v2_steps.py"
_spec = importlib.util.spec_from_file_location("_e88_dpo", str(_DPO_PATH))
_dpo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dpo)
_lse_hpwl = _dpo._lse_hpwl
_grid_density = _dpo._grid_density
_rudy_congestion = _dpo._rudy_congestion
_extract_net_data = _dpo._extract_net_data
NetData = _dpo.NetData


class DiffProxy:
    """Differentiable proxy on a fixed (benchmark, plc) pair.

    Construct once per bench; call `.cost(positions)` repeatedly under autograd.
    """

    def __init__(
        self,
        benchmark,
        plc,
        device: str | torch.device = "cpu",
        gamma_frac: float = 0.0005,
    ):
        self.benchmark = benchmark
        self.plc = plc
        self.device = torch.device(device)
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.sizes = benchmark.macro_sizes.to(self.device)
        self.half_sizes = self.sizes / 2.0
        self.num_hard = int(benchmark.num_hard_macros)

        nd = _extract_net_data(benchmark, plc)
        for attr_name in ("pin_macro_idx", "pin_offsets", "mask", "weights"):
            v = getattr(nd, attr_name)
            if isinstance(v, torch.Tensor):
                setattr(nd, attr_name, v.to(self.device))
        self.net_data: NetData = nd
        self.gamma = gamma_frac * self.cw

        gr, gc = benchmark.grid_rows, benchmark.grid_cols
        cell_w = self.cw / gc
        cell_h = self.ch / gr
        self.cell_area = cell_w * cell_h
        self.cell_x_min = torch.arange(gc, dtype=torch.float32, device=self.device) * cell_w
        self.cell_x_max = self.cell_x_min + cell_w
        self.cell_y_min = torch.arange(gr, dtype=torch.float32, device=self.device) * cell_h
        self.cell_y_max = self.cell_y_min + cell_h
        self.grid_h_routes = cell_h * benchmark.hroutes_per_micron
        self.grid_v_routes = cell_w * benchmark.vroutes_per_micron
        self.port_base = torch.zeros(1, 2, device=self.device)
        self.wl_norm = (self.cw + self.ch) * self.net_data.total_net_count
        self.grid_rows = gr
        self.grid_cols = gc

    def _clamp_to_canvas(self, positions: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                positions[:, 0].clamp(self.half_sizes[:, 0], self.cw - self.half_sizes[:, 0]),
                positions[:, 1].clamp(self.half_sizes[:, 1], self.ch - self.half_sizes[:, 1]),
            ],
            dim=1,
        )

    def cost(self, positions: torch.Tensor, *, include_congestion: bool = True) -> torch.Tensor:
        """Smooth canonical proxy. Differentiable w.r.t. positions."""
        if positions.device != self.device:
            positions = positions.to(self.device)
        clamped = self._clamp_to_canvas(positions)
        wl = _lse_hpwl(clamped, self.net_data, self.port_base, self.gamma) / self.wl_norm
        density = _grid_density(
            clamped,
            self.sizes,
            self.cell_x_min,
            self.cell_x_max,
            self.cell_y_min,
            self.cell_y_max,
            self.cell_area,
            self.grid_rows,
            self.grid_cols,
        )
        if include_congestion:
            cong = _rudy_congestion(
                clamped,
                self.net_data,
                self.port_base,
                self.gamma,
                self.cell_x_min,
                self.cell_x_max,
                self.cell_y_min,
                self.cell_y_max,
                self.grid_h_routes,
                self.grid_v_routes,
                self.grid_rows,
                self.grid_cols,
            )
        else:
            cong = torch.tensor(0.0, device=self.device)
        return wl + 0.5 * density + 0.5 * cong, {
            "wl": wl.detach(),
            "density": density.detach(),
            "cong": cong.detach(),
        }

    def overlap_penalty(self, positions: torch.Tensor) -> torch.Tensor:
        """Sum of unsigned hard-macro pairwise overlap areas.

        Computed on CLAMPED positions so positions escaping the canvas don't
        give spurious zero overlap (large dx/dy when positions are off-canvas).

        Smooth (no clamp on dx/dy, just ReLU on per-axis separation deficit).
        Soft macros excluded.
        """
        n = self.num_hard
        if n <= 1:
            return torch.tensor(0.0, device=self.device)
        clamped = self._clamp_to_canvas(positions)
        pos = clamped[:n]
        hw = self.half_sizes[:n, 0]
        hh = self.half_sizes[:n, 1]
        dx = (pos[:, 0].unsqueeze(0) - pos[:, 0].unsqueeze(1)).abs()
        dy = (pos[:, 1].unsqueeze(0) - pos[:, 1].unsqueeze(1)).abs()
        sep_x = hw.unsqueeze(0) + hw.unsqueeze(1)
        sep_y = hh.unsqueeze(0) + hh.unsqueeze(1)
        ox = torch.clamp(sep_x - dx, min=0.0)
        oy = torch.clamp(sep_y - dy, min=0.0)
        area = ox * oy
        mask = torch.triu(torch.ones_like(area, dtype=torch.bool), diagonal=1)
        return area[mask].sum()

    def out_of_canvas_penalty(self, positions: torch.Tensor) -> torch.Tensor:
        """Quadratic penalty for positions leaving the canvas (per macro).

        Returns sum of (over-edge distance)^2 across all macros and both axes.
        Differentiable everywhere; gradient pushes escaped positions back.
        """
        x = positions[:, 0]
        y = positions[:, 1]
        x_min = self.half_sizes[:, 0]
        x_max = self.cw - self.half_sizes[:, 0]
        y_min = self.half_sizes[:, 1]
        y_max = self.ch - self.half_sizes[:, 1]
        below_x = torch.clamp(x_min - x, min=0.0)
        above_x = torch.clamp(x - x_max, min=0.0)
        below_y = torch.clamp(y_min - y, min=0.0)
        above_y = torch.clamp(y - y_max, min=0.0)
        return (below_x ** 2 + above_x ** 2 + below_y ** 2 + above_y ** 2).sum()


def loss_with_penalty(
    proxy: DiffProxy,
    positions: torch.Tensor,
    overlap_lambda: float,
    *,
    include_congestion: bool = True,
    boundary_lambda: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    smooth_cost, parts = proxy.cost(positions, include_congestion=include_congestion)
    penalty = proxy.overlap_penalty(positions)
    boundary = proxy.out_of_canvas_penalty(positions)
    # Normalize by canvas area so lambdas are roughly scale-free.
    canvas_area = proxy.cw * proxy.ch
    penalty_norm = penalty / canvas_area
    boundary_norm = boundary / canvas_area
    total = smooth_cost + overlap_lambda * penalty_norm + boundary_lambda * boundary_norm
    parts["overlap_area_raw"] = penalty.detach()
    parts["overlap_area_norm"] = penalty_norm.detach()
    parts["boundary_raw"] = boundary.detach()
    parts["boundary_norm"] = boundary_norm.detach()
    parts["smooth_cost"] = smooth_cost.detach()
    return total, parts
