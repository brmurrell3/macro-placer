"""E131 — fp64-aware copy of FastDiffProxy (from E115).

Identical math; adds a `dtype` parameter so cell-coordinate tensors are
built in the requested float dtype (default float32, used as float64 for
fp64 descent). Pin / weight tensors inherit the dtype via the
FP64-aware PerNetTraceCongestionFp64 backbone.

Changes vs E115 fast_proxy.py:
  - `FastDiffProxy.__init__` accepts `dtype: torch.dtype = torch.float32`
  - cell_x_min / cell_x_max / cell_y_min / cell_y_max use that dtype
  - sizes / half_sizes / port_base / pin_offsets / weights converted to that dtype
  - uses PerNetTraceCongestionFp64 (dtype-aware) as the trace congestion backbone

Behavior with `dtype=torch.float32` is bit-equivalent to E115 FastDiffProxy.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

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

import importlib.util
_DPO_PATH = _ROOT / "writeup" / "archive" / "submissions" / "dpo" / "ablation_v2_steps.py"
_spec = importlib.util.spec_from_file_location("_e131_dpo_fp", str(_DPO_PATH))
_dpo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dpo)
_extract_net_data = _dpo._extract_net_data
NetData = _dpo.NetData

from per_net_trace_proxy_fp64 import PerNetTraceCongestionFp64


def _gather_pin_positions(all_pos: torch.Tensor, idx_2d: torch.Tensor) -> torch.Tensor:
    flat = idx_2d.reshape(-1)
    out = torch.index_select(all_pos, 0, flat)
    return out.reshape(*idx_2d.shape, *all_pos.shape[1:])


def fast_lse_hpwl(
    positions: torch.Tensor, net_data: NetData, port_base: torch.Tensor, gamma: float
) -> torch.Tensor:
    if len(net_data.weights) == 0:
        return torch.tensor(0.0, device=positions.device, dtype=positions.dtype)

    all_pos = torch.cat([positions, port_base], dim=0)
    pin_pos = _gather_pin_positions(all_pos, net_data.pin_macro_idx) + net_data.pin_offsets

    pin_x = pin_pos[..., 0]
    pin_y = pin_pos[..., 1]
    mask = net_data.mask
    big = 1e10
    neg_big = torch.tensor(-big, device=pin_x.device, dtype=pin_x.dtype)
    pos_big = torch.tensor(big, device=pin_x.device, dtype=pin_x.dtype)

    x_max = torch.where(mask, pin_x, neg_big)
    x_min = torch.where(mask, pin_x, pos_big)
    y_max = torch.where(mask, pin_y, neg_big)
    y_min = torch.where(mask, pin_y, pos_big)

    lse_max_x = gamma * torch.logsumexp(x_max / gamma, dim=1)
    lse_min_x = -gamma * torch.logsumexp(-x_min / gamma, dim=1)
    lse_max_y = gamma * torch.logsumexp(y_max / gamma, dim=1)
    lse_min_y = -gamma * torch.logsumexp(-y_min / gamma, dim=1)

    hpwl = (lse_max_x - lse_min_x) + (lse_max_y - lse_min_y)
    return (net_data.weights * hpwl).sum()


def fast_grid_density(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    cell_x_min: torch.Tensor,
    cell_x_max: torch.Tensor,
    cell_y_min: torch.Tensor,
    cell_y_max: torch.Tensor,
    cell_area: float,
    grid_rows: int,
    grid_cols: int,
) -> torch.Tensor:
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    mx_min = positions[:, 0] - half_w
    mx_max = positions[:, 0] + half_w
    my_min = positions[:, 1] - half_h
    my_max = positions[:, 1] + half_h

    x_overlap = torch.clamp(
        torch.minimum(mx_max.unsqueeze(1), cell_x_max.unsqueeze(0))
        - torch.maximum(mx_min.unsqueeze(1), cell_x_min.unsqueeze(0)),
        min=0,
    )
    y_overlap = torch.clamp(
        torch.minimum(my_max.unsqueeze(1), cell_y_max.unsqueeze(0))
        - torch.maximum(my_min.unsqueeze(1), cell_y_min.unsqueeze(0)),
        min=0,
    )

    overlap_area = y_overlap.unsqueeze(2) * x_overlap.unsqueeze(1)
    density_grid = overlap_area.sum(dim=0) / cell_area
    flat = density_grid.flatten()
    k = max(1, int(0.10 * len(flat)))
    top_k, _ = torch.topk(flat, k)
    return 0.5 * top_k.mean()


class FastDiffProxyFp64:
    """dtype-aware drop-in for FastDiffProxy. See E115 for the math."""

    def __init__(
        self,
        benchmark,
        plc,
        device: str | torch.device = "cuda",
        gamma_frac: float = 5e-4,
        trace_kwargs: Optional[Dict] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.benchmark = benchmark
        self.plc = plc
        self.device = torch.device(device)
        self.dtype = dtype
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.sizes = benchmark.macro_sizes.to(self.device).to(dtype)
        self.half_sizes = self.sizes / 2.0
        self.num_hard = int(benchmark.num_hard_macros)

        nd = _extract_net_data(benchmark, plc)
        for attr_name in ("pin_macro_idx", "pin_offsets", "mask", "weights"):
            v = getattr(nd, attr_name)
            if isinstance(v, torch.Tensor):
                v = v.to(self.device)
                # cast only the float-typed attributes
                if v.dtype.is_floating_point:
                    v = v.to(dtype)
                setattr(nd, attr_name, v)
        self.net_data: NetData = nd
        self.gamma = gamma_frac * self.cw

        gr, gc = benchmark.grid_rows, benchmark.grid_cols
        cell_w = self.cw / gc
        cell_h = self.ch / gr
        self.cell_area = cell_w * cell_h
        self.cell_x_min = torch.arange(gc, dtype=dtype, device=self.device) * cell_w
        self.cell_x_max = self.cell_x_min + cell_w
        self.cell_y_min = torch.arange(gr, dtype=dtype, device=self.device) * cell_h
        self.cell_y_max = self.cell_y_min + cell_h
        self.grid_h_routes = cell_h * benchmark.hroutes_per_micron
        self.grid_v_routes = cell_w * benchmark.vroutes_per_micron
        self.port_base = torch.zeros(1, 2, device=self.device, dtype=dtype)
        try:
            canonical_net_cnt = float(plc.net_cnt) if plc.net_cnt else self.net_data.total_net_count
        except AttributeError:
            canonical_net_cnt = self.net_data.total_net_count
        self.wl_norm = (self.cw + self.ch) * max(1.0, canonical_net_cnt)
        self.grid_rows = gr
        self.grid_cols = gc

        trace_kwargs = dict(trace_kwargs or {})
        # Force single-pass: huge chunk to keep the loop body running once.
        trace_kwargs.setdefault("pair_chunk_size", 2_000_000)
        self.trace = PerNetTraceCongestionFp64(
            benchmark, plc, device=device, dtype=dtype, **trace_kwargs
        )

    def set_gamma_frac(self, gamma_frac: float) -> None:
        self.gamma = gamma_frac * self.cw

    def _clamp_to_canvas(self, positions: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                positions[:, 0].clamp(self.half_sizes[:, 0], self.cw - self.half_sizes[:, 0]),
                positions[:, 1].clamp(self.half_sizes[:, 1], self.ch - self.half_sizes[:, 1]),
            ],
            dim=1,
        )

    def cost(self, positions: torch.Tensor, *, include_congestion: bool = True):
        if positions.device != self.device:
            positions = positions.to(self.device)
        clamped = self._clamp_to_canvas(positions)
        wl = fast_lse_hpwl(clamped, self.net_data, self.port_base, self.gamma) / self.wl_norm
        density = fast_grid_density(
            clamped, self.sizes,
            self.cell_x_min, self.cell_x_max,
            self.cell_y_min, self.cell_y_max,
            self.cell_area, self.grid_rows, self.grid_cols,
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

    def overlap_penalty(self, positions: torch.Tensor) -> torch.Tensor:
        n = self.num_hard
        if n <= 1:
            return torch.tensor(0.0, device=self.device, dtype=positions.dtype)
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


def fast_loss_with_penalty(
    proxy: FastDiffProxyFp64,
    positions: torch.Tensor,
    overlap_lambda: float,
    *,
    include_congestion: bool = True,
    boundary_lambda: float = 1.0,
):
    smooth_cost, parts = proxy.cost(positions, include_congestion=include_congestion)
    penalty = proxy.overlap_penalty(positions)
    boundary = proxy.out_of_canvas_penalty(positions)
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
