"""GPU-native smooth proxy for E87 GPU CD polish.

Standalone copy of the DPO smooth-proxy (LSE-HPWL + grid-density + RUDY-cong)
with proper device handling — every torch.zeros/torch.tensor call uses the
input placement's device.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util
_DPO_PATH = _ROOT / "writeup" / "archive" / "submissions" / "dpo" / "ablation_v2_steps.py"
_DPO_SPEC = importlib.util.spec_from_file_location("dpo_v2", str(_DPO_PATH))
_dpo = importlib.util.module_from_spec(_DPO_SPEC)
_DPO_SPEC.loader.exec_module(_dpo)
_lse_hpwl = _dpo._lse_hpwl
_grid_density = _dpo._grid_density
_extract_net_data = _dpo._extract_net_data


def _rudy_congestion_device(positions, net_data, port_base, gamma,
                             cell_x_min, cell_x_max, cell_y_min, cell_y_max,
                             grid_h_routes, grid_v_routes,
                             grid_rows, grid_cols):
    """Device-aware RUDY congestion. Same logic as DPO original but creates
    accumulator tensors on positions.device."""
    num_nets = len(net_data.weights)
    device = positions.device
    if num_nets == 0 or grid_h_routes == 0 or grid_v_routes == 0:
        return torch.tensor(0.0, device=device)

    all_pos = torch.cat([positions, port_base], dim=0)
    pin_pos = all_pos[net_data.pin_macro_idx] + net_data.pin_offsets
    pin_x = pin_pos[:, :, 0]
    pin_y = pin_pos[:, :, 1]
    mask = net_data.mask
    big = 1e10
    x_for_max = pin_x.clone(); x_for_max[~mask] = -big
    x_for_min = pin_x.clone(); x_for_min[~mask] = big
    y_for_max = pin_y.clone(); y_for_max[~mask] = -big
    y_for_min = pin_y.clone(); y_for_min[~mask] = big
    bbox_x_max = gamma * torch.logsumexp(x_for_max / gamma, dim=1)
    bbox_x_min = -gamma * torch.logsumexp(-x_for_min / gamma, dim=1)
    bbox_y_max = gamma * torch.logsumexp(y_for_max / gamma, dim=1)
    bbox_y_min = -gamma * torch.logsumexp(-y_for_min / gamma, dim=1)
    bbox_w = (bbox_x_max - bbox_x_min).clamp(min=1e-6)
    bbox_h = (bbox_y_max - bbox_y_min).clamp(min=1e-6)
    w = net_data.weights
    h_coeff = w / (bbox_w * grid_h_routes)
    v_coeff = w / (bbox_h * grid_v_routes)

    # **DEVICE-AWARE** accumulators
    h_cong = torch.zeros(grid_rows, grid_cols, device=device)
    v_cong = torch.zeros(grid_rows, grid_cols, device=device)
    batch_size = 2000
    for b_start in range(0, num_nets, batch_size):
        b_end = min(b_start + batch_size, num_nets)
        b = slice(b_start, b_end)
        x_ol = torch.clamp(
            torch.min(bbox_x_max[b].unsqueeze(1), cell_x_max.unsqueeze(0))
            - torch.max(bbox_x_min[b].unsqueeze(1), cell_x_min.unsqueeze(0)),
            min=0,
        )
        y_ol = torch.clamp(
            torch.min(bbox_y_max[b].unsqueeze(1), cell_y_max.unsqueeze(0))
            - torch.max(bbox_y_min[b].unsqueeze(1), cell_y_min.unsqueeze(0)),
            min=0,
        )
        ol_area = y_ol.unsqueeze(2) * x_ol.unsqueeze(1)
        h_cong = h_cong + (h_coeff[b].unsqueeze(1).unsqueeze(2) * ol_area).sum(0)
        v_cong = v_cong + (v_coeff[b].unsqueeze(1).unsqueeze(2) * ol_area).sum(0)
    combined = h_cong + v_cong
    flat = combined.flatten()
    k = max(1, int(0.05 * len(flat)))
    top_k, _ = torch.topk(flat, k)
    return top_k.mean()


class GPUSmoothProxy:
    """Smooth proxy for autograd: WL + 0.5*density + 0.5*congestion. Device-aware."""

    def __init__(self, benchmark, plc, device="cuda", gamma_frac=0.0005):
        self.benchmark = benchmark
        self.plc = plc
        self.device = torch.device(device)
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.sizes = benchmark.macro_sizes.to(self.device)
        self.half_sizes = self.sizes / 2.0
        # Extract net_data on CPU then move tensors to device
        nd = _extract_net_data(benchmark, plc)
        for attr_name in dir(nd):
            if attr_name.startswith("_"):
                continue
            v = getattr(nd, attr_name, None)
            if isinstance(v, torch.Tensor):
                setattr(nd, attr_name, v.to(self.device))
        self.net_data = nd
        self.gamma = gamma_frac * self.cw

        gr, gc = benchmark.grid_rows, benchmark.grid_cols
        cell_w = self.cw / gc
        cell_h = self.ch / gr
        self.cell_area = cell_w * cell_h
        self.cell_x_min = (torch.arange(gc, dtype=torch.float32, device=self.device) * cell_w)
        self.cell_x_max = self.cell_x_min + cell_w
        self.cell_y_min = (torch.arange(gr, dtype=torch.float32, device=self.device) * cell_h)
        self.cell_y_max = self.cell_y_min + cell_h
        self.grid_h_routes = cell_h * benchmark.hroutes_per_micron
        self.grid_v_routes = cell_w * benchmark.vroutes_per_micron
        self.port_base = torch.zeros(1, 2, device=self.device)
        self.wl_norm = (self.cw + self.ch) * self.net_data.total_net_count
        self.grid_rows = gr
        self.grid_cols = gc

    def cost(self, positions):
        # positions might be on CPU; move
        if positions.device != self.device:
            positions = positions.to(self.device)
        clamped = torch.stack([
            positions[:, 0].clamp(self.half_sizes[:, 0], self.cw - self.half_sizes[:, 0]),
            positions[:, 1].clamp(self.half_sizes[:, 1], self.ch - self.half_sizes[:, 1]),
        ], dim=1)
        wl = _lse_hpwl(clamped, self.net_data, self.port_base, self.gamma) / self.wl_norm
        density = _grid_density(
            clamped, self.sizes,
            self.cell_x_min, self.cell_x_max, self.cell_y_min, self.cell_y_max,
            self.cell_area, self.grid_rows, self.grid_cols,
        )
        cong = _rudy_congestion_device(
            clamped, self.net_data, self.port_base, self.gamma,
            self.cell_x_min, self.cell_x_max, self.cell_y_min, self.cell_y_max,
            self.grid_h_routes, self.grid_v_routes, self.grid_rows, self.grid_cols,
        )
        return wl + 0.5 * density + 0.5 * cong
