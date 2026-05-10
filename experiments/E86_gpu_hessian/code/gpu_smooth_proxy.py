"""Device-parametric smooth proxy. Drop-in replacement for E74's SmoothProxy
that runs on MPS (Apple Silicon) or CUDA via a device param.

Numerical parity with the CPU version is the primary test: same Hessian
eigenvalues to within ~1e-4 tolerance.

The DPO primitives `_lse_hpwl`, `_grid_density`, `_rudy_congestion`,
`_extract_net_data` come from the same upstream as E74's SmoothProxy and
are themselves device-agnostic (they're pure torch tensor ops); we just
need to move all the cached tensor state in __init__ onto the chosen
device and ensure input positions go to the device too.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DPO_PATH = _ROOT / "writeup" / "archive" / "submissions" / "dpo" / "ablation_v2_steps.py"
_spec = importlib.util.spec_from_file_location("dpo_v2", str(_DPO_PATH))
_dpo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dpo)
_lse_hpwl = _dpo._lse_hpwl
_grid_density = _dpo._grid_density
_extract_net_data = _dpo._extract_net_data


def _rudy_congestion_device(positions, net_data, port_base, gamma,
                            cell_x_min, cell_x_max, cell_y_min, cell_y_max,
                            grid_h_routes, grid_v_routes, grid_rows, grid_cols):
    """Device-aware reimplementation of DPO's _rudy_congestion.

    Same math; explicit device for accumulator tensors so MPS/CUDA work.
    """
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


_rudy_congestion = _rudy_congestion_device


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class _NetDataOnDevice:
    """Wrap NetData (a dataclass with tensor fields) onto a device."""
    def __init__(self, nd, device):
        self.pin_macro_idx = nd.pin_macro_idx.to(device)
        self.pin_offsets = nd.pin_offsets.to(device)
        self.mask = nd.mask.to(device)
        self.weights = nd.weights.to(device)
        self.num_macros = nd.num_macros
        self.total_net_count = nd.total_net_count


class GPUSmoothProxy:
    """Smooth proxy on user-chosen device.

    Same math as the CPU version in E74's hessian_saddle.SmoothProxy.
    """

    def __init__(self, benchmark, plc, *, gamma_frac=0.0005, device: Optional[str] = None):
        self.device = device or default_device()
        self.benchmark = benchmark
        self.plc = plc
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.sizes = benchmark.macro_sizes.to(self.device)
        self.half_sizes = self.sizes / 2.0
        nd = _extract_net_data(benchmark, plc)
        self.net_data = _NetDataOnDevice(nd, self.device)
        self.gamma = gamma_frac * self.cw

        gr, gc = benchmark.grid_rows, benchmark.grid_cols
        self.cell_w = self.cw / gc
        self.cell_h = self.ch / gr
        self.cell_area = self.cell_w * self.cell_h
        self.cell_x_min = (torch.arange(gc, dtype=torch.float32) * self.cell_w).to(self.device)
        self.cell_x_max = self.cell_x_min + self.cell_w
        self.cell_y_min = (torch.arange(gr, dtype=torch.float32) * self.cell_h).to(self.device)
        self.cell_y_max = self.cell_y_min + self.cell_h
        self.grid_h_routes = self.cell_h * benchmark.hroutes_per_micron
        self.grid_v_routes = self.cell_w * benchmark.vroutes_per_micron
        self.port_base = torch.zeros(1, 2, device=self.device)
        self.wl_norm = (self.cw + self.ch) * self.net_data.total_net_count
        self.grid_rows = gr
        self.grid_cols = gc

    def cost(self, positions: torch.Tensor) -> torch.Tensor:
        # Ensure on-device
        if positions.device != torch.device(self.device):
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
        cong = _rudy_congestion(
            clamped, self.net_data, self.port_base, self.gamma,
            self.cell_x_min, self.cell_x_max, self.cell_y_min, self.cell_y_max,
            self.grid_h_routes, self.grid_v_routes, self.grid_rows, self.grid_cols,
        )
        return wl + 0.5 * density + 0.5 * cong
