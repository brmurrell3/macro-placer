"""E131 — fp64-aware copy of PerNetTraceCongestion (from E111).

Identical math; adds a `dtype` parameter so cell-coordinate tensors and
pin-offset/weight tensors are built in the requested float dtype
(torch.float32 by default, torch.float64 for fp64 descent).

The only changes vs E111 per_net_trace_proxy.py are:
  - constructor accepts `dtype: torch.dtype = torch.float32`
  - `cell_x_ctr` / `cell_y_ctr` / `col_idx_f` / `row_idx_f` use that dtype
  - empty-pair fallback tensors (src_offset, snk_offset, pair_w) use it
  - pin_offsets / weights cast to it (rather than to float32)

NOT modified: anything mathematical, indexing logic, smoothing kernels.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

_DPO_PATH = _ROOT / "writeup" / "archive" / "submissions" / "dpo" / "ablation_v2_steps.py"
_spec = importlib.util.spec_from_file_location("_e131_dpo", str(_DPO_PATH))
_dpo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dpo)
_extract_net_data = _dpo._extract_net_data
NetData = _dpo.NetData


class PerNetTraceCongestionFp64:
    """dtype-aware copy of PerNetTraceCongestion. See E111 for the math."""

    def __init__(
        self,
        benchmark,
        plc,
        device: str | torch.device = "cpu",
        sigma_cell_frac: float = 0.5,
        beta_range_per_cell: float = 4.0,
        beta_minmax: float = 6.0,
        pair_chunk_size: int = 4096,
        abu_k: float = 0.05,
        include_macro_routing: bool = True,
        include_smoothing: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        self.benchmark = benchmark
        self.plc = plc
        self.device = torch.device(device)
        self.dtype = dtype
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.num_macros = int(benchmark.num_macros)
        self.num_hard = int(benchmark.num_hard_macros)
        self.gr = int(benchmark.grid_rows)
        self.gc = int(benchmark.grid_cols)
        self.cell_w = self.cw / self.gc
        self.cell_h = self.ch / self.gr

        self.cell_x_ctr = (torch.arange(self.gc, dtype=dtype, device=self.device) + 0.5) * self.cell_w
        self.cell_y_ctr = (torch.arange(self.gr, dtype=dtype, device=self.device) + 0.5) * self.cell_h
        self.col_idx_f = torch.arange(self.gc, dtype=dtype, device=self.device)
        self.row_idx_f = torch.arange(self.gr, dtype=dtype, device=self.device)

        self.grid_h_routes = self.cell_h * benchmark.hroutes_per_micron
        self.grid_v_routes = self.cell_w * benchmark.vroutes_per_micron

        self.hrouting_alloc = float(getattr(plc, "hrouting_alloc", 1.0))
        self.vrouting_alloc = float(getattr(plc, "vrouting_alloc", 1.0))

        self.sigma_cell_frac = float(sigma_cell_frac)
        self.beta_range_per_cell = float(beta_range_per_cell)
        self.beta_minmax = float(beta_minmax)
        self.pair_chunk_size = int(pair_chunk_size)
        self.abu_k = float(abu_k)
        self.include_macro_routing = bool(include_macro_routing)
        self.include_smoothing = bool(include_smoothing)
        self.smooth_range = int(getattr(plc, "smooth_range", 2))

        nd = _extract_net_data(benchmark, plc)
        self._build_pair_data(nd)

        self.macro_sizes = benchmark.macro_sizes.to(self.device).to(dtype)
        self.macro_fixed = benchmark.macro_fixed.to(self.device)
        self.cell_x_min = self.cell_x_ctr - self.cell_w / 2.0
        self.cell_x_max = self.cell_x_ctr + self.cell_w / 2.0
        self.cell_y_min = self.cell_y_ctr - self.cell_h / 2.0
        self.cell_y_max = self.cell_y_ctr + self.cell_h / 2.0

    def _build_pair_data(self, nd: NetData) -> None:
        if nd.weights.numel() == 0:
            self.n_pairs = 0
            self.src_macro = torch.zeros(0, dtype=torch.long, device=self.device)
            self.snk_macro = torch.zeros(0, dtype=torch.long, device=self.device)
            self.src_offset = torch.zeros(0, 2, dtype=self.dtype, device=self.device)
            self.snk_offset = torch.zeros(0, 2, dtype=self.dtype, device=self.device)
            self.pair_w = torch.zeros(0, dtype=self.dtype, device=self.device)
            return

        pin_macro_idx = nd.pin_macro_idx.to(self.device)
        pin_offsets = nd.pin_offsets.to(self.device).to(self.dtype)
        mask = nd.mask.to(self.device)
        weights = nd.weights.to(self.device).to(self.dtype)

        num_nets, max_pins = pin_macro_idx.shape
        src_macro_net = pin_macro_idx[:, 0]
        src_offset_net = pin_offsets[:, 0]

        sink_mask = mask[:, 1:]
        sink_macro = pin_macro_idx[:, 1:]
        sink_offset = pin_offsets[:, 1:]

        n_sinks = sink_mask.sum(dim=1)
        net_idx = torch.repeat_interleave(
            torch.arange(num_nets, device=self.device), n_sinks
        )
        sm_flat = sink_macro[sink_mask]
        so_flat = sink_offset[sink_mask]
        src_m = src_macro_net[net_idx]
        src_o = src_offset_net[net_idx]
        pair_w = weights[net_idx]

        self.n_pairs = int(net_idx.numel())
        self.src_macro = src_m
        self.snk_macro = sm_flat
        self.src_offset = src_o
        self.snk_offset = so_flat
        self.pair_w = pair_w

    def _soft_cell_assign(self, pin_coord: torch.Tensor, axis: str) -> torch.Tensor:
        if axis == "x":
            ctr = self.cell_x_ctr
            sz = self.cell_w
        else:
            ctr = self.cell_y_ctr
            sz = self.cell_h
        sigma = self.sigma_cell_frac * sz
        diff = (pin_coord.view(-1, 1) - ctr.view(1, -1)) / sigma
        scores = -(diff ** 2)
        return torch.softmax(scores, dim=1)

    def _soft_min_max(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = self.beta_minmax
        stacked = torch.stack([a, b], dim=0)
        soft_min = -torch.logsumexp(-beta * stacked, dim=0) / beta
        soft_max = torch.logsumexp(beta * stacked, dim=0) / beta
        return soft_min, soft_max

    def _trace_route_congestion(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n_pairs == 0:
            return (
                torch.zeros(self.gr, self.gc, device=self.device, dtype=positions.dtype),
                torch.zeros(self.gr, self.gc, device=self.device, dtype=positions.dtype),
            )

        device = positions.device
        port_base = torch.zeros(1, 2, device=device, dtype=positions.dtype)
        all_pos = torch.cat([positions, port_base], dim=0)

        H_route = torch.zeros(self.gr, self.gc, device=device, dtype=positions.dtype)
        V_route = torch.zeros(self.gr, self.gc, device=device, dtype=positions.dtype)

        cs = self.pair_chunk_size
        n_pairs = self.n_pairs
        for b_start in range(0, n_pairs, cs):
            b_end = min(b_start + cs, n_pairs)
            src_m = self.src_macro[b_start:b_end]
            snk_m = self.snk_macro[b_start:b_end]
            src_o = self.src_offset[b_start:b_end]
            snk_o = self.snk_offset[b_start:b_end]
            w = self.pair_w[b_start:b_end]

            src_pos = all_pos[src_m] + src_o
            snk_pos = all_pos[snk_m] + snk_o

            row_src = self._soft_cell_assign(src_pos[:, 1], axis="y")
            col_snk = self._soft_cell_assign(snk_pos[:, 0], axis="x")
            col_src = self._soft_cell_assign(src_pos[:, 0], axis="x")
            row_snk = self._soft_cell_assign(snk_pos[:, 1], axis="y")

            col_src_exp = (col_src * self.col_idx_f.view(1, -1)).sum(dim=1)
            col_snk_exp = (col_snk * self.col_idx_f.view(1, -1)).sum(dim=1)
            row_src_exp = (row_src * self.row_idx_f.view(1, -1)).sum(dim=1)
            row_snk_exp = (row_snk * self.row_idx_f.view(1, -1)).sum(dim=1)

            col_min, col_max = self._soft_min_max(col_src_exp, col_snk_exp)
            row_min, row_max = self._soft_min_max(row_src_exp, row_snk_exp)
            col_min = col_min.unsqueeze(1)
            col_max = col_max.unsqueeze(1)
            row_min = row_min.unsqueeze(1)
            row_max = row_max.unsqueeze(1)

            br = self.beta_range_per_cell
            col_ind = self.col_idx_f.view(1, -1)
            row_ind = self.row_idx_f.view(1, -1)
            h_col_range = (
                torch.sigmoid(br * (col_ind - col_min + 0.5))
                - torch.sigmoid(br * (col_ind - col_max + 0.5))
            )
            v_row_range = (
                torch.sigmoid(br * (row_ind - row_min + 0.5))
                - torch.sigmoid(br * (row_ind - row_max + 0.5))
            )

            row_src_w = row_src * w.unsqueeze(1)
            col_snk_w = col_snk * w.unsqueeze(1)
            H_route = H_route + row_src_w.transpose(0, 1) @ h_col_range
            V_route = V_route + v_row_range.transpose(0, 1) @ col_snk_w

        return V_route, H_route

    def _macro_route_congestion(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.include_macro_routing or self.num_hard == 0:
            return (
                torch.zeros(self.gr, self.gc, device=positions.device, dtype=positions.dtype),
                torch.zeros(self.gr, self.gc, device=positions.device, dtype=positions.dtype),
            )
        sizes = self.macro_sizes[:self.num_hard]
        pos = positions[:self.num_hard]
        half_w = sizes[:, 0] / 2.0
        half_h = sizes[:, 1] / 2.0
        mx_min = pos[:, 0] - half_w
        mx_max = pos[:, 0] + half_w
        my_min = pos[:, 1] - half_h
        my_max = pos[:, 1] + half_h

        x_ol = torch.clamp(
            torch.min(mx_max.unsqueeze(1), self.cell_x_max.unsqueeze(0))
            - torch.max(mx_min.unsqueeze(1), self.cell_x_min.unsqueeze(0)),
            min=0,
        )
        y_ol = torch.clamp(
            torch.min(my_max.unsqueeze(1), self.cell_y_max.unsqueeze(0))
            - torch.max(my_min.unsqueeze(1), self.cell_y_min.unsqueeze(0)),
            min=0,
        )

        y_present = (y_ol / self.cell_h).clamp(max=1.0)
        x_present = (x_ol / self.cell_w).clamp(max=1.0)
        V_macro = (y_present.transpose(0, 1) @ x_ol) * self.vrouting_alloc
        H_macro = (y_ol.transpose(0, 1) @ x_present) * self.hrouting_alloc
        return V_macro, H_macro

    def _box_smooth(self, M: torch.Tensor, axis: str) -> torch.Tensor:
        sr = self.smooth_range
        if sr <= 0 or not self.include_smoothing:
            return M
        ksize = 2 * sr + 1
        if axis == "V":
            ones = torch.ones_like(M)
            count = F.conv2d(
                ones.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, 1, ksize, device=M.device, dtype=M.dtype),
                padding=(0, sr),
            )[0, 0]
            M_norm = M / count
            return F.conv2d(
                M_norm.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, 1, ksize, device=M.device, dtype=M.dtype),
                padding=(0, sr),
            )[0, 0]
        elif axis == "H":
            ones = torch.ones_like(M)
            count = F.conv2d(
                ones.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, ksize, 1, device=M.device, dtype=M.dtype),
                padding=(sr, 0),
            )[0, 0]
            M_norm = M / count
            return F.conv2d(
                M_norm.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, ksize, 1, device=M.device, dtype=M.dtype),
                padding=(sr, 0),
            )[0, 0]
        return M

    def compute_congestion(self, positions: torch.Tensor) -> torch.Tensor:
        V_route, H_route = self._trace_route_congestion(positions)
        V_route = V_route / max(self.grid_v_routes, 1e-9)
        H_route = H_route / max(self.grid_h_routes, 1e-9)

        V_macro, H_macro = self._macro_route_congestion(positions)
        V_macro = V_macro / max(self.grid_v_routes, 1e-9)
        H_macro = H_macro / max(self.grid_h_routes, 1e-9)

        V_route = self._box_smooth(V_route, axis="V")
        H_route = self._box_smooth(H_route, axis="H")

        V_total = V_route + V_macro
        H_total = H_route + H_macro

        combined = torch.cat([V_total.flatten(), H_total.flatten()])
        k = max(1, int(self.abu_k * combined.numel()))
        top_k, _ = torch.topk(combined, k)
        return top_k.mean()
