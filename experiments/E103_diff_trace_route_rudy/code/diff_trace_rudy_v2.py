"""DiffTraceRudy v2: uses canonical pin extraction.

Builds pin pair list from `extract_canon_nets()` (matching plc.get_routing walk),
then computes L-route congestion per pair. Each canonical net is 2-pin in this
codebase (per the count check on ibm01); future bench types may need __l_routing/
__t_routing for 3-pin nets, but for now we use 2-pin star-from-source for any
N>2 case.

Also includes macro routing contribution (from __macro_route_over_grid_cell).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, NamedTuple, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from canon_pin_extract import extract_canon_nets


class DiffTraceRudyV2:
    def __init__(self, benchmark, plc, device: str | torch.device = "cpu",
                 beta_cell: float = 1.0, beta_range: float = 4.0):
        self.benchmark = benchmark
        self.plc = plc
        self.device = torch.device(device)
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.num_macros = int(benchmark.num_macros)
        self.num_hard = int(benchmark.num_hard_macros)
        self.gr = int(benchmark.grid_rows)
        self.gc = int(benchmark.grid_cols)
        self.cell_w = self.cw / self.gc
        self.cell_h = self.ch / self.gr
        self.cell_x_ctr = (torch.arange(self.gc, dtype=torch.float32, device=self.device) + 0.5) * self.cell_w
        self.cell_y_ctr = (torch.arange(self.gr, dtype=torch.float32, device=self.device) + 0.5) * self.cell_h
        self.grid_h_routes = self.cell_h * benchmark.hroutes_per_micron
        self.grid_v_routes = self.cell_w * benchmark.vroutes_per_micron
        self.beta_cell = beta_cell / self.cell_w
        self.beta_range = beta_range
        self.smooth_range = int(getattr(plc, "smooth_range", 2))

        # Extract canonical 2-pin pair data
        canon_nets = extract_canon_nets(benchmark, plc)
        # For each net, generate 2-pin pairs (star from source)
        src_macros, src_offsets = [], []
        snk_macros, snk_offsets = [], []
        weights = []
        for net in canon_nets:
            for snk_m, snk_o in zip(net.sink_macro_indices, net.sink_offsets):
                src_macros.append(net.src_macro_idx)
                src_offsets.append(net.src_offset)
                snk_macros.append(snk_m)
                snk_offsets.append(snk_o)
                weights.append(net.weight)
        self.n_pairs = len(src_macros)
        if self.n_pairs == 0:
            self.src_macro = torch.zeros(0, dtype=torch.long, device=self.device)
            self.snk_macro = torch.zeros(0, dtype=torch.long, device=self.device)
            self.src_offset = torch.zeros(0, 2, dtype=torch.float32, device=self.device)
            self.snk_offset = torch.zeros(0, 2, dtype=torch.float32, device=self.device)
            self.pair_w = torch.zeros(0, dtype=torch.float32, device=self.device)
            return
        self.src_macro = torch.tensor(src_macros, dtype=torch.long, device=self.device)
        self.snk_macro = torch.tensor(snk_macros, dtype=torch.long, device=self.device)
        self.src_offset = torch.tensor(src_offsets, dtype=torch.float32, device=self.device)
        self.snk_offset = torch.tensor(snk_offsets, dtype=torch.float32, device=self.device)
        self.pair_w = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Pre-compute hard macro sizes for macro routing congestion
        self.macro_sizes = benchmark.macro_sizes.to(self.device)
        self.macro_fixed = benchmark.macro_fixed.to(self.device)

    def trace_congestion(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable trace-route congestion (H and V per grid cell)."""
        if self.n_pairs == 0:
            return (torch.zeros(self.gr, self.gc, device=self.device),
                    torch.zeros(self.gr, self.gc, device=self.device))
        device = positions.device
        port_base = torch.zeros(1, 2, device=device, dtype=positions.dtype)
        all_pos = torch.cat([positions, port_base], dim=0)
        # In canonical, PORT pins use mod.get_pos() directly (ABSOLUTE position).
        # In our extraction, src_offset for PORTs is the absolute position (parent at origin),
        # and for MACRO_PINs it's relative to parent macro center.
        # Since port_base is at (0,0), all_pos[port_idx]+offset = offset = absolute. ✓
        src_pos = all_pos[self.src_macro] + self.src_offset
        snk_pos = all_pos[self.snk_macro] + self.snk_offset

        # Compute integer grid cell indices (smooth softmax over Gaussian distance)
        # σ=0.15 → mass concentrated in nearest cell
        sigma_cell = 0.15
        sx_src = -((src_pos[:, 0:1] - self.cell_x_ctr.view(1, -1)) / (sigma_cell * self.cell_w)) ** 2
        sx_snk = -((snk_pos[:, 0:1] - self.cell_x_ctr.view(1, -1)) / (sigma_cell * self.cell_w)) ** 2
        sy_src = -((src_pos[:, 1:2] - self.cell_y_ctr.view(1, -1)) / (sigma_cell * self.cell_h)) ** 2
        sy_snk = -((snk_pos[:, 1:2] - self.cell_y_ctr.view(1, -1)) / (sigma_cell * self.cell_h)) ** 2

        col_src = torch.softmax(sx_src, dim=1)
        col_snk = torch.softmax(sx_snk, dim=1)
        row_src = torch.softmax(sy_src, dim=1)
        row_snk = torch.softmax(sy_snk, dim=1)

        # Soft expected positions
        col_idx = torch.arange(self.gc, dtype=torch.float32, device=device)
        row_idx = torch.arange(self.gr, dtype=torch.float32, device=device)
        col_src_exp = (col_src * col_idx.view(1, -1)).sum(dim=1, keepdim=True)
        col_snk_exp = (col_snk * col_idx.view(1, -1)).sum(dim=1, keepdim=True)
        row_src_exp = (row_src * row_idx.view(1, -1)).sum(dim=1, keepdim=True)
        row_snk_exp = (row_snk * row_idx.view(1, -1)).sum(dim=1, keepdim=True)

        # Soft min/max
        beta_mm = 8.0
        def soft_min(a, b):
            return -torch.logsumexp(torch.stack([-beta_mm*a, -beta_mm*b], dim=0), dim=0) / beta_mm
        def soft_max(a, b):
            return torch.logsumexp(torch.stack([beta_mm*a, beta_mm*b], dim=0), dim=0) / beta_mm
        col_min = soft_min(col_src_exp, col_snk_exp)
        col_max = soft_max(col_src_exp, col_snk_exp)
        row_min = soft_min(row_src_exp, row_snk_exp)
        row_max = soft_max(row_src_exp, row_snk_exp)

        # Range indicators (sigmoid difference)
        beta_r = self.beta_range
        h_in_col = (torch.sigmoid(beta_r * (col_idx.view(1, -1) - col_min + 0.5))
                    - torch.sigmoid(beta_r * (col_idx.view(1, -1) - col_max + 0.5)))
        v_in_row = (torch.sigmoid(beta_r * (row_idx.view(1, -1) - row_min + 0.5))
                    - torch.sigmoid(beta_r * (row_idx.view(1, -1) - row_max + 0.5)))

        # Accumulate per cell
        weight = self.pair_w.view(-1, 1, 1)
        # H route lives on source_row, cols in range
        h_cong = (weight * row_src.unsqueeze(2) * h_in_col.unsqueeze(1)).sum(dim=0)
        # V route lives on sink_col, rows in range
        v_cong = (weight * v_in_row.unsqueeze(2) * col_snk.unsqueeze(1)).sum(dim=0)

        # Normalize by per-grid routing capacity
        h_cong = h_cong / self.grid_h_routes
        v_cong = v_cong / self.grid_v_routes
        return h_cong, v_cong

    def macro_route_congestion(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable macro footprint routing contribution.

        Canonical: for each hard MACRO, add macro's V/H routing density across the
        cells its footprint overlaps. Each cell gets weight ~ (cell_overlap_area /
        cell_area) × vroutes_per_micron × cell_height (and similar for H).
        """
        device = positions.device
        if self.num_hard == 0:
            return (torch.zeros(self.gr, self.gc, device=device),
                    torch.zeros(self.gr, self.gc, device=device))
        hard_pos = positions[:self.num_hard]   # [n_hard, 2]
        hard_sz = self.macro_sizes[:self.num_hard]   # [n_hard, 2]
        # Macro footprint
        x_min = hard_pos[:, 0:1] - hard_sz[:, 0:1] / 2  # [n_hard, 1]
        x_max = hard_pos[:, 0:1] + hard_sz[:, 0:1] / 2
        y_min = hard_pos[:, 1:2] - hard_sz[:, 1:2] / 2
        y_max = hard_pos[:, 1:2] + hard_sz[:, 1:2] / 2

        # Cell boundaries
        cell_x_min = (self.cell_x_ctr - self.cell_w / 2).view(1, -1)   # [1, gc]
        cell_x_max = (self.cell_x_ctr + self.cell_w / 2).view(1, -1)
        cell_y_min = (self.cell_y_ctr - self.cell_h / 2).view(1, -1)   # [1, gr]
        cell_y_max = (self.cell_y_ctr + self.cell_h / 2).view(1, -1)

        # Overlap (smooth via clamp)
        x_ol = torch.clamp(torch.minimum(x_max, cell_x_max) - torch.maximum(x_min, cell_x_min), min=0)
        y_ol = torch.clamp(torch.minimum(y_max, cell_y_max) - torch.maximum(y_min, cell_y_min), min=0)
        # x_ol: [n_hard, gc], y_ol: [n_hard, gr]
        # cell_overlap_area = x_ol[h, c] * y_ol[h, r] → [n_hard, gr, gc]
        ol_area = y_ol.unsqueeze(2) * x_ol.unsqueeze(1)
        # macro contribution: per cell, routes_per_micron × cell_height × overlap_fraction
        # Canonical adds 1 per fully-covered cell; for partial, ratio.
        cell_area = self.cell_w * self.cell_h
        macro_cong_h = (ol_area / cell_area).sum(dim=0)   # [gr, gc]
        macro_cong_v = macro_cong_h.clone()
        return macro_cong_h, macro_cong_v

    def congestion_cost(self, positions: torch.Tensor, abu_k: float = 0.05) -> torch.Tensor:
        h_net, v_net = self.trace_congestion(positions)
        h_macro, v_macro = self.macro_route_congestion(positions)
        total = (h_net + h_macro) + (v_net + v_macro)
        flat = total.flatten()
        k = max(1, int(len(flat) * abu_k))
        top, _ = torch.topk(flat, k)
        return top.mean()


def quick_smoke(bench_name: str = "ibm01"):
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.bench_paths import find_benchmark_dir
    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))
    rudy = DiffTraceRudyV2(bench, plc, device="cpu")
    print(f"{bench_name}: n_pairs={rudy.n_pairs}, grid={rudy.gr}x{rudy.gc}")
    pos = bench.macro_positions.clone().float().requires_grad_(True)
    cost = rudy.congestion_cost(pos)
    print(f"  smooth ABU-5: {float(cost):.5f}")
    cost.backward()
    print(f"  grad norm: {pos.grad.norm():.5f}")


if __name__ == "__main__":
    quick_smoke(sys.argv[1] if len(sys.argv) > 1 else "ibm01")
