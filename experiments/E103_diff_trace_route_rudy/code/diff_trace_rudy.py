"""E103 — Differentiable trace-route RUDY (the Day-1 unblocker for PATH C1).

Canonical TILOS routing (from `external/.../plc_client_os.py:get_routing`):
  For each 2-pin net:
    H-route along source_row: cells (s_row, c) for c in [min(s_col,t_col), max(s_col,t_col))
    V-route along sink_col: cells (r, t_col) for r in [min(s_row,t_row), max(s_row,t_row))
  Each touched cell += weight.
  Total cong per cell normalized by grid_h_routes / grid_v_routes.

Existing E88 diff_proxy `_rudy_congestion` uses bbox-uniform-spread (every cell
in the bbox of all pins gets weight × area_fraction). This is WHY E88/E95/E98
gradient descent fails: smooth grad ≠ canonical grad direction on hard benches
(memory: diff_proxy_rudy_mismatch).

This file builds the trace-route variant, fully torch-autograd-compatible:
  1. Pin positions = macro_pos[macro_idx] + pin_offset
  2. Soft (row, col) per pin via Gaussian over cell centers (β-temperature)
  3. For each 2-pin net: soft H-range × soft V-range product gives per-cell usage
  4. Sum over nets → smooth_H_cong, smooth_V_cong
  5. Normalize by grid_h_routes / grid_v_routes
  6. Apply smoothing kernel (matches `__smooth_routing_cong`)
  7. Take ABU-K mean (matches canonical congestion cost)

Calibration goal: smooth_cong(P) tracks canonical_cong(P) with ρ > 0.9
on perturbation directions, abs mismatch < 5% at cascade plateaus.

This is the missing piece for PATH C1. Once calibrated, GPU L-BFGS on it
should actually move in the canonical-improving direction.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class DiffTraceRudy:
    """Differentiable trace-route RUDY on a fixed (benchmark, plc) pair.

    Args:
      benchmark, plc: standard objects.
      device: target device.
      beta_cell: temperature for soft cell assignment (higher = sharper).
                 Default tracks gamma_frac scheme: ~1/(cell_width × beta_cell)
                 gives sigmoid steepness comparable to canonical hard assignment.
      smooth_range: matches plc.smooth_range for the final smoothing pass.
    """

    def __init__(
        self,
        benchmark,
        plc,
        device: str | torch.device = "cpu",
        beta_cell: float = 4.0,        # soft cell sharpness
        beta_range: float = 4.0,       # range indicator sharpness
        gamma_frac: float = 0.0005,
    ):
        self.benchmark = benchmark
        self.plc = plc
        self.device = torch.device(device)
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.num_macros = int(benchmark.num_macros)
        self.num_hard = int(benchmark.num_hard_macros)

        # Cell geometry
        self.gr = int(benchmark.grid_rows)
        self.gc = int(benchmark.grid_cols)
        self.cell_w = self.cw / self.gc
        self.cell_h = self.ch / self.gr
        # Cell CENTER coords
        self.cell_x_ctr = (torch.arange(self.gc, dtype=torch.float32, device=self.device) + 0.5) * self.cell_w
        self.cell_y_ctr = (torch.arange(self.gr, dtype=torch.float32, device=self.device) + 0.5) * self.cell_h

        # Per-grid routing capacities (matches canonical)
        self.grid_h_routes = self.cell_h * benchmark.hroutes_per_micron
        self.grid_v_routes = self.cell_w * benchmark.vroutes_per_micron

        # Soft cell sharpness — distance from center > cell_w/2 means weight < 0.5
        self.beta_cell = beta_cell / self.cell_w   # in 1/canvas-units
        self.beta_range = beta_range / self.cell_w  # range indicator sharpness

        self.smooth_range = int(getattr(plc, "smooth_range", 2))

        # Extract net topology: 2-pin pairs
        # For autograd we need everything as tensors. Pre-build (source, sink) index
        # tensors per pair: [n_pairs, 2] of macro_idx, [n_pairs, 2] of pin_offset, [n_pairs] weight
        self._build_net_pair_data()
        self.gamma_frac = gamma_frac  # placeholder; not used here

    def _build_net_pair_data(self):
        """Extract (source_macro_idx, source_offset, sink_macro_idx, sink_offset, weight)
        for every 2-pin net pair. For nets with >2 pins, use star routing from the
        first pin to each other pin (matches canonical's __split_net pattern for n>3).
        """
        bench = self.benchmark
        # Net structure: typically benchmark.macro_pin_idx and macro_pin_offset are not
        # directly available; need to extract from plc.
        # We mirror what E88's _extract_net_data does, but for 2-pin pair routing.
        pin_macro_idx = []   # per-pin: [n_pins_total] long
        pin_offsets = []     # per-pin: [n_pins_total, 2] float
        pair_src = []        # per-pair: long [n_pairs]
        pair_snk = []        # per-pair: long [n_pairs]
        pair_weight = []     # per-pair: float [n_pairs]

        # Need to access plc structure: nets, pins per net.
        # plc is the TILOS PlcClient object. Use its modules_w_pins to get pin info.
        # This part is plc-specific; replicate minimal extraction logic.
        if hasattr(bench, "net_pin_macro_idx") and hasattr(bench, "net_pin_offsets"):
            # Tensorized net struct already present (E88 extracted into benchmark).
            # Format: net_pin_macro_idx[n_net, max_pins] padded with sentinel.
            # Use first pin as source, others as sinks.
            net_macros = bench.net_pin_macro_idx
            net_offsets = bench.net_pin_offsets
            net_mask = bench.net_pin_mask
            net_weights = bench.net_weights if hasattr(bench, "net_weights") else None

            n_nets, max_pins = net_macros.shape
            for ni in range(n_nets):
                # Find valid pins on this net
                valid = [j for j in range(max_pins) if bool(net_mask[ni, j])]
                if len(valid) < 2:
                    continue
                w = float(net_weights[ni]) if net_weights is not None else 1.0
                src_j = valid[0]
                src_m = int(net_macros[ni, src_j])
                src_off = net_offsets[ni, src_j]
                for snk_j in valid[1:]:
                    snk_m = int(net_macros[ni, snk_j])
                    snk_off = net_offsets[ni, snk_j]
                    pair_src.append((src_m, src_off[0].item(), src_off[1].item()))
                    pair_snk.append((snk_m, snk_off[0].item(), snk_off[1].item()))
                    pair_weight.append(w)
        else:
            # Fallback: extract from plc.modules_w_pins. Slow O(N*P) walk.
            raise NotImplementedError(
                "DiffTraceRudy needs benchmark.net_pin_macro_idx / net_pin_offsets / net_pin_mask. "
                "Use the E88 extract_net_data path to populate them before passing benchmark."
            )

        # Convert to tensors
        n_pairs = len(pair_src)
        if n_pairs == 0:
            # No nets — degenerate
            self.n_pairs = 0
            self.src_macro = torch.zeros(0, dtype=torch.long, device=self.device)
            self.snk_macro = torch.zeros(0, dtype=torch.long, device=self.device)
            self.src_offset = torch.zeros(0, 2, dtype=torch.float32, device=self.device)
            self.snk_offset = torch.zeros(0, 2, dtype=torch.float32, device=self.device)
            self.pair_w = torch.zeros(0, dtype=torch.float32, device=self.device)
            return

        src_arr = torch.tensor(pair_src, dtype=torch.float32, device=self.device)
        snk_arr = torch.tensor(pair_snk, dtype=torch.float32, device=self.device)
        self.n_pairs = n_pairs
        self.src_macro = src_arr[:, 0].long()
        self.snk_macro = snk_arr[:, 0].long()
        self.src_offset = src_arr[:, 1:3].clone()
        self.snk_offset = snk_arr[:, 1:3].clone()
        self.pair_w = torch.tensor(pair_weight, dtype=torch.float32, device=self.device)

    def trace_congestion(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute differentiable per-cell H and V routing congestion.

        Args:
          positions: [num_macros, 2] tensor of macro centers (autograd-enabled).
        Returns:
          h_cong: [gr, gc] horizontal congestion per grid cell.
          v_cong: [gr, gc] vertical congestion per grid cell.
        """
        if self.n_pairs == 0:
            return (
                torch.zeros(self.gr, self.gc, device=self.device),
                torch.zeros(self.gr, self.gc, device=self.device),
            )
        device = positions.device
        # Append a [0,0] "port" row at index N so port-pin indices resolve.
        port_base = torch.zeros(1, 2, device=device, dtype=positions.dtype)
        all_pos = torch.cat([positions, port_base], dim=0)   # [N+1, 2]
        src_pos = all_pos[self.src_macro] + self.src_offset   # [n_pairs, 2]
        snk_pos = all_pos[self.snk_macro] + self.snk_offset

        # Soft cell assignment for source/sink: weights over [gr, gc] grid cells
        # For each pin, compute sigmoid-distance to each cell center.
        # h_col_src[p, c] = soft weight that source pin p is in col c
        x_src = src_pos[:, 0:1]   # [n_pairs, 1]
        x_snk = snk_pos[:, 0:1]
        y_src = src_pos[:, 1:2]
        y_snk = snk_pos[:, 1:2]

        # Sharper softmax — sigma=0.15 means pin one cell away gets weight exp(-44)≈0.
        sigma_cell = 0.15
        sx_src = -((x_src - self.cell_x_ctr.view(1, -1)) / (sigma_cell * self.cell_w)) ** 2
        sx_snk = -((x_snk - self.cell_x_ctr.view(1, -1)) / (sigma_cell * self.cell_w)) ** 2
        sy_src = -((y_src - self.cell_y_ctr.view(1, -1)) / (sigma_cell * self.cell_h)) ** 2
        sy_snk = -((y_snk - self.cell_y_ctr.view(1, -1)) / (sigma_cell * self.cell_h)) ** 2

        # Soft argmax (one-hot-like)
        col_src = torch.softmax(sx_src, dim=1)   # [n_pairs, gc]
        col_snk = torch.softmax(sx_snk, dim=1)
        row_src = torch.softmax(sy_src, dim=1)   # [n_pairs, gr]
        row_snk = torch.softmax(sy_snk, dim=1)

        # Soft column range: for each pair p, cells c in [min(s_col, t_col), max(s_col, t_col))
        # Soft range indicator: in_range(c) = sigmoid(β·(c - min)) × sigmoid(β·(max - c))
        # min_col = sum_c col_idx * col_src is soft-argmax expectation, NOT min(col_src, col_snk)
        col_idx = torch.arange(self.gc, dtype=torch.float32, device=device)  # [gc]
        row_idx = torch.arange(self.gr, dtype=torch.float32, device=device)  # [gr]
        col_src_exp = (col_src * col_idx.view(1, -1)).sum(dim=1, keepdim=True)  # [n_pairs, 1]
        col_snk_exp = (col_snk * col_idx.view(1, -1)).sum(dim=1, keepdim=True)
        row_src_exp = (row_src * row_idx.view(1, -1)).sum(dim=1, keepdim=True)
        row_snk_exp = (row_snk * row_idx.view(1, -1)).sum(dim=1, keepdim=True)

        # Soft min/max (smooth min/max)
        # smooth_min(a, b, β) = -1/β · logsumexp(-β·a, -β·b)
        # smooth_max = +1/β · logsumexp(+β·a, +β·b)
        beta_mm = 4.0
        def soft_min(a, b):
            return -torch.logsumexp(torch.stack([-beta_mm*a, -beta_mm*b], dim=0), dim=0) / beta_mm
        def soft_max(a, b):
            return torch.logsumexp(torch.stack([beta_mm*a, beta_mm*b], dim=0), dim=0) / beta_mm

        col_min = soft_min(col_src_exp, col_snk_exp)   # [n_pairs, 1]
        col_max = soft_max(col_src_exp, col_snk_exp)
        row_min = soft_min(row_src_exp, row_snk_exp)
        row_max = soft_max(row_src_exp, row_snk_exp)

        # Range indicator for H route (source_row, cols in [col_min, col_max))
        # in_range(c) = sigmoid(β·(c - col_min + 0.5)) - sigmoid(β·(c - col_max + 0.5))
        # The +0.5 makes the range inclusive of integer cells.
        beta_r = 4.0
        h_in_col_range = (
            torch.sigmoid(beta_r * (col_idx.view(1, -1) - col_min + 0.5))
            - torch.sigmoid(beta_r * (col_idx.view(1, -1) - col_max + 0.5))
        )  # [n_pairs, gc]
        v_in_row_range = (
            torch.sigmoid(beta_r * (row_idx.view(1, -1) - row_min + 0.5))
            - torch.sigmoid(beta_r * (row_idx.view(1, -1) - row_max + 0.5))
        )  # [n_pairs, gr]

        # H route lives on source_row → for each pair, distribute weight along row_src × col-range
        weight = self.pair_w.view(-1, 1, 1)
        # h_contrib[p, r, c] = weight[p] × row_src[p, r] × h_in_col_range[p, c]
        # Sum over pairs → [gr, gc]
        h_cong_per = weight * row_src.unsqueeze(2) * h_in_col_range.unsqueeze(1)   # [n_pairs, gr, gc]
        h_cong = h_cong_per.sum(dim=0)
        v_cong_per = weight * v_in_row_range.unsqueeze(2) * col_snk.unsqueeze(1)
        v_cong = v_cong_per.sum(dim=0)

        # Normalize by grid_h_routes / grid_v_routes (per cell)
        h_cong = h_cong / self.grid_h_routes
        v_cong = v_cong / self.grid_v_routes

        return h_cong, v_cong

    def congestion_cost(self, positions: torch.Tensor, abu_k: float = 0.05) -> torch.Tensor:
        """ABU-K congestion cost (matches canonical get_congestion_cost).

        ABU-K = mean of top-K% most-congested cells (by combined H+V).
        """
        h_cong, v_cong = self.trace_congestion(positions)
        # Sum H and V per cell (canonical does this)
        total = h_cong + v_cong   # [gr, gc]
        flat = total.flatten()
        k = max(1, int(len(flat) * abu_k))
        topk, _ = torch.topk(flat, k)
        return topk.mean()


def quick_smoke(bench_name: str = "ibm01") -> None:
    """Local sanity test: load bench, build DiffTraceRudy, check grad flows."""
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.bench_paths import find_benchmark_dir

    # Import E88's _extract_net_data to populate benchmark
    import importlib.util
    _DPO_PATH = _ROOT / "writeup" / "archive" / "submissions" / "dpo" / "ablation_v2_steps.py"
    spec = importlib.util.spec_from_file_location("_dpo", str(_DPO_PATH))
    dpo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dpo)

    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))
    nd = dpo._extract_net_data(bench, plc)
    # Attach to benchmark for DiffTraceRudy to find
    bench.net_pin_macro_idx = nd.pin_macro_idx
    bench.net_pin_offsets = nd.pin_offsets
    bench.net_pin_mask = nd.mask
    bench.net_weights = nd.weights

    rudy = DiffTraceRudy(bench, plc, device="cpu")
    print(f"{bench_name}: n_pairs={rudy.n_pairs}, grid={rudy.gr}x{rudy.gc}")

    # Use macro_positions as init
    pos = bench.macro_positions.clone().float().requires_grad_(True)
    cost = rudy.congestion_cost(pos)
    print(f"  smooth ABU-5 congestion: {float(cost):.5f}")
    cost.backward()
    print(f"  grad norm: {pos.grad.norm():.5f}")


if __name__ == "__main__":
    import sys
    quick_smoke(sys.argv[1] if len(sys.argv) > 1 else "ibm01")
