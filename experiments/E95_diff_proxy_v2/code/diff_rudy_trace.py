"""Differentiable RUDY congestion — Stage 1: scalar match canonical.

This implements canonical's `PlacementCost.get_routing` algorithm using
torch ops. **Stage 1** uses hard cell membership (integer floor of pin
position / cell_size) — NOT differentiable yet. Goal is scalar match
to within 1 % of canonical on cached cascade outputs.

**Stage 2** (future work, per DIFF_RUDY_REWRITE_SPEC.md):
  - Soft pin-to-cell membership via Gaussian kernel
  - Soft trace path indicator via sigmoid soft_between
  - Vectorize per-net loop into batched torch ops
  - Validate gradient direction aligns with canonical

For now this validates that the algorithm specification is correct
before adding the differentiability layer.

Usage:
    proxy = DiffRudyTrace(benchmark, plc)
    cong_cost = proxy.compute_congestion(positions)
    # Should match `compute_proxy_cost(positions, benchmark, plc)["congestion_cost"]`
    # to within ~1 % on cached cascade outputs.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F


def _net_data_from_plc(benchmark, plc):
    """Extract per-net (pin_macro_idx, pin_offset, weight) like _extract_net_data
    but keep python lists for the Stage 1 hard-cell implementation.

    Returns: nets (list of (pin_specs, weight)) where
        pin_specs: list of ("macro", bench_idx, off_x, off_y) or ("port", x, y)
    """
    # plc node-id → bench tensor index
    plc_to_tensor = {}
    for t_idx, p_idx in enumerate(benchmark.hard_macro_indices):
        plc_to_tensor[p_idx] = t_idx
    for i, p_idx in enumerate(benchmark.soft_macro_indices):
        plc_to_tensor[p_idx] = benchmark.num_hard_macros + i

    nets = []
    for driver_name, sink_names in plc.nets.items():
        d_idx = plc.mod_name_to_indices.get(driver_name)
        if d_idx is None:
            continue
        driver_pin = plc.modules_w_pins[d_idx]
        weight = float(driver_pin.get_weight())

        pin_specs = []
        for pin_name in [driver_name] + list(sink_names):
            pidx = plc.mod_name_to_indices.get(pin_name)
            if pidx is None:
                continue
            pin = plc.modules_w_pins[pidx]
            ptype = pin.get_type()
            if ptype in ("MACRO_PIN", "SOFT_MACRO_PIN"):
                parent_name = pin_name.split("/")[0]
                parent_pidx = plc.mod_name_to_indices.get(parent_name)
                if parent_pidx is None or parent_pidx not in plc_to_tensor:
                    continue
                t_idx = plc_to_tensor[parent_pidx]
                xo, yo = pin.get_offset()
                pin_specs.append(("macro", t_idx, float(xo), float(yo)))
            elif ptype == "PORT":
                x, y = pin.get_pos()
                pin_specs.append(("port", float(x), float(y)))
        if len(pin_specs) >= 2:
            nets.append((pin_specs, weight))
    return nets


class DiffRudyTrace:
    """Stage 1 implementation. Hard cell membership; no autograd flow."""

    def __init__(self, benchmark, plc, smooth_range: int = None):
        self.benchmark = benchmark
        self.plc = plc
        self.canvas_w = float(benchmark.canvas_width)
        self.canvas_h = float(benchmark.canvas_height)
        self.grid_cols = int(benchmark.grid_cols)
        self.grid_rows = int(benchmark.grid_rows)
        self.cell_w = self.canvas_w / self.grid_cols
        self.cell_h = self.canvas_h / self.grid_rows
        self.hroutes_per_micron = float(benchmark.hroutes_per_micron)
        self.vroutes_per_micron = float(benchmark.vroutes_per_micron)
        self.grid_h_routes = self.cell_h * self.hroutes_per_micron
        self.grid_v_routes = self.cell_w * self.vroutes_per_micron

        # Macro routing allocation constants (read from plc)
        self.hrouting_alloc = float(plc.hrouting_alloc)
        self.vrouting_alloc = float(plc.vrouting_alloc)

        # Smoothing range (canonical's smooth_range — typically 2)
        if smooth_range is None:
            smooth_range = int(plc.get_congestion_smooth_range())
        self.smooth_range = int(smooth_range)

        # Per-net pin data
        self.nets = _net_data_from_plc(benchmark, plc)

        # Macro sizes
        self.macro_sizes = benchmark.macro_sizes  # [num_macros, 2]
        self.num_hard = int(benchmark.num_hard_macros)
        self.num_macros = int(benchmark.num_macros)

    # ----------------------------------------------------------- pin → cell

    def _pin_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Canonical __get_grid_cell_location with bounds clamp."""
        row = int(math.floor(y / self.cell_h))
        col = int(math.floor(x / self.cell_w))
        row = max(0, min(row, self.grid_rows - 1))
        col = max(0, min(col, self.grid_cols - 1))
        return row, col

    # ------------------------------------------------------- trace per net

    def _two_pin_trace(self, source_rc, sink_rc, weight: float,
                       H: torch.Tensor, V: torch.Tensor):
        """Canonical __two_pin_net_routing: L-shape, H at source row."""
        sr, sc = source_rc
        kr, kc = sink_rc
        col_lo, col_hi = min(sc, kc), max(sc, kc)
        for c in range(col_lo, col_hi):  # range exclusive of col_hi (matches canon)
            H[sr, c] += weight
        row_lo, row_hi = min(sr, kr), max(sr, kr)
        for r in range(row_lo, row_hi):
            V[r, kc] += weight

    def _l_routing(self, sorted_gcells, weight, H, V):
        """Canonical __l_routing for 3-pin (3 gcells sorted by (col, row))."""
        y1, x1 = sorted_gcells[0]
        y2, x2 = sorted_gcells[1]
        y3, x3 = sorted_gcells[2]
        for c in range(x1, x2): H[y1, c] += weight
        for c in range(x2, x3): H[y2, c] += weight
        for r in range(min(y1, y2), max(y1, y2)): V[r, x2] += weight
        for r in range(min(y2, y3), max(y2, y3)): V[r, x3] += weight

    def _t_routing(self, sorted_gcells, weight, H, V):
        """Canonical __t_routing for 3-pin (sorted by (row, col))."""
        # canonical sorts via list.sort() (default tuple comparison → (row, col))
        y1, x1 = sorted_gcells[0]
        y2, x2 = sorted_gcells[1]
        y3, x3 = sorted_gcells[2]
        xmin = min(x1, x2, x3)
        xmax = max(x1, x2, x3)
        for c in range(xmin, xmax): H[y2, c] += weight
        for r in range(min(y1, y2), max(y1, y2)): V[r, x1] += weight
        for r in range(min(y2, y3), max(y2, y3)): V[r, x3] += weight

    def _three_pin_trace(self, gcells, weight, H, V):
        """Canonical __three_pin_net_routing dispatch."""
        # Sorted by (col, row) for first check
        gc = sorted(gcells, key=lambda x: (x[1], x[0]))
        y1, x1 = gc[0]
        y2, x2 = gc[1]
        y3, x3 = gc[2]
        if x1 < x2 < x3 and min(y1, y3) < y2 < max(y1, y3):
            self._l_routing(gc, weight, H, V)
        elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
            for c in range(x1, x2): H[y1, c] += weight
            for r in range(y1, max(y2, y3)): V[r, x2] += weight
        elif y2 == y3:
            for c in range(x1, x2): H[y1, c] += weight
            for c in range(x2, x3): H[y2, c] += weight
            for r in range(min(y2, y1), max(y2, y1)): V[r, x2] += weight
        else:
            # Canonical re-sorts here via list.sort() (default tuple) → (row, col)
            gc_rs = sorted(gcells)
            self._t_routing(gc_rs, weight, H, V)

    # ------------------------------------------------ macro routing fill-in

    def _macro_routes(self, positions, H_macro: torch.Tensor, V_macro: torch.Tensor):
        """Canonical __macro_route_over_grid_cell — only for HARD macros."""
        sizes = self.macro_sizes
        for i in range(self.num_hard):
            mx = float(positions[i, 0])
            my = float(positions[i, 1])
            mw = float(sizes[i, 0])
            mh = float(sizes[i, 1])
            # Two corners
            ur_x, ur_y = mx + mw / 2, my + mh / 2
            bl_x, bl_y = mx - mw / 2, my - mh / 2

            ur_row, ur_col = self._pin_cell(ur_x, ur_y)
            bl_row, bl_col = self._pin_cell(bl_x, bl_y)

            # canonical's OOB handling
            if ur_row < 0 or ur_col < 0: continue
            if bl_row < 0: bl_row = 0
            if bl_col < 0: bl_col = 0
            if bl_row < 0 or bl_col < 0: continue
            if ur_row > self.grid_rows - 1: ur_row = self.grid_rows - 1
            if ur_col > self.grid_cols - 1: ur_col = self.grid_cols - 1

            partial_v = False
            partial_h = False

            x_max = mx + mw / 2
            x_min = mx - mw / 2
            y_max = my + mh / 2
            y_min = my - mh / 2

            for r_i in range(bl_row, ur_row + 1):
                for c_i in range(bl_col, ur_col + 1):
                    cell_x_min = c_i * self.cell_w
                    cell_x_max = (c_i + 1) * self.cell_w
                    cell_y_min = r_i * self.cell_h
                    cell_y_max = (r_i + 1) * self.cell_h
                    x_dist = max(0.0, min(x_max, cell_x_max) - max(x_min, cell_x_min))
                    y_dist = max(0.0, min(y_max, cell_y_max) - max(y_min, cell_y_min))

                    if ur_row != bl_row:
                        if (r_i == bl_row and abs(y_dist - self.cell_h) > 1e-5) or \
                           (r_i == ur_row and abs(y_dist - self.cell_h) > 1e-5):
                            partial_v = True
                    if ur_col != bl_col:
                        if (c_i == bl_col and abs(x_dist - self.cell_w) > 1e-5) or \
                           (c_i == ur_col and abs(x_dist - self.cell_w) > 1e-5):
                            partial_h = True

                    V_macro[r_i, c_i] += x_dist * self.vrouting_alloc
                    H_macro[r_i, c_i] += y_dist * self.hrouting_alloc

            if partial_v:
                r_i = ur_row
                for c_i in range(bl_col, ur_col + 1):
                    cell_x_min = c_i * self.cell_w
                    cell_x_max = (c_i + 1) * self.cell_w
                    x_dist = max(0.0, min(x_max, cell_x_max) - max(x_min, cell_x_min))
                    V_macro[r_i, c_i] -= x_dist * self.vrouting_alloc

            if partial_h:
                c_i = ur_col
                for r_i in range(bl_row, ur_row + 1):
                    cell_y_min = r_i * self.cell_h
                    cell_y_max = (r_i + 1) * self.cell_h
                    y_dist = max(0.0, min(y_max, cell_y_max) - max(y_min, cell_y_min))
                    H_macro[r_i, c_i] -= y_dist * self.hrouting_alloc

    # -------------------------------------------------- box smoothing

    def _smooth(self, M: torch.Tensor, axis: str) -> torch.Tensor:
        """Canonical __smooth_routing_cong axis-aligned ±smooth_range box smoothing.

        For V: smooth across COLs within each row.
        For H: smooth across ROWs within each col.
        """
        sr = self.smooth_range
        if sr <= 0:
            return M
        out = torch.zeros_like(M)
        R, C = M.shape
        if axis == "V":
            # for each (row, col), distribute M[row, col] uniformly across
            # [max(0, col-sr) .. min(C-1, col+sr)] inclusive
            for row in range(R):
                for col in range(C):
                    lp = max(0, col - sr)
                    rp = min(C - 1, col + sr)
                    gcell_cnt = rp - lp + 1
                    val = M[row, col].item() / gcell_cnt
                    for ptr in range(lp, rp + 1):
                        out[row, ptr] += val
        elif axis == "H":
            for col in range(C):
                for row in range(R):
                    lp = max(0, row - sr)
                    rp = min(R - 1, row + sr)
                    gcell_cnt = rp - lp + 1
                    val = M[row, col].item() / gcell_cnt
                    for ptr in range(lp, rp + 1):
                        out[ptr, col] += val
        return out

    # -------------------------------------------------- compute_congestion

    def compute_congestion(self, positions: torch.Tensor) -> torch.Tensor:
        """Return canonical-equivalent congestion cost (top 5% of V+H mean).

        For Stage 1 this matches canonical's scalar; gradient does NOT flow
        through pin → cell membership (intentionally — Stage 2 will fix).
        """
        positions = positions.detach().cpu()
        # Per-pin (x, y) computation
        H = torch.zeros(self.grid_rows, self.grid_cols, dtype=torch.float64)
        V = torch.zeros(self.grid_rows, self.grid_cols, dtype=torch.float64)

        for pin_specs, weight in self.nets:
            # Compute each pin's gcell
            gcells = set()
            source_gcell = None
            for k, spec in enumerate(pin_specs):
                if spec[0] == "macro":
                    _, bi, ox, oy = spec
                    x = float(positions[bi, 0]) + ox
                    y = float(positions[bi, 1]) + oy
                else:
                    _, x, y = spec
                gc = self._pin_cell(x, y)
                gcells.add(gc)
                if k == 0:
                    source_gcell = gc

            n = len(gcells)
            if n <= 1:
                continue
            elif n == 2:
                gc_list = list(gcells)
                sink = gc_list[1] if gc_list[0] == source_gcell else gc_list[0]
                self._two_pin_trace(source_gcell, sink, weight, H, V)
            elif n == 3:
                self._three_pin_trace(list(gcells), weight, H, V)
            else:
                # Split into 2-pin pairs from source
                for gc in gcells:
                    if gc != source_gcell:
                        self._two_pin_trace(source_gcell, gc, weight, H, V)

        # Macro routing contribution
        H_macro = torch.zeros(self.grid_rows, self.grid_cols, dtype=torch.float64)
        V_macro = torch.zeros(self.grid_rows, self.grid_cols, dtype=torch.float64)
        self._macro_routes(positions, H_macro, V_macro)

        # Normalize by route capacity
        V = V / self.grid_v_routes
        H = H / self.grid_h_routes
        V_macro = V_macro / self.grid_v_routes
        H_macro = H_macro / self.grid_h_routes

        # Box smoothing (canonical's __smooth_routing_cong, V-axis on V map)
        V = self._smooth(V, axis="V")
        H = self._smooth(H, axis="H")

        # Sum macro + net contributions per cell
        V = V + V_macro
        H = H + H_macro

        # ABU 5% — top 5 % of concat([V, H]) (canonical uses python list
        # concatenation `V + H` on lists, NOT element-wise tensor add).
        # This means the top-K is over 2*N_cells values, not N_cells.
        combined = torch.cat([V.flatten(), H.flatten()])
        k = max(1, int(0.05 * combined.numel()))
        top_k, _ = torch.topk(combined, k)
        return top_k.mean()


if __name__ == "__main__":
    import json
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost

    results = []
    benches = ["ibm01", "ibm10"]
    if len(sys.argv) > 1:
        benches = sys.argv[1:]

    for bench in benches:
        bench_dir = Path(f"external/MacroPlacement/Testcases/ICCAD04/{bench}")
        benchmark, plc = load_benchmark_from_dir(str(bench_dir))
        trace = DiffRudyTrace(benchmark, plc)
        cached = torch.load(
            f"experiments/E84_cascading_saddle/results/cascade_{bench}.pt",
            weights_only=False, map_location="cpu",
        )
        pos = cached["placement"].clone().float()

        smooth_cong = float(trace.compute_congestion(pos))
        canon = compute_proxy_cost(pos, benchmark, plc)
        canon_cong = float(canon["congestion_cost"])
        # canonical's get_congestion_cost returns mean of top-5% of V+H combined;
        # compute_proxy_cost weights it by 0.5 by default. We want unweighted match.
        # Check what compute_proxy_cost returns: it returns `congestion_cost` = raw cong.
        gap_pct = (smooth_cong - canon_cong) / canon_cong * 100.0
        print(f"{bench}: trace={smooth_cong:.5f}  canon={canon_cong:.5f}  gap={gap_pct:+.2f}%")
        results.append({
            "bench": bench,
            "trace_cong": smooth_cong,
            "canon_cong": canon_cong,
            "gap_pct": gap_pct,
        })

    out_path = Path("experiments/E95_diff_proxy_v2/results/diff_rudy_trace_calibrate.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")
