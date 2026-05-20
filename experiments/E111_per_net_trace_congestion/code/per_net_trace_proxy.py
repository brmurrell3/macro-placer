"""E111 — Per-net-trace differentiable RUDY proxy.

Canonical `PlacementCost.get_routing` (plc_client_os.py:1514) computes
routing congestion as follows:

  for each net N:
      assign each pin to its gcell via floor(y/cell_h), floor(x/cell_w)
      if |unique_gcells| == 2: L-route — H stripe at source_row from
          src_col..sink_col, V stripe at sink_col from src_row..sink_row.
      if |unique_gcells| == 3: L- or T-shape Steiner.
      if |unique_gcells| > 3: split into (source, sink) 2-pin pairs, L-route each.
  for each hard MACRO: add footprint × *routing_alloc to V_macro / H_macro.
  normalize: V /= grid_v_routes,  H /= grid_h_routes.
  box-smooth ±smooth_range over cols (V) / rows (H).
  V_total = V_routes + V_macro,  H_total = H_routes + H_macro.
  congestion = mean of top-5% of (V_total + H_total)  # LIST CONCAT, 2N cells.

The DPO-era smooth `_rudy_congestion` (ablation_v2_steps.py:505) spreads
each net's weight UNIFORMLY across the cells of its bounding box. That's
geometrically wrong: the canonical only routes one row (source row) and
one column (sink col), not the whole bbox. On hard benches (ibm10/12/17)
the smooth surrogate diverges 3-4× from canonical — gradient descent then
walks toward smooth-favorable basins that canonical doesn't reward.

This file implements a differentiable approximation that **traces** the
L-route per (source, sink) pair, plus the macro footprint contribution
and the ±smooth_range smoothing. Reuses ideas from E103 v2
(diff_trace_rudy_v2.py) but fixes the memory blow-up (chunked accumulation
instead of full [n_pairs, gr, gc] materialization), adds the macro
footprint and the box smoothing, and uses concat-top-K.

Designed as a drop-in replacement for `_rudy_congestion(...)` in
DiffProxyV2. Constructor needs (benchmark, plc) only; pin / net
extraction matches E88 / DPO conventions (port pins at index num_macros).
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
_spec = importlib.util.spec_from_file_location("_e111_dpo", str(_DPO_PATH))
_dpo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dpo)
_extract_net_data = _dpo._extract_net_data
NetData = _dpo.NetData


class PerNetTraceCongestion:
    """Differentiable per-net L-route congestion proxy.

    Mirrors canonical `get_routing` topology:
      - For each net with k pins (k >= 2), star-routed from pin0.
      - Each (source, sink) pair contributes one H-stripe and one V-stripe.
      - Hard macros contribute footprint routing.
      - ±smooth_range box smoothing (2D conv).
      - ABU-5% over concat(V, H).

    All ops differentiable w.r.t. `positions`.

    Memory: O(B * (gr + gc)) per chunk + O(gr * gc) for cong maps.
    No [n_pairs, gr, gc] materialization.
    """

    def __init__(
        self,
        benchmark,
        plc,
        device: str | torch.device = "cpu",
        sigma_cell_frac: float = 0.5,  # Gaussian sharpness for pin-to-cell soft assign
        beta_range_per_cell: float = 4.0,  # sigmoid steepness for in-range indicator
        beta_minmax: float = 6.0,  # softmin/softmax steepness for range endpoints
        pair_chunk_size: int = 4096,  # process this many (src, sink) pairs at a time
        abu_k: float = 0.05,
        include_macro_routing: bool = True,
        include_smoothing: bool = True,
    ):
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

        # Cell CENTER coords (canonical floor() function gives cell index where
        # cell starts at i*cell_w; cell center is (i+0.5)*cell_w).
        self.cell_x_ctr = (torch.arange(self.gc, dtype=torch.float32, device=self.device) + 0.5) * self.cell_w
        self.cell_y_ctr = (torch.arange(self.gr, dtype=torch.float32, device=self.device) + 0.5) * self.cell_h
        # Cell INDICES (for range indicators in cell-index space)
        self.col_idx_f = torch.arange(self.gc, dtype=torch.float32, device=self.device)
        self.row_idx_f = torch.arange(self.gr, dtype=torch.float32, device=self.device)

        # Routing capacity (canonical normalizes by this)
        self.grid_h_routes = self.cell_h * benchmark.hroutes_per_micron
        self.grid_v_routes = self.cell_w * benchmark.vroutes_per_micron

        # Macro routing allocation (canonical-side constants)
        self.hrouting_alloc = float(getattr(plc, "hrouting_alloc", 1.0))
        self.vrouting_alloc = float(getattr(plc, "vrouting_alloc", 1.0))

        # Soft-assignment hyperparameters
        self.sigma_cell_frac = float(sigma_cell_frac)
        self.beta_range_per_cell = float(beta_range_per_cell)
        self.beta_minmax = float(beta_minmax)
        self.pair_chunk_size = int(pair_chunk_size)
        self.abu_k = float(abu_k)
        self.include_macro_routing = bool(include_macro_routing)
        self.include_smoothing = bool(include_smoothing)
        self.smooth_range = int(getattr(plc, "smooth_range", 2))

        # Extract per-net data once (E88's NetData), then expand to per-pair (star).
        nd = _extract_net_data(benchmark, plc)
        self._build_pair_data(nd)

        # Macro sizes (for footprint contribution)
        self.macro_sizes = benchmark.macro_sizes.to(self.device)
        self.macro_fixed = benchmark.macro_fixed.to(self.device)
        # Cell edges (for the soft macro footprint overlap)
        self.cell_x_min = self.cell_x_ctr - self.cell_w / 2.0
        self.cell_x_max = self.cell_x_ctr + self.cell_w / 2.0
        self.cell_y_min = self.cell_y_ctr - self.cell_h / 2.0
        self.cell_y_max = self.cell_y_ctr + self.cell_h / 2.0

    def _build_pair_data(self, nd: NetData) -> None:
        """Expand NetData (padded per-net) into per-pair (source, sink) tensors.

        Star routing from pin0 to each other pin in the net. Matches canonical
        __split_net + __two_pin_net_routing for >2-pin nets. For 3-pin nets
        canonical does a Steiner (L/T) shape; we still approximate as star —
        a small fidelity loss accepted in exchange for vectorization.
        """
        if nd.weights.numel() == 0:
            self.n_pairs = 0
            self.src_macro = torch.zeros(0, dtype=torch.long, device=self.device)
            self.snk_macro = torch.zeros(0, dtype=torch.long, device=self.device)
            self.src_offset = torch.zeros(0, 2, dtype=torch.float32, device=self.device)
            self.snk_offset = torch.zeros(0, 2, dtype=torch.float32, device=self.device)
            self.pair_w = torch.zeros(0, dtype=torch.float32, device=self.device)
            return

        # nd.pin_macro_idx: [num_nets, max_pins] long
        # nd.pin_offsets:   [num_nets, max_pins, 2]
        # nd.mask:          [num_nets, max_pins] bool
        # nd.weights:       [num_nets]
        pin_macro_idx = nd.pin_macro_idx.to(self.device)
        pin_offsets = nd.pin_offsets.to(self.device).to(torch.float32)
        mask = nd.mask.to(self.device)
        weights = nd.weights.to(self.device).to(torch.float32)

        num_nets, max_pins = pin_macro_idx.shape
        # Source is pin 0 (always valid in mask).
        src_macro_net = pin_macro_idx[:, 0]  # [num_nets]
        src_offset_net = pin_offsets[:, 0]   # [num_nets, 2]

        # Sinks are pins 1..max_pins-1 where mask is True.
        sink_mask = mask[:, 1:]               # [num_nets, max_pins-1]
        sink_macro = pin_macro_idx[:, 1:]      # [num_nets, max_pins-1]
        sink_offset = pin_offsets[:, 1:]       # [num_nets, max_pins-1, 2]

        # Build per-pair flat lists via boolean indexing.
        n_sinks = sink_mask.sum(dim=1)         # [num_nets]
        net_idx = torch.repeat_interleave(
            torch.arange(num_nets, device=self.device), n_sinks
        )                                       # [n_pairs]
        # Select valid (net, sink_j) entries via mask.
        sm_flat = sink_macro[sink_mask]         # [n_pairs]
        so_flat = sink_offset[sink_mask]        # [n_pairs, 2]
        src_m = src_macro_net[net_idx]          # [n_pairs]
        src_o = src_offset_net[net_idx]         # [n_pairs, 2]
        pair_w = weights[net_idx]               # [n_pairs]

        self.n_pairs = int(net_idx.numel())
        self.src_macro = src_m
        self.snk_macro = sm_flat
        self.src_offset = src_o
        self.snk_offset = so_flat
        self.pair_w = pair_w

    # --------------------------------------------------------- soft helpers

    def _soft_cell_assign(self, pin_coord: torch.Tensor, axis: str) -> torch.Tensor:
        """Return [B, n_cells] soft cell-assignment for pins along one axis.

        Uses a Gaussian over (pin - cell_center)^2 with σ = sigma_cell_frac × cell_size.
        σ ≈ 0.5 means a pin half-way between two cells gets ~26% / ~26% mass on each;
        σ ≈ 0.3 makes it sharper (~95% on one cell at midpoint), but loses gradient
        info when pins are near cell centers. Default 0.5 is a balance.
        """
        if axis == "x":
            ctr = self.cell_x_ctr
            sz = self.cell_w
        else:
            ctr = self.cell_y_ctr
            sz = self.cell_h
        sigma = self.sigma_cell_frac * sz
        # diff: [B, n_cells]
        diff = (pin_coord.view(-1, 1) - ctr.view(1, -1)) / sigma
        # Gaussian un-normalized, then softmax for probability simplex.
        scores = -(diff ** 2)
        return torch.softmax(scores, dim=1)

    def _soft_min_max(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Smooth (min, max) of (a, b)."""
        beta = self.beta_minmax
        stacked = torch.stack([a, b], dim=0)
        soft_min = -torch.logsumexp(-beta * stacked, dim=0) / beta
        soft_max = torch.logsumexp(beta * stacked, dim=0) / beta
        return soft_min, soft_max

    # --------------------------------------------------- trace congestion

    def _trace_route_congestion(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sum L-route H and V demand across all (source, sink) pairs.

        Returns (V_route, H_route) of shape [gr, gc] each, NOT yet normalized
        by route capacity. Differentiable w.r.t. positions.
        """
        if self.n_pairs == 0:
            return (
                torch.zeros(self.gr, self.gc, device=self.device),
                torch.zeros(self.gr, self.gc, device=self.device),
            )

        device = positions.device
        # Port pseudo-macro at origin so port pins resolve to their absolute pos
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
            w = self.pair_w[b_start:b_end]                 # [B]

            src_pos = all_pos[src_m] + src_o               # [B, 2]
            snk_pos = all_pos[snk_m] + snk_o

            # Soft cell assignments for source row & sink column (the key
            # axes that determine where the L-route lives).
            row_src = self._soft_cell_assign(src_pos[:, 1], axis="y")  # [B, gr]
            col_snk = self._soft_cell_assign(snk_pos[:, 0], axis="x")  # [B, gc]
            # Also need source col and sink row to compute the range [src, snk].
            col_src = self._soft_cell_assign(src_pos[:, 0], axis="x")  # [B, gc]
            row_snk = self._soft_cell_assign(snk_pos[:, 1], axis="y")  # [B, gr]

            # Soft cell-index of each endpoint (E[i]).
            col_src_exp = (col_src * self.col_idx_f.view(1, -1)).sum(dim=1)  # [B]
            col_snk_exp = (col_snk * self.col_idx_f.view(1, -1)).sum(dim=1)
            row_src_exp = (row_src * self.row_idx_f.view(1, -1)).sum(dim=1)
            row_snk_exp = (row_snk * self.row_idx_f.view(1, -1)).sum(dim=1)

            col_min, col_max = self._soft_min_max(col_src_exp, col_snk_exp)
            row_min, row_max = self._soft_min_max(row_src_exp, row_snk_exp)
            col_min = col_min.unsqueeze(1)   # [B, 1]
            col_max = col_max.unsqueeze(1)
            row_min = row_min.unsqueeze(1)
            row_max = row_max.unsqueeze(1)

            # Smooth range indicators: sigmoid(β(c - lo + 0.5)) - sigmoid(β(c - hi + 0.5))
            # peaks at 1 inside [lo, hi-1], decays outside. The +0.5 shift makes
            # the range center-aligned with integer cell indices.
            br = self.beta_range_per_cell
            col_ind = self.col_idx_f.view(1, -1)            # [1, gc]
            row_ind = self.row_idx_f.view(1, -1)            # [1, gr]
            # Canonical: cell c is in route if c in range(lo, hi) = {lo, ..., hi-1}.
            # I(c) = sigmoid(b*(c - lo + 0.5)) - sigmoid(b*(c - hi + 0.5))
            # ≈ 1 for c in {lo .. hi-1}, 0 otherwise. The +0.5 offsets
            # center the soft-step at integer half-cells.
            h_col_range = (
                torch.sigmoid(br * (col_ind - col_min + 0.5))
                - torch.sigmoid(br * (col_ind - col_max + 0.5))
            )                                                # [B, gc]
            v_row_range = (
                torch.sigmoid(br * (row_ind - row_min + 0.5))
                - torch.sigmoid(br * (row_ind - row_max + 0.5))
            )                                                # [B, gr]

            # Per-pair H demand on grid:
            #   H[r, c] += w_p * row_src[p, r] * h_col_range[p, c]
            # Vectorize the chunk sum into a matmul to avoid the [B, gr, gc]
            # 3-D materialization:
            #   H_chunk[gr, gc] = sum_p w_p * row_src[p, gr] * h_col_range[p, gc]
            #                  = (row_src · diag(w)).T @ h_col_range
            row_src_w = row_src * w.unsqueeze(1)             # [B, gr]
            col_snk_w = col_snk * w.unsqueeze(1)             # [B, gc]
            # H lives on source row, distributed over cols in [src, snk]:
            H_route = H_route + row_src_w.transpose(0, 1) @ h_col_range
            # V lives on sink col, distributed over rows in [src, snk]:
            # V[r, c] = sum_p w_p * v_row_range[p, r] * col_snk[p, c]
            V_route = V_route + v_row_range.transpose(0, 1) @ col_snk_w

        return V_route, H_route

    # ------------------------------------------------ macro footprint route

    def _macro_route_congestion(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add each HARD macro's footprint contribution to V_macro, H_macro.

        Canonical: for each macro footprint cell, add x_dist*vrouting_alloc to V
        and y_dist*hrouting_alloc to H, where x_dist/y_dist are the macro/cell
        overlap distances. We use the smooth clamp-based overlap formula.

        (Canonical has a partial-overlap fix-up at the edges; for the smooth
        proxy we omit that — the smooth proxy is meant to match scalar +
        gradient direction, not be a bit-exact recompute.)
        """
        if not self.include_macro_routing or self.num_hard == 0:
            return (
                torch.zeros(self.gr, self.gc, device=positions.device),
                torch.zeros(self.gr, self.gc, device=positions.device),
            )
        sizes = self.macro_sizes[:self.num_hard]              # [n_hard, 2]
        pos = positions[:self.num_hard]                       # [n_hard, 2]
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
        )  # [n_hard, gc]
        y_ol = torch.clamp(
            torch.min(my_max.unsqueeze(1), self.cell_y_max.unsqueeze(0))
            - torch.max(my_min.unsqueeze(1), self.cell_y_min.unsqueeze(0)),
            min=0,
        )  # [n_hard, gr]

        # Canonical: for each cell (r, c) in macro m's footprint, add x_dist
        # (the macro/cell horizontal overlap distance, in microns) to V_macro
        # and y_dist (vertical overlap) to H_macro. The "cell is in footprint"
        # condition is x_dist > 0 AND y_dist > 0.
        #
        # Differentiable presence: clip(overlap / cell_size, 0, 1). Equals 1
        # when the macro fully covers the cell (the dominant case for hard
        # macros >> cell size); fractional at footprint edges. Differentiable
        # everywhere via clamp's subgradient.
        y_present = (y_ol / self.cell_h).clamp(max=1.0)        # [n_hard, gr]
        x_present = (x_ol / self.cell_w).clamp(max=1.0)        # [n_hard, gc]
        # V_macro[r, c] = sum_m x_ol[m, c] * I(y_ol[m, r] > 0) * vroute_alloc
        # using fractional y_present as the soft indicator. Matmul contracts m.
        V_macro = (y_present.transpose(0, 1) @ x_ol) * self.vrouting_alloc
        H_macro = (y_ol.transpose(0, 1) @ x_present) * self.hrouting_alloc
        return V_macro, H_macro

    # ------------------------------------------------------ box smoothing

    def _box_smooth(self, M: torch.Tensor, axis: str) -> torch.Tensor:
        """Canonical __smooth_routing_cong: distribute each cell's value
        uniformly across ±smooth_range cells along axis (with clipped boundary).

        For V: smooth across COLS within each row.
        For H: smooth across ROWS within each col.

        Implementation: depth-1 1-D convolution along the smoothing axis with
        a uniform kernel of width (2*smooth_range + 1) and normalization by the
        per-cell kernel count (which equals the kernel width except at the boundary
        where canonical clips).
        """
        sr = self.smooth_range
        if sr <= 0 or not self.include_smoothing:
            return M
        ksize = 2 * sr + 1
        if axis == "V":
            # Smooth across cols. Canonical does:
            #   val = M[r, c] / gcell_cnt(c);  for ptr in [c-sr..c+sr]: out[r, ptr] += val
            # where gcell_cnt(c) is the active-window size at c. That's
            # mathematically convolve(M / count, ones), where count[c] is the
            # zero-pad convolution of ones with a length-ksize kernel.
            ones = torch.ones_like(M)
            count = F.conv2d(
                ones.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, 1, ksize, device=M.device),
                padding=(0, sr),
            )[0, 0]                                                    # [gr, gc]
            M_norm = M / count
            return F.conv2d(
                M_norm.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, 1, ksize, device=M.device),
                padding=(0, sr),
            )[0, 0]
        elif axis == "H":
            # Smooth across rows (same boundary-correction logic as V branch).
            ones = torch.ones_like(M)
            count = F.conv2d(
                ones.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, ksize, 1, device=M.device),
                padding=(sr, 0),
            )[0, 0]
            M_norm = M / count
            return F.conv2d(
                M_norm.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, ksize, 1, device=M.device),
                padding=(sr, 0),
            )[0, 0]
        return M

    # --------------------------------------------------- full congestion

    def compute_congestion(self, positions: torch.Tensor) -> torch.Tensor:
        """Return ABU-5% scalar congestion cost (differentiable)."""
        # Route trace contribution (un-normalized counts)
        V_route, H_route = self._trace_route_congestion(positions)
        # Normalize by per-cell route capacity (canonical does this).
        V_route = V_route / max(self.grid_v_routes, 1e-9)
        H_route = H_route / max(self.grid_h_routes, 1e-9)

        # Macro footprint contribution (already in canonical-scale units after
        # dividing by capacity).
        V_macro, H_macro = self._macro_route_congestion(positions)
        V_macro = V_macro / max(self.grid_v_routes, 1e-9)
        H_macro = H_macro / max(self.grid_h_routes, 1e-9)

        # ±smooth_range box smoothing — net contribution only (canonical applies
        # the smoothing pass BEFORE summing with macro).
        V_route = self._box_smooth(V_route, axis="V")
        H_route = self._box_smooth(H_route, axis="H")

        V_total = V_route + V_macro
        H_total = H_route + H_macro

        # Canonical: list-concatenate V + H, then ABU-5% of all 2*N values.
        combined = torch.cat([V_total.flatten(), H_total.flatten()])
        k = max(1, int(self.abu_k * combined.numel()))
        top_k, _ = torch.topk(combined, k)
        return top_k.mean()


# ---------------------------------------------------------------------------
# Self-test (calibration vs canonical)
# ---------------------------------------------------------------------------

def calibrate(bench_name: str = "ibm17", n_perturb: int = 16, perturb_frac: float = 0.02):
    """Compare smooth vs canonical congestion on cached cascade placements.

    Outputs:
      - absolute mismatch %
      - Pearson + Spearman ρ on Δcong from random perturbations
    """
    import numpy as np
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.objective import compute_proxy_cost

    print(f"\n=== E111 calibrate: {bench_name} ===", flush=True)
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    # Load cached cascade plateau if available; else use macro_positions.
    cached = _ROOT / "experiments" / "E84_cascading_saddle" / "results" / f"cascade_{bench_name}.pt"
    if cached.exists():
        data = torch.load(cached, weights_only=False, map_location="cpu")
        start = (data.get("placement") if "placement" in data else data["polished_placement"]).to(torch.float32)
        print(f"  using cached cascade placement", flush=True)
    else:
        start = benchmark.macro_positions.clone().float()
        print(f"  using macro_positions", flush=True)

    proxy = PerNetTraceCongestion(benchmark, plc, device="cpu")
    print(f"  built proxy: n_pairs={proxy.n_pairs}, grid={proxy.gr}x{proxy.gc}", flush=True)

    can = compute_proxy_cost(start, benchmark, plc)
    can_cong = float(can["congestion_cost"])
    with torch.no_grad():
        sm_cong = float(proxy.compute_congestion(start))
    rel_pct = (sm_cong - can_cong) / max(abs(can_cong), 1e-6) * 100.0
    print(f"  canonical cong: {can_cong:.5f}", flush=True)
    print(f"  smooth   cong:  {sm_cong:.5f}", flush=True)
    print(f"  rel mismatch:   {rel_pct:+.1f}%", flush=True)

    rng = np.random.default_rng(42)
    fixed = benchmark.macro_fixed.cpu().numpy()
    cw = float(benchmark.canvas_width)
    movable_hard = [i for i in range(benchmark.num_hard_macros) if not bool(fixed[i])]
    scale = perturb_frac * cw

    can_deltas = []
    sm_deltas = []
    base_can = can_cong
    base_sm = sm_cong
    for k in range(n_perturb):
        p = start.clone()
        targets = rng.choice(movable_hard, size=min(5, len(movable_hard)), replace=False)
        for t in targets:
            dx = float(rng.normal(0.0, scale))
            dy = float(rng.normal(0.0, scale))
            x0 = float(benchmark.macro_sizes[t, 0]) / 2.0
            y0 = float(benchmark.macro_sizes[t, 1]) / 2.0
            p[t, 0] = torch.clamp(p[t, 0] + dx, x0, cw - x0)
            p[t, 1] = torch.clamp(p[t, 1] + dy, y0, float(benchmark.canvas_height) - y0)
        c_can = float(compute_proxy_cost(p, benchmark, plc)["congestion_cost"])
        with torch.no_grad():
            c_sm = float(proxy.compute_congestion(p))
        can_deltas.append(c_can - base_can)
        sm_deltas.append(c_sm - base_sm)

    can_arr = np.asarray(can_deltas)
    sm_arr = np.asarray(sm_deltas)
    pearson = float(np.corrcoef(can_arr, sm_arr)[0, 1]) if len(can_arr) > 2 else float("nan")
    rk_c = np.argsort(np.argsort(can_arr))
    rk_s = np.argsort(np.argsort(sm_arr))
    spearman = float(np.corrcoef(rk_c, rk_s)[0, 1]) if len(can_arr) > 2 else float("nan")
    sign_agree = float(np.mean(np.sign(can_arr) == np.sign(sm_arr)))
    print(f"  Δcong correlation across {n_perturb} perturbations:", flush=True)
    print(f"    Pearson  ρ = {pearson:+.3f}", flush=True)
    print(f"    Spearman ρ = {spearman:+.3f}", flush=True)
    print(f"    sign_agree = {sign_agree:.0%}", flush=True)
    return {
        "bench": bench_name,
        "canonical": can_cong,
        "smooth": sm_cong,
        "rel_pct": rel_pct,
        "pearson": pearson,
        "spearman": spearman,
        "sign_agree": sign_agree,
        "n_perturb": n_perturb,
    }


if __name__ == "__main__":
    benches = sys.argv[1:] if len(sys.argv) > 1 else ["ibm17"]
    for b in benches:
        calibrate(b)
