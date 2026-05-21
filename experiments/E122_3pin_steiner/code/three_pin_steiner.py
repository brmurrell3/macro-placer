"""E122 — 3-pin Steiner T-route, surgical fix on top of E111 per-net-trace.

E111 (`per_net_trace_proxy.py`) routes every net as a STAR of L-routes from
pin0 to each other pin. That matches canonical for k=2 and k>3, but for
k=3 the canonical `PlacementCost.__three_pin_net_routing` (plc_client_os.py
lines 1354-1390) computes a Steiner shape (L or T):

  - Sort pins by (x, y). Let sorted be p1=(y1,x1), p2=(y2,x2), p3=(y3,x3)
    with x1 <= x2 <= x3.

  - L-routing (when x1<x2<x3 and y2 strictly between y1,y3):
      H stripe at row=y1 over [x1, x2]
      H stripe at row=y2 over [x2, x3]
      V stripe at col=x2 over [min(y1,y2), max(y1,y2)]
      V stripe at col=x3 over [min(y2,y3), max(y2,y3)]

  - T-routing (default for the typical case):
      H trunk at row=y2 over [xmin, xmax]  (xmin=x1, xmax=x3)
      V branch at col=x1 over [min(y1,y2), max(y1,y2)]
      V branch at col=x3 over [min(y2,y3), max(y2,y3)]

25-40% of nets in IBM benchmarks are 3-pin, and canonical's Steiner
re-uses the middle pin's row/col as a routing trunk — so star L-routing
(what we do) systematically over-counts demand. The diagnostic survey
(docs/research/2026-05-20_congestion_model_survey.md) identifies this as
the #1 source of the remaining 14-25% canonical bias on hard benches.

This file implements `PerNetTraceCongestionSteiner3Pin`, a drop-in
subclass of `PerNetTraceCongestion` that:

1. Separates nets into k=2, k=3, k>=4 groups at init time.
2. For k=3 nets, builds a soft T-route Steiner shape (the canonical
   default branch). Soft-sorts the 3 pins by x to identify (left, middle,
   right) and uses the middle's y as the trunk row.
3. For k=2 nets and k>=4 nets, keeps the existing star L-route from E111.

The simpler T-route only variant per Survey IMPROVEMENT 1 "Simpler
alternative" recommendation: catches the largest topology bias without
the engineering cost of the full 4-case split.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _ROOT,
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from per_net_trace_proxy import PerNetTraceCongestion  # noqa: E402

import importlib.util  # noqa: E402

_DPO_PATH = _ROOT / "writeup" / "archive" / "submissions" / "dpo" / "ablation_v2_steps.py"
_spec = importlib.util.spec_from_file_location("_e122_dpo", str(_DPO_PATH))
_dpo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dpo)
_extract_net_data = _dpo._extract_net_data
NetData = _dpo.NetData


class PerNetTraceCongestionSteiner3Pin(PerNetTraceCongestion):
    """E111 + 3-pin Steiner T-route surgical fix.

    Nets are split into 3 groups at init:
      - 2-pin: handled by inherited `_trace_route_congestion` star L-route
      - 3-pin: handled here by a soft T-route Steiner shape
      - 4+pin: handled by inherited star L-route (canonical-faithful)

    The 3-pin Steiner uses a SOFT median selection on x-coords to identify
    the middle pin (which provides the horizontal trunk's y-row). For each
    3-pin net we contribute:

      H stripe at row = median_x_pin.y, over [x_left, x_right]
      V stripe at col = x_left,         over [min(y_left, y_mid), max(...)]
      V stripe at col = x_right,        over [min(y_mid, y_right), max(...)]

    Compared to E111's star L-route on the same 3-pin net (which adds 2
    L-routes: pin0->pin1 + pin0->pin2, each with full H + V demand), the
    Steiner T-route saves ~30-50% routing demand on the trunk, matching
    canonical's behavior.

    Soft-median identification:
      Given 3 x-coordinates a, b, c, we want soft weights w_left, w_mid,
      w_right summing to 1 such that the soft middle x is sum(w * x).
      We use:
        - x_left  = softmin(a, b, c)     via -LSE(-beta * x) / beta
        - x_right = softmax(a, b, c)     via  LSE( beta * x) / beta
        - x_mid   = sum(x) - x_left - x_right    (algebraic complement)
      Similarly for y.
    """

    def __init__(self, *args, beta_steiner: float = 6.0, **kwargs):
        """beta_steiner controls sharpness of soft min/max in Steiner selection."""
        self.beta_steiner = float(beta_steiner)
        # We re-extract NetData here too because we need access to the
        # underlying nets before super().__init__() builds pair tensors.
        # Save it before super init may use it.
        super().__init__(*args, **kwargs)
        # After super init, also build dedicated 3-pin tensors.
        self._build_three_pin_data()

    # ---------------------------------------------------- 3-pin data build

    def _build_three_pin_data(self) -> None:
        """Extract per-3-pin-net tensors for the Steiner T-route.

        Each 3-pin net contributes 3 pin (macro_idx, offset_x, offset_y, weight)
        rows, in canonical (pin0, pin1, pin2) order.
        """
        # Re-extract NetData (cheap — single pass over plc.nets).
        nd = _extract_net_data(self.benchmark, self.plc)
        if nd.weights.numel() == 0:
            self._set_empty_three_pin()
            return

        pin_macro_idx = nd.pin_macro_idx.to(self.device)
        pin_offsets = nd.pin_offsets.to(self.device).to(torch.float32)
        mask = nd.mask.to(self.device)
        weights = nd.weights.to(self.device).to(torch.float32)

        # n_pins per net
        n_pins = mask.sum(dim=1)                       # [num_nets]
        is_3pin = (n_pins == 3) & (weights > 0)        # [num_nets]
        idx_3pin = torch.nonzero(is_3pin, as_tuple=False).flatten()

        self.n_3pin = int(idx_3pin.numel())
        if self.n_3pin == 0:
            self._set_empty_three_pin()
            return

        # Slice 3-pin nets: each has exactly 3 valid pins at positions 0,1,2.
        tp_macro = pin_macro_idx[idx_3pin, :3]           # [n_3pin, 3]
        tp_offsets = pin_offsets[idx_3pin, :3]           # [n_3pin, 3, 2]
        tp_w = weights[idx_3pin]                         # [n_3pin]

        self.tp_macro_a = tp_macro[:, 0]
        self.tp_macro_b = tp_macro[:, 1]
        self.tp_macro_c = tp_macro[:, 2]
        self.tp_off_a = tp_offsets[:, 0]                 # [n_3pin, 2]
        self.tp_off_b = tp_offsets[:, 1]
        self.tp_off_c = tp_offsets[:, 2]
        self.tp_w = tp_w

        # Also need to MASK these nets out of the inherited pair-list so we
        # don't double-count. Rebuild the pair tensors excluding 3-pin nets.
        self._rebuild_pair_data_excluding_3pin(nd, is_3pin)

    def _set_empty_three_pin(self) -> None:
        self.n_3pin = 0
        z_l = torch.zeros(0, dtype=torch.long, device=self.device)
        z_o = torch.zeros(0, 2, dtype=torch.float32, device=self.device)
        z_w = torch.zeros(0, dtype=torch.float32, device=self.device)
        self.tp_macro_a = z_l
        self.tp_macro_b = z_l
        self.tp_macro_c = z_l
        self.tp_off_a = z_o
        self.tp_off_b = z_o
        self.tp_off_c = z_o
        self.tp_w = z_w

    def _rebuild_pair_data_excluding_3pin(self, nd: NetData, is_3pin: torch.Tensor) -> None:
        """Rebuild the inherited (src, snk) pair lists excluding 3-pin nets.

        The base class built pair lists for ALL nets. For 3-pin nets we now
        do Steiner instead of star, so they must be removed from the pair
        list to avoid double-counting.
        """
        pin_macro_idx = nd.pin_macro_idx.to(self.device)
        pin_offsets = nd.pin_offsets.to(self.device).to(torch.float32)
        mask = nd.mask.to(self.device)
        weights = nd.weights.to(self.device).to(torch.float32)

        # Only keep nets that are NOT 3-pin and have weight>0.
        keep_net = (~is_3pin) & (weights > 0)
        n_pins_each = mask.sum(dim=1)
        keep_net = keep_net & (n_pins_each >= 2)  # must have at least 2 pins

        if not keep_net.any():
            self.n_pairs = 0
            z_l = torch.zeros(0, dtype=torch.long, device=self.device)
            z_o = torch.zeros(0, 2, dtype=torch.float32, device=self.device)
            z_w = torch.zeros(0, dtype=torch.float32, device=self.device)
            self.src_macro = z_l
            self.snk_macro = z_l
            self.src_offset = z_o
            self.snk_offset = z_o
            self.pair_w = z_w
            return

        kept_idx = torch.nonzero(keep_net, as_tuple=False).flatten()
        pmi = pin_macro_idx[kept_idx]                  # [n_keep, max_pins]
        po = pin_offsets[kept_idx]                      # [n_keep, max_pins, 2]
        m = mask[kept_idx]                              # [n_keep, max_pins]
        w = weights[kept_idx]                           # [n_keep]

        num_nets = pmi.shape[0]
        # Source is pin 0.
        src_macro_net = pmi[:, 0]
        src_offset_net = po[:, 0]

        sink_mask = m[:, 1:]
        sink_macro = pmi[:, 1:]
        sink_offset = po[:, 1:]

        n_sinks = sink_mask.sum(dim=1)
        net_idx = torch.repeat_interleave(
            torch.arange(num_nets, device=self.device), n_sinks
        )
        sm_flat = sink_macro[sink_mask]
        so_flat = sink_offset[sink_mask]
        src_m = src_macro_net[net_idx]
        src_o = src_offset_net[net_idx]
        pair_w = w[net_idx]

        self.n_pairs = int(net_idx.numel())
        self.src_macro = src_m
        self.snk_macro = sm_flat
        self.src_offset = src_o
        self.snk_offset = so_flat
        self.pair_w = pair_w

    # --------------------------------------------- 3-pin steiner contribution

    def _trace_three_pin_steiner(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute V_route, H_route contributions from 3-pin nets via T-route.

        For each 3-pin net with pins (a, b, c) and positions (xa, ya), (xb, yb),
        (xc, yc):

          1. Soft-sort x's: x_left, x_mid, x_right via softmin/softmax/sum.
          2. Identify y_mid: the y of the pin whose x is the median. We do
             this with a soft permutation matrix P[3,3] where P[i,j] = prob
             that pin j is at sorted position i. Then y_mid = sum_j P[1,j]*y_j.
             Approximation: y_mid ~ sum(y) - y_at_left - y_at_right where
             y_at_left/right are softly identified by attention over -x/+x.
          3. H trunk at y_mid over [x_left, x_right].
          4. V branch at x_left over [min(y_at_left, y_mid), max(...)].
          5. V branch at x_right over [min(y_mid, y_at_right), max(...)].

        Returns (V_3pin, H_3pin) tensors of shape [gr, gc].
        """
        if self.n_3pin == 0:
            return (
                torch.zeros(self.gr, self.gc, device=positions.device, dtype=positions.dtype),
                torch.zeros(self.gr, self.gc, device=positions.device, dtype=positions.dtype),
            )

        device = positions.device
        port_base = torch.zeros(1, 2, device=device, dtype=positions.dtype)
        all_pos = torch.cat([positions, port_base], dim=0)

        # Compute the 3 pin positions for each 3-pin net: [n_3pin, 2] each.
        pos_a = all_pos[self.tp_macro_a] + self.tp_off_a
        pos_b = all_pos[self.tp_macro_b] + self.tp_off_b
        pos_c = all_pos[self.tp_macro_c] + self.tp_off_c

        xa, ya = pos_a[:, 0], pos_a[:, 1]
        xb, yb = pos_b[:, 0], pos_b[:, 1]
        xc, yc = pos_c[:, 0], pos_c[:, 1]
        w = self.tp_w                                       # [n_3pin]

        # --- Step 1: soft sort to identify the (x_left, x_mid, x_right)
        # and the y-value of the pin at each x-rank.
        #
        # Soft attention: pin j is "the leftmost" with weight
        #   alpha_left[j] = softmax(-beta * x_j)
        # over j in {a, b, c}. Similarly for rightmost.
        # The middle pin's weights are 1 - alpha_left - alpha_right
        # (rebalanced over the 3 pins).
        beta = self.beta_steiner
        x_stack = torch.stack([xa, xb, xc], dim=1)         # [n_3pin, 3]
        y_stack = torch.stack([ya, yb, yc], dim=1)         # [n_3pin, 3]

        # Soft "is leftmost" probability (over pins a, b, c). 3 unnormalized
        # logits = -beta * x; softmax across 3 gives prob of being the min.
        left_weights = torch.softmax(-beta * x_stack, dim=1)    # [n_3pin, 3]
        right_weights = torch.softmax(beta * x_stack, dim=1)    # [n_3pin, 3]
        # Middle weights: 1 - left - right (clip to nonneg for safety).
        # In the case left == right (e.g., all pins coincide), middle = -1 < 0;
        # we use a renormalized softmax of |x - sum/3| to handle this edge case.
        mid_weights = (1.0 - left_weights - right_weights).clamp(min=0.0)
        # Renormalize middle in case left + right > 1.
        s = mid_weights.sum(dim=1, keepdim=True)
        # Fall back to uniform if soft sort collapsed.
        mid_weights = torch.where(
            s > 1e-6,
            mid_weights / s.clamp(min=1e-6),
            torch.full_like(mid_weights, 1.0 / 3.0),
        )

        # Soft x_left, x_mid, x_right
        x_left = (left_weights * x_stack).sum(dim=1)        # [n_3pin]
        x_right = (right_weights * x_stack).sum(dim=1)
        # x_mid not actually needed for routing — only y_mid matters for
        # the H trunk row (y of the median-x pin).

        # The y at left/mid/right ranks: weight y by the same soft ranks.
        y_left = (left_weights * y_stack).sum(dim=1)
        y_right = (right_weights * y_stack).sum(dim=1)
        y_mid = (mid_weights * y_stack).sum(dim=1)

        # --- Step 2: convert continuous (x, y) endpoints to soft cell indices
        # (in canonical-cell index space, where cell i has center (i+0.5)*cell_w).
        # Use the same _soft_cell_assign + expectation as the parent class.
        col_left = self._soft_cell_index(x_left, axis="x")        # [n_3pin]
        col_right = self._soft_cell_index(x_right, axis="x")
        row_left = self._soft_cell_index(y_left, axis="y")
        row_right = self._soft_cell_index(y_right, axis="y")
        row_mid = self._soft_cell_index(y_mid, axis="y")

        # Soft min/max for canonical range endpoints.
        col_lo, col_hi = self._soft_min_max(col_left, col_right)
        row_lo_l, row_hi_l = self._soft_min_max(row_left, row_mid)
        row_lo_r, row_hi_r = self._soft_min_max(row_mid, row_right)

        col_lo = col_lo.unsqueeze(1)
        col_hi = col_hi.unsqueeze(1)
        row_lo_l = row_lo_l.unsqueeze(1)
        row_hi_l = row_hi_l.unsqueeze(1)
        row_lo_r = row_lo_r.unsqueeze(1)
        row_hi_r = row_hi_r.unsqueeze(1)

        br = self.beta_range_per_cell
        col_ind = self.col_idx_f.view(1, -1)              # [1, gc]
        row_ind = self.row_idx_f.view(1, -1)              # [1, gr]

        # H trunk col range [col_lo, col_hi]
        h_col_range = (
            torch.sigmoid(br * (col_ind - col_lo + 0.5))
            - torch.sigmoid(br * (col_ind - col_hi + 0.5))
        )                                                  # [n_3pin, gc]
        # V left col cell-assignment (sharp on col_left)
        col_left_sa = self._soft_cell_assign(x_left, axis="x")        # [n_3pin, gc]
        col_right_sa = self._soft_cell_assign(x_right, axis="x")      # [n_3pin, gc]
        # H trunk row cell-assignment (sharp on row_mid)
        row_mid_sa = self._soft_cell_assign(y_mid, axis="y")          # [n_3pin, gr]
        # V left row range [min(y_left, y_mid), max(...)]
        v_row_range_left = (
            torch.sigmoid(br * (row_ind - row_lo_l + 0.5))
            - torch.sigmoid(br * (row_ind - row_hi_l + 0.5))
        )                                                  # [n_3pin, gr]
        v_row_range_right = (
            torch.sigmoid(br * (row_ind - row_lo_r + 0.5))
            - torch.sigmoid(br * (row_ind - row_hi_r + 0.5))
        )                                                  # [n_3pin, gr]

        # H[r, c] = sum_n w_n * row_mid_sa[n, r] * h_col_range[n, c]
        #        = (row_mid_sa * w[:,None]).T @ h_col_range
        w1 = w.unsqueeze(1)
        H_3pin = (row_mid_sa * w1).transpose(0, 1) @ h_col_range
        # V[r, c] = sum_n w_n * (
        #     v_row_range_left[n, r]  * col_left_sa[n, c]    # left V branch
        #   + v_row_range_right[n, r] * col_right_sa[n, c]   # right V branch
        # )
        V_3pin = (
            v_row_range_left.transpose(0, 1) @ (col_left_sa * w1)
            + v_row_range_right.transpose(0, 1) @ (col_right_sa * w1)
        )

        return V_3pin, H_3pin

    def _soft_cell_index(self, pin_coord: torch.Tensor, axis: str) -> torch.Tensor:
        """Soft cell index (real-valued) of `pin_coord` along `axis`.

        Mirrors what `_trace_route_congestion` does inline: soft assignment
        Gaussian → expectation over cell indices = soft cell index.
        """
        sa = self._soft_cell_assign(pin_coord, axis=axis)              # [B, n_cells]
        if axis == "x":
            idx_f = self.col_idx_f
        else:
            idx_f = self.row_idx_f
        return (sa * idx_f.view(1, -1)).sum(dim=1)                     # [B]

    # ---------------------------------------------------- override route trace

    def _trace_route_congestion(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sum L-route demand (k=2 & k>=4 nets) + Steiner T-route demand (k=3).

        Calls the parent (which now sees only k!=3 nets via the rebuilt pair
        list) + adds 3-pin Steiner contribution.
        """
        V_route, H_route = super()._trace_route_congestion(positions)
        V_3, H_3 = self._trace_three_pin_steiner(positions)
        return V_route + V_3, H_route + H_3


# ---------------------------------------------------------------------------
# Self-test (calibration vs canonical) — mirrors E111's calibrate()
# ---------------------------------------------------------------------------

def calibrate(
    bench_name: str = "ibm17",
    n_perturb: int = 16,
    perturb_frac: float = 0.02,
    show_stats: bool = True,
):
    """Compare baseline E111 vs E122 (3-pin Steiner) vs canonical."""
    import numpy as np
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.objective import compute_proxy_cost

    print(f"\n=== E122 3-pin Steiner calibrate: {bench_name} ===", flush=True)
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    cached = _ROOT / "experiments" / "E84_cascading_saddle" / "results" / f"cascade_{bench_name}.pt"
    if cached.exists():
        data = torch.load(cached, weights_only=False, map_location="cpu")
        start = (data.get("placement") if "placement" in data else data["polished_placement"]).to(torch.float32)
        print(f"  using cached cascade placement", flush=True)
    else:
        start = benchmark.macro_positions.clone().float()
        print(f"  using macro_positions", flush=True)

    print(f"  building proxies (E111 base + E122 Steiner)…", flush=True)
    base = PerNetTraceCongestion(benchmark, plc, device="cpu")
    e122 = PerNetTraceCongestionSteiner3Pin(benchmark, plc, device="cpu")

    if show_stats:
        print(f"  E111 n_pairs={base.n_pairs}", flush=True)
        print(f"  E122 n_pairs={e122.n_pairs}  n_3pin={e122.n_3pin}", flush=True)
        # Fraction of nets that are 3-pin (per-net, not per-pair).
        nd = _extract_net_data(benchmark, plc)
        n_pins_each = nd.mask.sum(dim=1)
        n_3 = ((n_pins_each == 3) & (nd.weights > 0)).sum().item()
        n_total = (nd.weights > 0).sum().item()
        n_2 = ((n_pins_each == 2) & (nd.weights > 0)).sum().item()
        n_4plus = ((n_pins_each >= 4) & (nd.weights > 0)).sum().item()
        print(
            f"  pin-degree distribution: total={n_total}  "
            f"2-pin={n_2} ({100*n_2/max(1,n_total):.1f}%)  "
            f"3-pin={n_3} ({100*n_3/max(1,n_total):.1f}%)  "
            f"4+pin={n_4plus} ({100*n_4plus/max(1,n_total):.1f}%)",
            flush=True,
        )

    can = compute_proxy_cost(start, benchmark, plc)
    can_cong = float(can["congestion_cost"])
    with torch.no_grad():
        sm_base = float(base.compute_congestion(start))
        sm_e122 = float(e122.compute_congestion(start))
    rel_base = (sm_base - can_cong) / max(abs(can_cong), 1e-6) * 100.0
    rel_e122 = (sm_e122 - can_cong) / max(abs(can_cong), 1e-6) * 100.0
    print(f"  canonical cong:    {can_cong:.5f}", flush=True)
    print(f"  E111 cong (base):  {sm_base:.5f}  ({rel_base:+.1f}%)", flush=True)
    print(f"  E122 cong (3-pin): {sm_e122:.5f}  ({rel_e122:+.1f}%)", flush=True)

    # Rank correlation
    rng = np.random.default_rng(42)
    fixed = benchmark.macro_fixed.cpu().numpy()
    cw = float(benchmark.canvas_width)
    movable_hard = [i for i in range(benchmark.num_hard_macros) if not bool(fixed[i])]
    scale = perturb_frac * cw

    can_deltas, base_deltas, e122_deltas = [], [], []
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
            c_base = float(base.compute_congestion(p))
            c_e122 = float(e122.compute_congestion(p))
        can_deltas.append(c_can - can_cong)
        base_deltas.append(c_base - sm_base)
        e122_deltas.append(c_e122 - sm_e122)

    def report(label, sm_arr):
        can_arr = np.asarray(can_deltas)
        sm_arr = np.asarray(sm_arr)
        pearson = float(np.corrcoef(can_arr, sm_arr)[0, 1])
        rk_c = np.argsort(np.argsort(can_arr))
        rk_s = np.argsort(np.argsort(sm_arr))
        spearman = float(np.corrcoef(rk_c, rk_s)[0, 1])
        sign_agree = float(np.mean(np.sign(can_arr) == np.sign(sm_arr)))
        print(f"  {label:8s} pearson={pearson:+.3f}  spearman={spearman:+.3f}  sign={sign_agree:.0%}", flush=True)

    print(f"  Δcong correlation across {n_perturb} perturbations:", flush=True)
    report("E111", base_deltas)
    report("E122", e122_deltas)

    return {
        "bench": bench_name,
        "canonical": can_cong,
        "base": sm_base,
        "base_rel_pct": rel_base,
        "e122": sm_e122,
        "e122_rel_pct": rel_e122,
        "n_perturb": n_perturb,
    }


if __name__ == "__main__":
    benches = sys.argv[1:] if len(sys.argv) > 1 else ["ibm17"]
    for b in benches:
        calibrate(b)
