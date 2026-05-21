"""E121 — Multi-Commodity Flow (MCF) Congestion Proxy.

CROSS-DOMAIN ANGLE: VLSI placement literature uses RUDY/ABU congestion
(Spindler & Johannes 2007, ICCAD) — a deterministic L-route per pair.
Network optimization literature treats routing as **multi-commodity
flow** (Ford-Fulkerson 1956; Garg-Konemann 2007 multiplicative weights).
This module brings the MCF framing to placement: each net is a
commodity with one unit of flow distributed over MANY candidate paths,
weighted by softmax over a temperature.

Key contrast vs RUDY/E111:
  RUDY: one L-route per (src, sink). Cell at (src_row, sink_col) gets
        the full L-corner peak. Cells off the L get zero.
  MCF:  cells across the WHOLE bbox get probability mass, weighted by
        the path-count distribution. Peak shifts to the bbox diagonal.

Implementation choice: L-route MIXTURE with parameterized bend column.
For each pair, we consider all monotonic single-bend paths
parameterized by bend column c_b. Each path uses:
  - H stripe on src row from src_col to c_b
  - V stripe on c_b from src_row to snk_row
  - H stripe on snk row from c_b to snk_col

This is a discrete family of size |Δcol| + 1 (clipped by canvas).
Weights:
  w(c_b) ∝ exp(-|c_b - c_diag|² / τ²)

where c_diag is the geometric midpoint column. At τ → 0, mass
collapses to the L-route corner (RUDY-equivalent). At τ → ∞, mass
spreads uniformly across bend columns, peaking flow on the bbox
DIAGONAL — the canonical MCF profile.

Per-pair contribution to demand decomposes (separable in r, c):
  H[r, c] = δ(r=src_r) * Σ_{cb >= c, cb >= src_c} w(cb) * I_h1
          + δ(r=snk_r) * Σ_{cb <  c, cb <  snk_c} w(cb) * I_h2
  V[r, c] = δ(c=c_b) * w(c_b) * I(min(src_r, snk_r) <= r <= max)

Using soft (Gaussian) cell assignments for δ(...) and sigmoid range
indicators (E111's pattern), the whole thing stays differentiable.

Memory: O(B * (gr + gc)) per chunk + O(gr * gc) for cong maps.
No [n_pairs, gr, gc] materialization.

Validation:
  - smooth_MCF vs canonical scalar within ±20% on cached placements.
  - Δcong Pearson > 0.5 on random perturbations.
  - At τ → 0, must match E111 / RUDY (sanity check).
"""
from __future__ import annotations

import importlib.util
import math
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
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

_DPO_PATH = _ROOT / "writeup" / "archive" / "submissions" / "dpo" / "ablation_v2_steps.py"
_spec = importlib.util.spec_from_file_location("_e121_dpo", str(_DPO_PATH))
_dpo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dpo)
_extract_net_data = _dpo._extract_net_data
NetData = _dpo.NetData


class MCFCongestion:
    """Multi-commodity flow congestion: L-route mixture with bend
    column softmax. Drop-in replacement for `PerNetTraceCongestion`.

    Constructor matches E111 + adds:
      tau_frac: float — bend column softmax temperature, in units of
        canvas cols. tau_frac → 0 collapses to RUDY-equivalent
        L-routes; tau_frac → 1 spreads uniformly over monotonic
        single-bend paths.
      n_bend_samples: int — # of bend columns to evaluate per pair.
        Default 7 (5 evenly spaced + 2 endpoint anchors). Larger →
        smoother MCF distribution; small overhead.
    """

    def __init__(
        self,
        benchmark,
        plc,
        device: str | torch.device = "cpu",
        sigma_cell_frac: float = 0.5,
        beta_range_per_cell: float = 4.0,
        beta_minmax: float = 6.0,
        pair_chunk_size: int = 2048,
        abu_k: float = 0.05,
        include_macro_routing: bool = True,
        include_smoothing: bool = True,
        # NEW MCF hyperparameters:
        tau_frac: float = 0.30,
        n_bend_samples: int = 7,
        rudy_mix: float = 0.0,  # 0 → pure MCF; 1 → pure RUDY (E111-equivalent)
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

        self.cell_x_ctr = (torch.arange(self.gc, dtype=torch.float32, device=self.device) + 0.5) * self.cell_w
        self.cell_y_ctr = (torch.arange(self.gr, dtype=torch.float32, device=self.device) + 0.5) * self.cell_h
        self.col_idx_f = torch.arange(self.gc, dtype=torch.float32, device=self.device)
        self.row_idx_f = torch.arange(self.gr, dtype=torch.float32, device=self.device)

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

        # MCF params
        self.tau_frac = float(tau_frac)
        self.n_bend_samples = int(n_bend_samples)
        self.rudy_mix = float(rudy_mix)
        # tau in cell units (gc * tau_frac), squared for the Gaussian denominator
        self.tau_cells = max(0.5, self.tau_frac * self.gc)

        nd = _extract_net_data(benchmark, plc)
        self._build_pair_data(nd)

        self.macro_sizes = benchmark.macro_sizes.to(self.device)
        self.macro_fixed = benchmark.macro_fixed.to(self.device)
        self.cell_x_min = self.cell_x_ctr - self.cell_w / 2.0
        self.cell_x_max = self.cell_x_ctr + self.cell_w / 2.0
        self.cell_y_min = self.cell_y_ctr - self.cell_h / 2.0
        self.cell_y_max = self.cell_y_ctr + self.cell_h / 2.0

    def _build_pair_data(self, nd: NetData) -> None:
        """Same as E111 — expand NetData into per-pair (source, sink)
        tensors (star routing from pin0)."""
        if nd.weights.numel() == 0:
            self.n_pairs = 0
            self.src_macro = torch.zeros(0, dtype=torch.long, device=self.device)
            self.snk_macro = torch.zeros(0, dtype=torch.long, device=self.device)
            self.src_offset = torch.zeros(0, 2, dtype=torch.float32, device=self.device)
            self.snk_offset = torch.zeros(0, 2, dtype=torch.float32, device=self.device)
            self.pair_w = torch.zeros(0, dtype=torch.float32, device=self.device)
            return

        pin_macro_idx = nd.pin_macro_idx.to(self.device)
        pin_offsets = nd.pin_offsets.to(self.device).to(torch.float32)
        mask = nd.mask.to(self.device)
        weights = nd.weights.to(self.device).to(torch.float32)

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

    # --------------------------------------------------------- soft helpers

    def _soft_cell_assign(self, pin_coord: torch.Tensor, axis: str) -> torch.Tensor:
        """[B, n_cells] soft cell assignment for pins along one axis."""
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
        """Smooth (min, max) of (a, b)."""
        beta = self.beta_minmax
        stacked = torch.stack([a, b], dim=0)
        soft_min = -torch.logsumexp(-beta * stacked, dim=0) / beta
        soft_max = torch.logsumexp(beta * stacked, dim=0) / beta
        return soft_min, soft_max

    # ============================================================ MCF core

    def _mcf_route_congestion(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-commodity flow: per-pair L-route mixture with bend
        column softmax temperature. Returns (V_route, H_route) on grid.

        Per-pair, per-bend-col c_b, the L-route has demand:
          H[src_r, c] += w_pair * w_bend(c_b) * I(c in [src_c, c_b])
          H[snk_r, c] += w_pair * w_bend(c_b) * I(c in [c_b, snk_c])
          V[r, c_b]   += w_pair * w_bend(c_b) * I(r in [src_r, snk_r])

        With n_bend_samples = K, we enumerate K candidate bend cols
        per pair: c_b^k = col_min + k * (col_max - col_min) / (K - 1).
        These K candidates form the **support** of the MCF distribution.
        Weights = softmax_τ(-((c_b - c_diag) / τ)^2) where
        c_diag = (col_min + col_max) / 2.

        At τ → 0: only c_b = c_diag has weight → "I-route" through diag.
        At τ → ∞: all K candidates equal weight → uniform MCF spread.
        For matching RUDY/E111, set rudy_mix = 1 (forces c_b = src_c).
        """
        if self.n_pairs == 0:
            return (
                torch.zeros(self.gr, self.gc, device=self.device),
                torch.zeros(self.gr, self.gc, device=self.device),
            )

        device = positions.device
        port_base = torch.zeros(1, 2, device=device, dtype=positions.dtype)
        all_pos = torch.cat([positions, port_base], dim=0)

        H_route = torch.zeros(self.gr, self.gc, device=device, dtype=positions.dtype)
        V_route = torch.zeros(self.gr, self.gc, device=device, dtype=positions.dtype)

        cs = self.pair_chunk_size
        n_pairs = self.n_pairs
        K = self.n_bend_samples
        tau = self.tau_cells

        # Pre-compute bend sample offsets (k / (K-1)) in [0, 1]
        if K > 1:
            bend_t = torch.linspace(0.0, 1.0, K, device=device)  # [K]
        else:
            bend_t = torch.tensor([0.5], device=device)

        br = self.beta_range_per_cell
        col_ind = self.col_idx_f.view(1, 1, -1)              # [1, 1, gc]
        row_ind = self.row_idx_f.view(1, 1, -1)              # [1, 1, gr]

        for b_start in range(0, n_pairs, cs):
            b_end = min(b_start + cs, n_pairs)
            src_m = self.src_macro[b_start:b_end]
            snk_m = self.snk_macro[b_start:b_end]
            src_o = self.src_offset[b_start:b_end]
            snk_o = self.snk_offset[b_start:b_end]
            w = self.pair_w[b_start:b_end]                   # [B]

            src_pos = all_pos[src_m] + src_o                 # [B, 2]
            snk_pos = all_pos[snk_m] + snk_o

            # Soft cell assignments
            row_src = self._soft_cell_assign(src_pos[:, 1], axis="y")  # [B, gr]
            col_snk = self._soft_cell_assign(snk_pos[:, 0], axis="x")  # [B, gc]
            col_src = self._soft_cell_assign(src_pos[:, 0], axis="x")  # [B, gc]
            row_snk = self._soft_cell_assign(snk_pos[:, 1], axis="y")  # [B, gr]

            # Soft cell-index of endpoints
            col_src_exp = (col_src * self.col_idx_f.view(1, -1)).sum(dim=1)  # [B]
            col_snk_exp = (col_snk * self.col_idx_f.view(1, -1)).sum(dim=1)
            row_src_exp = (row_src * self.row_idx_f.view(1, -1)).sum(dim=1)
            row_snk_exp = (row_snk * self.row_idx_f.view(1, -1)).sum(dim=1)

            col_min, col_max = self._soft_min_max(col_src_exp, col_snk_exp)
            row_min, row_max = self._soft_min_max(row_src_exp, row_snk_exp)
            # Diag column: midpoint of [col_min, col_max]
            col_diag = 0.5 * (col_min + col_max)               # [B]

            # K bend candidates: c_b^k = col_min + bend_t[k] * (col_max - col_min)
            #   [B, K]
            c_bs = col_min.unsqueeze(1) + bend_t.unsqueeze(0) * (col_max - col_min).unsqueeze(1)

            # Apply RUDY mix: blend c_bs toward c_snk (canonical L-route
            # places the bend at (src_row, snk_col); the H stripe lives
            # on src row, the V stripe on snk col).
            if self.rudy_mix > 0.0:
                c_bs = (1.0 - self.rudy_mix) * c_bs + self.rudy_mix * col_snk_exp.unsqueeze(1)

            # Bend weights: w_k ∝ exp(-((c_bs - col_diag)/τ)^2)
            bend_diff = (c_bs - col_diag.unsqueeze(1)) / tau
            bend_logits = -(bend_diff ** 2)                      # [B, K]
            bend_w = torch.softmax(bend_logits, dim=1)           # [B, K]

            # === Per-bend-col MCF accumulation ===
            # For each pair p and bend k:
            #   H[src_r, c] += w_p * bend_w[p,k] * I(c in [c_src, c_b^k])
            #   H[snk_r, c] += w_p * bend_w[p,k] * I(c in [c_b^k, c_snk])
            #   V[r, c_b^k] += w_p * bend_w[p,k] * I(r in [r_src, r_snk])
            #
            # Use EXACT min/max here (not _soft_min_max) so that degenerate
            # bends (c_b == c_src or c_b == c_snk) produce exactly zero
            # H1 or H2 mass. _soft_min_max would create a phantom stripe
            # of width ~2*log(2)/beta ≈ 0.23 cells at degenerate bends.
            # The smooth indicator below still gives gradient flow via
            # the sigmoid steepness β.
            c_src_b = col_src_exp.unsqueeze(1)                   # [B, 1]
            c_snk_b = col_snk_exp.unsqueeze(1)                   # [B, 1]
            # H1 stripe (src row, cols [min(c_src, c_b), max(c_src, c_b)])
            h1_lo = torch.minimum(c_src_b, c_bs)                  # [B, K]
            h1_hi = torch.maximum(c_src_b, c_bs)                  # [B, K]
            # H2 stripe (snk row, cols [min(c_b, c_snk), max(c_b, c_snk)])
            h2_lo = torch.minimum(c_bs, c_snk_b)                  # [B, K]
            h2_hi = torch.maximum(c_bs, c_snk_b)                  # [B, K]

            # H1 range mask: [B, K, gc]
            h1_mask = (
                torch.sigmoid(br * (col_ind - h1_lo.unsqueeze(2) + 0.5))
                - torch.sigmoid(br * (col_ind - h1_hi.unsqueeze(2) + 0.5))
            )
            h2_mask = (
                torch.sigmoid(br * (col_ind - h2_lo.unsqueeze(2) + 0.5))
                - torch.sigmoid(br * (col_ind - h2_hi.unsqueeze(2) + 0.5))
            )

            # Weighted sum over K (bend dimension): [B, gc]
            h1_col_weights = (bend_w.unsqueeze(2) * h1_mask).sum(dim=1)
            h2_col_weights = (bend_w.unsqueeze(2) * h2_mask).sum(dim=1)

            # H accumulation
            #   H[r, c] = sum_p w_p * row_src[p, r] * h1_col_weights[p, c]
            #          + sum_p w_p * row_snk[p, r] * h2_col_weights[p, c]
            w_pair = w.unsqueeze(1)                              # [B, 1]
            row_src_w = row_src * w_pair                         # [B, gr]
            row_snk_w = row_snk * w_pair                         # [B, gr]
            H_route = H_route + row_src_w.transpose(0, 1) @ h1_col_weights
            H_route = H_route + row_snk_w.transpose(0, 1) @ h2_col_weights

            # V accumulation
            # V[r, c] += sum_p sum_k w_p * bend_w[p,k] * v_row_mask[p, r] * col_mask_at_cb[p, k, c]
            # v_row_mask is shared across K (only depends on r_src, r_snk).
            v_row_mask = (
                torch.sigmoid(br * (row_ind - row_min.view(-1, 1, 1) + 0.5))
                - torch.sigmoid(br * (row_ind - row_max.view(-1, 1, 1) + 0.5))
            ).squeeze(1)                                          # [B, gr]
            # col-mask at c_b^k: Gaussian-softmax centered at c_b (matches
            # E111's PerNetTraceCongestion soft cell assignment width).
            # sigma_col = sigma_cell_frac (in cell units) — same as E111.
            sigma_col = self.sigma_cell_frac
            col_diff = (col_ind - c_bs.unsqueeze(2)) / sigma_col  # [B, K, gc]
            col_logits = -(col_diff ** 2)
            col_mask_cb = torch.softmax(col_logits, dim=2)        # [B, K, gc] sums to 1 over gc
            # Aggregate bend weights into per-col profile: [B, gc]
            v_col_weights = (bend_w.unsqueeze(2) * col_mask_cb).sum(dim=1)

            # V[r, c] = sum_p w_p * v_row_mask[p, r] * v_col_weights[p, c]
            v_row_w = v_row_mask * w_pair                        # [B, gr]
            V_route = V_route + v_row_w.transpose(0, 1) @ v_col_weights

        return V_route, H_route

    # ------------------------------------------------ macro footprint route
    # (Same as E111 — macros contribute their footprint to H/V routing.)

    def _macro_route_congestion(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.include_macro_routing or self.num_hard == 0:
            return (
                torch.zeros(self.gr, self.gc, device=positions.device),
                torch.zeros(self.gr, self.gc, device=positions.device),
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

    # ------------------------------------------------------ box smoothing

    def _box_smooth(self, M: torch.Tensor, axis: str) -> torch.Tensor:
        sr = self.smooth_range
        if sr <= 0 or not self.include_smoothing:
            return M
        ksize = 2 * sr + 1
        if axis == "V":
            ones = torch.ones_like(M)
            count = F.conv2d(
                ones.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, 1, ksize, device=M.device),
                padding=(0, sr),
            )[0, 0]
            M_norm = M / count
            return F.conv2d(
                M_norm.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, 1, ksize, device=M.device),
                padding=(0, sr),
            )[0, 0]
        elif axis == "H":
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
        V_route, H_route = self._mcf_route_congestion(positions)
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


# ---------------------------------------------------------------------------
# Self-test (calibration vs canonical + RUDY/E111 sanity)
# ---------------------------------------------------------------------------

def calibrate(bench_name: str = "ibm17", n_perturb: int = 16, perturb_frac: float = 0.02,
              tau_frac: float = 0.30, n_bend_samples: int = 7, rudy_mix: float = 0.0):
    """Compare smooth MCF vs canonical congestion on cached cascade placements."""
    import numpy as np
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.objective import compute_proxy_cost

    print(f"\n=== E121 calibrate: {bench_name} ===", flush=True)
    print(f"  config: tau_frac={tau_frac}  n_bend={n_bend_samples}  rudy_mix={rudy_mix}", flush=True)
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

    proxy = MCFCongestion(
        benchmark, plc, device="cpu",
        tau_frac=tau_frac, n_bend_samples=n_bend_samples, rudy_mix=rudy_mix,
    )
    print(f"  built proxy: n_pairs={proxy.n_pairs}, grid={proxy.gr}x{proxy.gc}", flush=True)

    can = compute_proxy_cost(start, benchmark, plc)
    can_cong = float(can["congestion_cost"])
    with torch.no_grad():
        sm_cong = float(proxy.compute_congestion(start))
    rel_pct = (sm_cong - can_cong) / max(abs(can_cong), 1e-6) * 100.0
    print(f"  canonical cong: {can_cong:.5f}", flush=True)
    print(f"  MCF smooth:     {sm_cong:.5f}", flush=True)
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
        "tau_frac": tau_frac,
        "n_bend": n_bend_samples,
        "rudy_mix": rudy_mix,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("benches", nargs="*", default=["ibm17"])
    parser.add_argument("--tau", type=float, default=0.30)
    parser.add_argument("--nbend", type=int, default=7)
    parser.add_argument("--rudy_mix", type=float, default=0.0)
    args = parser.parse_args()
    for b in args.benches:
        calibrate(b, tau_frac=args.tau, n_bend_samples=args.nbend, rudy_mix=args.rudy_mix)
