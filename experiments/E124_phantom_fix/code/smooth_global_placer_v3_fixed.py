"""E124 — SmoothGlobalPlacerV3Fixed: V3 with phantom-stripe-fixed congestion.

Drop-in replacement for E111's SmoothGlobalPlacerV3 that uses
`PerNetTraceCongestionFixed` (E124) in place of `PerNetTraceCongestion`
(E111). All other behavior — LSE-HPWL, grid density, γ-anneal, overlap
penalty ramp, boundary penalty, init, optimizer, legalize — is
identical to V3.

The only behavioral change is the congestion gradient at degenerate
L-routes (col_src == col_snk, row_src == row_snk). The fix replaces
soft-min/soft-max (which leaks ~0.23 cells of phantom stripe mass)
with exact torch.minimum/torch.maximum.
"""
from __future__ import annotations

import sys
import time
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
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, sdf_init
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from diff_proxy_v2 import DiffProxyV2  # noqa: E402
from macro_legalizer import greedy_macro_legalize  # noqa: E402
from per_net_trace_proxy_fixed import PerNetTraceCongestionFixed  # noqa: E402


class DiffProxyV3Fixed(DiffProxyV2):
    """DiffProxyV2 with PerNetTraceCongestionFixed replacing _rudy_congestion."""

    def __init__(
        self,
        benchmark,
        plc,
        device: str = "cpu",
        gamma_frac: float = 5e-4,
        trace_kwargs: Optional[Dict] = None,
    ):
        super().__init__(benchmark, plc, device=device, gamma_frac=gamma_frac)
        trace_kwargs = trace_kwargs or {}
        self.trace = PerNetTraceCongestionFixed(
            benchmark, plc, device=device, **trace_kwargs
        )

    def cost(self, positions: torch.Tensor, *, include_congestion: bool = True):
        if positions.device != self.device:
            positions = positions.to(self.device)
        clamped = self._clamp_to_canvas(positions)
        from diff_proxy import _lse_hpwl, _grid_density  # noqa: E402
        wl = _lse_hpwl(clamped, self.net_data, self.port_base, self.gamma) / self.wl_norm
        density = _grid_density(
            clamped,
            self.sizes,
            self.cell_x_min,
            self.cell_x_max,
            self.cell_y_min,
            self.cell_y_max,
            self.cell_area,
            self.grid_rows,
            self.grid_cols,
        )
        if include_congestion:
            cong = self.trace.compute_congestion(clamped)
        else:
            cong = torch.tensor(0.0, device=self.device)
        return wl + 0.5 * density + 0.5 * cong, {
            "wl": wl.detach(),
            "density": density.detach(),
            "cong": cong.detach(),
        }


def loss_with_penalty_v3_fixed(
    proxy: DiffProxyV3Fixed,
    positions: torch.Tensor,
    overlap_lambda: float,
    *,
    include_congestion: bool = True,
    boundary_lambda: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
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


class SmoothGlobalPlacerV3Fixed:
    """E110 SmoothGlobalPlacer + E124 phantom-fixed per-net-trace congestion.

    Same constructor args as SmoothGlobalPlacerV3 (E111).
    """

    def __init__(
        self,
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-4,
        overlap_lambda_start: float = 0.0,
        overlap_lambda_end: float = 50.0,
        overlap_ramp_pct: float = 0.7,
        boundary_lambda: float = 50.0,
        include_congestion: bool = True,
        init: str = "sdf",
        device: str = "cpu",
        rng_seed: int = 42,
        verbose: bool = True,
        log_every: int = 50,
        legalize_radius_steps: int = 80,
        legalize_step_frac: float = 0.02,
        adaptive_num_steps: bool = False,
        steps_per_macro: float = 2.0,
        trace_kwargs: Optional[Dict] = None,
    ):
        self.num_steps = num_steps
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_start = overlap_lambda_start
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.boundary_lambda = boundary_lambda
        self.include_congestion = include_congestion
        self.init = init
        self.device = device
        self.rng_seed = rng_seed
        self.verbose = verbose
        self.log_every = log_every
        self.legalize_radius_steps = legalize_radius_steps
        self.legalize_step_frac = legalize_step_frac
        self.adaptive_num_steps = adaptive_num_steps
        self.steps_per_macro = steps_per_macro
        self.trace_kwargs = trace_kwargs

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _init_positions(self, benchmark: Benchmark, plc=None) -> torch.Tensor:
        if self.init == "sdf":
            pos = sdf_init(benchmark)
            pos, n_iter = project_overlaps(pos, benchmark)
            self._log(f"  init=sdf: project_overlaps n_iter={n_iter}")
            return pos
        elif self.init == "dpo":
            sys.path.insert(0, str(_ROOT / "experiments" / "E18_dpo_init" / "code"))
            from cd_lns_sa_dpo_init import _best_of_v2_init
            assert plc is not None
            pos = _best_of_v2_init(benchmark, plc, seed=self.rng_seed)
            pos, n_iter = project_overlaps(pos, benchmark)
            self._log(f"  init=dpo: project_overlaps n_iter={n_iter}")
            return pos
        elif self.init == "random":
            g = torch.Generator().manual_seed(self.rng_seed)
            cw = float(benchmark.canvas_width)
            ch = float(benchmark.canvas_height)
            sizes = benchmark.macro_sizes
            half_w = sizes[:, 0] / 2.0
            half_h = sizes[:, 1] / 2.0
            x = torch.rand(benchmark.num_macros, generator=g) * (cw - 2 * half_w) + half_w
            y = torch.rand(benchmark.num_macros, generator=g) * (ch - 2 * half_h) + half_h
            pos = torch.stack([x, y], dim=1)
            mask = benchmark.macro_fixed.bool()
            pos[mask] = benchmark.macro_positions[mask]
            return pos
        elif self.init == "center":
            cw = float(benchmark.canvas_width)
            ch = float(benchmark.canvas_height)
            pos = torch.zeros(benchmark.num_macros, 2)
            pos[:, 0] = cw / 2.0
            pos[:, 1] = ch / 2.0
            mask = benchmark.macro_fixed.bool()
            pos[mask] = benchmark.macro_positions[mask]
            return pos
        else:
            raise ValueError(f"Unknown init: {self.init!r}")

    def descend(
        self,
        benchmark: Benchmark,
        plc,
    ) -> Tuple[torch.Tensor, Dict]:
        t0 = time.time()
        init_pos = self._init_positions(benchmark, plc=plc)
        cw = float(benchmark.canvas_width)
        lr = self.lr_frac * cw

        if self.adaptive_num_steps:
            n_macros = int(benchmark.num_hard_macros)
            adaptive_steps = max(200, int(n_macros * self.steps_per_macro))
            self._log(
                f"  adaptive_num_steps: {self.num_steps} → {adaptive_steps} "
                f"(n_hard={n_macros}, spm={self.steps_per_macro})"
            )
            num_steps = adaptive_steps
        else:
            num_steps = self.num_steps

        if self.device == "mps" and not torch.backends.mps.is_available():
            self._log("  MPS not available, falling back to CPU")
            self.device = "cpu"
        device = torch.device(self.device)
        proxy = DiffProxyV3Fixed(
            benchmark, plc, device=str(device),
            gamma_frac=self.gamma_start_frac,
            trace_kwargs=self.trace_kwargs,
        )

        positions = init_pos.clone().detach().to(device).requires_grad_(True)
        fixed_mask = benchmark.macro_fixed.bool().to(device)

        optimizer = torch.optim.Adam([positions], lr=lr)

        history = []
        last_parts = None
        last_loss = None
        for step in range(num_steps):
            t = step / max(1, num_steps - 1)
            gamma_frac = self.gamma_start_frac + t * (
                self.gamma_end_frac - self.gamma_start_frac
            )
            proxy.set_gamma_frac(gamma_frac)
            ramp_t = min(1.0, step / max(1, num_steps * self.overlap_ramp_pct))
            overlap_lambda = self.overlap_lambda_start + ramp_t * (
                self.overlap_lambda_end - self.overlap_lambda_start
            )

            optimizer.zero_grad()
            loss, parts = loss_with_penalty_v3_fixed(
                proxy, positions, overlap_lambda,
                include_congestion=self.include_congestion,
                boundary_lambda=self.boundary_lambda,
            )
            loss.backward()
            with torch.no_grad():
                if positions.grad is not None:
                    positions.grad[fixed_mask] = 0.0
            optimizer.step()

            last_parts = parts
            last_loss = float(loss.item())
            if step % self.log_every == 0 or step == num_steps - 1:
                self._log(
                    f"  step {step:4d}  loss={last_loss:.5f}  "
                    f"smooth={parts['smooth_cost'].item():.5f}  "
                    f"wl={parts['wl'].item():.4f}  "
                    f"d={parts['density'].item():.4f}  "
                    f"c={parts['cong'].item():.4f}  "
                    f"ovl_area={parts['overlap_area_raw'].item():.0f}  "
                    f"γ={gamma_frac:.4f}  λ_ovl={overlap_lambda:.1f}"
                )
                history.append({
                    "step": step,
                    "loss": last_loss,
                    "smooth": parts["smooth_cost"].item(),
                    "wl": parts["wl"].item(),
                    "density": parts["density"].item(),
                    "cong": parts["cong"].item(),
                    "overlap_area": parts["overlap_area_raw"].item(),
                    "gamma_frac": gamma_frac,
                    "overlap_lambda": overlap_lambda,
                })

        positions_final = positions.detach().cpu()
        stats = {
            "history": history,
            "final_loss": last_loss,
            "final_smooth": last_parts["smooth_cost"].item() if last_parts else None,
            "final_overlap_area": last_parts["overlap_area_raw"].item() if last_parts else None,
            "descend_wall_s": time.time() - t0,
        }
        return positions_final, stats

    def legalize(
        self,
        positions: torch.Tensor,
        benchmark: Benchmark,
    ) -> Tuple[torch.Tensor, Dict]:
        t0 = time.time()
        legal_pos, leg_stats = greedy_macro_legalize(
            positions, benchmark,
            search_radius_steps=self.legalize_radius_steps,
            step_size_frac=self.legalize_step_frac,
            verbose=False,
        )
        ovl = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        if ovl > 0:
            self._log(
                f"  greedy_legalize left {ovl} overlaps "
                f"(moved={leg_stats.get('n_moved')} failed={leg_stats.get('n_failed')}); "
                f"running project_overlaps"
            )
            legal_pos, _ = project_overlaps(legal_pos, benchmark)
            ovl = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        leg_stats["final_overlaps"] = ovl
        leg_stats["legalize_wall_s"] = time.time() - t0
        return legal_pos, leg_stats

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        self._log(f"=== SmoothGlobalPlacerV3Fixed ({benchmark.name}) ===")
        self._log(
            f"  config: steps={self.num_steps} lr_frac={self.lr_frac} "
            f"γ={self.gamma_start_frac}→{self.gamma_end_frac} "
            f"λ_ovl={self.overlap_lambda_start}→{self.overlap_lambda_end}"
            f" (ramp {self.overlap_ramp_pct*100:.0f}%) init={self.init} "
            f"device={self.device}"
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        positions, descend_stats = self.descend(benchmark, plc)
        self._log(
            f"  descent done: final_smooth={descend_stats['final_smooth']:.5f} "
            f"ovl_area={descend_stats['final_overlap_area']:.0f} "
            f"wall={descend_stats['descend_wall_s']:.1f}s"
        )

        legal_pos, leg_stats = self.legalize(positions, benchmark)
        self._log(
            f"  legalize: n_moved={leg_stats.get('n_moved')} "
            f"n_failed={leg_stats.get('n_failed')} "
            f"max_disp={leg_stats.get('max_displacement', -1):.1f} "
            f"final_overlaps={leg_stats['final_overlaps']} "
            f"wall={leg_stats['legalize_wall_s']:.1f}s"
        )

        proxy_cost = float(compute_proxy_cost(legal_pos, benchmark, plc)["proxy_cost"])
        self._log(
            f"  Final: proxy={proxy_cost:.5f} ovl={leg_stats['final_overlaps']} "
            f"total_wall={time.time()-t0:.1f}s"
        )

        return legal_pos
