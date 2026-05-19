"""CDLNSSACascadeStackedWireMaskPeripheryPlacer — Option C + WireMask polish.

Layers (in order):
  1. Stacked (E25 + E41 → plateau → cascade saddle → portfolio saddle)
  2. NEW: WireMask greedy reposition polish (criticality-ranked 2D
     atomic moves; conservative accept).
  3. Periphery push + CD polish (strict-conservative accept).

The WireMask polish runs on the post-portfolio plateau. Each pass
processes movable macros in descending WL contribution order, finds
each macro's best grid candidate via incremental delta_cost, and
commits if zero-overlap + proxy improved. Multiple passes until
plateau or budget exhausted. Falls back to base placement if any
exception or if no improvement.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

_STACKED_PATH = _ROOT / "submissions" / "cd_lns_sa_cascade_stacked" / "placer.py"
_spec = importlib.util.spec_from_file_location("stacked_placer", str(_STACKED_PATH))
_stacked = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stacked)
CDLNSSACascadeStackedPlacer = _stacked.CDLNSSACascadeStackedPlacer

_WIREMASK_CODE = _ROOT / "experiments" / "E113_wiremask_greedy" / "code"
if str(_WIREMASK_CODE) not in sys.path:
    sys.path.insert(0, str(_WIREMASK_CODE))
from wiremask_polish import wiremask_polish


def _periphery_push_and_polish(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    alpha: float = 0.01,
    cd_budget_s: float = 180.0,
    log=None,
):
    if log is None:
        log = lambda s: None
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    pos = placement.cpu().numpy().copy()
    half_w = (benchmark.macro_sizes[:, 0] / 2).cpu().numpy()
    half_h = (benchmark.macro_sizes[:, 1] / 2).cpu().numpy()
    fixed = benchmark.macro_fixed.cpu().numpy()

    for i in range(pos.shape[0]):
        if fixed[i]:
            continue
        x, y = pos[i, 0], pos[i, 1]
        dx_l = x - half_w[i]; dx_r = (cw - half_w[i]) - x
        dy_b = y - half_h[i]; dy_t = (ch - half_h[i]) - y
        dists = [dx_l, dx_r, dy_b, dy_t]
        nearest = int(np.argmin([abs(d) for d in dists]))
        if nearest == 0:   pos[i, 0] = x - alpha * dx_l
        elif nearest == 1: pos[i, 0] = x + alpha * dx_r
        elif nearest == 2: pos[i, 1] = y - alpha * dy_b
        else:              pos[i, 1] = y + alpha * dy_t
        pos[i, 0] = np.clip(pos[i, 0], half_w[i], cw - half_w[i])
        pos[i, 1] = np.clip(pos[i, 1], half_h[i], ch - half_h[i])

    pushed = torch.tensor(pos, dtype=torch.float32)
    legal, _ = project_overlaps(pushed, benchmark)
    movable_idx = [i for i in range(benchmark.num_macros) if not bool(fixed[i])]
    evaluator = IncrementalProxyEvaluator(benchmark, plc, legal.clone())
    run_cd_adaptive(
        evaluator, benchmark, plc, movable_idx,
        min_time_s=min(30.0, cd_budget_s),
        hard_cap_s=cd_budget_s,
        patience=3,
        plateau_threshold=0.001,
        log_fn=None,
    )
    polished = evaluator.placement.detach().clone().to(torch.float32)
    proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
    ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
    return polished, proxy, ovl


class CDLNSSACascadeStackedWireMaskPeripheryPlacer:
    """Option C + WireMask polish layer."""

    def __init__(
        self,
        cascade_max_iters: int = 5,
        cascade_eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
        cascade_polish_budget: float = 180.0,
        portfolio_max_iters: int = 2,
        portfolio_K_eps: int = 2,
        portfolio_eps_scale: float = 1.0,
        portfolio_polish_budget: float = 60.0,
        budget_seconds: Optional[float] = 3300.0,
        rng_seed: int = 42,
        periphery_alpha: float = 0.01,
        periphery_polish_s: float = 120.0,
        wiremask_n_per_axis: int = 5,
        wiremask_local_radius_frac: float = 0.15,
        wiremask_top_k: Optional[int] = 300,
        wiremask_max_passes: int = 8,
        wiremask_budget_s: float = 180.0,
        verbose: bool = True,
    ):
        self.cascade_max_iters = cascade_max_iters
        self.cascade_eps_values = cascade_eps_values
        self.cascade_polish_budget = cascade_polish_budget
        self.portfolio_max_iters = portfolio_max_iters
        self.portfolio_K_eps = portfolio_K_eps
        self.portfolio_eps_scale = portfolio_eps_scale
        self.portfolio_polish_budget = portfolio_polish_budget
        self.budget_seconds = budget_seconds
        self.rng_seed = rng_seed
        self.periphery_alpha = periphery_alpha
        self.periphery_polish_s = periphery_polish_s
        self.wiremask_n_per_axis = wiremask_n_per_axis
        self.wiremask_local_radius_frac = wiremask_local_radius_frac
        self.wiremask_top_k = wiremask_top_k
        self.wiremask_max_passes = wiremask_max_passes
        self.wiremask_budget_s = wiremask_budget_s
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSACascadeStackedWireMaskPeripheryPlacer ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        # Budget split: reserve wiremask + periphery.
        reserved = self.wiremask_budget_s + self.periphery_polish_s + 30.0
        stacked_budget = (
            self.budget_seconds - reserved if self.budget_seconds is not None else None
        )
        log(f"  budget split: stacked={stacked_budget}s, wiremask={self.wiremask_budget_s}s, "
            f"periphery={self.periphery_polish_s}s")

        stacked = CDLNSSACascadeStackedPlacer(
            cascade_max_iters=self.cascade_max_iters,
            cascade_eps_values=self.cascade_eps_values,
            cascade_polish_budget=self.cascade_polish_budget,
            portfolio_max_iters=self.portfolio_max_iters,
            portfolio_K_eps=self.portfolio_K_eps,
            portfolio_eps_scale=self.portfolio_eps_scale,
            portfolio_polish_budget=self.portfolio_polish_budget,
            budget_seconds=stacked_budget,
            rng_seed=self.rng_seed,
            verbose=self.verbose,
        )
        base_placement = stacked.place(benchmark)

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        base_proxy = float(compute_proxy_cost(base_placement, benchmark, plc)["proxy_cost"])
        base_ovl = compute_overlap_metrics(base_placement, benchmark)["overlap_count"]
        log(f"  base done: proxy={base_proxy:.5f} ovl={base_ovl}"
            f" (wall={time.time() - t0:.0f}s)")

        if base_ovl > 0:
            log(f"  WARNING: base has ovl={base_ovl}; skipping wiremask+periphery")
            return base_placement

        # ── WireMask polish layer ────────────────────────────────────────
        current_placement = base_placement
        current_proxy = base_proxy
        if deadline is not None:
            remaining = deadline - time.time()
            wm_budget = min(self.wiremask_budget_s,
                            remaining - self.periphery_polish_s - 30.0)
        else:
            wm_budget = self.wiremask_budget_s

        if wm_budget < 30.0:
            log(f"  SKIPPING wiremask (only {wm_budget:.0f}s left)")
        else:
            try:
                log(f"  WireMask polish (budget {wm_budget:.0f}s, n_per_axis={self.wiremask_n_per_axis}, "
                    f"local_radius_frac={self.wiremask_local_radius_frac})")
                fixed = benchmark.macro_fixed.cpu().numpy()
                movable_all = [i for i in range(benchmark.num_macros) if not bool(fixed[i])]
                ev_wm = IncrementalProxyEvaluator(benchmark, plc, base_placement.clone())
                stats = wiremask_polish(
                    ev_wm, benchmark, movable_all,
                    n_per_axis=self.wiremask_n_per_axis,
                    local_radius_frac=self.wiremask_local_radius_frac,
                    top_k=self.wiremask_top_k,
                    max_passes=self.wiremask_max_passes,
                    time_budget_s=wm_budget,
                    log=lambda s: None,
                )
                wm_placement = ev_wm.placement.detach().clone().to(torch.float32)
                wm_proxy = float(compute_proxy_cost(wm_placement, benchmark, plc)["proxy_cost"])
                wm_ovl = int(compute_overlap_metrics(wm_placement, benchmark)["overlap_count"])
                log(f"  wiremask done: proxy={wm_proxy:.5f} ovl={wm_ovl} "
                    f"Δ={wm_proxy-base_proxy:+.5f} ({100*(wm_proxy-base_proxy)/base_proxy:+.3f}%) "
                    f"(wall={time.time() - t0:.0f}s)")

                if wm_ovl == 0 and wm_proxy < current_proxy - 1e-7:
                    log(f"  ACCEPT wiremask: {current_proxy:.5f} → {wm_proxy:.5f}")
                    current_placement = wm_placement
                    current_proxy = wm_proxy
                else:
                    log(f"  REJECT wiremask (ovl={wm_ovl} or no improvement)")
            except Exception as exc:
                log(f"  wiremask FAILED: {exc}; keeping base")

        # ── Periphery push layer (unchanged from Option C) ───────────────
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining < 60.0:
                log(f"  SKIPPING periphery polish (only {remaining:.0f}s left)")
                return current_placement
            polish_s = min(self.periphery_polish_s, remaining - 20.0)
        else:
            polish_s = self.periphery_polish_s

        log(f"  Periphery push α={self.periphery_alpha} + CD polish (budget {polish_s:.0f}s)")
        try:
            periph, periph_proxy, periph_ovl = _periphery_push_and_polish(
                current_placement, benchmark, plc,
                alpha=self.periphery_alpha,
                cd_budget_s=polish_s,
                log=log,
            )
        except Exception as exc:
            log(f"  periphery push FAILED: {exc}; returning current")
            return current_placement

        improvement = current_proxy - periph_proxy
        log(f"  periphery done: proxy={periph_proxy:.5f} ({improvement:+.5f}) "
            f"ovl={periph_ovl} (total wall={time.time() - t0:.0f}s)")

        if periph_ovl == 0 and periph_proxy < current_proxy:
            log(f"  ACCEPT periphery: {current_proxy:.5f} → {periph_proxy:.5f}")
            return periph
        log(f"  REJECT periphery (ovl={periph_ovl}, Δ={improvement:+.5f}); keeping current")
        return current_placement


# Convenience alias for compatibility with launcher.
Placer = CDLNSSACascadeStackedWireMaskPeripheryPlacer
