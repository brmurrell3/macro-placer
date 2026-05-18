"""CDLNSSACascadeLevyPeripheryPlacer — Lévy cascade + periphery-bias post-pass.

Wraps `cd_lns_sa_cascade_levy/placer.py` (E97 Lévy cascade). After the
Lévy cascade returns a placement, applies a tiny periphery-bias push
(move each movable macro α fraction toward its nearest canvas edge),
then CD-polishes the result. Keep new placement only if (a) zero
overlaps AND (b) proxy strictly improved.

Motivation: DAC25-ReMaP (2025) reports +34% WNS / +65% TNS via periphery
relocation. Local E107 spike on ariane133 at α=0.01 measured proxy
-0.95% and edge_dist constant (-0.08%) vs control's -0.28%, with zero
overlaps. The periphery direction is informative even at small α —
periphery push beats random kick by 0.33%.

This placer is a strictly-conservative wrapper: if the periphery polish
doesn't strictly improve proxy AND maintain zero overlaps, it returns
the original Lévy cascade placement unchanged. No regression risk on
benches where periphery doesn't help.
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

# Reuse the cascade-stacked-6w placer (E84 cascade + E110 6-weight portfolio).
_STACKED_PATH = _ROOT / "submissions" / "cd_lns_sa_cascade_stacked_6w" / "placer.py"
_spec = importlib.util.spec_from_file_location("stacked_placer_6w", str(_STACKED_PATH))
_stacked = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stacked)
CDLNSSACascadeStacked6WPlacer = _stacked.CDLNSSACascadeStacked6WPlacer


def periphery_push_and_polish(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    alpha: float = 0.01,
    cd_budget_s: float = 180.0,
    log=None,
):
    """Push each movable macro α fraction toward nearest canvas edge,
    re-legalize, and CD-polish. Returns (polished, proxy, overlap_count).

    Args:
      placement: [N, 2] tensor of macro centers
      benchmark: benchmark with canvas_width/height, macro_sizes, macro_fixed
      plc: TILOS placement client
      alpha: push fraction (0.0 = no change; 0.01 default = tiny push)
      cd_budget_s: CD polish hard cap
    """
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


class CDLNSSACascadeStackedPeriphery6WK2Placer:
    """Cascade-stacked (cascade + portfolio) + periphery-bias post-pass."""

    def __init__(
        self,
        cascade_max_iters: int = 5,
        cascade_eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
        cascade_polish_budget: float = 180.0,
        portfolio_max_iters: int = 1,        # 6w_k2: 1 iter with 6 weights
        portfolio_K_eps: int = 2,            # 6w_k2: K=2 (deadline-limited)
        portfolio_eps_scale: float = 1.0,
        portfolio_polish_budget: float = 60.0,
        budget_seconds: Optional[float] = 3300.0,
        rng_seed: int = 42,
        periphery_alpha: float = 0.01,
        periphery_polish_s: float = 180.0,
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
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSACascadeStackedPeriphery6WK2Placer ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        # Reserve periphery_polish_s for the post-pass.
        stacked_budget = (
            self.budget_seconds - self.periphery_polish_s - 30.0
            if self.budget_seconds is not None else None
        )
        log(f"  budget split: stacked={stacked_budget}s, periphery polish={self.periphery_polish_s}s")

        stacked = CDLNSSACascadeStacked6WPlacer(
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
        log(f"  Lévy cascade done: proxy={base_proxy:.5f} ovl={base_ovl}"
            f" (wall={time.time() - t0:.0f}s)")

        if base_ovl > 0:
            log(f"  WARNING: Lévy cascade returned ovl={base_ovl}; skipping periphery push")
            return base_placement

        if deadline is not None:
            remaining = deadline - time.time()
            if remaining < 90.0:
                log(f"  SKIPPING periphery polish (only {remaining:.0f}s left)")
                return base_placement
            polish_s = min(self.periphery_polish_s, remaining - 20.0)
        else:
            polish_s = self.periphery_polish_s

        log(f"  Periphery push α={self.periphery_alpha} + CD polish (budget {polish_s:.0f}s)")
        try:
            periph, periph_proxy, periph_ovl = periphery_push_and_polish(
                base_placement, benchmark, plc,
                alpha=self.periphery_alpha,
                cd_budget_s=polish_s,
                log=log,
            )
        except Exception as exc:
            log(f"  periphery push FAILED: {exc}; returning Lévy result")
            return base_placement

        improvement = base_proxy - periph_proxy
        log(f"  periphery done: proxy={periph_proxy:.5f} ({improvement:+.5f}) ovl={periph_ovl}"
            f" (total wall={time.time() - t0:.0f}s)")

        # Strictly conservative: only accept if zero overlaps AND lower proxy.
        if periph_ovl == 0 and periph_proxy < base_proxy:
            log(f"  ACCEPT periphery: {base_proxy:.5f} → {periph_proxy:.5f}"
                f" ({improvement / base_proxy * 100:+.2f}%)")
            return periph
        log(f"  REJECT periphery (ovl={periph_ovl}, Δ={improvement:+.5f}); keeping Lévy")
        return base_placement
