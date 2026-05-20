"""CDLNSSACascadeStackedPeripheryE110Placer — Option C + E110 gradient lane.

Wraps the existing stacked_periphery (Option C) and adds **E110
SmoothGlobalPlacer** as a parallel candidate. After stacked_periphery
returns its placement, runs E110 with a short CD polish on the
remaining budget; pick the better of {stacked_periphery, E110+CD}.

Safe by construction:
- If E110 lane crashes → return stacked_periphery output unchanged.
- If E110+CD doesn't strictly improve OR has overlaps → keep stacked_periphery.
- If budget tight → skip E110 lane entirely.

Initial benchmark (M3, --fast 60s CD): E110 wins 3/4 (ibm01 −4.1%,
ibm09 −1.4%, ibm13 −2.3%) and loses 1 (ibm04 +2.6%). The plateau-pick
absorbs the loss. Expected --all aggregate lift: −0.5% to −1.5%.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# Base placer: stacked_periphery (current Option C).
_PERI_PATH = (
    _ROOT / "submissions" / "cd_lns_sa_cascade_stacked_periphery" / "placer.py"
)
_spec = importlib.util.spec_from_file_location("periphery_placer", str(_PERI_PATH))
_peri_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_peri_mod)
CDLNSSACascadeStackedPeripheryPlacer = _peri_mod.CDLNSSACascadeStackedPeripheryPlacer

# E110 SmoothGlobalPlacer.
_E110_CODE = _ROOT / "experiments" / "E110_smooth_global_placer" / "code"
if str(_E110_CODE) not in sys.path:
    sys.path.insert(0, str(_E110_CODE))
from smooth_global_placer import SmoothGlobalPlacer


class CDLNSSACascadeStackedPeripheryE110Placer:
    """Option C + E110 gradient lane as a parallel candidate."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 3300.0,
        # E110 hyperparameters (default = tonight's first-pass winner).
        e110_num_steps: int = 500,
        e110_lr_frac: float = 0.005,
        e110_gamma_start_frac: float = 5e-3,
        e110_gamma_end_frac: float = 5e-5,
        e110_overlap_lambda_end: float = 50.0,
        e110_overlap_ramp_pct: float = 0.7,
        e110_legalize_step_frac: float = 0.005,
        e110_legalize_radius_steps: int = 200,
        e110_init: str = "sdf",
        # Budget split.
        e110_lane_budget_s: float = 600.0,  # E110 descent + CD polish + buffer
        e110_cd_polish_s: float = 240.0,    # CD polish on E110 basin
        # Pass-through to Option C (rest of budget).
        cascade_max_iters: int = 5,
        cascade_eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
        cascade_polish_budget: float = 180.0,
        portfolio_max_iters: int = 2,
        portfolio_K_eps: int = 2,
        portfolio_eps_scale: float = 1.0,
        portfolio_polish_budget: float = 60.0,
        periphery_alpha: float = 0.01,
        periphery_polish_s: float = 180.0,
        rng_seed: int = 42,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.e110_num_steps = e110_num_steps
        self.e110_lr_frac = e110_lr_frac
        self.e110_gamma_start_frac = e110_gamma_start_frac
        self.e110_gamma_end_frac = e110_gamma_end_frac
        self.e110_overlap_lambda_end = e110_overlap_lambda_end
        self.e110_overlap_ramp_pct = e110_overlap_ramp_pct
        self.e110_legalize_step_frac = e110_legalize_step_frac
        self.e110_legalize_radius_steps = e110_legalize_radius_steps
        self.e110_init = e110_init
        self.e110_lane_budget_s = e110_lane_budget_s
        self.e110_cd_polish_s = e110_cd_polish_s
        self.cascade_max_iters = cascade_max_iters
        self.cascade_eps_values = cascade_eps_values
        self.cascade_polish_budget = cascade_polish_budget
        self.portfolio_max_iters = portfolio_max_iters
        self.portfolio_K_eps = portfolio_K_eps
        self.portfolio_eps_scale = portfolio_eps_scale
        self.portfolio_polish_budget = portfolio_polish_budget
        self.periphery_alpha = periphery_alpha
        self.periphery_polish_s = periphery_polish_s
        self.rng_seed = rng_seed
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _run_e110_lane(
        self,
        benchmark: Benchmark,
        plc,
        budget_s: float,
    ) -> Tuple[Optional[torch.Tensor], Optional[float], Optional[int]]:
        """Run E110 descent + CD polish under budget_s. Returns
        (placement, proxy, overlaps) or (None, None, None) on failure.
        """
        t0 = time.time()
        try:
            placer = SmoothGlobalPlacer(
                num_steps=self.e110_num_steps,
                lr_frac=self.e110_lr_frac,
                gamma_start_frac=self.e110_gamma_start_frac,
                gamma_end_frac=self.e110_gamma_end_frac,
                overlap_lambda_end=self.e110_overlap_lambda_end,
                overlap_ramp_pct=self.e110_overlap_ramp_pct,
                legalize_step_frac=self.e110_legalize_step_frac,
                legalize_radius_steps=self.e110_legalize_radius_steps,
                init=self.e110_init,
                rng_seed=self.rng_seed,
                verbose=False,
            )
            pos = placer.place(benchmark)
        except Exception as exc:
            self._log(f"  E110 descent FAILED: {exc}")
            return None, None, None

        descent_wall = time.time() - t0
        cd_budget = max(60.0, budget_s - descent_wall - 30.0)
        cd_budget = min(cd_budget, self.e110_cd_polish_s)
        self._log(
            f"  E110 descent: wall={descent_wall:.0f}s; "
            f"CD polish budget={cd_budget:.0f}s"
        )

        try:
            evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
            movable = [
                i for i in range(benchmark.num_macros)
                if not bool(benchmark.macro_fixed[i])
            ]
            run_cd_adaptive(
                evaluator, benchmark, plc, movable,
                min_time_s=cd_budget, hard_cap_s=cd_budget,
                patience=3, plateau_threshold=0.001,
            )
            pos = evaluator.placement.detach().clone().to(torch.float32)
        except Exception as exc:
            self._log(f"  E110+CD polish FAILED: {exc}")
            return None, None, None

        proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        return pos, proxy, ovl

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = self._log
        log(f"=== CDLNSSACascadeStackedPeripheryE110Placer ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        # Carve out E110 lane budget from Option C base.
        peri_budget = (
            self.budget_seconds - self.e110_lane_budget_s
            if self.budget_seconds else None
        )
        log(
            f"  budget split: stacked_periphery={peri_budget}s, "
            f"E110 lane={self.e110_lane_budget_s}s"
        )

        # Lane 1-3 + cascade + portfolio + periphery (Option C).
        peri_placer = CDLNSSACascadeStackedPeripheryPlacer(
            cascade_max_iters=self.cascade_max_iters,
            cascade_eps_values=self.cascade_eps_values,
            cascade_polish_budget=self.cascade_polish_budget,
            portfolio_max_iters=self.portfolio_max_iters,
            portfolio_K_eps=self.portfolio_K_eps,
            portfolio_eps_scale=self.portfolio_eps_scale,
            portfolio_polish_budget=self.portfolio_polish_budget,
            budget_seconds=peri_budget,
            rng_seed=self.rng_seed,
            periphery_alpha=self.periphery_alpha,
            periphery_polish_s=self.periphery_polish_s,
            verbose=self.verbose,
        )
        peri_pos = peri_placer.place(benchmark)

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        peri_proxy = float(compute_proxy_cost(peri_pos, benchmark, plc)["proxy_cost"])
        peri_ovl = compute_overlap_metrics(peri_pos, benchmark)["overlap_count"]
        log(
            f"  stacked_periphery: proxy={peri_proxy:.5f} ovl={peri_ovl} "
            f"(wall={time.time() - t0:.0f}s)"
        )

        # Lane 4: E110 + CD polish.
        if deadline is not None:
            remaining = deadline - time.time()
        else:
            remaining = self.e110_lane_budget_s
        if remaining < 90.0:
            log(f"  E110 lane SKIPPED (only {remaining:.0f}s left)")
            return peri_pos

        e110_budget = min(self.e110_lane_budget_s, remaining - 30.0)
        log(f"  E110 lane: budget={e110_budget:.0f}s")
        e110_pos, e110_proxy, e110_ovl = self._run_e110_lane(
            benchmark, plc, budget_s=e110_budget,
        )

        if e110_pos is None or e110_ovl is None:
            log(f"  E110 lane returned no valid output; keeping stacked_periphery")
            return peri_pos
        log(
            f"  E110+CD: proxy={e110_proxy:.5f} ovl={e110_ovl} "
            f"(total wall={time.time() - t0:.0f}s)"
        )

        # Plateau pick (strict: zero overlaps + strict improvement).
        if e110_ovl == 0 and e110_proxy < peri_proxy:
            log(
                f"  ACCEPT E110 lane: {peri_proxy:.5f} → {e110_proxy:.5f} "
                f"({(e110_proxy - peri_proxy) / peri_proxy * 100:+.2f}%)"
            )
            return e110_pos
        log(
            f"  REJECT E110 lane (ovl={e110_ovl}, "
            f"Δ={e110_proxy - peri_proxy:+.5f}); keeping stacked_periphery"
        )
        return peri_pos
