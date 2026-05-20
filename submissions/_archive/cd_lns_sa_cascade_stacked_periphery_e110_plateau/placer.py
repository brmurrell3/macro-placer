"""CDLNSSACascadeStackedPeripheryE110PlateauPlacer — Variant B + periphery.

Wraps CDLNSSACascadeStackedE110PlateauPlacer (Variant B, which puts E110
in the plateau pick before cascade) with the strict-conservative
periphery polish from Option C.

This is the "fully integrated" placer where E110 is a first-class lane.
Cascade may compound the E110 lift if E110 wins the plateau.

Use this if Variant A (cd_lns_sa_cascade_stacked_periphery_e110 — E110
as 4th candidate AFTER cascade) is marginal.
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
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# Base: Variant B (cascade_stacked with E110 plateau).
_STACKED_PATH = (
    _ROOT / "submissions" / "common"
    / "cd_lns_sa_cascade_stacked_e110_plateau" / "placer.py"
)
_spec = importlib.util.spec_from_file_location("stacked_e110_plateau", str(_STACKED_PATH))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CDLNSSACascadeStackedE110PlateauPlacer = _mod.CDLNSSACascadeStackedE110PlateauPlacer

# Periphery push from Option C.
_PERI_PATH = (
    _ROOT / "submissions" / "cd_lns_sa_cascade_stacked_periphery" / "placer.py"
)
_pspec = importlib.util.spec_from_file_location("periphery_placer", str(_PERI_PATH))
_pmod = importlib.util.module_from_spec(_pspec)
_pspec.loader.exec_module(_pmod)
periphery_push_and_polish = _pmod.periphery_push_and_polish


class CDLNSSACascadeStackedPeripheryE110PlateauPlacer:
    """Variant B + periphery wrapper. E110 in plateau-pick before cascade."""

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
        periphery_polish_s: float = 180.0,
        verbose: bool = True,
        # E110 hyperparams.
        e110_num_steps: int = 500,
        e110_lr_frac: float = 0.005,
        e110_gamma_start_frac: float = 5e-3,
        e110_gamma_end_frac: float = 5e-5,
        e110_overlap_lambda_end: float = 50.0,
        e110_overlap_ramp_pct: float = 0.7,
        e110_legalize_step_frac: float = 0.005,
        e110_legalize_radius_steps: int = 200,
        e110_init: str = "sdf",
        e110_cd_polish_s: float = 120.0,
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
        self.e110_num_steps = e110_num_steps
        self.e110_lr_frac = e110_lr_frac
        self.e110_gamma_start_frac = e110_gamma_start_frac
        self.e110_gamma_end_frac = e110_gamma_end_frac
        self.e110_overlap_lambda_end = e110_overlap_lambda_end
        self.e110_overlap_ramp_pct = e110_overlap_ramp_pct
        self.e110_legalize_step_frac = e110_legalize_step_frac
        self.e110_legalize_radius_steps = e110_legalize_radius_steps
        self.e110_init = e110_init
        self.e110_cd_polish_s = e110_cd_polish_s

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSACascadeStackedPeripheryE110PlateauPlacer ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        stacked_budget = (
            self.budget_seconds - self.periphery_polish_s - 30.0
            if self.budget_seconds is not None else None
        )
        log(f"  budget split: stacked={stacked_budget}s, "
            f"periphery={self.periphery_polish_s}s")

        stacked = CDLNSSACascadeStackedE110PlateauPlacer(
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
            e110_num_steps=self.e110_num_steps,
            e110_lr_frac=self.e110_lr_frac,
            e110_gamma_start_frac=self.e110_gamma_start_frac,
            e110_gamma_end_frac=self.e110_gamma_end_frac,
            e110_overlap_lambda_end=self.e110_overlap_lambda_end,
            e110_overlap_ramp_pct=self.e110_overlap_ramp_pct,
            e110_legalize_step_frac=self.e110_legalize_step_frac,
            e110_legalize_radius_steps=self.e110_legalize_radius_steps,
            e110_init=self.e110_init,
            e110_cd_polish_s=self.e110_cd_polish_s,
        )
        base_placement = stacked.place(benchmark)

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        base_proxy = float(compute_proxy_cost(base_placement, benchmark, plc)["proxy_cost"])
        base_ovl = compute_overlap_metrics(base_placement, benchmark)["overlap_count"]
        log(f"  Variant B done: proxy={base_proxy:.5f} ovl={base_ovl}"
            f" (wall={time.time() - t0:.0f}s)")

        if base_ovl > 0:
            log(f"  WARNING: Variant B returned ovl={base_ovl}; skipping periphery push")
            return base_placement

        if deadline is not None:
            remaining = deadline - time.time()
            if remaining < 90.0:
                log(f"  SKIPPING periphery polish (only {remaining:.0f}s left)")
                return base_placement
            polish_s = min(self.periphery_polish_s, remaining - 20.0)
        else:
            polish_s = self.periphery_polish_s

        log(f"  Periphery push α={self.periphery_alpha} + CD polish "
            f"(budget {polish_s:.0f}s)")
        try:
            periph, periph_proxy, periph_ovl = periphery_push_and_polish(
                base_placement, benchmark, plc,
                alpha=self.periphery_alpha,
                cd_budget_s=polish_s,
                log=log,
            )
        except Exception as exc:
            log(f"  periphery push FAILED: {exc}; returning Variant B result")
            return base_placement

        improvement = base_proxy - periph_proxy
        log(f"  periphery done: proxy={periph_proxy:.5f} "
            f"({improvement:+.5f}) ovl={periph_ovl} "
            f"(total wall={time.time() - t0:.0f}s)")

        if periph_ovl == 0 and periph_proxy < base_proxy:
            log(f"  ACCEPT periphery: {base_proxy:.5f} → {periph_proxy:.5f} "
                f"({improvement / base_proxy * 100:+.2f}%)")
            return periph
        log(f"  REJECT periphery (ovl={periph_ovl}, Δ={improvement:+.5f}); "
            f"keeping Variant B")
        return base_placement
