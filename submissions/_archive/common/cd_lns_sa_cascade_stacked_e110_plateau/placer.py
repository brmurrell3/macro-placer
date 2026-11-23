"""CDLNSSACascadeStackedE110PlateauPlacer — Variant B.

Same as CDLNSSACascadeStackedPlacer but adds E110 SmoothGlobalPlacer
as a third init lane. The plateau pick now ranges over
{E25, E41, E110+CD} before cascade_saddle runs.

If E110 wins the plateau, cascade_saddle (and portfolio_saddle) operate
on the E110 basin — potentially compounding the gradient-lane lift.
If E110 loses the plateau, behavior is identical to the original
CDLNSSACascadeStackedPlacer.

Safe by construction:
- E110 lane wrapped in try/except; on failure, plateau picks from {E25, E41}.
- Cascade/portfolio always evaluate against canonical proxy; any lane
  producing overlaps is rejected by the final candidate scan.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

_E25_PATH = _ROOT / "submissions" / "common" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer_b", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer

from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import CDLNSSADPOKJointPlacer

_E84_CODE = _ROOT / "experiments" / "E84_cascading_saddle" / "code"
if str(_E84_CODE) not in sys.path:
    sys.path.insert(0, str(_E84_CODE))
from cascading_saddle import cascading_saddle_escape

_E100_CODE = _ROOT / "experiments" / "E100_weight_portfolio_saddle" / "code"
if str(_E100_CODE) not in sys.path:
    sys.path.insert(0, str(_E100_CODE))
from portfolio_saddle import portfolio_saddle_escape

_E110_CODE = _ROOT / "experiments" / "E110_smooth_global_placer" / "code"
if str(_E110_CODE) not in sys.path:
    sys.path.insert(0, str(_E110_CODE))
from smooth_global_placer import SmoothGlobalPlacer


class CDLNSSACascadeStackedE110PlateauPlacer:
    """E25/E41/E110 plateau-pick + cascade + portfolio."""

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
        e110_device: str = "cpu",
        e110_cd_polish_s: float = 120.0,  # CD polish on E110 basin before plateau
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
        self.verbose = verbose
        # Three non-canonical weights: canonical is handled by cascade_saddle.
        self.portfolio = [
            (1.0, 0.0, 1.0),  # cong-focus
            (1.0, 1.0, 0.0),  # density-focus
            (0.0, 1.0, 1.0),  # non-WL
        ]
        self.e110_num_steps = e110_num_steps
        self.e110_lr_frac = e110_lr_frac
        self.e110_gamma_start_frac = e110_gamma_start_frac
        self.e110_gamma_end_frac = e110_gamma_end_frac
        self.e110_overlap_lambda_end = e110_overlap_lambda_end
        self.e110_overlap_ramp_pct = e110_overlap_ramp_pct
        self.e110_legalize_step_frac = e110_legalize_step_frac
        self.e110_legalize_radius_steps = e110_legalize_radius_steps
        self.e110_init = e110_init
        self.e110_device = e110_device
        self.e110_cd_polish_s = e110_cd_polish_s

    def _e110_lane(
        self,
        benchmark: Benchmark,
        plc,
        cd_polish_s: float,
        log,
    ) -> Tuple[Optional[torch.Tensor], float]:
        """Run E110 + CD polish. Returns (placement, proxy) or (None, inf)."""
        try:
            t0 = time.time()
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
                device=self.e110_device,
                rng_seed=self.rng_seed,
                verbose=False,
            )
            pos = placer.place(benchmark)
            descent_wall = time.time() - t0
            log(f"  E110 descent done: wall={descent_wall:.0f}s, "
                f"CD polish budget={cd_polish_s:.0f}s")

            ev = IncrementalProxyEvaluator(benchmark, plc, pos)
            movable = [
                i for i in range(benchmark.num_macros)
                if not bool(benchmark.macro_fixed[i])
            ]
            run_cd_adaptive(
                ev, benchmark, plc, movable,
                min_time_s=cd_polish_s, hard_cap_s=cd_polish_s,
                patience=3, plateau_threshold=0.001,
            )
            pos = ev.placement.detach().clone().to(torch.float32)
            proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
            ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
            if ovl > 0:
                log(f"  E110+CD has {ovl} overlaps; rejecting lane")
                return None, float("inf")
            return pos, proxy
        except Exception as exc:
            log(f"  E110 lane EXCEPTION: {exc}")
            return None, float("inf")

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSACascadeStackedE110PlateauPlacer ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        if self.budget_seconds is not None:
            B = self.budget_seconds
            # New split with E110 lane: shrink E25/E41 by ~7-10% each.
            e25_kwargs = dict(
                cd_hard_cap_s=B * 0.16, lns_budget_s=B * 0.05, sa_budget_s=B * 0.05,
            )
            e41_kwargs = dict(
                cd_hard_cap_s=B * 0.16, lns_budget_s=B * 0.04, sa_budget_s=B * 0.04,
                kjoint_budget_s=B * 0.04,
            )
            e110_budget_target = B * 0.18         # ~594s for E110 + CD polish
            cascade_budget_target = B * 0.18      # ~594s
            portfolio_budget_target = B * 0.11    # ~363s
            log(f"  budget {self.budget_seconds:.0f}s; "
                f"E25={sum(e25_kwargs.values()):.0f}s "
                f"E41={sum(e41_kwargs.values()):.0f}s "
                f"E110≈{e110_budget_target:.0f}s "
                f"cascade≈{cascade_budget_target:.0f}s "
                f"portfolio≈{portfolio_budget_target:.0f}s")
        else:
            e25_kwargs = e41_kwargs = {}
            e110_budget_target = None
            cascade_budget_target = portfolio_budget_target = None

        # Phase 1: E25 lane.
        log("  Phase 1: E25 (CDLNSSAPlacer)")
        e25 = None
        e25_proxy = float("inf")
        try:
            e25 = CDLNSSAPlacer(**e25_kwargs).place(benchmark)
            e25_proxy = float(compute_proxy_cost(e25, benchmark, plc)["proxy_cost"])
            log(f"  E25 done: proxy={e25_proxy:.5f} (wall={time.time() - t0:.0f}s)")
        except Exception as exc:
            log(f"  E25 FAILED ({exc}); continuing")

        # Phase 2: E41 lane.
        e41 = None
        e41_proxy = float("inf")
        if deadline is not None and (deadline - time.time()) < 300.0:
            log(f"  Phase 2: SKIPPED ({deadline - time.time():.0f}s left)")
        else:
            log("  Phase 2: E41 (CDLNSSADPOKJointPlacer)")
            try:
                e41 = CDLNSSADPOKJointPlacer(**e41_kwargs).place(benchmark)
                e41_proxy = float(compute_proxy_cost(e41, benchmark, plc)["proxy_cost"])
                log(f"  E41 done: proxy={e41_proxy:.5f} (wall={time.time() - t0:.0f}s)")
            except Exception as exc:
                log(f"  E41 FAILED ({exc}); skipping")

        # Phase 2.5: E110 lane (NEW).
        e110 = None
        e110_proxy = float("inf")
        if deadline is not None:
            remaining = deadline - time.time()
            e110_total = min(
                e110_budget_target if e110_budget_target else remaining,
                max(remaining - (cascade_budget_target or 0)
                    - (portfolio_budget_target or 0) - 90.0,
                    self.e110_cd_polish_s + 60.0),
            )
        else:
            e110_total = (e110_budget_target or 600.0)
        if e110_total < (self.e110_cd_polish_s + 30.0):
            log(f"  Phase 2.5 E110: SKIPPED (budget {e110_total:.0f}s too small)")
        else:
            cd_for_e110 = min(self.e110_cd_polish_s, e110_total - 30.0)
            log(f"  Phase 2.5: E110+CD (total budget {e110_total:.0f}s, "
                f"CD polish {cd_for_e110:.0f}s)")
            e110, e110_proxy = self._e110_lane(
                benchmark, plc, cd_polish_s=cd_for_e110, log=log,
            )
            if e110 is not None:
                log(f"  E110+CD done: proxy={e110_proxy:.5f} "
                    f"(wall={time.time() - t0:.0f}s)")

        if e25 is None and e41 is None and e110 is None:
            raise RuntimeError("All init lanes failed")

        # Plateau pick.
        candidates_pre = [
            (e25_proxy, e25, "E25"),
            (e41_proxy, e41, "E41"),
            (e110_proxy, e110, "E110"),
        ]
        candidates_pre = [(p, x, n) for p, x, n in candidates_pre if x is not None]
        candidates_pre.sort(key=lambda c: c[0])
        plateau_proxy, plateau, plateau_label = candidates_pre[0]
        log(f"  plateau = {plateau_label} ({plateau_proxy:.5f})")

        # Phase 3a: cascade_saddle on plateau.
        cascade_state = plateau
        cascade_proxy = plateau_proxy
        if deadline is not None:
            remaining = deadline - time.time()
            cascade_budget = min(
                cascade_budget_target if cascade_budget_target else remaining,
                max(remaining - (portfolio_budget_target or 0) - 60.0, 60.0),
            )
        else:
            cascade_budget = 600.0

        if cascade_budget < 60.0:
            log(f"  Phase 3a: cascade SKIPPED (budget {cascade_budget:.0f}s)")
        else:
            log(f"  Phase 3a: cascade_saddle canonical "
                f"(budget {cascade_budget:.0f}s, max_iters={self.cascade_max_iters})")
            try:
                cascade_state, stats = cascading_saddle_escape(
                    plateau, benchmark, plc,
                    max_iters=self.cascade_max_iters,
                    eps_values=self.cascade_eps_values,
                    polish_budget=self.cascade_polish_budget,
                    total_budget_s=cascade_budget,
                    log=lambda s: None,
                )
                cascade_proxy = float(compute_proxy_cost(
                    cascade_state, benchmark, plc)["proxy_cost"])
                log(f"  cascade done: {cascade_proxy:.5f} "
                    f"iters={stats.get('iters_run', '?')}")
            except Exception as exc:
                log(f"  cascade failed: {exc}; falling back to plateau")
                cascade_state = plateau
                cascade_proxy = plateau_proxy

        # Phase 3b: portfolio_saddle.
        portfolio_state = cascade_state
        portfolio_proxy = cascade_proxy
        if deadline is not None:
            remaining = deadline - time.time()
            portfolio_budget = remaining - 30.0
        else:
            portfolio_budget = 600.0

        if portfolio_budget < 90.0:
            log(f"  Phase 3b: portfolio SKIPPED (budget {portfolio_budget:.0f}s)")
        else:
            log(f"  Phase 3b: portfolio_saddle (budget {portfolio_budget:.0f}s)")
            try:
                portfolio_state, stats = portfolio_saddle_escape(
                    cascade_state, benchmark, plc,
                    max_iters=self.portfolio_max_iters,
                    portfolio=self.portfolio,
                    K_eps=self.portfolio_K_eps,
                    eps_scale=self.portfolio_eps_scale,
                    polish_budget=self.portfolio_polish_budget,
                    total_budget_s=portfolio_budget,
                    rng_seed=self.rng_seed,
                    log=lambda s: None,
                )
                portfolio_proxy = float(compute_proxy_cost(
                    portfolio_state, benchmark, plc)["proxy_cost"])
                log(f"  portfolio done: {portfolio_proxy:.5f}")
            except Exception as exc:
                log(f"  portfolio failed: {exc}; falling back to cascade plateau")
                portfolio_state = cascade_state
                portfolio_proxy = cascade_proxy

        candidates = [
            (cascade_proxy, cascade_state, "cascade"),
            (portfolio_proxy, portfolio_state, "portfolio"),
        ]
        if e25 is not None:
            candidates.append((e25_proxy, e25, "E25"))
        if e41 is not None:
            candidates.append((e41_proxy, e41, "E41"))
        if e110 is not None:
            candidates.append((e110_proxy, e110, "E110"))
        candidates.sort(key=lambda c: c[0])

        chosen = None
        for proxy_i, placement_i, name_i in candidates:
            ovl_i = compute_overlap_metrics(placement_i, benchmark)["overlap_count"]
            if ovl_i == 0:
                chosen = (proxy_i, placement_i, name_i)
                break
            log(f"  WARNING: {name_i} has {ovl_i} overlaps; falling through")
        if chosen is None:
            raise RuntimeError("Pipeline returned no overlap-free candidate")
        best_proxy, best_placement, best_name = chosen

        log(f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"({', '.join(f'{n}={p:.5f}' for p, _, n in candidates)})  "
            f"total wall={time.time() - t0:.0f}s")
        return best_placement
