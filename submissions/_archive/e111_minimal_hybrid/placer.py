"""e111_minimal_hybrid — V3 basin + CD polish + cascade saddle + portfolio saddle.

Hybrid experiment: tries the E84 cascade saddle escape (canonical eigvec
Hessian-based perturbations) + E100 portfolio saddle (3 non-canonical
Hessian weights) on top of the V3 (thinkorplace-v2) basin.

Pipeline:
  1. V3 smooth descent + greedy_legalize (~30-60s)
  2. CD polish on V3 basin (~300s)
  3. cascade_saddle_escape on CD-polished result (up to 600s, max_iters=5)
  4. portfolio_saddle_escape (3 non-canonical weights, up to 400s)
  5. Final CD polish if time remains
  Total budget: 2400s (40 min/bench)

References:
  - submissions/thinkorplace-v2/placer.py (V3 basin + CD)
  - submissions/common/cd_lns_sa_cascade_stacked/placer.py (cascade+portfolio)
  - experiments/E84_cascading_saddle/code/cascading_saddle.py
  - experiments/E100_weight_portfolio_saddle/code/portfolio_saddle.py

Comparison target:
  - thinkorplace-v2 M3 IBM 1.00279 / EPYC 1.008
  - V3 alone on ibm17 ≈ 1.20 (hardest IBM)

Success criteria:
  - ibm17: hybrid < 1.18 (V3 alone is 1.20)
  - --all avg: hybrid < 1.000 (V3 alone is 1.003)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E74_hessian_saddle" / "code",
    _ROOT / "experiments" / "E84_cascading_saddle" / "code",
    _ROOT / "experiments" / "E100_weight_portfolio_saddle" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v3 import SmoothGlobalPlacerV3  # noqa: E402
from cascading_saddle import cascading_saddle_escape  # noqa: E402
from portfolio_saddle import portfolio_saddle_escape  # noqa: E402


class Placer:
    """V3 basin → CD polish → cascade saddle → portfolio saddle → final CD."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 2400.0,           # 40 min/bench
        # V3 descent params (match thinkorplace-v2).
        v3_num_steps: int = 500,
        v3_lr_frac: float = 0.005,
        v3_gamma_start_frac: float = 5e-3,
        v3_gamma_end_frac: float = 5e-5,
        v3_overlap_lambda_end: float = 10.0,
        v3_overlap_ramp_pct: float = 0.7,
        v3_init: str = "sdf",
        # CD polish on V3 basin.
        cd_polish_basin_s: float = 300.0,
        # Cascade saddle params.
        cascade_max_iters: int = 5,
        cascade_eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
        cascade_polish_budget: float = 180.0,
        cascade_total_budget: float = 600.0,
        # Portfolio saddle params (3 non-canonical weights only — cascade does
        # the canonical direction).
        portfolio_max_iters: int = 2,
        portfolio_K_eps: int = 2,
        portfolio_eps_scale: float = 1.0,
        portfolio_polish_budget: float = 60.0,
        portfolio_total_budget: float = 400.0,
        # Final CD polish.
        final_cd_polish_s: float = 200.0,
        cd_plateau_threshold: float = 0.001,
        rng_seed: int = 42,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.v3_num_steps = v3_num_steps
        self.v3_lr_frac = v3_lr_frac
        self.v3_gamma_start_frac = v3_gamma_start_frac
        self.v3_gamma_end_frac = v3_gamma_end_frac
        self.v3_overlap_lambda_end = v3_overlap_lambda_end
        self.v3_overlap_ramp_pct = v3_overlap_ramp_pct
        self.v3_init = v3_init
        self.cd_polish_basin_s = cd_polish_basin_s
        self.cascade_max_iters = cascade_max_iters
        self.cascade_eps_values = cascade_eps_values
        self.cascade_polish_budget = cascade_polish_budget
        self.cascade_total_budget = cascade_total_budget
        self.portfolio_max_iters = portfolio_max_iters
        self.portfolio_K_eps = portfolio_K_eps
        self.portfolio_eps_scale = portfolio_eps_scale
        self.portfolio_polish_budget = portfolio_polish_budget
        self.portfolio_total_budget = portfolio_total_budget
        self.final_cd_polish_s = final_cd_polish_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.rng_seed = rng_seed
        self.verbose = verbose
        # Three non-canonical weights: canonical is handled by cascade_saddle.
        self.portfolio = [
            (1.0, 0.0, 1.0),  # cong-focus
            (1.0, 1.0, 0.0),  # density-focus
            (0.0, 1.0, 1.0),  # non-WL
        ]

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _safe_fallback_place(self, benchmark, plc):
        """Defensive fallback: SDF init + project_overlaps."""
        self._log("  fallback: SDF + project_overlaps")
        pos = sdf_init(benchmark)
        pos, _ = project_overlaps(pos, benchmark)
        ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        if ovl > 0:
            self._log(f"  WARNING: SDF+project still has {ovl} overlaps")
        return pos

    def _run_v3(self, benchmark, plc, deadline):
        """V3 smooth descent + greedy_legalize with retries + fallback."""
        pos = None
        for attempt, cfg in enumerate([
            dict(overlap_lambda_end=self.v3_overlap_lambda_end),
            dict(overlap_lambda_end=50.0),
            dict(overlap_lambda_end=100.0, num_steps=300),
        ]):
            try:
                num_steps = cfg.get("num_steps", self.v3_num_steps)
                ovl_end = cfg["overlap_lambda_end"]
                self._log(f"  V3 attempt {attempt+1}: ovl_end={ovl_end} steps={num_steps}")
                descender = SmoothGlobalPlacerV3(
                    num_steps=num_steps,
                    lr_frac=self.v3_lr_frac,
                    gamma_start_frac=self.v3_gamma_start_frac,
                    gamma_end_frac=self.v3_gamma_end_frac,
                    overlap_lambda_end=ovl_end,
                    overlap_ramp_pct=self.v3_overlap_ramp_pct,
                    init=self.v3_init,
                    rng_seed=self.rng_seed + attempt * 100,
                    verbose=False,
                )
                pos_try = descender.place(benchmark)
                ovl_try = compute_overlap_metrics(pos_try, benchmark)["overlap_count"]
                if ovl_try == 0:
                    pos_try2, _ = project_overlaps(pos_try, benchmark)
                    ovl2 = compute_overlap_metrics(pos_try2, benchmark)["overlap_count"]
                    if ovl2 == 0:
                        pos = pos_try2
                        self._log(f"  V3 attempt {attempt+1}: ovl=0 (post-project)")
                        break
                else:
                    pos_try2, _ = project_overlaps(pos_try, benchmark)
                    ovl2 = compute_overlap_metrics(pos_try2, benchmark)["overlap_count"]
                    if ovl2 == 0:
                        pos = pos_try2
                        self._log(f"  V3 attempt {attempt+1}: ovl=0 after project_overlaps")
                        break
                    self._log(f"  V3 attempt {attempt+1}: still {ovl_try}/{ovl2} overlaps")
            except Exception as exc:
                self._log(f"  V3 attempt {attempt+1} EXCEPTION: {exc}")
                continue

            if deadline is not None and time.time() > deadline - 60:
                break

        if pos is None:
            pos = self._safe_fallback_place(benchmark, plc)
        return pos

    def _cd_polish(self, pos, benchmark, plc, budget_s):
        """Run CD polish; returns new placement."""
        evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        run_cd_adaptive(
            evaluator, benchmark, plc, movable,
            min_time_s=budget_s * 0.5,
            hard_cap_s=budget_s,
            patience=5,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=None,
        )
        return evaluator.placement.detach().clone().to(torch.float32)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== e111_minimal_hybrid ({benchmark.name}) ===")
        if deadline is not None:
            self._log(f"  budget={self.budget_seconds:.0f}s")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Phase 1: V3 descent + greedy legalize.
        self._log("  Phase 1: V3 smooth descent + greedy_legalize")
        v3_pos = self._run_v3(benchmark, plc, deadline)
        v3_wall = time.time() - t0
        v3_proxy = float(compute_proxy_cost(v3_pos, benchmark, plc)["proxy_cost"])
        v3_ovl = compute_overlap_metrics(v3_pos, benchmark)["overlap_count"]
        self._log(
            f"  V3 basin: proxy={v3_proxy:.5f} ovl={v3_ovl} wall={v3_wall:.0f}s"
        )
        best_state, best_proxy, best_name = v3_pos, v3_proxy, "V3"

        # Phase 2: CD polish on V3 basin.
        if deadline is not None:
            remaining = deadline - time.time()
            cd_basin_budget = max(
                30.0,
                min(remaining - 600.0, self.cd_polish_basin_s),
            )
        else:
            cd_basin_budget = self.cd_polish_basin_s
        if cd_basin_budget < 30.0:
            self._log(f"  Phase 2: CD polish SKIPPED (budget={cd_basin_budget:.0f}s)")
            cd_pos = v3_pos
            cd_proxy = v3_proxy
        else:
            self._log(f"  Phase 2: CD polish on V3 basin (budget={cd_basin_budget:.0f}s)")
            try:
                cd_pos = self._cd_polish(v3_pos, benchmark, plc, cd_basin_budget)
                cd_proxy = float(compute_proxy_cost(cd_pos, benchmark, plc)["proxy_cost"])
                cd_ovl = compute_overlap_metrics(cd_pos, benchmark)["overlap_count"]
                self._log(
                    f"  CD polished: proxy={cd_proxy:.5f} ovl={cd_ovl} "
                    f"wall={time.time()-t0:.0f}s"
                )
                if cd_ovl == 0 and cd_proxy < best_proxy:
                    best_state, best_proxy, best_name = cd_pos, cd_proxy, "CD"
            except Exception as exc:
                self._log(f"  CD polish FAILED ({exc}); using V3 basin")
                cd_pos = v3_pos
                cd_proxy = v3_proxy

        # Phase 3: cascade saddle escape on CD-polished result.
        if deadline is not None:
            remaining = deadline - time.time()
            # Reserve some for portfolio + final CD.
            cascade_budget = max(
                60.0,
                min(
                    self.cascade_total_budget,
                    remaining - self.portfolio_total_budget - self.final_cd_polish_s - 30.0,
                ),
            )
        else:
            cascade_budget = self.cascade_total_budget
        if cascade_budget < 90.0:
            self._log(f"  Phase 3: cascade SKIPPED (budget={cascade_budget:.0f}s)")
            cascade_state = cd_pos
            cascade_proxy = cd_proxy
        else:
            self._log(
                f"  Phase 3: cascade_saddle canonical (budget={cascade_budget:.0f}s, "
                f"max_iters={self.cascade_max_iters})"
            )
            try:
                cascade_state, stats = cascading_saddle_escape(
                    cd_pos, benchmark, plc,
                    max_iters=self.cascade_max_iters,
                    eps_values=self.cascade_eps_values,
                    polish_budget=self.cascade_polish_budget,
                    total_budget_s=cascade_budget,
                    log=lambda s: None,
                )
                cascade_proxy = float(compute_proxy_cost(
                    cascade_state, benchmark, plc)["proxy_cost"])
                cascade_ovl = compute_overlap_metrics(cascade_state, benchmark)["overlap_count"]
                self._log(
                    f"  cascade done: proxy={cascade_proxy:.5f} ovl={cascade_ovl} "
                    f"iters={stats.get('iters_run', '?')} wall={time.time()-t0:.0f}s"
                )
                if cascade_ovl == 0 and cascade_proxy < best_proxy:
                    best_state, best_proxy, best_name = cascade_state, cascade_proxy, "cascade"
            except Exception as exc:
                self._log(f"  cascade FAILED ({exc}); using CD plateau")
                cascade_state = cd_pos
                cascade_proxy = cd_proxy

        # Phase 4: portfolio saddle escape on cascade plateau.
        if deadline is not None:
            remaining = deadline - time.time()
            portfolio_budget = max(
                60.0,
                min(
                    self.portfolio_total_budget,
                    remaining - self.final_cd_polish_s - 30.0,
                ),
            )
        else:
            portfolio_budget = self.portfolio_total_budget
        if portfolio_budget < 90.0:
            self._log(f"  Phase 4: portfolio SKIPPED (budget={portfolio_budget:.0f}s)")
            portfolio_state = cascade_state
            portfolio_proxy = cascade_proxy
        else:
            self._log(
                f"  Phase 4: portfolio_saddle 3 non-canonical weights "
                f"(budget={portfolio_budget:.0f}s, K_eps={self.portfolio_K_eps})"
            )
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
                portfolio_ovl = compute_overlap_metrics(
                    portfolio_state, benchmark)["overlap_count"]
                self._log(
                    f"  portfolio done: proxy={portfolio_proxy:.5f} "
                    f"ovl={portfolio_ovl} iters={stats.get('iters_run', '?')} "
                    f"wall={time.time()-t0:.0f}s"
                )
                if portfolio_ovl == 0 and portfolio_proxy < best_proxy:
                    best_state, best_proxy, best_name = (
                        portfolio_state, portfolio_proxy, "portfolio")
            except Exception as exc:
                self._log(f"  portfolio FAILED ({exc}); using cascade plateau")
                portfolio_state = cascade_state
                portfolio_proxy = cascade_proxy

        # Phase 5: Final CD polish on best state.
        if deadline is not None:
            remaining = deadline - time.time()
            final_cd_budget = max(0.0, min(remaining - 10.0, self.final_cd_polish_s))
        else:
            final_cd_budget = self.final_cd_polish_s
        if final_cd_budget >= 30.0:
            self._log(f"  Phase 5: final CD polish (budget={final_cd_budget:.0f}s)")
            try:
                final_pos = self._cd_polish(
                    best_state, benchmark, plc, final_cd_budget)
                final_proxy = float(compute_proxy_cost(
                    final_pos, benchmark, plc)["proxy_cost"])
                final_ovl = compute_overlap_metrics(
                    final_pos, benchmark)["overlap_count"]
                self._log(
                    f"  final CD: proxy={final_proxy:.5f} ovl={final_ovl} "
                    f"wall={time.time()-t0:.0f}s"
                )
                if final_ovl == 0 and final_proxy < best_proxy:
                    best_state, best_proxy, best_name = (
                        final_pos, final_proxy, "final_CD")
            except Exception as exc:
                self._log(f"  final CD FAILED ({exc}); using prior best")
        else:
            self._log(f"  Phase 5: final CD SKIPPED (budget={final_cd_budget:.0f}s)")

        # Validate winner has zero overlaps.
        ovl_check = compute_overlap_metrics(best_state, benchmark)["overlap_count"]
        if ovl_check > 0:
            self._log(f"  WARNING: winner {best_name} has {ovl_check} overlaps; "
                      f"falling back to V3 basin")
            best_state = v3_pos
            best_proxy = v3_proxy
            best_name = "V3 (fallback)"
            ovl_check = compute_overlap_metrics(best_state, benchmark)["overlap_count"]
            if ovl_check > 0:
                raise RuntimeError(f"V3 fallback also has {ovl_check} overlaps")

        self._log(
            f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"(V3={v3_proxy:.5f} CD={cd_proxy:.5f} "
            f"cascade={cascade_proxy:.5f} portfolio={portfolio_proxy:.5f}) "
            f"total_wall={time.time()-t0:.0f}s"
        )
        return best_state
