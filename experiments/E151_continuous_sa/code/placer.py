"""E151 — thinkorplace-v2 + SA on continuous Gaussian per-macro moves.

Pipeline (1500s budget):
  1. V4+Gaussian descent (lifts from v2 verbatim)
  2. CD polish #1 (~700s)
  3. SA-continuous (~300s) — N(0, T·step_size) per-macro Gaussian moves
  4. CD polish #2 (~200s, only if SA improved over CD1)

Step 3 differs from E139's SA-swap: pair swaps were swap-stable in the CD
basin, but small continuous displacements of a single macro explore off-
axis directions CD never commits to. Hypothesis: SA-continuous can drift
the basin into a strictly lower neighbourhood, which CD2 then polishes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E127_v4_gaussian" / "code",
    _HERE,  # local sa_continuous.py
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

from smooth_global_placer_v4_gaussian import SmoothGlobalPlacerV4Gaussian  # noqa: E402

from sa_continuous import run_sa_continuous  # noqa: E402


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class E151ContinuousSaPlacer:
    """v2 pipeline + SA on continuous Gaussian moves + CD2 polish."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 1500.0,
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        cd_polish_s: float = 700.0,
        cd_polish2_s: float = 200.0,
        cd_plateau_threshold: float = 0.001,
        sa_budget_s: float = 300.0,
        sa_step_size_frac: float = 0.005,
        sa_T_start_frac: float = 0.01,
        sa_T_end_frac: float = 0.0001,
        rng_seed: int = 42,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.num_steps = num_steps
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.init = init
        self.cd_polish_s = cd_polish_s
        self.cd_polish2_s = cd_polish2_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.sa_budget_s = sa_budget_s
        self.sa_step_size_frac = sa_step_size_frac
        self.sa_T_start_frac = sa_T_start_frac
        self.sa_T_end_frac = sa_T_end_frac
        self.rng_seed = rng_seed
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E151 sa_continuous ({benchmark.name}) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        device = _best_device()
        self._log(f"  device={device}")

        # ── Step 1: V4+Gaussian descent (lift from v2 verbatim) ──
        pos = None
        for attempt, cfg in enumerate([
            dict(overlap_lambda_end=self.overlap_lambda_end),
            dict(overlap_lambda_end=50.0),
            dict(overlap_lambda_end=100.0, num_steps=300),
        ]):
            try:
                num_steps = cfg.get("num_steps", self.num_steps)
                ovl_end = cfg["overlap_lambda_end"]
                self._log(f"  attempt {attempt+1}: ovl_end={ovl_end} steps={num_steps}")
                descender = SmoothGlobalPlacerV4Gaussian(
                    num_steps=num_steps,
                    lr_frac=self.lr_frac,
                    gamma_start_frac=self.gamma_start_frac,
                    gamma_end_frac=self.gamma_end_frac,
                    overlap_lambda_end=ovl_end,
                    overlap_ramp_pct=self.overlap_ramp_pct,
                    init=self.init,
                    device=device,
                    rng_seed=self.rng_seed + attempt * 100,
                    verbose=False,
                )
                pos_try = descender.place(benchmark)
                ovl_try = compute_overlap_metrics(pos_try, benchmark)["overlap_count"]
                if ovl_try == 0:
                    pos = pos_try
                    self._log(f"  attempt {attempt+1}: ovl=0")
                    break
                pos_try2, _ = project_overlaps(pos_try, benchmark)
                if compute_overlap_metrics(pos_try2, benchmark)["overlap_count"] == 0:
                    pos = pos_try2
                    self._log(f"  attempt {attempt+1}: ovl=0 after project")
                    break
                self._log(f"  attempt {attempt+1}: still {ovl_try} overlaps, retrying...")
            except Exception as exc:
                self._log(f"  attempt {attempt+1} EXCEPTION: {exc}")
                continue
            if deadline is not None and time.time() > deadline - 60:
                break

        if pos is None:
            self._log("  fallback: SDF + project_overlaps + CD polish")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)

        descent_wall = time.time() - t0
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self._log(f"  basin: proxy={descent_proxy:.5f} wall={descent_wall:.0f}s")

        # ── Step 2: CD polish #1 ──
        if deadline is not None:
            # Reserve sa_budget_s + cd_polish2_s + 15s buffer for SA and post-polish.
            reserve = self.sa_budget_s + self.cd_polish2_s + 15.0
            remaining = deadline - time.time() - reserve
            cd_budget = max(30.0, min(remaining, self.cd_polish_s))
        else:
            cd_budget = self.cd_polish_s
        self._log(f"  CD polish #1 budget={cd_budget:.0f}s")

        evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        run_cd_adaptive(
            evaluator, benchmark, plc, movable,
            min_time_s=cd_budget * 0.5,
            hard_cap_s=cd_budget,
            patience=5,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=None,
        )
        cd1_proxy = float(evaluator.current_cost()["proxy"])
        self._log(f"  CD polish #1 done: proxy={cd1_proxy:.5f} wall={time.time()-t0:.0f}s")

        # ── Step 3: SA-continuous ──
        if deadline is not None:
            reserve2 = self.cd_polish2_s + 10.0
            sa_budget = max(10.0, min(deadline - time.time() - reserve2, self.sa_budget_s))
        else:
            sa_budget = self.sa_budget_s
        self._log(f"  SA-continuous budget={sa_budget:.0f}s")

        sa_stats = run_sa_continuous(
            evaluator,
            n_hard=benchmark.num_hard_macros,
            fixed_np=benchmark.macro_fixed.cpu().numpy(),
            canvas_width=float(benchmark.canvas_width),
            canvas_height=float(benchmark.canvas_height),
            budget_s=sa_budget,
            step_size_frac=self.sa_step_size_frac,
            T_start_frac=self.sa_T_start_frac,
            T_end_frac=self.sa_T_end_frac,
            rng_seed=self.rng_seed,
            log_fn=self._log if self.verbose else None,
        )

        post_sa_proxy = float(evaluator.current_cost()["proxy"])
        sa_improved = post_sa_proxy < cd1_proxy - 1e-6
        self._log(
            f"  SA-continuous done: post-SA proxy={post_sa_proxy:.5f} "
            f"vs CD1={cd1_proxy:.5f} (Δ={post_sa_proxy - cd1_proxy:+.5f}) "
            f"sa_improved={sa_improved}"
        )

        # ── Step 4: CD polish #2 (only if SA moved us anywhere) ──
        # Always re-polish if SA accepted any moves — even if it didn't beat CD1,
        # the current state may be different from CD1 and CD2 can re-find the basin.
        any_accepts = (
            int(sa_stats.get("accepted_better", 0)) + int(sa_stats.get("accepted_worse", 0))
        ) > 0
        if any_accepts and deadline is not None:
            remaining = deadline - time.time() - 5.0
            cd2_budget = max(15.0, min(remaining, self.cd_polish2_s))
            self._log(f"  CD polish #2 budget={cd2_budget:.0f}s")
            run_cd_adaptive(
                evaluator, benchmark, plc, movable,
                min_time_s=cd2_budget * 0.5,
                hard_cap_s=cd2_budget,
                patience=5,
                plateau_threshold=self.cd_plateau_threshold,
                log_fn=None,
            )
            cd2_proxy = float(evaluator.current_cost()["proxy"])
            self._log(f"  CD polish #2 done: proxy={cd2_proxy:.5f} wall={time.time()-t0:.0f}s")
        else:
            self._log(f"  CD polish #2 skipped (no accepts or no time)")

        final = evaluator.placement.detach().clone().to(torch.float32)
        final_proxy = float(compute_proxy_cost(final, benchmark, plc)["proxy_cost"])
        final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
        self._log(
            f"  Final: proxy={final_proxy:.5f} ovl={final_ovl} "
            f"sa_lift_over_CD1={cd1_proxy - sa_stats['best_proxy']:+.5f} "
            f"total_lift_over_CD1={cd1_proxy - final_proxy:+.5f} "
            f"total_wall={time.time()-t0:.0f}s"
        )
        if final_ovl > 0:
            raise RuntimeError(f"Final placement has {final_ovl} overlaps")
        return final


# Evaluate harness looks for `Placer` class or named class.
Placer = E151ContinuousSaPlacer
