"""E140 — V4-Gaussian multi-seed basin pick + cascade saddle escape.

Combines two 2026-05-21 positive results: E138 (multi-seed basin
diversity) + E136 (cascade saddle escape on V4 basin).

Pipeline:
  1. Run V4 descent for each seed in {42, 123, 999}, score pre-CD basin.
  2. Pick lowest-proxy basin.
  3. CD polish that basin.
  4. Cascade saddle escape on polished result.

Isolated experiment: imports V4Placer and SmoothGlobalPlacerV4Gaussian
read-only; imports cascading_saddle_escape read-only. No modifications.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E18_dpo_init" / "code",
    _ROOT / "experiments" / "E84_cascading_saddle" / "code",
    _ROOT / "experiments" / "E74_hessian_saddle" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E127_v4_gaussian" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
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

from smooth_global_placer_v4_gaussian import SmoothGlobalPlacerV4Gaussian
from cascading_saddle import cascading_saddle_escape


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Placer:
    """V4 multi-seed basin pick + CD polish + cascade saddle escape."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 1500.0,
        seeds: Tuple[int, ...] = (42, 123, 999),
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        cd_polish_s: float = 450.0,
        cd_plateau_threshold: float = 0.001,
        cascade_reserve_s: float = 300.0,
        cascade_max_iters: int = 2,
        cascade_eps_values: Tuple[float, ...] = (0.5, 1.5),
        cascade_polish_budget: float = 90.0,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.seeds = seeds
        self.num_steps = num_steps
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.init = init
        self.cd_polish_s = cd_polish_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.cascade_reserve_s = cascade_reserve_s
        self.cascade_max_iters = cascade_max_iters
        self.cascade_eps_values = cascade_eps_values
        self.cascade_polish_budget = cascade_polish_budget
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _run_seed_basin(
        self,
        seed: int,
        benchmark: Benchmark,
        plc,
        device: str,
    ) -> Optional[Tuple[float, torch.Tensor]]:
        """V4 descent only (no CD polish yet); returns (canonical_proxy, pos)."""
        t_seed = time.time()
        try:
            desc = SmoothGlobalPlacerV4Gaussian(
                num_steps=self.num_steps,
                lr_frac=self.lr_frac,
                gamma_start_frac=self.gamma_start_frac,
                gamma_end_frac=self.gamma_end_frac,
                overlap_lambda_end=self.overlap_lambda_end,
                overlap_ramp_pct=self.overlap_ramp_pct,
                init=self.init,
                device=device,
                rng_seed=seed,
                verbose=False,
            )
            pos = desc.place(benchmark)
            ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
            if ovl > 0:
                pos, _ = project_overlaps(pos, benchmark)
                ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
            wall = time.time() - t_seed
            if ovl > 0:
                self._log(f"  [seed={seed}] SKIP: {ovl} residual overlaps wall={wall:.0f}s")
                return None
            canonical = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
            self._log(
                f"  [seed={seed}] basin canonical={canonical:.5f} ovl=0 wall={wall:.0f}s"
            )
            return canonical, pos
        except Exception as exc:
            wall = time.time() - t_seed
            self._log(f"  [seed={seed}] EXCEPTION {exc} wall={wall:.0f}s")
            return None

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_global = time.time()
        deadline = (t_global + self.budget_seconds) if self.budget_seconds else None
        self._log(
            f"=== E140 v4_multiseed_cascade ({benchmark.name}) "
            f"seeds={self.seeds} budget={self.budget_seconds}s ==="
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        device = _best_device()
        self._log(f"  device={device}")

        # Phase 1: multi-seed basin search (no CD polish, just descent)
        results: List[Tuple[float, int, torch.Tensor]] = []
        reserve_after_seeds = self.cd_polish_s + self.cascade_reserve_s + 60
        for seed in self.seeds:
            if deadline is not None and time.time() > deadline - reserve_after_seeds:
                self._log(f"  [budget] skip remaining seeds; need CD+cascade time")
                break
            r = self._run_seed_basin(seed, benchmark, plc, device)
            if r is not None:
                results.append((r[0], seed, r[1]))

        if not results:
            self._log("  All seeds failed; SDF fallback")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)
        else:
            results.sort(key=lambda r: r[0])
            best_canonical, best_seed, best_pos = results[0]
            per_seed = ", ".join(f"s={s}:{c:.5f}" for c, s, _ in results)
            self._log(
                f"  [PICK] seed={best_seed} basin (canonical={best_canonical:.5f}); "
                f"per-seed: {per_seed}"
            )
            pos = best_pos

        # Phase 2: CD polish the picked basin
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd_budget = max(30.0, min(remaining - self.cascade_reserve_s, self.cd_polish_s))
        else:
            cd_budget = self.cd_polish_s
        self._log(f"  CD polish budget={cd_budget:.0f}s")

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
        polished = evaluator.placement.detach().clone().to(torch.float32)
        polished_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
        polished_ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
        self._log(
            f"  CD polish done: proxy={polished_proxy:.5f} ovl={polished_ovl}"
        )

        if polished_ovl > 0:
            polished, _ = project_overlaps(polished, benchmark)
            polished_ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
            if polished_ovl > 0:
                raise RuntimeError(f"Polished basin has {polished_ovl} overlaps")
            polished_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])

        # Phase 3: cascade saddle escape on polished basin
        if deadline is not None:
            cascade_budget = max(60.0, deadline - time.time() - 30.0)
        else:
            cascade_budget = self.cascade_reserve_s
        self._log(f"  cascade budget={cascade_budget:.0f}s")

        try:
            saddled, stats = cascading_saddle_escape(
                polished.to(torch.float32),
                benchmark,
                plc,
                max_iters=self.cascade_max_iters,
                eps_values=self.cascade_eps_values,
                polish_budget=self.cascade_polish_budget,
                total_budget_s=cascade_budget,
                log=self._log if self.verbose else (lambda s: None),
            )
            final_proxy = float(compute_proxy_cost(saddled, benchmark, plc)["proxy_cost"])
            final_ovl = compute_overlap_metrics(saddled, benchmark)["overlap_count"]
            lift = polished_proxy - final_proxy
            self._log(
                f"  cascade: polished={polished_proxy:.5f} -> final={final_proxy:.5f} "
                f"lift={lift:+.5f} ({100*lift/polished_proxy:+.2f}%) "
                f"total_wall={time.time()-t_global:.0f}s ovl={final_ovl}"
            )
            if final_ovl == 0 and final_proxy < polished_proxy - 1e-7:
                return saddled
            return polished
        except Exception as exc:
            self._log(f"  cascade EXCEPTION: {exc}; returning polished basin")
            return polished
