"""E138 — V4-Gaussian best-of-N seeds for basin attractor diversity.

Runs V4-Gaussian descent with 3 different RNG seeds {42, 123, 999},
picks the best canonical-proxy basin (before CD polish), then CD-polishes
the winner. Hypothesis: V4's Adam descent is seed-sensitive on hard
benches, and best-of-N captures basin diversity that single-seed misses.

Isolated experiment — imports SmoothGlobalPlacerV4Gaussian read-only.
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


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Placer:
    """V4-Gaussian best-of-N seeds + CD polish."""

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
        cd_polish_s: float = 600.0,
        cd_plateau_threshold: float = 0.001,
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
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _run_seed(
        self,
        seed: int,
        benchmark: Benchmark,
        plc,
        device: str,
        deadline: Optional[float],
    ) -> Optional[Tuple[float, torch.Tensor]]:
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
                f"  [seed={seed}] canonical={canonical:.5f} ovl=0 wall={wall:.0f}s"
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
            f"=== E138 v4_multiseed ({benchmark.name}) "
            f"seeds={self.seeds} budget={self.budget_seconds}s ==="
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        device = _best_device()
        self._log(f"  device={device}")

        results: List[Tuple[float, int, torch.Tensor]] = []
        for seed in self.seeds:
            # Safety: leave at least cd_polish_s + 60s for CD
            if deadline is not None and time.time() > deadline - (self.cd_polish_s + 60):
                self._log(f"  [budget] skip remaining seeds; need CD time")
                break
            r = self._run_seed(seed, benchmark, plc, device, deadline)
            if r is not None:
                results.append((r[0], seed, r[1]))

        if not results:
            self._log("  All seeds failed; SDF fallback")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)
            return self._cd_polish_and_return(
                pos, benchmark, plc, deadline, t_global, best_seed=-1
            )

        results.sort(key=lambda r: r[0])
        best_canonical, best_seed, best_pos = results[0]
        per_seed = ", ".join(f"s={s}:{c:.5f}" for c, s, _ in results)
        self._log(
            f"  [PICK] seed={best_seed} basin (canonical={best_canonical:.5f}); "
            f"per-seed: {per_seed}"
        )
        return self._cd_polish_and_return(
            best_pos, benchmark, plc, deadline, t_global, best_seed=best_seed
        )

    def _cd_polish_and_return(
        self,
        pos: torch.Tensor,
        benchmark: Benchmark,
        plc,
        deadline: Optional[float],
        t_global: float,
        best_seed: int,
    ) -> torch.Tensor:
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd_budget = max(30.0, min(remaining, self.cd_polish_s))
        else:
            cd_budget = self.cd_polish_s
        self._log(f"  CD polish budget={cd_budget:.0f}s (from seed={best_seed})")

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
        final = evaluator.placement.detach().clone().to(torch.float32)
        final_proxy = float(compute_proxy_cost(final, benchmark, plc)["proxy_cost"])
        final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
        self._log(
            f"  Final: proxy={final_proxy:.5f} ovl={final_ovl} "
            f"seed={best_seed} total_wall={time.time()-t_global:.0f}s"
        )
        if final_ovl > 0:
            final, _ = project_overlaps(final, benchmark)
            final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
            if final_ovl > 0:
                raise RuntimeError(f"Final has {final_ovl} overlaps")
        return final
