"""E130 — Multi-restart V4+Gaussian basin + cascade saddle escape + CD polish.

Combines E128 (V4+Gaussian + cascade saddle + CD polish) with E129
(multi-restart best-of-N descent). Pipeline per bench:

  1. For seed in [42, 142, 242]:
       run V4+Gaussian descent + legalize, score by canonical proxy
  2. Pick best basin (min proxy)
  3. Run cascading_saddle_escape on best basin
  4. Run CD polish on saddle output

Total per-bench wall: N*descent + saddle + cd_polish
At N=3 + saddle=240 + cd=480: ~3*120 + 240 + 480 = 1080s = 18 min/bench.
Budget seconds 1800 leaves safety margin for slower benches.
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
    _ROOT,
    _ROOT / "experiments" / "E84_cascading_saddle" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E127_v4_gaussian" / "code",
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


class E130CascadeMultiRestartPlacer:
    """Multi-restart V4+Gaussian -> best basin -> cascade saddle escape -> CD polish."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 1800.0,  # 30 min/bench safety cap
        n_restarts: int = 3,
        seeds: Optional[List[int]] = None,
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        cd_polish_s: float = 480.0,  # 8 min CD
        saddle_budget_s: float = 240.0,  # 4 min saddle
        saddle_max_iters: int = 3,
        cd_plateau_threshold: float = 0.001,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.n_restarts = n_restarts
        self.seeds = seeds or [42 + 100 * i for i in range(n_restarts)]
        self.num_steps = num_steps
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.init = init
        self.cd_polish_s = cd_polish_s
        self.saddle_budget_s = saddle_budget_s
        self.saddle_max_iters = saddle_max_iters
        self.cd_plateau_threshold = cd_plateau_threshold
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _descend_one(self, benchmark: Benchmark, plc, seed: int, device: str
                    ) -> Tuple[Optional[torch.Tensor], float]:
        """Run one descent attempt; return (pos, basin_proxy) or (None, inf) on failure."""
        try:
            descender = SmoothGlobalPlacerV4Gaussian(
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
            pos = descender.place(benchmark)
            ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
            if ovl > 0:
                pos, _ = project_overlaps(pos, benchmark)
                ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
            if ovl > 0:
                return None, float("inf")
            proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
            return pos, proxy
        except Exception as exc:
            self._log(f"    seed {seed} EXCEPTION: {exc}")
            return None, float("inf")

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(
            f"=== E130 cascade+multirestart ({benchmark.name}) N={self.n_restarts} ==="
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        device = _best_device()
        self._log(f"  device={device}")

        # Phase 1: N restarts, score each basin
        results: List[Tuple[float, int, torch.Tensor]] = []
        for i, seed in enumerate(self.seeds[: self.n_restarts]):
            t_seed = time.time()
            pos, proxy = self._descend_one(benchmark, plc, seed, device)
            wall = time.time() - t_seed
            if pos is not None:
                results.append((proxy, seed, pos))
                self._log(f"  seed {seed}: basin={proxy:.5f} wall={wall:.0f}s")
            else:
                self._log(f"  seed {seed}: FAILED wall={wall:.0f}s")
            # Budget safety: reserve saddle + CD time
            reserve = self.saddle_budget_s + self.cd_polish_s + 30.0
            if deadline is not None and time.time() > deadline - reserve:
                self._log(f"  budget tight, stopping after {i+1} restarts")
                break

        if not results:
            self._log("  all restarts FAILED; SDF fallback")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)
            basin_proxy = float("inf")
            best_seed = -1
        else:
            results.sort(key=lambda r: r[0])
            basin_proxy, best_seed, pos = results[0]
            self._log(f"  BEST seed {best_seed}: basin={basin_proxy:.5f}")

        # Phase 2: Cascade saddle escape on best basin
        t_saddle = time.time()
        try:
            saddle_pos, saddle_info = cascading_saddle_escape(
                pos, benchmark, plc,
                max_iters=self.saddle_max_iters,
                total_budget_s=self.saddle_budget_s,
                log=lambda s: self._log(f"  [saddle] {s}"),
            )
            saddle_ovl = compute_overlap_metrics(saddle_pos, benchmark)["overlap_count"]
            if saddle_ovl == 0:
                saddle_proxy = float(
                    compute_proxy_cost(saddle_pos, benchmark, plc)["proxy_cost"]
                )
                if saddle_proxy <= basin_proxy:
                    pos = saddle_pos
                    self._log(
                        f"  saddle accept: {basin_proxy:.5f} -> {saddle_proxy:.5f}"
                    )
                else:
                    self._log(
                        f"  saddle reject: {saddle_proxy:.5f} > {basin_proxy:.5f}"
                    )
            else:
                self._log(f"  saddle output has {saddle_ovl} overlaps; rejecting")
        except Exception as exc:
            self._log(f"  saddle EXCEPTION: {exc}; skipping")
        self._log(f"  saddle wall={time.time()-t_saddle:.0f}s")

        # Phase 3: CD polish
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd_budget = max(30.0, min(remaining, self.cd_polish_s))
        else:
            cd_budget = self.cd_polish_s
        self._log(f"  CD polish budget={cd_budget:.0f}s")

        evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
        movable = [i for i in range(benchmark.num_macros) if not bool(benchmark.macro_fixed[i])]
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
            f"total_wall={time.time()-t0:.0f}s"
        )
        if final_ovl > 0:
            raise RuntimeError(f"Final has {final_ovl} overlaps")
        return final
