"""V3Minimal multi-seed ensemble.

Runs N seeds of V3 descent (random init + perturbation), each with
shorter CD polish. Picks best canonical proxy. Total budget split
evenly across seeds.

Hypothesis: random init diversity → different basins → ensemble pick
beats single-seed by 0.5-1.5%.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

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

_E111_CODE = _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code"
if str(_E111_CODE) not in sys.path:
    sys.path.insert(0, str(_E111_CODE))
from smooth_global_placer_v3 import SmoothGlobalPlacerV3


class E111MinimalMultiseedPlacer:
    """3-seed ensemble of V3+CD; pick best."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 720.0,  # 12 min total = 4 min/seed
        n_seeds: int = 3,
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 50.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.n_seeds = n_seeds
        self.num_steps = num_steps
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.init = init
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _one_seed(self, benchmark, plc, seed, cd_budget_s):
        try:
            placer = SmoothGlobalPlacerV3(
                num_steps=self.num_steps,
                lr_frac=self.lr_frac,
                gamma_start_frac=self.gamma_start_frac,
                gamma_end_frac=self.gamma_end_frac,
                overlap_lambda_end=self.overlap_lambda_end,
                overlap_ramp_pct=self.overlap_ramp_pct,
                init=self.init,
                rng_seed=seed,
                verbose=False,
            )
            pos = placer.place(benchmark)
            ev = IncrementalProxyEvaluator(benchmark, plc, pos)
            movable = [i for i in range(benchmark.num_macros)
                       if not bool(benchmark.macro_fixed[i])]
            run_cd_adaptive(
                ev, benchmark, plc, movable,
                min_time_s=cd_budget_s * 0.5,
                hard_cap_s=cd_budget_s,
                patience=3, plateau_threshold=0.001,
            )
            pos = ev.placement.detach().clone().to(torch.float32)
            proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
            ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
            return pos, proxy, ovl
        except Exception as exc:
            self._log(f"    seed={seed} FAILED: {exc}")
            return None, float("inf"), -1

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E111MinimalMultiseed ({benchmark.name}), N={self.n_seeds} ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        per_seed_total = self.budget_seconds / self.n_seeds
        per_seed_cd = max(60.0, per_seed_total - 60.0)

        best_pos, best_proxy, best_seed = None, float("inf"), None
        for i in range(self.n_seeds):
            seed = 42 + i * 1000
            if deadline is not None and (deadline - time.time()) < per_seed_total * 0.4:
                self._log(f"  seed {i}: SKIPPED (only {deadline - time.time():.0f}s left)")
                break
            self._log(f"  seed {i} (rng={seed}): CD budget {per_seed_cd:.0f}s")
            pos, proxy, ovl = self._one_seed(benchmark, plc, seed, per_seed_cd)
            self._log(f"  seed {i}: proxy={proxy:.5f} ovl={ovl}")
            if pos is not None and ovl == 0 and proxy < best_proxy:
                best_pos, best_proxy, best_seed = pos, proxy, i
        if best_pos is None:
            raise RuntimeError("All multiseed runs failed")

        self._log(f"  WINNER: seed {best_seed} proxy={best_proxy:.5f} "
                  f"total_wall={time.time() - t0:.0f}s")
        return best_pos
