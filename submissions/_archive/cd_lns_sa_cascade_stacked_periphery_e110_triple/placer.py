"""Triple-E110-lane variant: runs all 3 sweep-winning E110 cfgs and picks best.

Adds ALL THREE sweep top cfgs as parallel E110 lanes:
  1. ovl10:  lr=5e-3, steps=500, ovl_end=10
  2. nocong: lr=3e-3, steps=500, ovl_end=30, include_cong=False
  3. long:   lr=2e-3, steps=1500, ovl_end=50

Plateau-picks over {stacked_periphery, ovl10+CD, nocong+CD, long+CD}.

Time budget split (total ~3300s):
  - stacked_periphery: 2100s (vs default 2700)
  - 3 E110 lanes × 250s each = 750s
  - 30s buffer

The hope: different E110 cfgs find different basins, plateau picks the
best per bench, no per-bench tuning required.
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

_PERI_PATH = (
    _ROOT / "submissions" / "cd_lns_sa_cascade_stacked_periphery" / "placer.py"
)
_spec = importlib.util.spec_from_file_location("periphery_placer", str(_PERI_PATH))
_peri_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_peri_mod)
CDLNSSACascadeStackedPeripheryPlacer = _peri_mod.CDLNSSACascadeStackedPeripheryPlacer

_E110_CODE = _ROOT / "experiments" / "E110_smooth_global_placer" / "code"
if str(_E110_CODE) not in sys.path:
    sys.path.insert(0, str(_E110_CODE))
from smooth_global_placer import SmoothGlobalPlacer  # noqa: E402


E110_CFGS = [
    # (label, kwargs)
    ("ovl10", dict(
        num_steps=500, lr_frac=0.005, gamma_end_frac=5e-5,
        overlap_lambda_end=10.0, overlap_ramp_pct=0.7,
        legalize_step_frac=0.005, legalize_radius_steps=200,
        init="sdf", include_congestion=True,
    )),
    ("nocong", dict(
        num_steps=500, lr_frac=0.003, gamma_end_frac=5e-5,
        overlap_lambda_end=30.0, overlap_ramp_pct=0.7,
        legalize_step_frac=0.005, legalize_radius_steps=200,
        init="sdf", include_congestion=False,
    )),
    ("long", dict(
        num_steps=1500, lr_frac=0.002, gamma_end_frac=5e-5,
        overlap_lambda_end=50.0, overlap_ramp_pct=0.7,
        legalize_step_frac=0.005, legalize_radius_steps=200,
        init="sdf", include_congestion=True,
    )),
]


class CDLNSSACascadeStackedPeripheryE110TriplePlacer:
    """Option C + 3 parallel E110 lanes with different sweep-winning cfgs."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 3300.0,
        e110_total_budget_s: float = 900.0,         # 300s × 3 lanes
        e110_cd_polish_s: float = 150.0,
        rng_seed: int = 42,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.e110_total_budget_s = e110_total_budget_s
        self.e110_cd_polish_s = e110_cd_polish_s
        self.rng_seed = rng_seed
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _run_one_e110(self, benchmark, plc, label, kwargs, cd_budget_s):
        t0 = time.time()
        try:
            placer = SmoothGlobalPlacer(
                rng_seed=self.rng_seed, verbose=False, **kwargs,
            )
            pos = placer.place(benchmark)
        except Exception as exc:
            self._log(f"    E110[{label}] descent FAILED: {exc}")
            return None, float("inf")
        self._log(f"    E110[{label}] descent: wall={time.time()-t0:.0f}s")
        try:
            ev = IncrementalProxyEvaluator(benchmark, plc, pos)
            movable = [i for i in range(benchmark.num_macros)
                       if not bool(benchmark.macro_fixed[i])]
            run_cd_adaptive(
                ev, benchmark, plc, movable,
                min_time_s=cd_budget_s, hard_cap_s=cd_budget_s,
                patience=3, plateau_threshold=0.001,
            )
            pos = ev.placement.detach().clone().to(torch.float32)
        except Exception as exc:
            self._log(f"    E110[{label}]+CD FAILED: {exc}")
            return None, float("inf")
        proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        if ovl > 0:
            self._log(f"    E110[{label}] has {ovl} overlaps; rejecting")
            return None, float("inf")
        self._log(f"    E110[{label}] done: proxy={proxy:.5f}")
        return pos, proxy

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = self._log
        log(f"=== CDLNSSACascadeStackedPeripheryE110TriplePlacer ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        peri_budget = (
            self.budget_seconds - self.e110_total_budget_s
            if self.budget_seconds else None
        )
        log(f"  budget split: stacked_periphery={peri_budget}s, "
            f"3 E110 lanes={self.e110_total_budget_s}s")

        peri_placer = CDLNSSACascadeStackedPeripheryPlacer(
            budget_seconds=peri_budget,
            rng_seed=self.rng_seed,
            verbose=self.verbose,
        )
        peri_pos = peri_placer.place(benchmark)

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        peri_proxy = float(compute_proxy_cost(peri_pos, benchmark, plc)["proxy_cost"])
        peri_ovl = compute_overlap_metrics(peri_pos, benchmark)["overlap_count"]
        log(f"  stacked_periphery: proxy={peri_proxy:.5f} ovl={peri_ovl} "
            f"(wall={time.time() - t0:.0f}s)")

        # Run 3 E110 lanes if time remains.
        candidates = [(peri_proxy, peri_pos, "stacked_periphery")]
        per_lane = (deadline - time.time() - 30.0) / 3 if deadline else 300.0
        cd_for_lane = min(self.e110_cd_polish_s, per_lane - 60.0)

        if per_lane < 90.0:
            log(f"  E110 lanes SKIPPED (only {per_lane:.0f}s/lane left)")
        else:
            log(f"  E110 lanes: budget {per_lane:.0f}s/lane, CD polish {cd_for_lane:.0f}s")
            for label, kwargs in E110_CFGS:
                if deadline and (deadline - time.time()) < 90.0:
                    log(f"    E110[{label}] SKIPPED (budget exhausted)")
                    break
                pos, proxy = self._run_one_e110(
                    benchmark, plc, label, kwargs, cd_for_lane,
                )
                if pos is not None:
                    candidates.append((proxy, pos, f"E110[{label}]"))

        # Plateau pick: lowest proxy with zero overlaps.
        candidates.sort(key=lambda c: c[0])
        chosen = None
        for prx, pls, nm in candidates:
            ovl = compute_overlap_metrics(pls, benchmark)["overlap_count"]
            if ovl == 0:
                chosen = (prx, pls, nm)
                break
            log(f"    {nm} has {ovl} overlaps; falling through")
        if chosen is None:
            raise RuntimeError("Triple-E110 returned no overlap-free candidate")
        best_proxy, best_pos, best_name = chosen
        log(f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"({', '.join(f'{n}={p:.5f}' for p, _, n in candidates)})  "
            f"total wall={time.time() - t0:.0f}s")
        return best_pos
