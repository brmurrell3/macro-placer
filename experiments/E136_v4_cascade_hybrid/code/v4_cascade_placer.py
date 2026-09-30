"""E136 — V4-Gaussian basin + cascade saddle escape on top.

Wraps the production thinkorplace-v2 Placer (read-only import) to produce a
V4 + CD-polish basin, then applies E84 cascading_saddle_escape with
conservative settings to probe whether the V4 basin has soft-mode
curvature that CD polish missed.

Isolated experiment: imports only, no modifications to any submission or
shared library code. The saddle wrapper is monotonic — it can only improve
on the V4 basin, never regress.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_E84 = _ROOT / "experiments" / "E84_cascading_saddle" / "code"
_E74 = _ROOT / "experiments" / "E74_hessian_saddle" / "code"
_V4 = _ROOT / "submissions" / "thinkorplace-v2"
for p in (_ROOT, _E84, _E74, _V4):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

import importlib.util as _ilu
_v4_spec = _ilu.spec_from_file_location("_v4_placer_mod", str(_V4 / "placer.py"))
_v4_mod = _ilu.module_from_spec(_v4_spec)
_v4_spec.loader.exec_module(_v4_mod)
V4Placer = _v4_mod.Placer

from cascading_saddle import cascading_saddle_escape  # noqa: E402


class Placer:
    """E136 V4-Gaussian + cascade saddle escape hybrid.

    Total budget 1500s default (matches thinkorplace-v2). Allocates:
      - V4 descent + CD polish: budget - cascade_reserve - 30s
      - Cascade saddle escape:  cascade_reserve (default 360s)
    """

    def __init__(
        self,
        budget_seconds: Optional[float] = 1500.0,
        cascade_reserve_s: float = 360.0,
        cascade_max_iters: int = 2,
        cascade_eps_values: tuple = (0.5, 1.5),
        cascade_polish_budget: float = 120.0,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.cascade_reserve_s = cascade_reserve_s
        self.cascade_max_iters = cascade_max_iters
        self.cascade_eps_values = cascade_eps_values
        self.cascade_polish_budget = cascade_polish_budget
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        self._log(f"=== E136 v4_cascade_hybrid ({benchmark.name}) ===")

        v4_budget = None
        if self.budget_seconds is not None:
            v4_budget = max(60.0, self.budget_seconds - self.cascade_reserve_s - 30.0)
        v4 = V4Placer(budget_seconds=v4_budget, verbose=self.verbose)
        basin = v4.place(benchmark)
        basin_wall = time.time() - t0

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        basin_proxy = float(compute_proxy_cost(basin, benchmark, plc)["proxy_cost"])
        basin_ovl = compute_overlap_metrics(basin, benchmark)["overlap_count"]
        self._log(
            f"[E136] V4 basin: proxy={basin_proxy:.5f} ovl={basin_ovl} "
            f"wall={basin_wall:.0f}s"
        )

        if basin_ovl > 0:
            self._log(f"[E136] V4 basin has overlaps; returning basin as-is")
            return basin

        elapsed = time.time() - t0
        if self.budget_seconds is not None:
            cascade_budget = max(60.0, self.budget_seconds - elapsed - 30.0)
        else:
            cascade_budget = self.cascade_reserve_s
        self._log(f"[E136] cascade budget={cascade_budget:.0f}s")

        try:
            saddled, stats = cascading_saddle_escape(
                basin.to(torch.float32),
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
            lift = basin_proxy - final_proxy
            self._log(
                f"[E136] saddle: basin={basin_proxy:.5f} -> final={final_proxy:.5f} "
                f"lift={lift:+.5f} ({100*lift/basin_proxy:+.2f}%) ovl={final_ovl}"
            )
            if final_ovl == 0 and final_proxy < basin_proxy - 1e-7:
                return saddled
            return basin
        except Exception as exc:
            self._log(f"[E136] cascade EXCEPTION: {exc}; returning V4 basin")
            return basin


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", help="benchmark name (e.g., ibm03)")
    ap.add_argument("--budget", type=float, default=1500.0)
    args = ap.parse_args()

    bench_dir = find_benchmark_dir(args.bench)
    bench, _plc = load_benchmark_from_dir(str(bench_dir))
    placer = Placer(budget_seconds=args.budget)
    result = placer.place(bench)
    print(f"DONE: {args.bench} final_proxy="
          f"{compute_proxy_cost(result, bench, _plc)['proxy_cost']:.5f}")
