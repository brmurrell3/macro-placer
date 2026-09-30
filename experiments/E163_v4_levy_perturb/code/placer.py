"""E139 — V4-Gaussian basin + Lévy ε saddle escape (E97 mechanism).

Same V4 wrapper as E136 but uses levy_saddle_escape (heavy-tail ε
magnitudes) instead of cascading_saddle_escape (fixed Gaussian grid).
Tests whether Lévy's magnitude diversity helps on V4 basins specifically.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_E97 = _ROOT / "experiments" / "E97_levy_saddle" / "code"
_E84 = _ROOT / "experiments" / "E84_cascading_saddle" / "code"
_E74 = _ROOT / "experiments" / "E74_hessian_saddle" / "code"
_V4 = _ROOT / "submissions" / "thinkorplace-v2"
for p in (_ROOT, _E97, _E84, _E74, _V4):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

import importlib.util as _ilu
_v4_spec = _ilu.spec_from_file_location("_v4_placer_mod_e139", str(_V4 / "placer.py"))
_v4_mod = _ilu.module_from_spec(_v4_spec)
_v4_spec.loader.exec_module(_v4_mod)
V4Placer = _v4_mod.Placer

from levy_saddle import levy_saddle_escape  # noqa: E402


class Placer:
    """E139 V4-Gaussian + Lévy saddle escape hybrid."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 1500.0,
        cascade_reserve_s: float = 420.0,
        K_eps: int = 4,
        eps_scale: float = 1.0,
        max_iters: int = 2,
        polish_budget: float = 120.0,
        rng_seed: int = 42,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.cascade_reserve_s = cascade_reserve_s
        self.K_eps = K_eps
        self.eps_scale = eps_scale
        self.max_iters = max_iters
        self.polish_budget = polish_budget
        self.rng_seed = rng_seed
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        self._log(f"=== E139 v4_levy_perturb ({benchmark.name}) ===")

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
            f"[E139] V4 basin: proxy={basin_proxy:.5f} ovl={basin_ovl} "
            f"wall={basin_wall:.0f}s"
        )

        if basin_ovl > 0:
            self._log(f"[E139] V4 basin has overlaps; returning basin as-is")
            return basin

        elapsed = time.time() - t0
        if self.budget_seconds is not None:
            saddle_budget = max(60.0, self.budget_seconds - elapsed - 30.0)
        else:
            saddle_budget = self.cascade_reserve_s
        self._log(f"[E139] levy saddle budget={saddle_budget:.0f}s")

        try:
            saddled, stats = levy_saddle_escape(
                basin.to(torch.float32),
                benchmark,
                plc,
                max_iters=self.max_iters,
                K_eps=self.K_eps,
                eps_scale=self.eps_scale,
                polish_budget=self.polish_budget,
                total_budget_s=saddle_budget,
                rng_seed=self.rng_seed,
                log=self._log if self.verbose else (lambda s: None),
            )
            final_proxy = float(compute_proxy_cost(saddled, benchmark, plc)["proxy_cost"])
            final_ovl = compute_overlap_metrics(saddled, benchmark)["overlap_count"]
            lift = basin_proxy - final_proxy
            self._log(
                f"[E139] levy: basin={basin_proxy:.5f} -> final={final_proxy:.5f} "
                f"lift={lift:+.5f} ({100*lift/basin_proxy:+.2f}%) ovl={final_ovl}"
            )
            if final_ovl == 0 and final_proxy < basin_proxy - 1e-7:
                return saddled
            return basin
        except Exception as exc:
            self._log(f"[E139] levy EXCEPTION: {exc}; returning V4 basin")
            return basin
