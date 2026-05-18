"""Curvature-adaptive Lévy production placer.

Uses curvature_adaptive_levy_saddle_escape (E101): σ_iter = β/√|λ_min|
per cascade iter. Multi-iter (polish_budget=60). β=3.0.
"""
from __future__ import annotations

import importlib.util
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
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer

from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import CDLNSSADPOKJointPlacer

_E101 = _ROOT / "experiments" / "E101_levy_curvature_adaptive" / "code"
if str(_E101) not in sys.path:
    sys.path.insert(0, str(_E101))
from levy_curvature_adaptive import curvature_adaptive_levy_saddle_escape


class CDLNSSACascadeLevyCurvAdaptPlacer:
    """Curvature-adaptive Lévy cascade saddle."""

    def __init__(
        self,
        max_iters: int = 6,
        K_eps: int = 3,
        beta: float = 3.0,
        polish_budget: float = 60.0,
        budget_seconds: Optional[float] = 3300.0,
        rng_seed: int = 42,
        verbose: bool = True,
    ):
        self.max_iters = max_iters
        self.K_eps = K_eps
        self.beta = beta
        self.polish_budget = polish_budget
        self.budget_seconds = budget_seconds
        self.rng_seed = rng_seed
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSACascadeLevyCurvAdaptPlacer ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        if self.budget_seconds is not None:
            B = self.budget_seconds
            e25_kwargs = dict(cd_hard_cap_s=B*0.20, lns_budget_s=B*0.06, sa_budget_s=B*0.06)
            e41_kwargs = dict(cd_hard_cap_s=B*0.20, lns_budget_s=B*0.05, sa_budget_s=B*0.05,
                              kjoint_budget_s=B*0.05)
        else:
            e25_kwargs = e41_kwargs = {}

        log("  Phase 1: E25")
        e25 = CDLNSSAPlacer(**e25_kwargs).place(benchmark)
        e25_proxy = float(compute_proxy_cost(e25, benchmark, plc)["proxy_cost"])
        log(f"  E25 done: proxy={e25_proxy:.5f} (wall={time.time() - t0:.0f}s)")

        e41 = None
        e41_proxy = float("inf")
        if deadline is not None and (deadline - time.time()) < 300.0:
            log(f"  Phase 2: SKIPPED")
        else:
            log("  Phase 2: E41")
            e41 = CDLNSSADPOKJointPlacer(**e41_kwargs).place(benchmark)
            e41_proxy = float(compute_proxy_cost(e41, benchmark, plc)["proxy_cost"])
            log(f"  E41 done: proxy={e41_proxy:.5f} (wall={time.time() - t0:.0f}s)")

        if e41 is None or e25_proxy <= e41_proxy:
            plateau, plateau_label, plateau_proxy = e25, "E25", e25_proxy
        else:
            plateau, plateau_label, plateau_proxy = e41, "E41", e41_proxy
        log(f"  plateau = {plateau_label} ({plateau_proxy:.5f})")

        cascade_state = plateau
        cascade_proxy = plateau_proxy
        remaining = (deadline - time.time()) if deadline is not None else float('inf')
        if remaining < 60.0:
            log(f"  Phase 3: SKIPPED")
        else:
            cascade_budget = remaining - 10.0 if remaining != float('inf') else 3600.0
            log(f"  Phase 3: curvature-adaptive Lévy saddle (budget {cascade_budget:.0f}s, β={self.beta})")
            try:
                cascade_state, stats = curvature_adaptive_levy_saddle_escape(
                    plateau, benchmark, plc,
                    max_iters=self.max_iters, K_eps=self.K_eps, beta=self.beta,
                    polish_budget=self.polish_budget,
                    total_budget_s=cascade_budget,
                    rng_seed=self.rng_seed,
                    log=lambda s: None,
                )
                cascade_proxy = float(compute_proxy_cost(cascade_state, benchmark, plc)["proxy_cost"])
                log(f"  curv-Lévy cascade done: {cascade_proxy:.5f} iters={stats.get('iters_run', '?')}")
            except Exception as exc:
                log(f"  cascade failed: {exc}; falling back to plateau")
                cascade_state = plateau
                cascade_proxy = plateau_proxy

        candidates = [(e25_proxy, e25, "E25"), (cascade_proxy, cascade_state, "curv_levy_cascade")]
        if e41 is not None:
            candidates.append((e41_proxy, e41, "E41"))
        candidates.sort(key=lambda c: c[0])
        for proxy_i, placement_i, name_i in candidates:
            ovl_i = compute_overlap_metrics(placement_i, benchmark)["overlap_count"]
            if ovl_i == 0:
                log(f"  WINNER: {name_i} proxy={proxy_i:.5f} "
                    f"({', '.join(f'{n}={p:.5f}' for p, _, n in candidates)})  "
                    f"total wall={time.time() - t0:.0f}s")
                return placement_i
            log(f"  WARNING: {name_i} has {ovl_i} overlaps; falling through")
        raise RuntimeError("curv-Lévy pipeline returned no overlap-free candidate")
