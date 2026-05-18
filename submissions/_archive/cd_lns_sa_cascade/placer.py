"""CDLNSSACascadePlacer — E48 hybrid + cascading Hessian saddle escape.

Same pipeline as E74 (cd_lns_sa_hessian) except Phase 3 iterates the
saddle escape instead of running it once. After each ±ε polish, the
Hessian is recomputed at the new state and a new soft mode is found;
the cascade stops when

  - the smallest eigenvalue is non-negative (true local min reached), OR
  - the cascade iteration produced no improvement, OR
  - max_iters reached, OR
  - wall budget exhausted.

Empirical aggregate over 17 IBM (cached, no wall cap): 1.06119 vs E74
1.0666 (−0.51 %).

`budget_seconds` (default 3300s = 55 min) enforces a hard wall deadline.
Same enforcement model as the wall-safe E74 placer.
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
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# E25 placer.
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer

# E41 placer.
from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import CDLNSSADPOKJointPlacer

# Cascading saddle escape (E84).
_E84_CODE = _ROOT / "experiments" / "E84_cascading_saddle" / "code"
if str(_E84_CODE) not in sys.path:
    sys.path.insert(0, str(_E84_CODE))
from cascading_saddle import cascading_saddle_escape


class CDLNSSACascadePlacer:
    """E48 plateau pick + cascading Hessian saddle escape, wall-budgeted."""

    def __init__(
        self,
        max_iters: int = 5,
        eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
        polish_budget: float = 180.0,
        budget_seconds: Optional[float] = 3300.0,
        verbose: bool = True,
    ):
        self.max_iters = max_iters
        self.eps_values = eps_values
        self.polish_budget = polish_budget
        self.budget_seconds = budget_seconds
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSACascadePlacer ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        if self.budget_seconds is not None:
            B = self.budget_seconds
            e25_kwargs = dict(
                cd_hard_cap_s=B * 0.20, lns_budget_s=B * 0.06, sa_budget_s=B * 0.06,
            )
            e41_kwargs = dict(
                cd_hard_cap_s=B * 0.20, lns_budget_s=B * 0.05, sa_budget_s=B * 0.05,
                kjoint_budget_s=B * 0.05,
            )
            log(f"  budget enforced: {self.budget_seconds:.0f}s; "
                f"E25 cap≈{sum(e25_kwargs.values()):.0f}s, "
                f"E41 cap≈{sum(e41_kwargs.values()):.0f}s, "
                f"cascade≈{B * 0.33:.0f}s")
        else:
            e25_kwargs, e41_kwargs = {}, {}
            log(f"  no wall budget (offline mode)")

        # 1. E25.
        log("  Phase 1: E25 (CDLNSSAPlacer)")
        e25 = CDLNSSAPlacer(**e25_kwargs).place(benchmark)
        e25_proxy = float(compute_proxy_cost(e25, benchmark, plc)["proxy_cost"])
        log(f"  E25 done: proxy={e25_proxy:.5f} (wall={time.time() - t0:.0f}s)")

        # 2. E41.
        e41 = None
        e41_proxy = float("inf")
        if deadline is not None and (deadline - time.time()) < 300.0:
            log(f"  Phase 2: SKIPPED (only {deadline - time.time():.0f}s left)")
        else:
            log("  Phase 2: E41 (CDLNSSADPOKJointPlacer)")
            e41 = CDLNSSADPOKJointPlacer(**e41_kwargs).place(benchmark)
            e41_proxy = float(compute_proxy_cost(e41, benchmark, plc)["proxy_cost"])
            log(f"  E41 done: proxy={e41_proxy:.5f} (wall={time.time() - t0:.0f}s)")

        if e41 is None or e25_proxy <= e41_proxy:
            plateau, plateau_label, plateau_proxy = e25, "E25", e25_proxy
        else:
            plateau, plateau_label, plateau_proxy = e41, "E41", e41_proxy
        log(f"  E48 plateau = {plateau_label} ({plateau_proxy:.5f})")

        # 3. Cascading saddle escape.
        cascade_state = plateau
        cascade_proxy = plateau_proxy
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining < 60.0:
                log(f"  Phase 3: SKIPPED (only {remaining:.0f}s left)")
            else:
                cascade_budget = remaining - 10.0  # 10s safety buffer
                log(f"  Phase 3: cascading saddle (budget {cascade_budget:.0f}s, "
                    f"max_iters {self.max_iters})")
                try:
                    cascade_state, stats = cascading_saddle_escape(
                        plateau, benchmark, plc,
                        max_iters=self.max_iters,
                        eps_values=self.eps_values,
                        polish_budget=self.polish_budget,
                        total_budget_s=cascade_budget,
                        log=log if self.verbose else None,
                    )
                    cascade_proxy = float(compute_proxy_cost(
                        cascade_state, benchmark, plc)["proxy_cost"])
                    log(f"  cascade done: {cascade_proxy:.5f} "
                        f"iters={stats['iters_run']}")
                except Exception as exc:
                    log(f"  cascade failed: {exc}; falling back to plateau")
                    cascade_state = plateau
                    cascade_proxy = plateau_proxy
        else:
            log(f"  Phase 3: cascading saddle (offline, max_iters {self.max_iters})")
            try:
                cascade_state, stats = cascading_saddle_escape(
                    plateau, benchmark, plc,
                    max_iters=self.max_iters,
                    eps_values=self.eps_values,
                    polish_budget=self.polish_budget,
                    total_budget_s=3600.0,
                    log=log if self.verbose else None,
                )
                cascade_proxy = float(compute_proxy_cost(
                    cascade_state, benchmark, plc)["proxy_cost"])
                log(f"  cascade done: {cascade_proxy:.5f} iters={stats['iters_run']}")
            except Exception as exc:
                log(f"  cascade failed: {exc}; falling back to plateau")

        # 4. Best of all (overlap-validated fallback chain).
        # All three phases independently guarantee zero overlaps internally,
        # but defend against the edge case where cascade returns a state
        # whose overlap-count slipped past its internal validation (e.g.
        # float-precision wedge after legalization). If the lowest-proxy
        # candidate has overlaps, walk the sorted list until we find a
        # zero-overlap one. E25 is always present and always overlap-free,
        # so the chain is bounded.
        candidates = [(e25_proxy, e25, "E25"), (cascade_proxy, cascade_state, "cascade")]
        if e41 is not None:
            candidates.append((e41_proxy, e41, "E41"))
        candidates.sort(key=lambda c: c[0])
        best_proxy, best_placement, best_name = candidates[0]

        chosen = None
        for proxy_i, placement_i, name_i in candidates:
            ovl_i = compute_overlap_metrics(placement_i, benchmark)["overlap_count"]
            if ovl_i == 0:
                chosen = (proxy_i, placement_i, name_i)
                break
            log(f"  WARNING: {name_i} has {ovl_i} overlaps; falling through")
        if chosen is None:
            raise RuntimeError(
                "Cascade pipeline returned no overlap-free candidate "
                "(all of E25 / E41 / cascade have hard-macro overlaps)"
            )
        best_proxy, best_placement, best_name = chosen

        log(f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"({', '.join(f'{n}={p:.5f}' for p, _, n in candidates)})  "
            f"total wall={time.time() - t0:.0f}s")
        return best_placement
