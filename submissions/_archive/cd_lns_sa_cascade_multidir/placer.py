"""CDLNSSACascadeMultidirPlacer — cascade + multi-direction saddle polish.

Extends the wall-safe `CDLNSSACascadeAdaptivePlacer` (current submission
entry) with one extra polish phase: after cascade returns its best
placement, run the multi-direction saddle escape from E90 to find
basins single-direction cascade can't reach.

Empirical motivation (E90, 2026-05-12): on cached cascade ibm01 (canon
0.84528 from 3 single-direction cascade iters), the rank-2 multi-direction
sweep finds canon 0.84231 = +0.352 % lift. The key signal is that
*combining* eigenvectors of the smooth proxy reaches basins single-eigvec
escape misses, and the eps sweet spot is sign-vector dependent.

Wall budget split: cascade gets 80 % of `budget_seconds`, multidir gets
the rest. Multidir runs `cascade_multidir_escape` which iterates the
multi-direction saddle escape until plateau or budget exhausted.
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

from submissions.cd_lns_sa_cascade.placer_adaptive import (
    CDLNSSACascadeAdaptivePlacer,
)

# Multi-direction saddle escape (E90).
_E90_CODE = _ROOT / "experiments" / "E90_newton_cg_saddle" / "code"
if str(_E90_CODE) not in sys.path:
    sys.path.insert(0, str(_E90_CODE))
from cascade_multidir import cascade_multidir_escape


class CDLNSSACascadeMultidirPlacer(CDLNSSACascadeAdaptivePlacer):
    """Cascade-adaptive + multi-direction saddle polish.

    Designed so the cascade-multidir polish is purely additive on top of
    the existing wall-safe submission: if multidir times out or fails,
    the parent's result is returned unchanged.
    """

    def __init__(
        self,
        *,
        multidir_budget_frac: float = 0.20,
        multidir_K: int = 3,
        multidir_eps_values: Tuple[float, ...] = (0.5, 2.0),
        multidir_polish_budget_s: float = 60.0,
        multidir_max_iters: int = 2,
        only_rank_at_least: int = 2,
        budget_seconds: float = 3000.0,
        **kwargs,
    ):
        # Split the wall budget. Cascade gets (1 - multidir_frac), multidir
        # gets the rest. Parent stores `budget_seconds` and routes that
        # to cascade's wall budget.
        self._total_budget_s = float(budget_seconds)
        self._multidir_budget_frac = float(multidir_budget_frac)
        cascade_budget = self._total_budget_s * (1.0 - self._multidir_budget_frac)
        kwargs["budget_seconds"] = cascade_budget
        super().__init__(**kwargs)
        self.multidir_K = multidir_K
        self.multidir_eps_values = multidir_eps_values
        self.multidir_polish_budget_s = multidir_polish_budget_s
        self.multidir_max_iters = multidir_max_iters
        self.only_rank_at_least = only_rank_at_least

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        log = (lambda s: print(s, flush=True)) if getattr(self, "verbose", True) else (lambda s: None)
        log(f"=== CDLNSSACascadeMultidirPlacer ({benchmark.name}) ===")
        log(f"  total budget {self._total_budget_s:.0f}s; "
            f"cascade {(1 - self._multidir_budget_frac) * self._total_budget_s:.0f}s, "
            f"multidir {self._multidir_budget_frac * self._total_budget_s:.0f}s")

        # Phase A: cascade-adaptive (parent).
        cascade_result = super().place(benchmark)
        cascade_proxy = float(compute_proxy_cost(cascade_result, benchmark, plc=_load_plc(benchmark))["proxy_cost"])
        cascade_ovl = int(compute_overlap_metrics(cascade_result, benchmark)["overlap_count"])
        elapsed = time.time() - t0
        log(f"  cascade-adaptive done: proxy={cascade_proxy:.5f} ovl={cascade_ovl} "
            f"wall={elapsed:.0f}s")

        if cascade_ovl != 0:
            log("  cascade winner has overlaps — skipping multidir polish")
            return cascade_result

        remaining = self._total_budget_s - elapsed
        # Reserve 30 s safety buffer for return overhead.
        multidir_budget = max(0.0, remaining - 30.0)
        if multidir_budget < self.multidir_polish_budget_s * 4:
            log(f"  multidir budget too small ({multidir_budget:.0f}s); "
                f"returning cascade result")
            return cascade_result

        # Phase B: cascade-multidir polish.
        _, plc = load_benchmark_from_dir(str(find_benchmark_dir(benchmark.name)))
        log(f"  Phase B: cascade-multidir polish (budget {multidir_budget:.0f}s, "
            f"K={self.multidir_K}, eps={self.multidir_eps_values}, "
            f"polish={self.multidir_polish_budget_s}s, "
            f"only_rank_at_least={self.only_rank_at_least})")
        try:
            multidir_result, stats = cascade_multidir_escape(
                cascade_result, benchmark, plc,
                K=self.multidir_K,
                eps_values=self.multidir_eps_values,
                polish_budget_s=self.multidir_polish_budget_s,
                max_iters=self.multidir_max_iters,
                total_budget_s=multidir_budget,
                only_rank_at_least=self.only_rank_at_least,
                log=log,
            )
            md_proxy = float(compute_proxy_cost(multidir_result, benchmark, plc)["proxy_cost"])
            md_ovl = int(compute_overlap_metrics(multidir_result, benchmark)["overlap_count"])
            log(f"  multidir done: proxy={md_proxy:.5f} ovl={md_ovl} "
                f"iters={stats['iters_run']} wall={time.time() - t0:.0f}s")
        except Exception as exc:
            log(f"  multidir failed: {exc!r}; returning cascade result")
            return cascade_result

        # Best of {cascade, multidir}. Multidir is only chosen if it strictly
        # improves AND preserves zero overlaps.
        if md_ovl == 0 and md_proxy < cascade_proxy - 1e-7:
            log(f"  WINNER: multidir {md_proxy:.5f} (vs cascade {cascade_proxy:.5f})")
            return multidir_result
        log(f"  WINNER: cascade {cascade_proxy:.5f} (multidir didn't improve)")
        return cascade_result


def _load_plc(benchmark: Benchmark):
    """Load the PlacementCost for a benchmark by name."""
    bench_dir = find_benchmark_dir(benchmark.name)
    _, plc = load_benchmark_from_dir(str(bench_dir))
    return plc
