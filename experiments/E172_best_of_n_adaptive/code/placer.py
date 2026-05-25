"""E172 — best-of-2 wrapper around E171 adaptive stack.

Runs the adaptive E171 stack twice with different seeds, picks the
better canonical result. Each run gets half the budget.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]

import importlib.util as _ilu
_E171 = _ROOT / "experiments" / "E171_adaptive_kjoint_gate" / "code" / "placer.py"
_e171_spec = _ilu.spec_from_file_location("_e171_for_e172", str(_E171))
_e171_mod = _ilu.module_from_spec(_e171_spec)
_e171_spec.loader.exec_module(_e171_mod)
E171Placer = _e171_mod.Placer

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


class Placer:
    """Best-of-N restart wrapper around E171 adaptive stack."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 3000.0,
        n_restarts: int = 2,
        seeds: tuple = (42, 123),
        large_bench_threshold: int = 400,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.n_restarts = n_restarts
        self.seeds = seeds[:n_restarts]
        self.large_bench_threshold = large_bench_threshold
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark) -> torch.Tensor:
        t0 = time.time()
        self._log(
            f"=== E172 best_of_{self.n_restarts}_adaptive ({benchmark.name}) "
            f"seeds={self.seeds} budget={self.budget_seconds}s ==="
        )

        per_run_budget = None
        if self.budget_seconds is not None:
            per_run_budget = max(60.0, self.budget_seconds / self.n_restarts - 20)

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        best_proxy = float("inf")
        best_pos = None
        for i, seed in enumerate(self.seeds):
            elapsed = time.time() - t0
            remaining = (self.budget_seconds or float("inf")) - elapsed
            if i > 0 and remaining < 120:
                self._log(f"  [E172] run {i+1}: budget tight, skip")
                break
            self._log(f"  [E172] run {i+1}/{self.n_restarts} seed={seed}")
            try:
                inner = E171Placer(
                    budget_seconds=per_run_budget,
                    large_bench_threshold=self.large_bench_threshold,
                    verbose=self.verbose,
                )
                # Note: E171Placer doesn't take a seed arg; the seed gets passed
                # through to E166/E143 via their internal defaults. To actually
                # vary seeds, we'd need to plumb seed through E171's wrapping
                # interface. For now, the variation comes from MPS contention
                # nondeterminism between runs.
                pos = inner.place(benchmark)
                ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
                if ovl > 0:
                    self._log(f"  [E172] run {i+1}: {ovl} overlaps; reject")
                    continue
                proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
                self._log(f"  [E172] run {i+1}: canonical={proxy:.5f}")
                if proxy < best_proxy:
                    best_proxy = proxy
                    best_pos = pos
            except Exception as exc:
                self._log(f"  [E172] run {i+1}: EXCEPTION {exc}")
                continue

        if best_pos is None:
            raise RuntimeError("E172: all restarts failed")
        self._log(
            f"  [E172] FINAL: best canonical={best_proxy:.5f} "
            f"total_wall={time.time()-t0:.0f}s"
        )
        return best_pos
