"""E171 — adaptive E166/E143 selection by hard-macro count.

If benchmark.num_hard_macros > 400: run E143 stack (K-joint+SA productive).
Else: run E166 stack with extended portfolio budget.

NOT per-bench dispatch — uses runtime input dimension (analogous to
choosing block size in matrix multiplication).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]

import importlib.util as _ilu

_E166 = _ROOT / "experiments" / "E166_v4_full_stack" / "code" / "placer.py"
_e166_spec = _ilu.spec_from_file_location("_e166_for_e171", str(_E166))
_e166_mod = _ilu.module_from_spec(_e166_spec)
_e166_spec.loader.exec_module(_e166_mod)
E166Placer = _e166_mod.Placer

_E143 = _ROOT / "experiments" / "E143_v4_full_kjoint_sa" / "code" / "placer.py"
_e143_spec = _ilu.spec_from_file_location("_e143_for_e171", str(_E143))
_e143_mod = _ilu.module_from_spec(_e143_spec)
_e143_spec.loader.exec_module(_e143_mod)
E143Placer = _e143_mod.Placer


class Placer:
    """Adaptive: E143 stack on large benches, extended-E166 on small benches."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 3000.0,
        large_bench_threshold: int = 400,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.large_bench_threshold = large_bench_threshold
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark) -> torch.Tensor:
        n_hard = int(benchmark.num_hard_macros)
        self._log(
            f"=== E171 adaptive_kjoint_gate ({benchmark.name}) "
            f"hard_movables={n_hard} threshold={self.large_bench_threshold} ==="
        )
        if n_hard > self.large_bench_threshold:
            self._log(f"  LARGE bench: using E143 stack (K-joint+SA)")
            inner = E143Placer(
                budget_seconds=self.budget_seconds, verbose=self.verbose,
            )
        else:
            self._log(f"  SMALL bench: using E166 stack with extended portfolio")
            inner = E166Placer(
                budget_seconds=self.budget_seconds,
                cascade_reserve_s=400.0,
                portfolio_reserve_s=1100.0,  # absorbed K-joint+SA's allocation
                portfolio_max_iters=3,        # was 2; more iters with bigger budget
                cd_polish_s=500.0,
                verbose=self.verbose,
            )
        return inner.place(benchmark)
