"""E144 — E143 + periphery wrapper (strict-conservative).

Adds α=0.01 periphery push + CD-polish acceptance gate after E143's
SA polish v2 phase. Conservative: only accept if strictly better.

Isolated experiment — all primitives imported read-only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E18_dpo_init" / "code",
    _ROOT / "experiments" / "E39_kmacro_joint_lns" / "code",
    _ROOT / "experiments" / "E84_cascading_saddle" / "code",
    _ROOT / "experiments" / "E97_levy_saddle" / "code",
    _ROOT / "experiments" / "E100_weight_portfolio_saddle" / "code",
    _ROOT / "experiments" / "E107_periphery_bias" / "code",
    _ROOT / "experiments" / "E74_hessian_saddle" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E127_v4_gaussian" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Load E143 placer module directly
import importlib.util as _ilu
_E143 = _ROOT / "experiments" / "E143_v4_full_kjoint_sa" / "code" / "placer.py"
_spec = _ilu.spec_from_file_location("_e143_placer", str(_E143))
_e143_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_e143_mod)
E143Placer = _e143_mod.Placer

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost
from decompose_spike import push_periphery


class Placer:
    """E143 full stack + periphery wrapper (strict-conservative)."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 3200.0,
        periphery_alpha: float = 0.01,
        periphery_cd_budget: float = 120.0,
        verbose: bool = True,
        **e143_kwargs,
    ):
        self.budget_seconds = budget_seconds
        self.periphery_alpha = periphery_alpha
        self.periphery_cd_budget = periphery_cd_budget
        self.verbose = verbose
        # Reserve 200s for periphery; the rest goes to E143.
        inner_budget = (
            None if budget_seconds is None
            else max(60.0, budget_seconds - 200.0)
        )
        self._inner = E143Placer(
            budget_seconds=inner_budget, verbose=verbose, **e143_kwargs
        )

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_global = time.time()
        self._log(f"=== E144 v4_full_periphery ({benchmark.name}) ===")

        # Phase 1-6: full E143 stack
        inner = self._inner.place(benchmark)

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        inner_proxy = float(compute_proxy_cost(inner, benchmark, plc)["proxy_cost"])
        inner_ovl = compute_overlap_metrics(inner, benchmark)["overlap_count"]
        self._log(
            f"[E144] E143 done: proxy={inner_proxy:.5f} ovl={inner_ovl} "
            f"elapsed={time.time()-t_global:.0f}s"
        )

        if inner_ovl > 0:
            return inner

        # Phase 7: periphery push + polish + strict-conservative accept
        try:
            pushed = push_periphery(inner, benchmark, self.periphery_alpha)
            pushed, _ = project_overlaps(pushed, benchmark)
            pushed_ovl = compute_overlap_metrics(pushed, benchmark)["overlap_count"]
            if pushed_ovl > 0:
                self._log(f"[E144] periphery push left {pushed_ovl} overlaps; reject")
                return inner

            ev = IncrementalProxyEvaluator(benchmark, plc, pushed)
            movable = [
                i for i in range(benchmark.num_macros)
                if not bool(benchmark.macro_fixed[i])
            ]
            run_cd_adaptive(
                ev, benchmark, plc, movable,
                min_time_s=self.periphery_cd_budget * 0.5,
                hard_cap_s=self.periphery_cd_budget,
                patience=3, plateau_threshold=0.001, log_fn=None,
            )
            polished = ev.placement.detach().clone().to(torch.float32)
            polished_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
            polished_ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
            if polished_ovl == 0 and polished_proxy < inner_proxy - 1e-7:
                self._log(
                    f"[E144] periphery ACCEPT: {inner_proxy:.5f} -> {polished_proxy:.5f} "
                    f"({100*(inner_proxy-polished_proxy)/inner_proxy:+.2f}%) "
                    f"total_wall={time.time()-t_global:.0f}s"
                )
                return polished
            self._log(
                f"[E144] periphery REJECT: {polished_proxy:.5f} >= {inner_proxy:.5f}; "
                f"keep E143 output ({inner_proxy:.5f})"
            )
            return inner
        except Exception as exc:
            self._log(f"[E144] periphery EXCEPTION: {exc}; keep E143 output")
            return inner
