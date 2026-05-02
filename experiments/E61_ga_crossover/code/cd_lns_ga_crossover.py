"""E61 placer: GA crossover between E25 (SDF basin) and E41 (DPO basin)
solutions, polished via CD+LNS+SA-v2.

Pipeline (per benchmark):
  1. Run E25 (CDLNSSAPlacer) → e25_placement.
  2. Run E41 (CDLNSSADPOKJointPlacer) → e41_placement.
  3. Crossover: per hard-macro, random Bernoulli pick from E25 or E41.
  4. project_overlaps to repair the recombination.
  5. Polish via CD plateau (≤1200s) + LNS (≤300s) + SA-v2 (≤300s).
  6. Return lowest-proxy output among {E25, E41, polished_crossover}.

Hypothesis: macros position-mixed from two different verified basins,
followed by local polish, can settle into a local minimum that neither
parent finds alone. Standard GA crossover from crystal-structure
prediction (USPEX) and combinatorial optimization.

Reference:
- E25 — SDF basin pipeline (`submissions/cd_lns_sa/placer.py`).
- E41 — DPO basin pipeline.
- E48 hybrid (just picks max of E25/E41 per bench; no recombination).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import (
    project_overlaps, run_cd_adaptive,
)
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

import importlib.util
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer
run_lns_gridbin = _E25_MOD.run_lns_gridbin
run_sa_polish_v2 = _E25_MOD.run_sa_polish_v2

from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import (
    CDLNSSADPOKJointPlacer,
)


class CDLNSGACrossoverPlacer:
    """E61 placer: crossover-and-polish between E25 and E41 basins."""

    def __init__(
        self,
        crossover_seed: int = 42,
        polish_cd_cap_s: float = 1200.0,
        polish_lns_budget_s: float = 300.0,
        polish_sa_budget_s: float = 300.0,
        polish_sa_T0: float = 5e-4,
        polish_sa_Tf: float = 1e-6,
        verbose: bool = True,
        **kwargs,
    ):
        # Inner placers — pass through any common kwargs.
        e41_only_keys = {"kjoint_K", "kjoint_top_N", "kjoint_budget_s", "kjoint_seed"}
        e41_kwargs = {k: v for k, v in kwargs.items() if k in e41_only_keys}
        common_kwargs = {k: v for k, v in kwargs.items() if k not in e41_only_keys}
        self._e25 = CDLNSSAPlacer(**common_kwargs)
        self._e41 = CDLNSSADPOKJointPlacer(**common_kwargs, **e41_kwargs)

        self.crossover_seed = int(crossover_seed)
        self.polish_cd_cap_s = float(polish_cd_cap_s)
        self.polish_lns_budget_s = float(polish_lns_budget_s)
        self.polish_sa_budget_s = float(polish_sa_budget_s)
        self.polish_sa_T0 = float(polish_sa_T0)
        self.polish_sa_Tf = float(polish_sa_Tf)
        self.verbose = bool(verbose)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.perf_counter()
        self._log(
            f"=== CDLNSGACrossoverPlacer ({benchmark.name}): "
            f"E25 then E41 then crossover+polish (cd={self.polish_cd_cap_s:.0f}s, "
            f"lns={self.polish_lns_budget_s:.0f}s, sa={self.polish_sa_budget_s:.0f}s) ==="
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # 1. Run E25.
        t_e25 = time.perf_counter()
        e25_placement = self._e25.place(benchmark)
        e25_proxy = compute_proxy_cost(e25_placement, benchmark, plc)["proxy_cost"]
        e25_overlaps = compute_overlap_metrics(e25_placement, benchmark)["overlap_count"]
        self._log(
            f"  E25 done: proxy={e25_proxy:.5f} overlaps={e25_overlaps} "
            f"wall={time.perf_counter() - t_e25:.1f}s"
        )

        # 2. Run E41.
        t_e41 = time.perf_counter()
        e41_placement = self._e41.place(benchmark)
        e41_proxy = compute_proxy_cost(e41_placement, benchmark, plc)["proxy_cost"]
        e41_overlaps = compute_overlap_metrics(e41_placement, benchmark)["overlap_count"]
        self._log(
            f"  E41 done: proxy={e41_proxy:.5f} overlaps={e41_overlaps} "
            f"wall={time.perf_counter() - t_e41:.1f}s"
        )

        # 3. Crossover: per-hard-macro Bernoulli pick.
        rng = np.random.default_rng(seed=self.crossover_seed)
        n_hard = benchmark.num_hard_macros
        crossover_mask = rng.random(n_hard) < 0.5  # True = pick from E25
        crossover_placement = e41_placement.detach().clone().to(torch.float64)
        e25_pos = e25_placement.detach().to(torch.float64)
        for i in range(n_hard):
            if crossover_mask[i]:
                crossover_placement[i] = e25_pos[i]
        n_from_e25 = int(crossover_mask.sum())
        self._log(
            f"  crossover: {n_from_e25}/{n_hard} hard macros from E25, "
            f"{n_hard - n_from_e25} from E41 (seed={self.crossover_seed})"
        )

        # 4. Repair via project_overlaps.
        crossover_placement, proj_iters = project_overlaps(crossover_placement, benchmark)
        crossover_post_proj = compute_overlap_metrics(crossover_placement, benchmark)
        self._log(
            f"  project_overlaps: {proj_iters} iters, "
            f"residual={crossover_post_proj['overlap_count']}"
        )
        # If project_overlaps couldn't clear all overlaps in 50 iters, fall back
        # to the better of e25/e41.
        if crossover_post_proj["overlap_count"] > 0:
            self._log(
                f"  crossover unrecoverable ({crossover_post_proj['overlap_count']} "
                f"overlaps after 50 iters); falling back to min(E25, E41)"
            )
            if e25_proxy <= e41_proxy:
                return e25_placement
            return e41_placement

        # 5. Polish via CD + LNS + SA-v2.
        evaluator = IncrementalProxyEvaluator(
            benchmark, plc, crossover_placement.detach().clone()
        )
        crossover_init_proxy = evaluator.current_cost()["proxy"]
        self._log(f"  crossover init proxy={crossover_init_proxy:.5f}")

        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        hard_movable = [
            i for i in range(benchmark.num_hard_macros)
            if not bool(benchmark.macro_fixed[i])
        ]

        self._log(f"  starting polish-CD (cap={self.polish_cd_cap_s:.0f}s)")
        cd_stats = run_cd_adaptive(
            evaluator=evaluator, benchmark=benchmark, plc=plc,
            movable=movable, min_time_s=300.0,
            hard_cap_s=self.polish_cd_cap_s, patience=3,
            plateau_threshold=0.001,
            log_fn=self._log if self.verbose else None,
        )
        cd_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  polish-CD done: sweeps={cd_stats['sweeps']}, "
            f"exit={cd_stats['exit_reason']}, proxy={cd_proxy:.5f}"
        )

        self._log(f"  starting polish-LNS (budget={self.polish_lns_budget_s:.0f}s)")
        lns_stats = run_lns_gridbin(
            evaluator=evaluator, benchmark=benchmark, plc=plc,
            hard_movable=hard_movable, time_budget_s=self.polish_lns_budget_s,
            destroy_frac=0.05, destroy_cap=30, seed=42,
            log_fn=self._log if self.verbose else None,
        )
        lns_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  polish-LNS done: samples={lns_stats['samples']}, "
            f"Δ={lns_stats['total_improvement']:+.5f}, proxy={lns_proxy:.5f}"
        )

        self._log(f"  starting polish-SA (budget={self.polish_sa_budget_s:.0f}s)")
        sa_stats = run_sa_polish_v2(
            evaluator=evaluator, benchmark=benchmark, plc=plc,
            hard_movable=hard_movable, time_budget_s=self.polish_sa_budget_s,
            T0=self.polish_sa_T0, Tf=self.polish_sa_Tf, seed=42,
            log_fn=self._log if self.verbose else None,
        )
        sa_final_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  polish-SA done: best={sa_stats['best_proxy']:.5f}, "
            f"final proxy={sa_final_proxy:.5f}"
        )

        # 6. Pull polished placement; preserve fixed macros.
        polished_placement_f64 = evaluator.placement.detach().clone().cpu()
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            polished_placement_f64[fixed_mask] = original_positions[fixed_mask]
        polished_placement = polished_placement_f64.to(torch.float32)

        polished_overlaps = compute_overlap_metrics(polished_placement, benchmark)
        polished_proxy = compute_proxy_cost(polished_placement, benchmark, plc)["proxy_cost"]
        self._log(
            f"  polished_crossover: proxy={polished_proxy:.5f} "
            f"overlaps={polished_overlaps['overlap_count']}"
        )

        # 7. Return best of three.
        candidates = []
        if e25_overlaps == 0:
            candidates.append(("E25", e25_placement, e25_proxy))
        if e41_overlaps == 0:
            candidates.append(("E41", e41_placement, e41_proxy))
        if polished_overlaps["overlap_count"] == 0:
            candidates.append(("polished", polished_placement, polished_proxy))

        if not candidates:
            raise RuntimeError("E61: ALL three options had overlaps")

        candidates.sort(key=lambda x: x[2])
        winner_name, winner_placement, winner_proxy = candidates[0]
        all_proxies = ", ".join(f"{n}={p:.5f}" for n, _, p in candidates)
        self._log(
            f"  E61 winner: {winner_name} proxy={winner_proxy:.5f} "
            f"({all_proxies}) total_wall={time.perf_counter() - t0:.1f}s"
        )
        return winner_placement
