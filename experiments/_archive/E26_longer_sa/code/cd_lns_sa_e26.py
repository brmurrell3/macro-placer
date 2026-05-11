"""E26 — CDAdaptive + grid-bin LNS (300 s) + SA polish v2 (900 s).

E25's compositional pipeline (CD + LNS + SA-v2) hit avg `--fast` 0.9336
beating E12 prod on every benchmark. Detailed phase logs revealed:

  * **LNS converged early** on every fast bench: ibm01 LNS sample 4 at
    98 s of 600 s budget; ibm04, ibm09, ibm13 similarly. Most of the
    600 s LNS budget went unused.
  * **SA-v2 found best at t ≈ 599 s** on every bench where it improved
    (ibm01: 599.4 s, ibm04: 599.4 s, ibm09: 598.9 s). SA was *still
    actively decreasing proxy* when the 600 s budget expired.

Hypothesis: reallocating the unused LNS time to SA gives SA the
headroom to find further improvements. New split: CD ≤ 2400 s + LNS
≤ 300 s + SA ≤ 900 s = 3600 s legal cap. LNS budget is more than
enough for 4 samples (~100 s) on small benchmarks; SA budget is 1.5×
E25's, exposing whether the late-budget gain has real headroom.

Pipeline identical to E25 (`experiments/E25_lns_sa_compose/code/cd_lns_sa.py`)
except the budget split. All hyperparameters global. No per-benchmark
tuning. Hard limit: total wall ≤ 3600 s/benchmark.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import torch

# The eval harness loads placers via importlib.spec_from_file_location, which
# does NOT add the repo root to sys.path. Add it here, plus the E12 LNS dir
# and E24 SA-v2 dir so we can import their respective primitives.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_E12_DIR = str(_ROOT / "submissions" / "cd_lns_gridbin")
_E24_DIR = str(_ROOT / "experiments" / "E24_sa_polish_v2" / "code")
for _p in (_E12_DIR, _E24_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import (
    project_overlaps,
    run_cd_adaptive,
    sdf_init,
)
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics

from placer import run_lns_gridbin  # type: ignore  # noqa: E402
from cd_sa_polish_v2 import run_sa_polish_v2  # type: ignore  # noqa: E402


# ── Placer ──────────────────────────────────────────────────────────────────


class CDLNSSAE26Placer:
    """E26 placer: CD + LNS (300 s) + SA-v2 (900 s).

    Identical pipeline to E25 (CDLNSSAComposePlacer); only budget split
    differs. CD ≤ 2400 s, LNS ≤ 300 s, SA ≤ 900 s = 3600 s legal cap.
    """

    def __init__(
        self,
        cd_hard_cap_s: float = 2400.0,
        cd_min_time_s: float = 300.0,
        cd_patience: int = 3,
        cd_plateau_threshold: float = 0.001,
        # LNS: cut from 600 → 300; converged at sample 4 (~100 s) on
        # every E25 fast bench, so 300 s is ample headroom.
        lns_budget_s: float = 300.0,
        lns_destroy_frac: float = 0.05,
        lns_destroy_cap: int = 30,
        lns_destroy_strategy: str = "cost_aware",
        lns_seed: int = 42,
        # SA: bumped from 600 → 900; SA found best at t ≈ 599 s on every
        # E25 winning bench, so it was still improving at budget end.
        sa_budget_s: float = 900.0,
        sa_T0: float = 5e-4,
        sa_Tf: float = 1e-6,
        sa_seed: int = 42,
        sa_breakpoint_budget: int = 12,
        verbose: bool = True,
    ) -> None:
        self.cd_hard_cap_s = float(cd_hard_cap_s)
        self.cd_min_time_s = float(cd_min_time_s)
        self.cd_patience = int(cd_patience)
        self.cd_plateau_threshold = float(cd_plateau_threshold)
        self.lns_budget_s = float(lns_budget_s)
        self.lns_destroy_frac = float(lns_destroy_frac)
        self.lns_destroy_cap = int(lns_destroy_cap)
        self.lns_destroy_strategy = str(lns_destroy_strategy)
        self.lns_seed = int(lns_seed)
        self.sa_budget_s = float(sa_budget_s)
        self.sa_T0 = float(sa_T0)
        self.sa_Tf = float(sa_Tf)
        self.sa_seed = int(sa_seed)
        self.sa_breakpoint_budget = int(sa_breakpoint_budget)
        self.verbose = bool(verbose)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _load_plc_for(self, benchmark: Benchmark):
        bench_dir = find_benchmark_dir(benchmark.name)
        return load_benchmark_from_dir(str(bench_dir))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_total0 = time.perf_counter()
        self._log(
            f"=== CDLNSSAE26Placer ({benchmark.name}): "
            f"CD cap={self.cd_hard_cap_s:.0f}s, LNS={self.lns_budget_s:.0f}s, "
            f"SA={self.sa_budget_s:.0f}s (E26: 300/900 split) ==="
        )
        self._log(
            f"  num_macros={benchmark.num_macros}, "
            f"n_hard={benchmark.num_hard_macros}, "
            f"canvas={benchmark.canvas_width:.1f}x{benchmark.canvas_height:.1f}"
        )

        # 1. SDF init
        t_init0 = time.perf_counter()
        placement = sdf_init(benchmark)
        self._log(f"  SDF init wall = {time.perf_counter() - t_init0:.1f} s")

        # 2. Project overlaps
        placement, proj_iters = project_overlaps(placement, benchmark)
        init_overlaps = compute_overlap_metrics(placement, benchmark)
        self._log(
            f"  overlap projection: {proj_iters} iters, "
            f"residual count={init_overlaps['overlap_count']}"
        )

        # 3. Build evaluator
        _, plc = self._load_plc_for(benchmark)
        placement_f64 = placement.detach().clone().to(torch.float64)
        evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
        init_cost = evaluator.current_cost()
        self._log(
            f"  init proxy={init_cost['proxy']:.5f} "
            f"[wl={init_cost['wl']:.4f} d={init_cost['density']:.4f} "
            f"c={init_cost['congestion']:.4f}]"
        )

        # 4. CD phase
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        hard_movable = [
            i for i in range(benchmark.num_hard_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        self._log(f"  starting CD phase (cap={self.cd_hard_cap_s:.0f}s)")
        cd_stats = run_cd_adaptive(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            movable=movable,
            min_time_s=self.cd_min_time_s,
            hard_cap_s=self.cd_hard_cap_s,
            patience=self.cd_patience,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=self._log if self.verbose else None,
        )
        cd_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  CD done: exit={cd_stats['exit_reason']}, "
            f"sweeps={cd_stats['sweeps']}, accepts={cd_stats['total_moves']}, "
            f"wall={cd_stats['wall_total_s']:.1f}s, proxy={cd_proxy:.5f}"
        )

        # 5. LNS phase (300 s)
        self._log(f"  starting LNS phase (budget={self.lns_budget_s:.0f}s)")
        lns_stats = run_lns_gridbin(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.lns_budget_s,
            destroy_frac=self.lns_destroy_frac,
            destroy_cap=self.lns_destroy_cap,
            destroy_strategy=self.lns_destroy_strategy,
            seed=self.lns_seed,
            log_fn=self._log if self.verbose else None,
        )
        lns_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  LNS done: samples={lns_stats['samples']}, "
            f"Δ={lns_stats['total_improvement']:+.5f}, "
            f"wall={lns_stats['wall_total_s']:.1f}s, proxy={lns_proxy:.5f}"
        )

        # 6. SA-v2 phase (900 s)
        self._log(f"  starting SA-v2 phase (budget={self.sa_budget_s:.0f}s)")
        sa_stats = run_sa_polish_v2(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.sa_budget_s,
            T0=self.sa_T0,
            Tf=self.sa_Tf,
            seed=self.sa_seed,
            breakpoint_budget=self.sa_breakpoint_budget,
            log_fn=self._log if self.verbose else None,
        )
        final_cost = evaluator.current_cost()
        self._log(
            f"  SA-v2 done: lift_vs_init={sa_stats['improvement_vs_init']:+.5f}, "
            f"best={sa_stats['best_proxy']:.5f}, "
            f"final proxy={final_cost['proxy']:.5f}"
        )
        self._log(
            f"  TOTAL lifts: CD={init_cost['proxy'] - cd_proxy:+.5f}, "
            f"LNS={cd_proxy - lns_proxy:+.5f}, "
            f"SA={lns_proxy - final_cost['proxy']:+.5f}"
        )

        # 7. Pull placement back; preserve fixed macros
        final_placement_f64 = evaluator.placement.detach().clone().cpu()
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            final_placement_f64[fixed_mask] = original_positions[fixed_mask]
        final_placement = final_placement_f64.to(torch.float32)

        overlaps = compute_overlap_metrics(final_placement, benchmark)
        if overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"CDLNSSAE26Placer produced {overlaps['overlap_count']} "
                f"overlaps (area {overlaps['total_overlap_area']:.4f}) on "
                f"'{benchmark.name}'"
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(SDF + project + CD + LNS + SA + validate)"
        )
        return final_placement
