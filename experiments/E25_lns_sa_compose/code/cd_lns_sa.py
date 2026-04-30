"""E25 — CDAdaptive + grid-bin LNS + SA polish v2 (compositional test).

Hypothesis: E12's grid-bin LNS overlay (champion at avg `--all` 1.0990)
and E24's SA polish v2 (best-tracking, T₀ = 5e-4) provide *similar lift*
over CD plateau on `--fast` (E12-random 0.9372 vs E24 0.9366) — but they
may target DIFFERENT improvements. LNS uses (col, row) grid-cell-center
moves (outside CD's per-axis reachable set). SA-v2 uses per-axis
breakpoints (inside CD's reachable set, but Metropolis order, with
best-so-far tracking).

If the two mechanisms are compositional (find different improvements),
running them in sequence after CD should add their lifts:
- CD plateau ≈ 0.94 on `--fast`
- + LNS overlay → ~0.937 (E12-random)
- + SA-v2 polish → ~0.93 if compositional, ~0.937 if not

If E25 `--fast` ≈ E12 / E24 (within noise), then the two mechanisms hit
the same improvements; not compositional. If E25 < 0.93, compositional
and a likely champion candidate.

Pipeline (per benchmark):
  1. SDF init
  2. Project overlaps
  3. Build IncrementalProxyEvaluator
  4. CD phase (cd_hard_cap_s, plateau detection)
  5. LNS phase (grid-bin, cost-aware destroy — same recipe as E12)
  6. SA-v2 phase (Metropolis on per-axis breakpoints, best-so-far
     tracking — same recipe as E24)
  7. Validate (zero overlaps), return.

All hyperparameters global. No per-benchmark tuning.
Hard limit: total wall ≤ 3600 s/benchmark (contest legal limit).
Default split: CD ≤ 2400 s + LNS ≤ 600 s + SA ≤ 600 s.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import torch

# The eval harness loads placers via importlib.spec_from_file_location, which
# does NOT add the repo root to sys.path. We need it on sys.path so the
# `macro_place.*` imports below resolve.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Make the E12 LNS function and the E24 SA-v2 function importable. Both
# experiments live under `experiments/E*/code/`, so add those dirs to
# sys.path too. (Same pattern as the placer's _ROOT insertion.)
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

# Import the LNS function from the E12 champion placer module.
from placer import run_lns_gridbin  # type: ignore  # noqa: E402

# Import the SA-v2 function from E24 placer module.
from cd_sa_polish_v2 import run_sa_polish_v2  # type: ignore  # noqa: E402


# ── Placer ──────────────────────────────────────────────────────────────────


class CDLNSSAComposePlacer:
    """E25 placer: CDAdaptive + grid-bin LNS + SA polish v2.

    Time budget per benchmark:
      * CD phase: cd_hard_cap_s = 2400s.
      * LNS phase: lns_budget_s = 600s.
      * SA phase: sa_budget_s = 600s.
      * Total: ≤ 3600s (contest legal limit).
    """

    def __init__(
        self,
        cd_hard_cap_s: float = 2400.0,
        cd_min_time_s: float = 300.0,
        cd_patience: int = 3,
        cd_plateau_threshold: float = 0.001,
        # LNS phase (same defaults as E12 champion).
        lns_budget_s: float = 600.0,
        lns_destroy_frac: float = 0.05,
        lns_destroy_cap: int = 30,
        lns_destroy_strategy: str = "cost_aware",
        lns_seed: int = 42,
        # SA phase (same defaults as E24).
        sa_budget_s: float = 600.0,
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
            f"=== CDLNSSAComposePlacer ({benchmark.name}): "
            f"CD cap={self.cd_hard_cap_s:.0f}s, LNS={self.lns_budget_s:.0f}s, "
            f"SA={self.sa_budget_s:.0f}s ==="
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

        # 5. LNS phase (grid-bin, cost-aware destroy — same as E12)
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

        # 6. SA-v2 phase (best-so-far Metropolis on breakpoints — same as E24)
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
                f"CDLNSSAComposePlacer produced {overlaps['overlap_count']} "
                f"overlaps (area {overlaps['total_overlap_area']:.4f}) on "
                f"'{benchmark.name}'"
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(SDF + project + CD + LNS + SA + validate)"
        )
        return final_placement
