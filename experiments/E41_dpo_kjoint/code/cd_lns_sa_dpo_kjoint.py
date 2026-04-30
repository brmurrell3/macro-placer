"""E41 — CDLNSSA + DPO best_of_v2 init + K-macro joint LNS.

Composes the two independent --fast wins from 2026-04-29:
- E18 (DPO init, avg --fast 0.92542): DPO best_of_v2 init lands a basin
  different from SDF's; -0.91 % vs E25 with 3/4 per-bench wins, plus
  -1.67 % vs E12 on NG45 with 4/4 wins.
- E39 (K-macro joint LNS, avg --fast 0.93070): K=3 joint reinsertion
  extracts 0.001-0.005 lift per bench post-CD-LNS-SA — a move type SA
  alone misses.

Pipeline (per benchmark):
  1. DPO best_of_v2 init (replaces SDF; same as E18).
  2. project_overlaps to clear DPO residuals.
  3. Build IncrementalProxyEvaluator.
  4. CD adaptive phase (≤ 2400 s).
  5. Grid-bin LNS phase (≤ 600 s).
  6. SA-v2 phase (≤ 600 s, T₀=5e-4, best-so-far tracking).
  7. K-macro joint LNS (≤ 600 s, K=3, top_N=5).
  8. Validate (zero overlaps), preserve fixed macros, return.

Imports the DPO init helper from E18's module and the K-joint runner +
E25-shaped LNS/SA primitives from E39's module — both already register
the repo root on sys.path on import.

Reference:
- E18 — DPO init parent. `experiments/E18_dpo_init/code/cd_lns_sa_dpo_init.py`.
- E39 — K-joint parent. `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py`.
- E25 — shared CD+LNS+SA-v2 backbone. `submissions/cd_lns_sa/placer.py`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch

# The eval harness loads placers via importlib.spec_from_file_location, which
# does NOT add the repo root to sys.path. Add the repo root and the parent
# experiments' code dirs so the cross-experiment imports below resolve.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics

# Import the parent helpers directly. Both E18 and E39 modules register
# the repo root on import, so this is safe.
from experiments.E18_dpo_init.code.cd_lns_sa_dpo_init import _best_of_v2_init
from experiments.E39_kmacro_joint_lns.code.cd_lns_sa_kjoint import (
    run_kjoint_lns,
    run_lns_gridbin,
    run_sa_polish_v2,
)


# ── Placer ─────────────────────────────────────────────────────────────────


class CDLNSSADPOKJointPlacer:
    """E41 placer: DPO best_of_v2 init -> CD -> LNS -> SA-v2 -> K-joint.

    Composes E18's DPO basin lift with E39's joint-move escape. All
    hyperparameters match the parents (DPO seed=42, CD cap 2400 s, LNS
    600 s, SA-v2 600 s with T₀=5e-4, K-joint 600 s with K=3 and top_N=5).
    No per-benchmark tuning.

    Theoretical per-benchmark budget is 2400 + 600 + 600 + 600 = 4200 s
    (above the 1-hr legal cap on paper); in practice CD plateaus well
    below 2400 s on most benches so the realistic wall is ~1 hr/bench.
    """

    def __init__(
        self,
        cd_hard_cap_s: float = 2400.0,
        cd_min_time_s: float = 300.0,
        cd_patience: int = 3,
        cd_plateau_threshold: float = 0.001,
        lns_budget_s: float = 600.0,
        lns_destroy_frac: float = 0.05,
        lns_destroy_cap: int = 30,
        lns_seed: int = 42,
        sa_budget_s: float = 600.0,
        sa_T0: float = 5e-4,
        sa_Tf: float = 1e-6,
        sa_seed: int = 42,
        sa_breakpoint_budget: int = 12,
        kjoint_budget_s: float = 600.0,
        kjoint_K: int = 3,
        kjoint_top_N: int = 5,
        kjoint_seed: int = 42,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.cd_hard_cap_s = float(cd_hard_cap_s)
        self.cd_min_time_s = float(cd_min_time_s)
        self.cd_patience = int(cd_patience)
        self.cd_plateau_threshold = float(cd_plateau_threshold)
        self.lns_budget_s = float(lns_budget_s)
        self.lns_destroy_frac = float(lns_destroy_frac)
        self.lns_destroy_cap = int(lns_destroy_cap)
        self.lns_seed = int(lns_seed)
        self.sa_budget_s = float(sa_budget_s)
        self.sa_T0 = float(sa_T0)
        self.sa_Tf = float(sa_Tf)
        self.sa_seed = int(sa_seed)
        self.sa_breakpoint_budget = int(sa_breakpoint_budget)
        self.kjoint_budget_s = float(kjoint_budget_s)
        self.kjoint_K = int(kjoint_K)
        self.kjoint_top_N = int(kjoint_top_N)
        self.kjoint_seed = int(kjoint_seed)
        self.seed = int(seed)
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
            f"=== CDLNSSADPOKJointPlacer ({benchmark.name}): "
            f"DPO init -> CD cap={self.cd_hard_cap_s:.0f}s -> "
            f"LNS={self.lns_budget_s:.0f}s -> SA={self.sa_budget_s:.0f}s -> "
            f"KJoint={self.kjoint_budget_s:.0f}s ==="
        )
        self._log(
            f"  num_macros={benchmark.num_macros}, "
            f"n_hard={benchmark.num_hard_macros}, "
            f"canvas={benchmark.canvas_width:.1f}x{benchmark.canvas_height:.1f}"
        )

        # 1. DPO best_of_v2 init (replaces SDF; matches E18).
        _, plc = self._load_plc_for(benchmark)
        t_init0 = time.perf_counter()
        placement = _best_of_v2_init(
            benchmark, plc, seed=self.seed,
            log_fn=self._log if self.verbose else None,
        )
        self._log(f"  DPO init wall = {time.perf_counter() - t_init0:.1f} s")

        # 2. Project overlaps.
        placement, proj_iters = project_overlaps(placement, benchmark)
        init_overlaps = compute_overlap_metrics(placement, benchmark)
        self._log(
            f"  overlap projection: {proj_iters} iters, "
            f"residual count={init_overlaps['overlap_count']}"
        )

        # 3. Build evaluator.
        placement_f64 = placement.detach().clone().to(torch.float64)
        evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
        init_cost = evaluator.current_cost()
        self._log(
            f"  init proxy={init_cost['proxy']:.5f} "
            f"[wl={init_cost['wl']:.4f} d={init_cost['density']:.4f} "
            f"c={init_cost['congestion']:.4f}]"
        )

        # 4. CD phase.
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

        # 5. LNS phase.
        self._log(f"  starting LNS phase (budget={self.lns_budget_s:.0f}s)")
        lns_stats = run_lns_gridbin(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.lns_budget_s,
            destroy_frac=self.lns_destroy_frac,
            destroy_cap=self.lns_destroy_cap,
            seed=self.lns_seed,
            log_fn=self._log if self.verbose else None,
        )
        lns_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  LNS done: samples={lns_stats['samples']}, "
            f"Δ={lns_stats['total_improvement']:+.5f}, "
            f"wall={lns_stats['wall_total_s']:.1f}s, proxy={lns_proxy:.5f}"
        )

        # 6. SA-v2 phase.
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
        sa_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  SA-v2 done: lift_vs_init={sa_stats['improvement_vs_init']:+.5f}, "
            f"best={sa_stats['best_proxy']:.5f}, proxy={sa_proxy:.5f}"
        )

        # 7. K-macro joint LNS phase.
        self._log(
            f"  starting K-joint phase (budget={self.kjoint_budget_s:.0f}s, "
            f"K={self.kjoint_K}, top_N={self.kjoint_top_N})"
        )
        kj_stats = run_kjoint_lns(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.kjoint_budget_s,
            K=self.kjoint_K,
            top_N=self.kjoint_top_N,
            seed=self.kjoint_seed,
            log_fn=self._log if self.verbose else None,
        )
        final_cost = evaluator.current_cost()
        self._log(
            f"  K-joint done: passes={kj_stats['passes']}, "
            f"tuples_tried={kj_stats['ktuples_tried']}, "
            f"committed={kj_stats['ktuples_committed']}, "
            f"Δ={kj_stats['total_improvement']:+.5f}, "
            f"wall={kj_stats['wall_total_s']:.1f}s, "
            f"final proxy={final_cost['proxy']:.5f}"
        )
        self._log(
            f"  TOTAL lifts: CD={init_cost['proxy'] - cd_proxy:+.5f}, "
            f"LNS={cd_proxy - lns_proxy:+.5f}, "
            f"SA={lns_proxy - sa_proxy:+.5f}, "
            f"KJoint={sa_proxy - final_cost['proxy']:+.5f}"
        )

        # 8. Pull placement back; preserve fixed macros.
        final_placement_f64 = evaluator.placement.detach().clone().cpu()
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            final_placement_f64[fixed_mask] = original_positions[fixed_mask]
        final_placement = final_placement_f64.to(torch.float32)

        overlaps = compute_overlap_metrics(final_placement, benchmark)
        if overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"CDLNSSADPOKJointPlacer produced {overlaps['overlap_count']} "
                f"overlaps (area {overlaps['total_overlap_area']:.4f}) on "
                f"'{benchmark.name}'"
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(DPO + project + CD + LNS + SA + KJoint + validate)"
        )
        return final_placement
