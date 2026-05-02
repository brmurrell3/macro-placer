"""E60 placer: SDF + CD + LNS + REPLICA-EXCHANGE SA-v2 + validate.

Drop-in replacement for E25 (CDLNSSAPlacer) with the single-chain SA-v2
phase replaced by a K=4 replica-exchange (parallel tempering) SA-v2.

Reference:
- E25 — `submissions/cd_lns_sa/placer.py` (single-chain SA-v2 baseline).
- E60 replica_sa.py — replica-exchange implementation.

Hypothesis (full manifest in `experiments/E60_replica_sa/manifest.md`):
the SA-v2 plateau on E25/E41/E48 is from a single chain stuck at the
basin floor that cold moves alone can't escape. Replica exchange adds
hot chains that explore globally and a Metropolis swap mechanism that
lets the cold chain inherit deeper basins discovered by hot chains.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import (
    project_overlaps, run_cd_adaptive, sdf_init,
)
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics

# Reuse E25's run_lns_gridbin (LNS phase).
import importlib.util
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
run_lns_gridbin = _E25_MOD.run_lns_gridbin

# E60-local: replica SA implementation.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from replica_sa import run_replica_exchange_sa_v2  # noqa: E402


class CDLNSReplicaSAPlacer:
    """E60 placer: same as E25 but SA-v2 phase is replaced by K=4 replica
    exchange parallel tempering at T0 = [5e-4, 5e-3, 5e-2, 5e-1].
    """

    def __init__(
        self,
        cd_min_time_s: float = 300.0,
        cd_hard_cap_s: float = 2400.0,
        cd_patience: int = 3,
        cd_plateau_threshold: float = 0.001,
        lns_budget_s: float = 600.0,
        lns_destroy_frac: float = 0.05,
        lns_destroy_cap: int = 30,
        lns_destroy_strategy: str = "cost_aware",
        lns_seed: int = 42,
        sa_budget_s: float = 600.0,
        sa_T0_list: list = (5e-4, 1e-3, 2e-3, 5e-3),
        sa_Tf: float = 1e-6,
        sa_seed: int = 42,
        sa_breakpoint_budget: int = 12,
        sa_moves_per_round: int = 200,
        sa_swap_every_n_rounds: int = 4,
        verbose: bool = True,
    ):
        self.cd_min_time_s = cd_min_time_s
        self.cd_hard_cap_s = cd_hard_cap_s
        self.cd_patience = int(cd_patience)
        self.cd_plateau_threshold = float(cd_plateau_threshold)
        self.lns_budget_s = float(lns_budget_s)
        self.lns_destroy_frac = float(lns_destroy_frac)
        self.lns_destroy_cap = int(lns_destroy_cap)
        self.lns_destroy_strategy = str(lns_destroy_strategy)
        self.lns_seed = int(lns_seed)
        self.sa_budget_s = float(sa_budget_s)
        self.sa_T0_list = list(sa_T0_list)
        self.sa_Tf = float(sa_Tf)
        self.sa_seed = int(sa_seed)
        self.sa_breakpoint_budget = int(sa_breakpoint_budget)
        self.sa_moves_per_round = int(sa_moves_per_round)
        self.sa_swap_every_n_rounds = int(sa_swap_every_n_rounds)
        self.verbose = bool(verbose)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_total0 = time.perf_counter()
        self._log(
            f"=== CDLNSReplicaSAPlacer ({benchmark.name}): "
            f"SDF -> CD cap={self.cd_hard_cap_s:.0f}s -> "
            f"LNS={self.lns_budget_s:.0f}s -> "
            f"replica-SA={self.sa_budget_s:.0f}s K={len(self.sa_T0_list)} ==="
        )
        self._log(
            f"  num_macros={benchmark.num_macros}, "
            f"n_hard={benchmark.num_hard_macros}, "
            f"canvas={benchmark.canvas_width:.1f}x{benchmark.canvas_height:.1f}"
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # 1. SDF init
        t0 = time.perf_counter()
        placement = sdf_init(benchmark)
        self._log(f"  SDF init wall = {time.perf_counter() - t0:.1f} s")

        # 2. project_overlaps
        placement, proj_iters = project_overlaps(placement, benchmark)
        init_overlaps = compute_overlap_metrics(placement, benchmark)
        self._log(
            f"  overlap projection: {proj_iters} iters, "
            f"residual count={init_overlaps['overlap_count']}"
        )

        # 3. Build evaluator
        placement_f64 = placement.detach().clone().to(torch.float64)
        evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
        init_cost = evaluator.current_cost()
        self._log(
            f"  init proxy={init_cost['proxy']:.5f} "
            f"[wl={init_cost['wl']:.4f} d={init_cost['density']:.4f} "
            f"c={init_cost['congestion']:.4f}]"
        )

        # 4. CD plateau
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
            f"  CD done: sweeps={cd_stats['sweeps']}, "
            f"exit={cd_stats['exit_reason']}, proxy={cd_proxy:.5f}"
        )

        # 5. LNS phase
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
            f"wall={lns_stats['wall_total_s']:.1f}s, "
            f"proxy={lns_proxy:.5f}"
        )

        # 6. REPLICA-EXCHANGE SA-v2 phase (the new mechanism)
        self._log(
            f"  starting replica-exchange SA phase "
            f"(budget={self.sa_budget_s:.0f}s, K={len(self.sa_T0_list)}, "
            f"T0={self.sa_T0_list})"
        )
        post_lns_placement = evaluator.placement.detach().clone()
        best_placement, sa_stats = run_replica_exchange_sa_v2(
            base_placement=post_lns_placement,
            benchmark=benchmark,
            plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.sa_budget_s,
            T_list=self.sa_T0_list,
            Tf=self.sa_Tf,
            seed=self.sa_seed,
            breakpoint_budget=self.sa_breakpoint_budget,
            moves_per_round=self.sa_moves_per_round,
            swap_every_n_rounds=self.sa_swap_every_n_rounds,
            log_fn=self._log if self.verbose else None,
        )
        # Replace evaluator's state with the best placement found across chains.
        # Since replica-SA uses K independent evaluators internally, we need to
        # reset our master evaluator to the best placement.
        evaluator = IncrementalProxyEvaluator(
            benchmark, plc,
            best_placement.detach().clone().to(torch.float64),
        )
        sa_final_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  replica-SA done: best across K chains = {sa_stats['best_proxy']:.5f} "
            f"(chain {sa_stats['best_chain']}, T0={self.sa_T0_list[sa_stats['best_chain']]:.2e}); "
            f"swaps={sa_stats['swap_accepts']}/{sa_stats['swap_attempts']}"
        )
        self._log(
            f"  TOTAL lifts: CD={init_cost['proxy'] - cd_proxy:+.5f}, "
            f"LNS={cd_proxy - lns_proxy:+.5f}, "
            f"replicaSA={lns_proxy - sa_final_proxy:+.5f}"
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
                f"CDLNSReplicaSAPlacer produced {overlaps['overlap_count']} overlaps "
                f"(area={overlaps['total_overlap_area']:.6f}) on {benchmark.name}"
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(SDF + project + CD + LNS + replica-SA + validate)"
        )
        return final_placement
