"""E40 — CDLNSSAMultiSeedPlacer: fork SA-v2 across multiple seeds from a
shared post-LNS state, keep best.

Hypothesis. E25's pipeline (CD + grid-bin LNS + SA-v2 polish) is
deterministic through CD and LNS (modulo `lns_seed`); only SA-v2 is
genuinely stochastic via `sa_seed`. The post-LNS state is identical
across SA seeds. Forking 4 SA seeds {42, 1, 2, 3} from the *same* saved
post-LNS state and keeping the best per-bench should improve per-bench
proxy by 0.05–0.2 % on benches where SA-v2 found lift in E25 (ibm01,
ibm04, ibm08, ibm09, ibm10). Cheap: CD + LNS shared, only SA gets
multiplied.

Pipeline (per benchmark; 3600 s legal cap NOT respected — research probe):

  1. SDF init.
  2. Project overlaps.
  3. Build IncrementalProxyEvaluator.
  4. CD phase (≤ 2400 s, plateau threshold 0.001) — UNCHANGED from E25.
  5. LNS phase (≤ 600 s, grid-bin destroy/reinsert) — UNCHANGED from E25.
  6. Snapshot post-LNS placement.
  7. For each sa_seed in {42, 1, 2, 3}:
     a. Restore evaluator to snapshot via per-macro move() walks.
     b. Run SA-v2 polish (≤ 600 s, T0 = 5e-4, T_f = 1e-6) with seed.
     c. Record (sa_seed, final_proxy, placement_clone).
  8. Pick best fork. Walk evaluator state to best fork's placement.
  9. Validate (zero overlaps), preserve fixed macros, return.

State-restoration design choice. The evaluator caches V/H congestion
state in step with `move()` calls. Direct mutation of
`evaluator.placement` desyncs the cache. The E25 SA-v2 best-restore
pattern (per-macro `move()` for indices whose target differs from
current) is reused for both:
  (a) restoring forks to the saved post-LNS state, and
  (b) restoring the final evaluator state to the best-fork placement.

References:
- E25 — submissions/cd_lns_sa/placer.py (champion candidate 1.0954,
  primitives reused here directly).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch

# Eval harness loads via importlib.spec_from_file_location, so add repo root
# to sys.path so `macro_place.*` and `submissions.*` imports resolve.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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

# Reuse E25's inlined LNS + SA-v2 primitives directly (no behavior change).
from submissions.cd_lns_sa.placer import (  # noqa: E402
    run_lns_gridbin,
    run_sa_polish_v2,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _restore_evaluator_to(
    evaluator: IncrementalProxyEvaluator,
    target_placement: torch.Tensor,
    eps: float = 1e-9,
) -> int:
    """Walk per-macro move() for any index whose current pos differs from
    target. Returns the number of moves applied. Mirrors the SA-v2
    best-restore pattern: keeps the V/H congestion cache in sync (direct
    mutation of evaluator.placement would desync the cache)."""
    n_macros = int(evaluator.placement.shape[0])
    moves = 0
    for i in range(n_macros):
        tx = float(target_placement[i, 0])
        ty = float(target_placement[i, 1])
        cx = float(evaluator.placement[i, 0])
        cy = float(evaluator.placement[i, 1])
        if abs(tx - cx) > eps or abs(ty - cy) > eps:
            evaluator.move(i, (tx, ty))
            moves += 1
    return moves


# ── Placer ─────────────────────────────────────────────────────────────────


class CDLNSSAMultiSeedPlacer:
    """E40 placer: CD + LNS + multi-seed SA-v2 forks from shared post-LNS
    state, keep best.

    Time budget per benchmark (3600 s legal cap NOT enforced):
      * CD phase: ≤ 2400 s.
      * LNS phase: ≤ 600 s.
      * SA-v2 phase: |sa_seeds| × 600 s (4 × 600 s = 2400 s by default).

    All hyperparameters global. No per-benchmark tuning.
    """

    def __init__(
        self,
        # CD phase (E25 defaults).
        cd_hard_cap_s: float = 2400.0,
        cd_min_time_s: float = 300.0,
        cd_patience: int = 3,
        cd_plateau_threshold: float = 0.001,
        # LNS phase (E25 defaults).
        lns_budget_s: float = 600.0,
        lns_destroy_frac: float = 0.05,
        lns_destroy_cap: int = 30,
        lns_seed: int = 42,
        # SA-v2 phase (E25 defaults; multi-seed adds sa_seeds list).
        sa_budget_s: float = 600.0,
        sa_T0: float = 5e-4,
        sa_Tf: float = 1e-6,
        sa_seeds: Tuple[int, ...] = (42, 1, 2, 3),
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
        self.lns_seed = int(lns_seed)
        self.sa_budget_s = float(sa_budget_s)
        self.sa_T0 = float(sa_T0)
        self.sa_Tf = float(sa_Tf)
        self.sa_seeds: List[int] = [int(s) for s in sa_seeds]
        if len(self.sa_seeds) == 0:
            raise ValueError("sa_seeds must be a non-empty list")
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
            f"=== CDLNSSAMultiSeedPlacer ({benchmark.name}): "
            f"CD cap={self.cd_hard_cap_s:.0f}s, LNS={self.lns_budget_s:.0f}s, "
            f"SA={self.sa_budget_s:.0f}s × {len(self.sa_seeds)} seeds "
            f"({self.sa_seeds}) ==="
        )
        self._log(
            f"  num_macros={benchmark.num_macros}, "
            f"n_hard={benchmark.num_hard_macros}, "
            f"canvas={benchmark.canvas_width:.1f}x{benchmark.canvas_height:.1f}"
        )

        # 1. SDF init.
        t_init0 = time.perf_counter()
        placement = sdf_init(benchmark)
        self._log(f"  SDF init wall = {time.perf_counter() - t_init0:.1f} s")

        # 2. Project overlaps.
        placement, proj_iters = project_overlaps(placement, benchmark)
        init_overlaps = compute_overlap_metrics(placement, benchmark)
        self._log(
            f"  overlap projection: {proj_iters} iters, "
            f"residual count={init_overlaps['overlap_count']}"
        )

        # 3. Build evaluator.
        _, plc = self._load_plc_for(benchmark)
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

        # 6. Snapshot post-LNS state. This is the shared starting point for
        #    every SA fork. Floating-point exact clone — restore is per-macro
        #    move() so cache stays in sync.
        saved_placement = evaluator.placement.detach().clone()
        saved_proxy = lns_proxy
        self._log(
            f"  snapshot post-LNS state: proxy={saved_proxy:.5f}, "
            f"|placement|={int(saved_placement.shape[0])} macros"
        )

        # 7. SA-v2 fork loop. Each seed starts from saved_placement.
        fork_results: List[Tuple[int, float, torch.Tensor]] = []
        for fork_idx, sa_seed in enumerate(self.sa_seeds):
            # 7a. Restore evaluator to saved_placement.
            if fork_idx == 0:
                # Already at saved state; sanity check.
                cur_proxy = evaluator.current_cost()["proxy"]
                if abs(cur_proxy - saved_proxy) > 1e-9:
                    self._log(
                        f"  WARNING: fork 0 starts at proxy {cur_proxy:.5f}, "
                        f"saved was {saved_proxy:.5f}"
                    )
                restore_moves = 0
            else:
                restore_moves = _restore_evaluator_to(evaluator, saved_placement)
                cur_proxy = evaluator.current_cost()["proxy"]
                if abs(cur_proxy - saved_proxy) > 1e-6:
                    self._log(
                        f"  WARNING: fork {fork_idx} restored to proxy "
                        f"{cur_proxy:.5f}, saved was {saved_proxy:.5f} "
                        f"(Δ={cur_proxy - saved_proxy:+.2e})"
                    )

            self._log(
                f"  --- SA fork {fork_idx + 1}/{len(self.sa_seeds)} "
                f"(seed={sa_seed}, restore_moves={restore_moves}) ---"
            )

            # 7b. Run SA-v2 polish with this seed.
            sa_stats = run_sa_polish_v2(
                evaluator=evaluator,
                benchmark=benchmark,
                plc=plc,
                hard_movable=hard_movable,
                time_budget_s=self.sa_budget_s,
                T0=self.sa_T0,
                Tf=self.sa_Tf,
                seed=sa_seed,
                breakpoint_budget=self.sa_breakpoint_budget,
                log_fn=self._log if self.verbose else None,
            )

            # 7c. Record fork result.
            fork_proxy = evaluator.current_cost()["proxy"]
            fork_placement = evaluator.placement.detach().clone()
            fork_results.append((sa_seed, fork_proxy, fork_placement))
            self._log(
                f"  fork {fork_idx + 1} (seed={sa_seed}): "
                f"final_proxy={fork_proxy:.5f}, "
                f"sa_best={sa_stats['best_proxy']:.5f}, "
                f"lift_vs_lns={saved_proxy - fork_proxy:+.5f}, "
                f"wall={sa_stats['wall_total_s']:.1f}s"
            )

        # 8. Pick best fork. Walk evaluator state to best.
        best_fork_idx, best_seed, best_proxy, best_placement = min(
            ((i, s, p, pl) for i, (s, p, pl) in enumerate(fork_results)),
            key=lambda t: t[2],
        )
        self._log(
            f"  fork summary: "
            + ", ".join(
                f"seed={s}: {p:.5f}{' *' if i == best_fork_idx else ''}"
                for i, (s, p, _) in enumerate(fork_results)
            )
        )
        self._log(
            f"  best fork: seed={best_seed}, proxy={best_proxy:.5f} "
            f"(lift_vs_lns={saved_proxy - best_proxy:+.5f})"
        )

        # If the current evaluator state is already the best fork, no walk.
        cur_after_last_fork = evaluator.current_cost()["proxy"]
        if best_fork_idx != len(fork_results) - 1 or abs(cur_after_last_fork - best_proxy) > 1e-9:
            walk_moves = _restore_evaluator_to(evaluator, best_placement)
            self._log(
                f"  restored evaluator to best fork "
                f"(seed={best_seed}, walk_moves={walk_moves})"
            )

        final_cost = evaluator.current_cost()
        self._log(
            f"  final: proxy={final_cost['proxy']:.5f} "
            f"[wl={final_cost['wl']:.4f} d={final_cost['density']:.4f} "
            f"c={final_cost['congestion']:.4f}]"
        )
        self._log(
            f"  TOTAL lifts: CD={init_cost['proxy'] - cd_proxy:+.5f}, "
            f"LNS={cd_proxy - lns_proxy:+.5f}, "
            f"SA(best of {len(self.sa_seeds)})={lns_proxy - final_cost['proxy']:+.5f}"
        )

        # 9. Pull placement back; preserve fixed macros; validate.
        final_placement_f64 = evaluator.placement.detach().clone().cpu()
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            final_placement_f64[fixed_mask] = original_positions[fixed_mask]
        final_placement = final_placement_f64.to(torch.float32)

        overlaps = compute_overlap_metrics(final_placement, benchmark)
        if overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"CDLNSSAMultiSeedPlacer produced {overlaps['overlap_count']} "
                f"overlaps (area {overlaps['total_overlap_area']:.4f}) on "
                f"'{benchmark.name}'"
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(SDF + project + CD + LNS + {len(self.sa_seeds)}× SA + validate)"
        )
        return final_placement
