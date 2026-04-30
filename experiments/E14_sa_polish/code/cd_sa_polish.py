"""E14 — CDAdaptive + SA polish (simulated annealing on per-axis breakpoints).

Hypothesis (per ADR-003): CD plateaus when every macro is at its per-axis
fixed point. Different mechanism than LNS (structural destroy/reinsert):
SA accepts worsening moves probabilistically — explicit tunneling through
CD's fixed point. Same candidate set as CD (per-axis breakpoints), but
Metropolis acceptance lets the search jump out of the basin CD is stuck in.

Algorithm (per benchmark):
  1. Run CDAdaptive (plateau threshold=0.001, cap=3000s — leaves 600s for SA).
  2. SA polish phase (sa_budget_s seconds):
       a. Pick a random hard movable macro (uniform).
       b. Pick a random axis (x or y, uniform).
       c. Compute the per-axis legal range and breakpoint candidate set
          via cd_core.legal_axis_range + cd_core.axis_breakpoints (same set
          CD already searched — so we tunnel through CD's fixed point with
          the same primitives, just under Metropolis rather than greedy).
       d. Pick one breakpoint uniformly at random (excluding current pos).
       e. Move via evaluator.move(idx, (new_x, new_y)); query proxy.
          Metropolis: accept if Δproxy ≤ 0; else accept with
          probability exp(-Δproxy / T). On reject, evaluator.revert().
       f. Cool: T_k = T0 * (Tf / T0) ** (k / N).
  3. Validate zero overlaps, restore fixed macro positions, return.

T0 / Tf rationale (proxy-units): Δproxy magnitudes observed at CD plateau
are ~0.001. We want early acceptance of typical worsening moves and near-
zero late acceptance:
    early: exp(-0.001 / 0.01)   = exp(-0.1)   ≈ 0.905   (mostly accept)
    late : exp(-0.001 / 1e-5)   = exp(-100)   ≈ 0       (greedy)
So T0=0.01, Tf=1e-5 brackets a ~3-decade cooling.

All hyperparameters global. No per-benchmark tuning.
Hard limit: total wall ≤ 3600s/benchmark (contest legal limit).
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

# The eval harness loads placers via importlib.spec_from_file_location, which
# does NOT add the repo root to sys.path. We need it on sys.path so the
# `macro_place.*` imports below resolve. This is the only sys.path tweak
# the placer needs; all shared CD code lives in macro_place.cd_core.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import (
    _grid_lines,
    axis_breakpoints,
    legal_axis_range,
    project_overlaps,
    run_cd_adaptive,
    sdf_init,
)
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics


# ── SA polish primitive ─────────────────────────────────────────────────────


def run_sa_polish(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    hard_movable: List[int],
    time_budget_s: float,
    T0: float = 0.01,
    Tf: float = 1e-5,
    seed: int = 42,
    breakpoint_budget: int = 12,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """SA polish over per-axis breakpoint moves with Metropolis acceptance.

    Move set: identical to CD's (legal range + breakpoint candidates from
    ``axis_breakpoints``). Acceptance: Δ ≤ 0 always; else exp(-Δ/T).

    The cooling schedule is geometric over an *estimated* number of moves N.
    Because per-move wall time depends on the picked macro's net degree, we
    don't know N in advance — instead, we cool against (elapsed / budget):

        T(t) = T0 * (Tf / T0) ** (t / time_budget_s)

    so T traverses [T0, Tf] exactly once over the budget regardless of move
    rate. This is equivalent to the geometric-on-step schedule when steps
    are uniform in time, which is the regime we're in.
    """
    rng = np.random.default_rng(seed=seed)
    grid_lines_x, grid_lines_y = _grid_lines(plc)
    n_hard = benchmark.num_hard_macros

    if not hard_movable:
        if log_fn is not None:
            log_fn("  SA: no hard movable macros; skipping")
        return {
            "proposed": 0, "accepted_better": 0, "accepted_worse": 0,
            "rejected": 0, "skipped": 0, "total_improvement": 0.0,
            "wall_total_s": 0.0,
        }

    if log_fn is not None:
        log_fn(
            f"  SA budget={time_budget_s:.0f}s, T0={T0:.2e}, Tf={Tf:.2e}, "
            f"|H|={len(hard_movable)}, seed={seed}"
        )

    proposed = 0
    accepted_better = 0
    accepted_worse = 0
    rejected = 0
    skipped = 0
    init_proxy = evaluator.current_cost()["proxy"]
    cur_proxy = init_proxy

    log_ratio = math.log(Tf / T0)
    t_start = time.perf_counter()
    last_log_t = t_start

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:
            break

        # Geometric temperature on elapsed/budget.
        frac = elapsed / time_budget_s
        T = T0 * math.exp(log_ratio * frac)

        # Pick macro & axis uniformly at random.
        idx = int(hard_movable[rng.integers(0, len(hard_movable))])
        axis = int(rng.integers(0, 2))

        # Legal range on chosen axis.
        lo, hi = legal_axis_range(
            idx, evaluator.placement, evaluator.macro_sizes,
            benchmark.macro_fixed, n_hard, axis=axis,
            canvas_w=benchmark.canvas_width,
            canvas_h=benchmark.canvas_height,
        )
        if hi - lo < 1e-5:
            skipped += 1
            continue

        cur_axis_val = float(evaluator.placement[idx, axis])
        grid_lines = grid_lines_x if axis == 0 else grid_lines_y
        cands = axis_breakpoints(
            idx, axis, evaluator, grid_lines, lo, hi,
            max_breakpoints=breakpoint_budget, cur_axis=cur_axis_val,
        )
        # Exclude current position (within tol).
        if len(cands) > 0:
            mask = np.abs(cands - cur_axis_val) > 1e-6
            cands = cands[mask]
        if len(cands) == 0:
            skipped += 1
            continue

        new_axis_val = float(cands[rng.integers(0, len(cands))])
        cur_xy = (float(evaluator.placement[idx, 0]),
                  float(evaluator.placement[idx, 1]))
        new_xy = list(cur_xy)
        new_xy[axis] = new_axis_val

        # Probe via incremental evaluator. Legality on the chosen axis is
        # guaranteed by legal_axis_range (the perpendicular axis is
        # unchanged; the breakpoint set is clamped to [lo, hi]).
        proposed += 1
        new_proxy = evaluator.move(idx, tuple(new_xy))["proxy"]
        delta = new_proxy - cur_proxy

        if delta <= 0.0:
            # Strict improvement (or tie) — keep.
            cur_proxy = new_proxy
            accepted_better += 1
        else:
            accept_p = math.exp(-delta / T) if T > 0 else 0.0
            if rng.random() < accept_p:
                cur_proxy = new_proxy
                accepted_worse += 1
            else:
                evaluator.revert()
                rejected += 1

        # Periodic log (every ~30s).
        now = time.perf_counter()
        if log_fn is not None and (now - last_log_t) >= 30.0:
            last_log_t = now
            log_fn(
                f"  SA t={now - t_start:6.1f}s T={T:.2e} "
                f"proposed={proposed} better={accepted_better} "
                f"worse={accepted_worse} rejected={rejected} "
                f"skipped={skipped} proxy={cur_proxy:.5f}"
            )

    wall = time.perf_counter() - t_start
    final_proxy = evaluator.current_cost()["proxy"]
    return {
        "proposed": proposed,
        "accepted_better": accepted_better,
        "accepted_worse": accepted_worse,
        "rejected": rejected,
        "skipped": skipped,
        "total_improvement": final_proxy - init_proxy,
        "wall_total_s": wall,
    }


# ── Placer ──────────────────────────────────────────────────────────────────


class CDSAPolishPlacer:
    """E14 placer: CDAdaptive + simulated-annealing polish on breakpoint moves.

    Time budget per benchmark:
      * CD phase: cd_hard_cap_s = 3000s (legal limit minus SA reservation).
      * SA phase: sa_budget_s = 600s.
      * Total: ≤ 3600s (contest legal limit).
    """

    def __init__(
        self,
        cd_hard_cap_s: float = 3000.0,
        cd_min_time_s: float = 300.0,
        cd_patience: int = 3,
        cd_plateau_threshold: float = 0.001,
        sa_budget_s: float = 600.0,
        sa_T0: float = 0.01,
        sa_Tf: float = 1e-5,
        sa_seed: int = 42,
        sa_breakpoint_budget: int = 12,
        verbose: bool = True,
    ) -> None:
        self.cd_hard_cap_s = float(cd_hard_cap_s)
        self.cd_min_time_s = float(cd_min_time_s)
        self.cd_patience = int(cd_patience)
        self.cd_plateau_threshold = float(cd_plateau_threshold)
        self.sa_budget_s = float(sa_budget_s)
        self.sa_T0 = float(sa_T0)
        self.sa_Tf = float(sa_Tf)
        self.sa_seed = int(sa_seed)
        self.sa_breakpoint_budget = int(sa_breakpoint_budget)
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _load_plc_for(self, benchmark: Benchmark):
        bench_dir = find_benchmark_dir(benchmark.name)
        return load_benchmark_from_dir(str(bench_dir))

    # ------------------------------------------------------------------

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_total0 = time.perf_counter()
        self._log(
            f"=== CDSAPolishPlacer ({benchmark.name}): "
            f"CD cap={self.cd_hard_cap_s:.0f}s, SA budget={self.sa_budget_s:.0f}s ==="
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

        # 5. SA polish phase
        self._log(f"  starting SA polish (budget={self.sa_budget_s:.0f}s)")
        sa_stats = run_sa_polish(
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
            f"  SA done: proposed={sa_stats['proposed']}, "
            f"better={sa_stats['accepted_better']}, "
            f"worse={sa_stats['accepted_worse']}, "
            f"rejected={sa_stats['rejected']}, "
            f"skipped={sa_stats['skipped']}, "
            f"Δ={sa_stats['total_improvement']:+.5f}, "
            f"wall={sa_stats['wall_total_s']:.1f}s, "
            f"final proxy={final_cost['proxy']:.5f}"
        )

        # 6. Pull placement back; preserve fixed macros
        final_placement_f64 = evaluator.placement.detach().clone().cpu()
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            final_placement_f64[fixed_mask] = original_positions[fixed_mask]
        final_placement = final_placement_f64.to(torch.float32)

        overlaps = compute_overlap_metrics(final_placement, benchmark)
        if overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"CDSAPolishPlacer produced {overlaps['overlap_count']} overlaps "
                f"(area {overlaps['total_overlap_area']:.4f}) on '{benchmark.name}'"
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(SDF + project + CD + SA + validate)"
        )
        return final_placement
