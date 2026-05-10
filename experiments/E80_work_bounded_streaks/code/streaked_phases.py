"""E80 streaked phase functions — work-bounded early termination.

Two phase functions with plateau/saturation streak termination:

  * run_lns_gridbin_streaked   — exit after 5 consecutive non-improving samples
                                 (vs current 1)
  * run_sa_polish_v2_streaked  — exit after 1000 consecutive moves with no
                                 global-best improvement (vs wall-only)

K-joint is NOT streaked here — TODO §Derisk Mitigation #4 says K-joint is
redundant when Hessian saddle follows, so E80's E41 lane skips K-joint
entirely (kjoint_budget_s=0).

Wall caps remain the same; streaks become the PRIMARY termination, wall the
SECONDARY.  These are drop-in replacements for the originals via monkey-patch.

Reference: TODO.md §Derisk Mitigations #3 + #4.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.cd_core import (
    _grid_lines,
    axis_breakpoints,
    legal_axis_range,
)
from macro_place.incremental_evaluator import IncrementalProxyEvaluator

# Reuse primitives unchanged.
from experiments.E39_kmacro_joint_lns.code.cd_lns_sa_kjoint import (
    _cost_aware_destroy,
    _gridbin_reinsert,
)


# ── Streaked LNS gridbin ────────────────────────────────────────────────────


def run_lns_gridbin_streaked(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    hard_movable: List[int],
    time_budget_s: float,
    destroy_frac: float = 0.05,
    destroy_cap: int = 30,
    seed: int = 42,
    log_fn: Optional[Callable[[str], None]] = None,
    plateau_streak: int = 5,
) -> dict:
    """Grid-bin LNS with N-streak plateau termination."""
    K = max(1, min(destroy_cap, int(destroy_frac * len(hard_movable))))
    macro_sizes_np = evaluator.macro_sizes.cpu().numpy()
    n_hard = benchmark.num_hard_macros

    if log_fn is not None:
        log_fn(
            f"  LNS budget={time_budget_s:.0f}s, destroy K={K} "
            f"(={destroy_frac*100:.1f}% of {len(hard_movable)} hard movables, "
            f"capped at {destroy_cap}), strategy=cost_aware, "
            f"plateau_streak={plateau_streak}"
        )

    sample = 0
    total_improvement = 0.0
    no_improve_streak = 0
    early_exit = False
    t_start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:
            break
        sample += 1
        sample_t0 = time.perf_counter()

        destroy = _cost_aware_destroy(evaluator, hard_movable, K)

        sample_delta = 0.0
        moves_this_sample = 0
        for idx in destroy:
            if time.perf_counter() - t_start >= time_budget_s:
                break
            delta = _gridbin_reinsert(
                evaluator, idx, plc, n_hard, macro_sizes_np,
                deadline_s=time_budget_s, t_start=t_start,
            )
            sample_delta += delta
            if delta < 0:
                moves_this_sample += 1

        total_improvement += sample_delta
        sample_wall = time.perf_counter() - sample_t0
        elapsed = time.perf_counter() - t_start
        cur_proxy = evaluator.current_cost()["proxy"]
        if log_fn is not None:
            log_fn(
                f"  LNS sample {sample}: K={K}, Δ={sample_delta:+.5f} "
                f"(moves={moves_this_sample}/{K}), proxy={cur_proxy:.5f}, "
                f"sample_wall={sample_wall:.1f}s, total elapsed={elapsed:.1f}s, "
                f"no_imp_streak={no_improve_streak}"
            )

        if abs(sample_delta) < 1e-7:
            no_improve_streak += 1
            if no_improve_streak >= plateau_streak:
                early_exit = True
                if log_fn is not None:
                    log_fn(
                        f"  LNS converged at sample {sample} "
                        f"({plateau_streak}-streak no improvement)"
                    )
                break
        else:
            no_improve_streak = 0

    return {
        "samples": sample,
        "total_improvement": total_improvement,
        "wall_total_s": time.perf_counter() - t_start,
        "early_exit": early_exit,
    }


# ── Streaked SA-v2 polish ───────────────────────────────────────────────────


def run_sa_polish_v2_streaked(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    hard_movable: List[int],
    time_budget_s: float,
    T0: float = 5e-4,
    Tf: float = 1e-6,
    seed: int = 42,
    breakpoint_budget: int = 12,
    log_fn: Optional[Callable[[str], None]] = None,
    no_improve_limit: int = 1000,
) -> dict:
    """SA-v2 with no-improvement streak termination (1000 moves default)."""
    rng = np.random.default_rng(seed=seed)
    grid_lines_x, grid_lines_y = _grid_lines(plc)
    n_hard = benchmark.num_hard_macros

    if not hard_movable:
        if log_fn is not None:
            log_fn("  SA: no hard movable macros; skipping")
        return {
            "proposed": 0, "accepted_better": 0, "accepted_worse": 0,
            "rejected": 0, "skipped": 0,
            "init_proxy": float("nan"), "best_proxy": float("nan"),
            "final_proxy": float("nan"), "improvement_vs_init": 0.0,
            "best_found_at_t": 0.0, "wall_total_s": 0.0,
            "best_restored": False, "early_exit": False,
        }

    if log_fn is not None:
        log_fn(
            f"  SA v2 budget={time_budget_s:.0f}s, T0={T0:.2e}, Tf={Tf:.2e}, "
            f"|H|={len(hard_movable)}, seed={seed} "
            f"(best-so-far ON, no_improve_limit={no_improve_limit})"
        )

    proposed = 0
    accepted_better = 0
    accepted_worse = 0
    rejected = 0
    skipped = 0
    init_proxy = evaluator.current_cost()["proxy"]
    cur_proxy = init_proxy
    best_proxy = init_proxy
    best_placement = evaluator.placement.detach().clone()
    best_found_at_t = 0.0
    moves_since_best = 0
    early_exit = False

    log_ratio = math.log(Tf / T0)
    t_start = time.perf_counter()
    last_log_t = t_start

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:
            break

        frac = elapsed / time_budget_s
        T = T0 * math.exp(log_ratio * frac)

        idx = int(hard_movable[rng.integers(0, len(hard_movable))])
        axis = int(rng.integers(0, 2))

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

        proposed += 1
        new_proxy = evaluator.move(idx, tuple(new_xy))["proxy"]
        delta = new_proxy - cur_proxy
        accepted = False

        if delta <= 0.0:
            cur_proxy = new_proxy
            accepted_better += 1
            accepted = True
        else:
            accept_p = math.exp(-delta / T) if T > 0 else 0.0
            if rng.random() < accept_p:
                cur_proxy = new_proxy
                accepted_worse += 1
                accepted = True
            else:
                evaluator.revert()
                rejected += 1

        if accepted and cur_proxy < best_proxy - 1e-12:
            best_proxy = cur_proxy
            best_placement = evaluator.placement.detach().clone()
            best_found_at_t = time.perf_counter() - t_start
            moves_since_best = 0
        else:
            moves_since_best += 1
            if moves_since_best >= no_improve_limit:
                early_exit = True
                if log_fn is not None:
                    log_fn(
                        f"  SA v2 early-exit: {no_improve_limit} moves with "
                        f"no global-best improvement (proposed={proposed}, "
                        f"elapsed={time.perf_counter() - t_start:.1f}s)"
                    )
                break

        now = time.perf_counter()
        if log_fn is not None and (now - last_log_t) >= 30.0:
            last_log_t = now
            log_fn(
                f"  SA v2 t={now - t_start:6.1f}s T={T:.2e} "
                f"proposed={proposed} better={accepted_better} "
                f"worse={accepted_worse} rejected={rejected} "
                f"skipped={skipped} cur={cur_proxy:.5f} best={best_proxy:.5f} "
                f"no_imp_streak={moves_since_best}"
            )

    wall = time.perf_counter() - t_start
    final_proxy_chain = evaluator.current_cost()["proxy"]

    best_restored = False
    if best_proxy < final_proxy_chain - 1e-12:
        n_macros = int(evaluator.placement.shape[0])
        for i in range(n_macros):
            tx = float(best_placement[i, 0])
            ty = float(best_placement[i, 1])
            cx = float(evaluator.placement[i, 0])
            cy = float(evaluator.placement[i, 1])
            if abs(tx - cx) > 1e-9 or abs(ty - cy) > 1e-9:
                evaluator.move(i, (tx, ty))
        best_restored = True

    final_proxy = evaluator.current_cost()["proxy"]
    if log_fn is not None:
        log_fn(
            f"  SA v2 done: proposed={proposed}, better={accepted_better}, "
            f"worse={accepted_worse}, rejected={rejected}, skipped={skipped}, "
            f"init={init_proxy:.5f}, best={best_proxy:.5f} "
            f"(found at t={best_found_at_t:.1f}s), "
            f"chain-final={final_proxy_chain:.5f}, "
            f"restored={best_restored}, evaluator-final={final_proxy:.5f}"
            + (" [EARLY EXIT]" if early_exit else "")
        )

    return {
        "proposed": proposed,
        "accepted_better": accepted_better,
        "accepted_worse": accepted_worse,
        "rejected": rejected,
        "skipped": skipped,
        "init_proxy": init_proxy,
        "best_proxy": best_proxy,
        "final_proxy": final_proxy,
        "improvement_vs_init": init_proxy - best_proxy,
        "best_found_at_t": best_found_at_t,
        "wall_total_s": wall,
        "best_restored": best_restored,
        "early_exit": early_exit,
    }
