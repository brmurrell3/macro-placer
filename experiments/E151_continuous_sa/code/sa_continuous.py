"""E151 — Simulated annealing on continuous per-macro Gaussian moves.

Applied after CD polish. Each iteration:
  - Sample a random movable hard macro i.
  - Propose dx, dy ~ N(0, T · step_size). New position = old + (dx, dy).
  - Clamp to canvas (with macro half-size margin).
  - Peek proxy via `IncrementalProxyEvaluator.delta_cost()` (non-mutating).
  - If Δ < 0 → accept; else Metropolis with p = exp(-Δ/T).
  - On accept: `move()` to commit; then check legality (bbox sweep). If
    illegal, `revert()`; otherwise update best-so-far.
  - On reject: just don't move (no mutation occurred).

Why this is different from E139:
  - E139 moves were *pair swaps* — provably outside CD's reachable set,
    but the CD basin turned out to be also a swap-stable local min.
  - E151 moves are *small continuous displacements of one macro* —
    CD only commits at per-axis grid-or-real best, so off-axis or
    non-grid neighbourhoods are NOT explored by CD. SA-continuous
    samples those directly.

Overlap handling:
  - delta_cost() is non-mutating so it CANNOT detect overlap (it only
    returns proxy). We commit via move() and check overlap_count
    post-hoc, reverting if illegal.
  - Since we only displace one macro, the overlap check reduces to a
    bbox sweep of macro i against all other hard macros (cheap).
  - revert() is single-step; we only ever move at most one macro per
    iteration, so single-step revert is safe.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch

from macro_place.incremental_evaluator import IncrementalProxyEvaluator


def _is_overlap_at(
    i: int,
    new_x: float,
    new_y: float,
    placement: torch.Tensor,
    macro_sizes_np: np.ndarray,
    n_hard: int,
    eps: float = 1e-4,
) -> bool:
    """Return True if placing macro i at (new_x, new_y) would overlap any
    other hard macro k != i. Soft macros (idx >= n_hard) are ignored — they
    overlap by design and CD/SA never legality-check them.
    """
    if i >= n_hard:
        return False  # soft macro can overlap freely

    pos = placement[:n_hard].cpu().numpy().astype(np.float64)
    sz = macro_sizes_np[:n_hard]

    half_wi = float(macro_sizes_np[i, 0]) / 2.0
    half_hi = float(macro_sizes_np[i, 1]) / 2.0

    dx = np.abs(pos[:, 0] - new_x)
    dy = np.abs(pos[:, 1] - new_y)
    min_dx = half_wi + sz[:, 0] / 2.0
    min_dy = half_hi + sz[:, 1] / 2.0
    blockers = (dx < min_dx - eps) & (dy < min_dy - eps)
    blockers[i] = False
    return bool(np.any(blockers))


def run_sa_continuous(
    evaluator: IncrementalProxyEvaluator,
    n_hard: int,
    fixed_np: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    budget_s: float,
    step_size_frac: float = 0.005,
    T_start_frac: float = 0.01,
    T_end_frac: float = 0.0001,
    pace_window_s: float = 10.0,
    rng_seed: int = 42,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Run SA on per-macro Gaussian moves until budget exhausted.

    Args:
        evaluator: CD-polished `IncrementalProxyEvaluator`.
        n_hard: number of hard macros.
        fixed_np: [num_macros] bool, True if fixed.
        canvas_width, canvas_height: canvas size for step-size scaling.
        budget_s: wall budget.
        step_size_frac: σ = step_size_frac · canvas_width (and · height for y).
        T_start_frac, T_end_frac: temperature schedule as fraction of init proxy.
        pace_window_s: first N seconds estimate iters/sec for T decay.
        rng_seed: PRNG.

    Returns dict with proposed, accepted_better, accepted_worse,
    rejected_proxy, rejected_illegal, init_proxy, best_proxy, final_proxy,
    best_restored, wall_total_s.
    """
    rng = np.random.default_rng(rng_seed)
    macro_sizes_np = evaluator.macro_sizes.cpu().numpy()

    # Movable hard macros only. Soft macros aren't represented in legality
    # logic; also we don't include them because CD also skips soft.
    movable = [
        i for i in range(n_hard)
        if not bool(fixed_np[i])
    ]
    if len(movable) < 1:
        if log_fn is not None:
            log_fn(f"  SA-cont: no movable hard macros, skipping")
        empty = {
            "proposed": 0, "accepted_better": 0, "accepted_worse": 0,
            "rejected_proxy": 0, "rejected_illegal": 0,
            "init_proxy": evaluator.current_cost()["proxy"],
            "best_proxy": evaluator.current_cost()["proxy"],
            "final_proxy": evaluator.current_cost()["proxy"],
            "best_restored": False, "best_found_at_t": 0.0,
            "wall_total_s": 0.0,
        }
        return empty

    init_proxy = float(evaluator.current_cost()["proxy"])
    cur_proxy = init_proxy
    best_proxy = init_proxy
    best_placement = evaluator.placement.detach().clone()
    best_found_at_t = 0.0

    T_start = T_start_frac * init_proxy
    T_end = T_end_frac * init_proxy
    if T_start <= 0.0:
        T_start = 1e-6
    if T_end <= 0.0:
        T_end = 1e-9

    step_x = step_size_frac * canvas_width
    step_y = step_size_frac * canvas_height

    proposed = 0
    accepted_better = 0
    accepted_worse = 0
    rejected_proxy = 0
    rejected_illegal = 0

    t_start = time.perf_counter()
    last_log_t = t_start

    T = T_start
    decay = 1.0
    pace_active = True
    pace_count_start = 0

    M = len(movable)
    while True:
        now = time.perf_counter()
        elapsed = now - t_start
        if elapsed >= budget_s:
            break

        # Switch from pace to schedule once pace_window elapsed.
        if pace_active and elapsed >= pace_window_s:
            pace_active = False
            pace_iters = proposed - pace_count_start
            remaining_s = budget_s - elapsed
            if pace_iters > 0 and pace_window_s > 0:
                rate = pace_iters / pace_window_s
                expected_remaining = int(rate * remaining_s)
            else:
                expected_remaining = 1
            expected_remaining = max(expected_remaining, 1)
            decay = (T_end / T_start) ** (1.0 / expected_remaining)
            if log_fn is not None:
                log_fn(
                    f"  SA-cont pace: {pace_iters} iters in {pace_window_s:.0f}s "
                    f"→ rate={pace_iters/pace_window_s:.1f}/s, "
                    f"expected_remaining_iters={expected_remaining}, decay={decay:.6f}"
                )

        # Sample a movable macro and a Gaussian displacement.
        ai = int(rng.integers(0, M))
        i = movable[ai]
        old_x = float(evaluator.placement[i, 0])
        old_y = float(evaluator.placement[i, 1])
        # σ ~ T·step. T runs 0.01·proxy → 0.0001·proxy. step is fixed.
        # Multiply by (T / T_start) so σ scales naturally with annealing.
        scale = T / T_start
        dx = float(rng.normal(0.0, step_x * scale))
        dy = float(rng.normal(0.0, step_y * scale))
        new_x = old_x + dx
        new_y = old_y + dy

        # Clamp to canvas (with half-size margin to keep macro fully inside).
        half_w = float(macro_sizes_np[i, 0]) / 2.0
        half_h = float(macro_sizes_np[i, 1]) / 2.0
        new_x = min(max(new_x, half_w), canvas_width - half_w)
        new_y = min(max(new_y, half_h), canvas_height - half_h)

        # Peek proxy (non-mutating).
        delta_costs = evaluator.delta_cost(i, (new_x, new_y))
        new_proxy = float(delta_costs["proxy"])
        delta = new_proxy - cur_proxy

        proposed += 1

        # Metropolis decision (before legality check, to avoid wasted overlap sweeps
        # on guaranteed-rejected moves).
        accept = False
        if delta < 0.0:
            accept = True
        else:
            p = math.exp(-delta / T) if T > 0 else 0.0
            if rng.random() < p:
                accept = True

        if not accept:
            rejected_proxy += 1
            if not pace_active:
                T *= decay
                if T < T_end:
                    T = T_end
            continue

        # Legality check — bbox sweep of macro i at (new_x, new_y) vs all
        # other hard macros.
        if _is_overlap_at(
            i, new_x, new_y,
            evaluator.placement, macro_sizes_np, n_hard,
        ):
            rejected_illegal += 1
            if not pace_active:
                T *= decay
                if T < T_end:
                    T = T_end
            continue

        # Commit.
        evaluator.move(i, (new_x, new_y))
        # delta_cost is correct under our pin-retargeting trick, so new_proxy
        # equals evaluator.current_cost()["proxy"] up to float roundoff. Trust it.
        cur_proxy = new_proxy

        if delta < 0.0:
            accepted_better += 1
            if cur_proxy < best_proxy - 1e-12:
                best_proxy = cur_proxy
                best_placement = evaluator.placement.detach().clone()
                best_found_at_t = elapsed
        else:
            accepted_worse += 1

        if not pace_active:
            T *= decay
            if T < T_end:
                T = T_end

        if log_fn is not None and (now - last_log_t) >= 30.0:
            last_log_t = now
            acc = accepted_better + accepted_worse
            acc_rate = (acc / proposed * 100.0) if proposed > 0 else 0.0
            log_fn(
                f"  SA-cont t={elapsed:6.1f}s T={T:.2e} σ={(step_x*T/T_start):.2f} "
                f"proposed={proposed} better={accepted_better} "
                f"worse={accepted_worse} rejProxy={rejected_proxy} "
                f"rejIlleg={rejected_illegal} acc%={acc_rate:.1f} "
                f"cur={cur_proxy:.5f} best={best_proxy:.5f}"
            )

    wall = time.perf_counter() - t_start

    # Restore best-so-far if exit state is worse.
    final_chain_proxy = float(evaluator.current_cost()["proxy"])
    best_restored = False
    if best_proxy < final_chain_proxy - 1e-12:
        n_macros = int(evaluator.placement.shape[0])
        for k in range(n_macros):
            tx = float(best_placement[k, 0])
            ty = float(best_placement[k, 1])
            cx = float(evaluator.placement[k, 0])
            cy = float(evaluator.placement[k, 1])
            if abs(tx - cx) > 1e-9 or abs(ty - cy) > 1e-9:
                evaluator.commit(k, (tx, ty))
        best_restored = True

    final_proxy = float(evaluator.current_cost()["proxy"])
    if log_fn is not None:
        acc = accepted_better + accepted_worse
        acc_rate = (acc / proposed * 100.0) if proposed > 0 else 0.0
        log_fn(
            f"  SA-cont done: proposed={proposed}, better={accepted_better}, "
            f"worse={accepted_worse}, rejProxy={rejected_proxy}, "
            f"rejIlleg={rejected_illegal}, acc%={acc_rate:.1f}, "
            f"init={init_proxy:.5f}, best={best_proxy:.5f} (t={best_found_at_t:.1f}s), "
            f"chain-final={final_chain_proxy:.5f}, "
            f"restored={best_restored}, eval-final={final_proxy:.5f}"
        )

    return {
        "proposed": proposed,
        "accepted_better": accepted_better,
        "accepted_worse": accepted_worse,
        "rejected_proxy": rejected_proxy,
        "rejected_illegal": rejected_illegal,
        "init_proxy": init_proxy,
        "best_proxy": best_proxy,
        "final_proxy": final_proxy,
        "best_restored": best_restored,
        "best_found_at_t": best_found_at_t,
        "wall_total_s": wall,
    }
