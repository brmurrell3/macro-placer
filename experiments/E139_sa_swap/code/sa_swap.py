"""E139 — Simulated annealing on macro pair-swap moves.

Designed to be appended after thinkorplace-v2's CD polish phase. The
target basin is already at the CD per-axis fixed point, so we need a
move type CD never considers. Swap = simultaneous 2-macro repositioning,
which is provably outside CD's reachable set.

Move accepted by Metropolis: better always, worse with exp(−Δ/T).
Schedule: T_start = 0.01·proxy, T_end = 0.0001·proxy, exponential decay.

Overlap handling:
- Pre-swap legality check via bbox test (excluding i, j from blockers).
  Identical to E15's `_is_legal_2d_excluding`.
- Post-swap state is guaranteed overlap-free because i ends at j's old
  position and j ends at i's — they swap, no third macro is touched.
- Reverts are 2-step `move()`: single-step `revert()` only undoes the
  most recent move. Move-back is bit-identical at the placement level
  (positions are exact, evaluator state is recomputed from positions).

Acceptance peek uses `delta_cost` (non-mutating) for the FIRST move,
then commits if the swap will likely be accepted; but the 2-step move
needs the intermediate state, so we use `move/move` and revert if rejected.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from macro_place.incremental_evaluator import IncrementalProxyEvaluator


def _is_legal_swap(
    i: int,
    j: int,
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    placement: torch.Tensor,
    macro_sizes_np: np.ndarray,
    n_hard: int,
    eps: float = 1e-4,
) -> bool:
    """Check whether moving i→pos_j and j→pos_i would create any overlap
    with a hard macro k ∉ {i, j}.

    For each of (i at pos_j) and (j at pos_i), do a bbox sweep against all
    other hard macros (excluding i, j). Since the final swapped state has
    i at j's old position and j at i's old position, the i-j pair itself
    is guaranteed not to overlap each other (assuming initial state was
    overlap-free, i and j sizes can differ but their *new* centers are at
    each other's old positions, which were non-overlapping).
    """
    # Soft macros (idx >= n_hard) overlap by design — skip legality.
    if i >= n_hard or j >= n_hard:
        return True

    pos = placement[:n_hard].cpu().numpy().astype(np.float64)
    sz = macro_sizes_np[:n_hard]

    half_wi = float(macro_sizes_np[i, 0]) / 2.0
    half_hi = float(macro_sizes_np[i, 1]) / 2.0
    half_wj = float(macro_sizes_np[j, 0]) / 2.0
    half_hj = float(macro_sizes_np[j, 1]) / 2.0

    # Check (i at pos_j) vs all third macros.
    dx_i = np.abs(pos[:, 0] - pos_j[0])
    dy_i = np.abs(pos[:, 1] - pos_j[1])
    min_dx_i = half_wi + sz[:, 0] / 2.0
    min_dy_i = half_hi + sz[:, 1] / 2.0
    blockers_i = (dx_i < min_dx_i - eps) & (dy_i < min_dy_i - eps)
    blockers_i[i] = False
    blockers_i[j] = False
    if bool(np.any(blockers_i)):
        return False

    # Check (j at pos_i) vs all third macros.
    dx_j = np.abs(pos[:, 0] - pos_i[0])
    dy_j = np.abs(pos[:, 1] - pos_i[1])
    min_dx_j = half_wj + sz[:, 0] / 2.0
    min_dy_j = half_hj + sz[:, 1] / 2.0
    blockers_j = (dx_j < min_dx_j - eps) & (dy_j < min_dy_j - eps)
    blockers_j[i] = False
    blockers_j[j] = False
    if bool(np.any(blockers_j)):
        return False

    return True


def run_sa_swap(
    evaluator: IncrementalProxyEvaluator,
    n_hard: int,
    fixed_np: np.ndarray,
    budget_s: float,
    T_start_frac: float = 0.01,
    T_end_frac: float = 0.0001,
    pace_window_s: float = 10.0,
    rng_seed: int = 42,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Run SA on pair-swap moves until budget exhausted.

    Args:
        evaluator: already CD-polished `IncrementalProxyEvaluator`.
        n_hard: number of hard macros (legality applies only here).
        fixed_np: shape [num_macros] bool, True if macro must not move.
        budget_s: wall budget in seconds.
        T_start_frac, T_end_frac: temperature schedule as fraction of
            initial proxy.
        pace_window_s: first N seconds used to estimate swaps/sec for
            T-decay calibration. After that, T decays exponentially toward
            T_end across the remaining budget.
        rng_seed: PRNG seed.

    Returns:
        dict with proposed, accepted_better, accepted_worse, rejected_proxy,
        rejected_illegal, init_proxy, best_proxy, final_proxy, best_restored,
        wall_total_s.
    """
    rng = np.random.default_rng(rng_seed)
    macro_sizes_np = evaluator.macro_sizes.cpu().numpy()

    # Movable hard macros (legality check requires hard; we skip soft because
    # soft macros overlap by design and CD doesn't move them either).
    movable = [
        i for i in range(n_hard)
        if not bool(fixed_np[i])
    ]
    if len(movable) < 2:
        if log_fn is not None:
            log_fn(f"  SA-swap: only {len(movable)} movable hard macros, skipping")
        return {
            "proposed": 0, "accepted_better": 0, "accepted_worse": 0,
            "rejected_proxy": 0, "rejected_illegal": 0,
            "init_proxy": evaluator.current_cost()["proxy"],
            "best_proxy": evaluator.current_cost()["proxy"],
            "final_proxy": evaluator.current_cost()["proxy"],
            "best_restored": False, "wall_total_s": 0.0,
        }

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

    proposed = 0
    accepted_better = 0
    accepted_worse = 0
    rejected_proxy = 0
    rejected_illegal = 0

    t_start = time.perf_counter()
    last_log_t = t_start

    # Pace phase: hold T at T_start to count swaps/sec.
    T = T_start
    decay = 1.0  # set after pace window
    pace_active = True
    pace_t0 = t_start
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
            # Estimate remaining iters based on observed rate.
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
                    f"  SA-swap pace: {pace_iters} iters in {pace_window_s:.0f}s "
                    f"→ rate={pace_iters/pace_window_s:.1f}/s, "
                    f"expected_remaining_iters={expected_remaining}, decay={decay:.6f}"
                )

        # Sample distinct pair i, j.
        ai = int(rng.integers(0, M))
        aj = int(rng.integers(0, M))
        while aj == ai:
            aj = int(rng.integers(0, M))
        i = movable[ai]
        j = movable[aj]

        pos_i = (
            float(evaluator.placement[i, 0]),
            float(evaluator.placement[i, 1]),
        )
        pos_j = (
            float(evaluator.placement[j, 0]),
            float(evaluator.placement[j, 1]),
        )

        # Legality check (third-party overlap).
        if not _is_legal_swap(
            i, j,
            np.array(pos_i, dtype=np.float64),
            np.array(pos_j, dtype=np.float64),
            evaluator.placement,
            macro_sizes_np,
            n_hard,
        ):
            rejected_illegal += 1
            proposed += 1
            if not pace_active:
                T *= decay
            continue

        # Apply 2-step swap.
        evaluator.move(i, pos_j)
        evaluator.move(j, pos_i)
        new_proxy = float(evaluator.current_cost()["proxy"])
        delta = new_proxy - cur_proxy

        proposed += 1
        if delta < 0.0:
            cur_proxy = new_proxy
            accepted_better += 1
            if cur_proxy < best_proxy - 1e-12:
                best_proxy = cur_proxy
                best_placement = evaluator.placement.detach().clone()
                best_found_at_t = elapsed
        else:
            # Metropolis accept.
            accept_p = math.exp(-delta / T) if T > 0 else 0.0
            if rng.random() < accept_p:
                cur_proxy = new_proxy
                accepted_worse += 1
            else:
                # Reject: 2-step revert. revert() is single-step, undoes only the
                # last move(). We need to undo BOTH. Move-back is bit-identical.
                evaluator.move(j, pos_j)
                evaluator.move(i, pos_i)
                rejected_proxy += 1

        if not pace_active:
            T *= decay
            if T < T_end:
                T = T_end

        if log_fn is not None and (now - last_log_t) >= 30.0:
            last_log_t = now
            log_fn(
                f"  SA-swap t={elapsed:6.1f}s T={T:.2e} "
                f"proposed={proposed} better={accepted_better} "
                f"worse={accepted_worse} rejProxy={rejected_proxy} "
                f"rejIlleg={rejected_illegal} cur={cur_proxy:.5f} "
                f"best={best_proxy:.5f}"
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
                # commit() avoids snapshot overhead; we don't need revert here.
                evaluator.commit(k, (tx, ty))
        best_restored = True

    final_proxy = float(evaluator.current_cost()["proxy"])
    if log_fn is not None:
        log_fn(
            f"  SA-swap done: proposed={proposed}, better={accepted_better}, "
            f"worse={accepted_worse}, rejProxy={rejected_proxy}, "
            f"rejIlleg={rejected_illegal}, init={init_proxy:.5f}, "
            f"best={best_proxy:.5f} (t={best_found_at_t:.1f}s), "
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
