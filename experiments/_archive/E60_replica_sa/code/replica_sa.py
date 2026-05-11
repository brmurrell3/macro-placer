"""Replica-exchange (parallel tempering) SA-v2 — drop-in replacement for the
single-chain SA-v2 phase used in E25/E41 pipelines.

K parallel chains at temperatures T₀ = [5e-4, 5e-3, 5e-2, 5e-1]; each
has its own IncrementalProxyEvaluator state initialized from the
post-LNS placement. Round-robin dispatch: each chain runs N moves at its
T, then attempt one swap between a random adjacent pair via Metropolis.
Best-tracked across all chains.

Mathematically equivalent to the SA-v2 baseline when K=1.
"""
from __future__ import annotations

import math
import time
from typing import Callable, List, Optional

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.cd_core import legal_axis_range, axis_breakpoints
from macro_place.incremental_evaluator import IncrementalProxyEvaluator


def _grid_lines(plc):
    """Mirror of cd_core's _grid_lines (kept here so this module is
    self-contained)."""
    grid_w = plc.width / plc.grid_col
    grid_h = plc.height / plc.grid_row
    grid_lines_x = np.array(
        [(c + 0.5) * grid_w for c in range(int(plc.grid_col))],
        dtype=np.float64,
    )
    grid_lines_y = np.array(
        [(r + 0.5) * grid_h for r in range(int(plc.grid_row))],
        dtype=np.float64,
    )
    return grid_lines_x, grid_lines_y


def _step_chain(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    grid_lines_x: np.ndarray,
    grid_lines_y: np.ndarray,
    hard_movable: List[int],
    n_moves: int,
    cur_proxy: float,
    T: float,
    rng: np.random.Generator,
    breakpoint_budget: int,
) -> tuple:
    """Run n_moves of SA-v2 at temperature T on this chain. Returns
    (cur_proxy, n_proposed, n_accepted_better, n_accepted_worse,
    n_rejected, n_skipped, best_proxy_in_chunk, best_placement_in_chunk).
    """
    n_hard = benchmark.num_hard_macros
    proposed = accepted_better = accepted_worse = rejected = skipped = 0
    best_proxy_in_chunk = cur_proxy
    best_placement_in_chunk = evaluator.placement.detach().clone()

    for _ in range(n_moves):
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

        if delta <= 0.0:
            cur_proxy = new_proxy
            accepted_better += 1
            if cur_proxy < best_proxy_in_chunk - 1e-12:
                best_proxy_in_chunk = cur_proxy
                best_placement_in_chunk = evaluator.placement.detach().clone()
        else:
            accept_p = math.exp(-delta / T) if T > 0 else 0.0
            if rng.random() < accept_p:
                cur_proxy = new_proxy
                accepted_worse += 1
            else:
                evaluator.revert()
                rejected += 1

    return (
        cur_proxy, proposed, accepted_better, accepted_worse,
        rejected, skipped, best_proxy_in_chunk, best_placement_in_chunk,
    )


def run_replica_exchange_sa_v2(
    base_placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    hard_movable: List[int],
    time_budget_s: float,
    T_list: List[float] = (5e-4, 1e-3, 2e-3, 5e-3),
    Tf: float = 1e-6,  # ignored in fixed-T mode; kept for compat
    seed: int = 42,
    breakpoint_budget: int = 12,
    moves_per_round: int = 200,
    swap_every_n_rounds: int = 4,
    log_fn: Optional[Callable[[str], None]] = None,
) -> tuple:
    """Replica-exchange parallel tempering SA on per-axis breakpoints.

    FIXED-T variant (standard PT): each chain k runs at constant T_k for
    the duration. Round-robin: every chain advances `moves_per_round`
    moves; every `swap_every_n_rounds`, attempt one swap between a
    random adjacent pair (k, k+1) via Metropolis. T is SWAPPED on
    accept, not state — equivalent dynamics with cheaper bookkeeping.

    T_list spacing recommendation: geometric factor ~2 between adjacent
    chains (default [5e-4, 1e-3, 2e-3, 5e-3]). Wider spacing causes
    aggressive swaps that disrupt cold-chain descent.

    Returns (best_placement, stats_dict).
    """
    T0_list = list(T_list)  # alias for backwards compat in stats
    K = len(T_list)
    rng_global = np.random.default_rng(seed=seed)
    chain_rngs = [
        np.random.default_rng(seed=seed + 1000 * k) for k in range(K)
    ]

    grid_lines_x, grid_lines_y = _grid_lines(plc)
    n_hard = benchmark.num_hard_macros

    if not hard_movable:
        if log_fn is not None:
            log_fn("  replica-SA: no hard movable macros; skipping")
        return base_placement.detach().clone(), {
            "K": K, "rounds": 0, "swap_attempts": 0, "swap_accepts": 0,
            "best_proxy": float("nan"), "best_chain": -1,
            "wall_total_s": 0.0,
        }

    # K independent evaluators, each initialized to the post-LNS placement.
    # IncrementalProxyEvaluator does its own state tracking.
    chain_evaluators: List[IncrementalProxyEvaluator] = []
    for k in range(K):
        ev = IncrementalProxyEvaluator(
            benchmark, plc,
            base_placement.detach().clone().to(torch.float64),
        )
        chain_evaluators.append(ev)

    # FIXED temperature schedule per chain INDEX. Chain k always anneals
    # from T_list[k] to Tf over the full budget. Swaps exchange chain
    # state (evaluator, proxy, RNG, best) — NOT temperatures.
    chain_proxies = [ev.current_cost()["proxy"] for ev in chain_evaluators]
    chain_best_proxy = list(chain_proxies)
    chain_best_placement = [
        ev.placement.detach().clone() for ev in chain_evaluators
    ]
    init_proxies = list(chain_proxies)

    if log_fn is not None:
        log_fn(
            f"  replica-SA budget={time_budget_s:.0f}s K={K} "
            f"T_fixed={T_list} N={moves_per_round} "
            f"swap_every={swap_every_n_rounds} rounds "
            f"|H|={len(hard_movable)} init_proxies={[f'{p:.5f}' for p in init_proxies]}"
        )

    t_start = time.perf_counter()
    last_log_t = t_start
    rounds = 0
    swap_attempts = 0
    swap_accepts = 0
    proposed_total = accepted_better_total = accepted_worse_total = 0
    rejected_total = skipped_total = 0

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:
            break

        rounds += 1
        # Each round: advance every chain by moves_per_round at its
        # annealed T (per fixed chain INDEX). T_k(t) = T_list[k] * exp(
        # log(Tf/T_list[k]) * t/total_time).
        for k in range(K):
            elapsed_now = time.perf_counter() - t_start
            if elapsed_now >= time_budget_s:
                break
            frac = elapsed_now / time_budget_s
            T0_k = T_list[k]
            T = T0_k * math.exp(math.log(Tf / T0_k) * frac) if T0_k > Tf else Tf

            (
                new_proxy, p, ab, aw, r, s,
                best_proxy_in_chunk, best_placement_in_chunk,
            ) = _step_chain(
                chain_evaluators[k], benchmark,
                grid_lines_x, grid_lines_y, hard_movable,
                moves_per_round, chain_proxies[k], T,
                chain_rngs[k], breakpoint_budget,
            )
            chain_proxies[k] = new_proxy
            proposed_total += p
            accepted_better_total += ab
            accepted_worse_total += aw
            rejected_total += r
            skipped_total += s
            if best_proxy_in_chunk < chain_best_proxy[k] - 1e-12:
                chain_best_proxy[k] = best_proxy_in_chunk
                chain_best_placement[k] = best_placement_in_chunk

        # Every swap_every_n_rounds, attempt one swap between a random
        # adjacent pair (k, k+1). Standard PT: SWAP STATES (evaluators,
        # proxies, RNGs, best-tracking) — temperatures stay fixed at
        # their indices. Acceptance uses CURRENT annealed T values.
        if K >= 2 and rounds % swap_every_n_rounds == 0:
            k = int(rng_global.integers(0, K - 1))
            frac = (time.perf_counter() - t_start) / time_budget_s
            T_k = T_list[k] * math.exp(math.log(Tf / T_list[k]) * frac) if T_list[k] > Tf else Tf
            T_kp1 = T_list[k + 1] * math.exp(math.log(Tf / T_list[k + 1]) * frac) if T_list[k + 1] > Tf else Tf
            E_k = chain_proxies[k]
            E_kp1 = chain_proxies[k + 1]
            # Standard replica-exchange acceptance:
            # P(accept) = min(1, exp((β_k - β_{k+1}) * (E_k - E_{k+1})))
            beta_diff = (1.0 / max(T_k, 1e-12)) - (1.0 / max(T_kp1, 1e-12))
            energy_diff = E_k - E_kp1
            log_p = beta_diff * energy_diff
            swap_attempts += 1
            if log_p >= 0 or rng_global.random() < math.exp(log_p):
                # Accept: swap entire chain state. Each chain is now
                # operating on the OTHER chain's prior placement, but at
                # its own (fixed) annealing schedule.
                chain_evaluators[k], chain_evaluators[k + 1] = (
                    chain_evaluators[k + 1], chain_evaluators[k]
                )
                chain_proxies[k], chain_proxies[k + 1] = (
                    chain_proxies[k + 1], chain_proxies[k]
                )
                chain_rngs[k], chain_rngs[k + 1] = (
                    chain_rngs[k + 1], chain_rngs[k]
                )
                chain_best_proxy[k], chain_best_proxy[k + 1] = (
                    chain_best_proxy[k + 1], chain_best_proxy[k]
                )
                chain_best_placement[k], chain_best_placement[k + 1] = (
                    chain_best_placement[k + 1], chain_best_placement[k]
                )
                swap_accepts += 1

        now = time.perf_counter()
        if log_fn is not None and (now - last_log_t) >= 30.0:
            last_log_t = now
            best_k = int(np.argmin(chain_best_proxy))
            log_fn(
                f"  replica-SA t={now - t_start:6.1f}s rounds={rounds} "
                f"swap_acc={swap_accepts}/{swap_attempts} "
                f"chain_proxies=[{', '.join(f'{p:.5f}' for p in chain_proxies)}] "
                f"chain_best=[{', '.join(f'{p:.5f}' for p in chain_best_proxy)}] "
                f"best_chain={best_k}"
            )

    # Pick the best across all chains.
    best_k = int(np.argmin(chain_best_proxy))
    best_proxy = float(chain_best_proxy[best_k])
    best_placement = chain_best_placement[best_k]

    if log_fn is not None:
        log_fn(
            f"  replica-SA done: rounds={rounds} swap={swap_accepts}/{swap_attempts} "
            f"proposed={proposed_total} better={accepted_better_total} "
            f"worse={accepted_worse_total} rejected={rejected_total} "
            f"skipped={skipped_total} init_proxies={[f'{p:.5f}' for p in init_proxies]} "
            f"final_best={best_proxy:.5f} (chain index {best_k} of {K}, T0={T_list[best_k]:.2e})"
        )

    return best_placement, {
        "K": K, "rounds": rounds,
        "swap_attempts": swap_attempts, "swap_accepts": swap_accepts,
        "proposed": proposed_total,
        "accepted_better": accepted_better_total,
        "accepted_worse": accepted_worse_total,
        "rejected": rejected_total, "skipped": skipped_total,
        "init_proxies": init_proxies,
        "best_proxy": best_proxy, "best_chain": best_k,
        "chain_best_proxies": list(chain_best_proxy),
        "wall_total_s": time.perf_counter() - t_start,
    }
