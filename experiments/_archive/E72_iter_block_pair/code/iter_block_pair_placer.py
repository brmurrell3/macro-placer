"""Iterative block-level + per-pair SP refinement.

Alternates E61 V2 spatial-block crossover with E69 SP-guided per-pair
swap, with short CD polish in between, K iterations. Tracks best
state seen and returns it.

Hypothesis: each mechanism alone is sub-threshold (E61 V2 -0.07% --all,
E69 -0.10% --fast). Alternation may compound: block escapes basin
coarsely, per-pair refines locally, polish absorbs. Each iteration
re-encodes SP of the new state vs the OTHER basin so disagreements
shift over time.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_E69 = _ROOT / "experiments" / "E69_sequence_pair_search" / "code"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_E69) not in sys.path:
    sys.path.insert(0, str(_E69))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sp_search_placer import sp_guided_swap_search

from macro_place.benchmark import Benchmark
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def spatial_block_crossover(
    state: torch.Tensor,
    other: torch.Tensor,
    benchmark: Benchmark,
    *,
    seed: int = 42,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, dict]:
    """Apply E61 V2 quadrant crossover: state ⊕ other.

    Returns (crossover_placement, stats). If project_overlaps fails to
    legalize, returns the state unchanged.
    """
    if log is None:
        log = lambda s: print(s, flush=True)

    rng = np.random.default_rng(seed=seed)
    n_hard = benchmark.num_hard_macros
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    mid_x, mid_y = canvas_w / 2.0, canvas_h / 2.0

    state_np = state.detach().to(torch.float64).cpu().numpy()
    quadrant = (
        (state_np[:n_hard, 0] >= mid_x).astype(int)
        + 2 * (state_np[:n_hard, 1] >= mid_y).astype(int)
    )
    quadrant_pick_state = rng.random(4) < 0.5
    if quadrant_pick_state.all() or (~quadrant_pick_state).all():
        flip_idx = int(rng.integers(4))
        quadrant_pick_state[flip_idx] = not quadrant_pick_state[flip_idx]

    crossover = other.detach().clone().to(torch.float64)
    n_from_state = 0
    state_f64 = state.detach().to(torch.float64)
    for i in range(n_hard):
        if quadrant_pick_state[quadrant[i]]:
            crossover[i] = state_f64[i]
            n_from_state += 1

    crossover, proj_iters = project_overlaps(crossover, benchmark)
    # Cast to float32 BEFORE checking feasibility — compute_overlap_metrics
    # uses float32 sizes (from benchmark.macro_sizes), so we need positions in
    # the same dtype to avoid boundary-precision flips.
    crossover_f32 = crossover.to(torch.float32)
    overlap = compute_overlap_metrics(crossover_f32, benchmark)

    # If residual overlaps exist due to float32/64 boundary noise, try ONE more
    # legalize pass with float32-input project_overlaps (more iterations of
    # the boundary-robust legalizer).
    if overlap["overlap_count"] > 0 and overlap["overlap_count"] <= 5:
        crossover_f32, _ = project_overlaps(crossover_f32, benchmark)
        overlap = compute_overlap_metrics(crossover_f32, benchmark)

    log(f"  [crossover] quadrant_pick_state={quadrant_pick_state.tolist()}, "
        f"{n_from_state}/{n_hard} from state, project_iters={proj_iters}, "
        f"residual_overlap={overlap['overlap_count']}")

    if overlap["overlap_count"] > 0:
        log(f"  [crossover] unrecoverable; returning state unchanged")
        return state.detach().clone(), {
            "n_from_state": n_from_state,
            "feasible": False,
            "residual_overlap": overlap["overlap_count"],
        }

    return crossover_f32, {
        "n_from_state": n_from_state,
        "feasible": True,
        "residual_overlap": 0,
    }


def cd_polish(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    budget_seconds: float = 300.0,
    log: Optional[Callable[[str], None]] = None,
) -> torch.Tensor:
    if log is None:
        log = lambda s: print(s, flush=True)
    placement = placement.detach().clone()
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement)
    n_hard = benchmark.num_hard_macros
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable = [i for i in range(n_hard) if not bool(fixed[i])]
    info = run_cd_adaptive(
        evaluator, benchmark, plc, movable,
        min_time_s=30.0,
        hard_cap_s=budget_seconds,
        patience=3,
        plateau_threshold=0.001,
        log_fn=None,  # quiet
    )
    log(f"  [cd-polish] {info.get('exit_reason')} after {info.get('total_moves', '?')} moves "
        f"in {info.get('wall_total_s', 0):.0f}s")
    return evaluator.placement.detach().clone().to(torch.float32)


def iter_block_pair_search(
    e25_placement: torch.Tensor,
    e41_placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    n_iters: int = 4,
    block_polish_budget: float = 300.0,
    pair_max_attempts: int = 300,
    pair_budget: float = 600.0,
    pair_polish_budget: float = 300.0,
    seed_base: int = 42,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, dict]:
    """Iterative block-level + per-pair SP refinement.

    Returns (best_placement, stats_dict).
    """
    if log is None:
        log = lambda s: print(s, flush=True)

    e25_proxy = float(compute_proxy_cost(e25_placement, benchmark, plc)["proxy_cost"])
    e41_proxy = float(compute_proxy_cost(e41_placement, benchmark, plc)["proxy_cost"])

    if e25_proxy <= e41_proxy:
        state = e25_placement.detach().clone()
        state_label = "E25"
        other = e41_placement
        other_label = "E41"
        state_proxy = e25_proxy
    else:
        state = e41_placement.detach().clone()
        state_label = "E41"
        other = e25_placement
        other_label = "E25"
        state_proxy = e41_proxy

    log(f"[iter] starting from {state_label} ({state_proxy:.5f}); other = {state_label}")
    log(f"[iter] running {n_iters} iterations")

    best_proxy = state_proxy
    best_placement = state.detach().clone()
    best_iter = -1
    iter_log: List[dict] = []

    for it in range(n_iters):
        log(f"\n--- iter {it+1}/{n_iters} ---")
        t_iter = time.time()

        # 1. Block-level crossover.
        log(f"  [block-crossover] mixing state ({state_label}) with {other_label}")
        crossed, x_stats = spatial_block_crossover(
            state, other, benchmark, seed=seed_base + it, log=log,
        )
        if not x_stats["feasible"]:
            log(f"  [iter {it+1}] crossover unrecoverable; skipping iter")
            iter_log.append({"iter": it, "skipped": True})
            continue

        # 2. CD polish after crossover.
        polished_block = cd_polish(crossed, benchmark, plc,
                                   budget_seconds=block_polish_budget, log=log)
        block_proxy = float(compute_proxy_cost(polished_block, benchmark, plc)["proxy_cost"])
        block_overlap = compute_overlap_metrics(polished_block, benchmark)["overlap_count"]
        log(f"  [block-polished] proxy={block_proxy:.5f} overlap={block_overlap}")

        if block_overlap > 0:
            log(f"  [iter {it+1}] block polish produced overlaps; skipping")
            iter_log.append({"iter": it, "block_proxy": block_proxy, "skipped_after_block": True})
            continue

        # 3. SP per-pair swap (axis-preserving) on polished_block, target = other.
        log(f"  [sp-swap] running per-pair on polished-block, target={other_label}")
        rotated, sp_stats = sp_guided_swap_search(
            polished_block, other, benchmark, plc,
            max_attempts=pair_max_attempts,
            budget_seconds=pair_budget,
            seed=seed_base + 1000 + it,
            axis_preserving_only=True,
            log=lambda s: None,  # quiet inside
        )
        log(f"  [sp-swap] tried={sp_stats['swaps_tried']} accepted={sp_stats['swaps_accepted']} "
            f"final={sp_stats['final_proxy']:.5f}")

        # 4. CD polish after per-pair.
        polished_pair = cd_polish(rotated, benchmark, plc,
                                  budget_seconds=pair_polish_budget, log=log)
        pair_proxy = float(compute_proxy_cost(polished_pair, benchmark, plc)["proxy_cost"])
        pair_overlap = compute_overlap_metrics(polished_pair, benchmark)["overlap_count"]
        log(f"  [pair-polished] proxy={pair_proxy:.5f} overlap={pair_overlap}")

        wall_iter = time.time() - t_iter
        iter_log.append({
            "iter": it,
            "block_proxy": block_proxy,
            "block_overlap": block_overlap,
            "rotated_proxy": sp_stats["final_proxy"],
            "swaps_accepted": sp_stats["swaps_accepted"],
            "pair_proxy": pair_proxy,
            "pair_overlap": pair_overlap,
            "wall_iter_s": wall_iter,
        })

        # Track best.
        if pair_overlap == 0 and pair_proxy < best_proxy - 1e-7:
            best_proxy = pair_proxy
            best_placement = polished_pair.detach().clone()
            best_iter = it
            log(f"  [iter {it+1}] NEW BEST: {best_proxy:.5f} "
                f"(Δ={best_proxy - state_proxy:+.5f} vs start)")
        elif block_overlap == 0 and block_proxy < best_proxy - 1e-7:
            # block alone improved; pair didn't help further
            best_proxy = block_proxy
            best_placement = polished_block.detach().clone()
            best_iter = it

        # Decide what to use as next iteration's state. Use best from this iter
        # (block or pair, whichever was lower), to continue compounding.
        if pair_overlap == 0 and pair_proxy < block_proxy:
            state = polished_pair
            state_label = f"iter{it+1}-pair"
            state_proxy = pair_proxy
        else:
            state = polished_block
            state_label = f"iter{it+1}-block"
            state_proxy = block_proxy

        log(f"  [iter {it+1} done] wall={wall_iter:.0f}s, current={state_proxy:.5f}, "
            f"best_so_far={best_proxy:.5f}")

    log(f"\n[iter] complete: best={best_proxy:.5f} found at iter {best_iter+1}, "
        f"start was {e25_proxy if state_label == 'E25' else e41_proxy:.5f}")

    return best_placement, {
        "n_iters_run": len(iter_log),
        "best_proxy": best_proxy,
        "best_iter": best_iter,
        "starting_proxy": min(e25_proxy, e41_proxy),
        "improvement": min(e25_proxy, e41_proxy) - best_proxy,
        "improvement_frac": (min(e25_proxy, e41_proxy) - best_proxy)
                             / min(e25_proxy, e41_proxy),
        "iter_log": iter_log,
    }
