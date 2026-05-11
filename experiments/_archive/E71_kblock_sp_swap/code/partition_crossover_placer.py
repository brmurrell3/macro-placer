"""Fine-grained partition crossover (E71 V2 — generalizes E61 V2 quadrant).

E61 V2 partitions canvas into 2x2 quadrants; per-quadrant Bernoulli pick from
state or other basin. Single-shot; one accepted result.

E71 V2 generalizes:
- NxN grid partition (N=2,3,4,5).
- Multi-trial: many random pickings per attempt, accept any that improves.
- Multi-grid: try several grid sizes within one run.

Hypothesis: finer-than-quadrant granularity preserves coherent local
clusters within each cell while allowing more total recombinations.
With 4x4=16 cells, 2^16 ≈ 65k possible pickings vs 2^4=16 for quadrant.
Larger search space, more chances to find a productive recombination.

The partition approach AVOIDS the cluster-transplant feasibility problem
(E70/E71 V1) because EVERY macro is assigned a position from ONE of the
two basins — there's no "patch" to legalize against the rest. Boundary
overlaps are usually small and resolved in 5-50 project_overlaps iters
(per E61 V2's track record).
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
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def _grid_assignment(positions: np.ndarray, n_hard: int,
                     canvas_w: float, canvas_h: float, grid_n: int) -> np.ndarray:
    """For each macro, return which grid cell its center falls in (0..grid_n²-1)."""
    cell_w = canvas_w / grid_n
    cell_h = canvas_h / grid_n
    col = np.clip((positions[:n_hard, 0] / cell_w).astype(int), 0, grid_n - 1)
    row = np.clip((positions[:n_hard, 1] / cell_h).astype(int), 0, grid_n - 1)
    return row * grid_n + col


def partition_crossover(
    state: torch.Tensor,
    other: torch.Tensor,
    benchmark: Benchmark,
    *,
    grid_n: int,
    pick_state_prob: float = 0.5,
    seed: int = 42,
) -> Tuple[Optional[torch.Tensor], dict]:
    """Single partition crossover trial.

    Returns (placement, stats) on success; (None, stats) if unrecoverable.
    """
    rng = np.random.default_rng(seed)
    n_hard = benchmark.num_hard_macros
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    state_f64 = state.detach().to(torch.float64)
    other_f64 = other.detach().to(torch.float64)
    state_np = state_f64.cpu().numpy()

    # Assign each macro to a grid cell by its current state position.
    cell = _grid_assignment(state_np, n_hard, canvas_w, canvas_h, grid_n)
    n_cells = grid_n * grid_n
    pick_state = rng.random(n_cells) < pick_state_prob
    # Force at least one cell flip (avoid degenerate all-state or all-other).
    if pick_state.all() or (~pick_state).all():
        pick_state[int(rng.integers(n_cells))] = not pick_state[int(rng.integers(n_cells))]

    # Build crossover: start from other, override picked cells with state.
    crossover = other_f64.clone()
    n_from_state = 0
    for i in range(n_hard):
        if pick_state[cell[i]]:
            crossover[i] = state_f64[i]
            n_from_state += 1

    crossover_f32 = crossover.to(torch.float32)
    crossover_f32, proj_iters = project_overlaps(crossover_f32, benchmark)
    overlap = compute_overlap_metrics(crossover_f32, benchmark)

    # Extended legalize: if residual overlaps remain, run jitter+project.
    if overlap["overlap_count"] > 0:
        from extended_legalize import extended_legalize
        crossover_f32, leg_stats = extended_legalize(
            crossover_f32, benchmark, max_passes=15, jitter_scale=0.3, seed=seed,
        )
        overlap = compute_overlap_metrics(crossover_f32, benchmark)

    stats = {
        "grid_n": grid_n,
        "n_cells": n_cells,
        "n_from_state": n_from_state,
        "n_from_other": n_hard - n_from_state,
        "proj_iters": proj_iters,
        "residual_overlap": overlap["overlap_count"],
        "feasible": overlap["overlap_count"] == 0,
        "pick_state_count": int(pick_state.sum()),
    }
    if overlap["overlap_count"] > 0:
        return None, stats
    return crossover_f32, stats


def partition_crossover_search(
    state: torch.Tensor,
    other: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    grid_sizes: Tuple[int, ...] = (2, 3, 4),
    max_trials: int = 100,
    polish_per_trial: float = 60.0,
    budget_seconds: float = 1800.0,
    log: Optional[Callable[[str], None]] = None,
    seed: int = 42,
) -> Tuple[torch.Tensor, dict]:
    """Multi-trial partition crossover search.

    For each trial:
      1. Pick grid_n from grid_sizes randomly.
      2. Pick random Bernoulli per cell.
      3. Crossover; if feasible, polish briefly with CD.
      4. Accept if proxy < current proxy.

    Tracks best placement seen. If best > start, return start.
    """
    if log is None:
        log = lambda s: print(s, flush=True)

    rng = np.random.default_rng(seed)
    starting_proxy = float(compute_proxy_cost(state, benchmark, plc)["proxy_cost"])
    log(f"[partition] starting proxy: {starting_proxy:.5f}")
    best_proxy = starting_proxy
    best_placement = state.detach().clone()

    feas_count = 0
    accepted = 0
    feas_failed = 0
    proxy_failed = 0
    accepts_by_grid: dict = {g: 0 for g in grid_sizes}
    feas_by_grid: dict = {g: 0 for g in grid_sizes}
    proxy_history = [starting_proxy]
    t0 = time.time()
    trial = 0

    for trial in range(max_trials):
        if (time.time() - t0) > budget_seconds:
            log(f"[partition] reached budget {budget_seconds:.0f}s")
            break

        grid_n = int(grid_sizes[rng.integers(len(grid_sizes))])
        trial_seed = seed + 10000 + trial
        crossed, x_stats = partition_crossover(
            state, other, benchmark,
            grid_n=grid_n, pick_state_prob=0.5, seed=trial_seed,
        )
        if crossed is None:
            feas_failed += 1
            continue

        feas_count += 1
        feas_by_grid[grid_n] += 1

        # Quick CD polish.
        evaluator = IncrementalProxyEvaluator(benchmark, plc, crossed.clone())
        n_hard = benchmark.num_hard_macros
        fixed = benchmark.macro_fixed.cpu().numpy()
        movable = [i for i in range(n_hard) if not bool(fixed[i])]
        info = run_cd_adaptive(
            evaluator, benchmark, plc, movable,
            min_time_s=15.0, hard_cap_s=polish_per_trial,
            patience=3, plateau_threshold=0.005,
            log_fn=None,
        )
        polished = evaluator.placement.detach().clone().to(torch.float32)
        polished_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
        polished_overlap = compute_overlap_metrics(polished, benchmark)["overlap_count"]

        if polished_overlap == 0 and polished_proxy < best_proxy - 1e-7:
            best_proxy = polished_proxy
            best_placement = polished
            accepted += 1
            accepts_by_grid[grid_n] += 1
            proxy_history.append(best_proxy)
            log(f"[partition] [{trial+1}/{max_trials}] grid={grid_n}x{grid_n} ACCEPT → "
                f"polished {polished_proxy:.5f} (Δ={polished_proxy - proxy_history[-2]:+.5f}, "
                f"accepted={accepted}, picked {x_stats['pick_state_count']}/{x_stats['n_cells']} from state)")
            # Use the new state as the next baseline for further crossovers.
            state = best_placement
        else:
            proxy_failed += 1

    wall = time.time() - t0
    log(f"[partition] done: trials={trial+1} feasible={feas_count} accepted={accepted} "
        f"feas_failed={feas_failed} proxy_failed={proxy_failed} wall={wall:.0f}s")
    log(f"[partition] feasible by grid: {feas_by_grid}")
    log(f"[partition] accepts by grid: {accepts_by_grid}")
    log(f"[partition] best proxy: {best_proxy:.5f} (vs starting {starting_proxy:.5f}, "
        f"Δ={best_proxy - starting_proxy:+.5f})")

    return best_placement, {
        "trials": trial + 1,
        "feasible_count": feas_count,
        "accepted": accepted,
        "feas_failed": feas_failed,
        "proxy_failed": proxy_failed,
        "feas_by_grid": feas_by_grid,
        "accepts_by_grid": accepts_by_grid,
        "starting_proxy": starting_proxy,
        "final_proxy": best_proxy,
        "improvement": starting_proxy - best_proxy,
        "improvement_frac": (starting_proxy - best_proxy) / starting_proxy if starting_proxy > 0 else 0.0,
        "wall_seconds": wall,
        "proxy_history": proxy_history,
    }


def cd_polish(
    placement: torch.Tensor, benchmark: Benchmark, plc,
    *, budget_seconds: float = 600.0, log=None,
) -> torch.Tensor:
    if log is None:
        log = lambda s: print(s, flush=True)
    log(f"[cd-polish] starting (budget {budget_seconds:.0f}s)...")
    placement = placement.detach().clone()
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement)
    n_hard = benchmark.num_hard_macros
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable = [i for i in range(n_hard) if not bool(fixed[i])]
    info = run_cd_adaptive(
        evaluator, benchmark, plc, movable,
        min_time_s=60.0, hard_cap_s=budget_seconds,
        patience=3, plateau_threshold=0.001, log_fn=None,
    )
    log(f"[cd-polish] done: {info.get('exit_reason')} after {info.get('total_moves', '?')} moves")
    return evaluator.placement.detach().clone().to(torch.float32)
