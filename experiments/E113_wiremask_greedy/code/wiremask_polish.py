"""WireMask greedy reposition polish (E113).

Inspired by WireMask-BBO (NeurIPS 2023, Shi/Yu/Qian) and EGPlace (ICML
2025, Deng et al.). For each movable macro, in order of current WL
contribution (criticality-ranked), enumerate a uniform grid of
candidate positions; pick the lowest-proxy position via incremental
delta_cost evaluation; commit if overlap-free.

Differs from CD search_axis in that 2D moves are atomic (simultaneous
x+y change). 2D moves can unlock placements blocked by intermediate
1D states with overlap.

This is a POST-PROCESSING layer — accepts any feasible placement,
returns a refined one with same or lower proxy and zero overlaps.
"""
from __future__ import annotations

import math
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics


def _criticality_order(ev: IncrementalProxyEvaluator, movable: List[int]) -> List[int]:
    """Order macros by descending estimated WL contribution.

    Cheap proxy for criticality: |macro_to_nets| * mean(net_hpwl over those nets).
    """
    scores = []
    for m in movable:
        nets = ev.macro_to_nets[m].tolist()
        if len(nets) == 0:
            scores.append((0.0, m))
            continue
        hpwl_sum = float(ev.net_hpwl[nets].sum())
        weighted = hpwl_sum * float(ev.net_weight[nets].mean())
        scores.append((weighted, m))
    scores.sort(reverse=True)
    return [m for _, m in scores]


def _generate_grid_candidates(
    cur_xy: Tuple[float, float],
    half_w: float, half_h: float,
    canvas_w: float, canvas_h: float,
    n_per_axis: int,
    local_radius: Optional[float] = None,
) -> np.ndarray:
    """Generate (N×N) grid candidates centered on cur_xy.

    If local_radius is None, grid spans the full feasible canvas
    region. Otherwise grid spans cur_xy ± local_radius (clipped to
    feasible region).
    """
    x_lo_full = half_w
    x_hi_full = canvas_w - half_w
    y_lo_full = half_h
    y_hi_full = canvas_h - half_h

    if local_radius is None:
        x_lo, x_hi = x_lo_full, x_hi_full
        y_lo, y_hi = y_lo_full, y_hi_full
    else:
        x_lo = max(x_lo_full, cur_xy[0] - local_radius)
        x_hi = min(x_hi_full, cur_xy[0] + local_radius)
        y_lo = max(y_lo_full, cur_xy[1] - local_radius)
        y_hi = min(y_hi_full, cur_xy[1] + local_radius)

    if x_hi <= x_lo or y_hi <= y_lo:
        return np.empty((0, 2), dtype=np.float64)

    xs = np.linspace(x_lo, x_hi, n_per_axis)
    ys = np.linspace(y_lo, y_hi, n_per_axis)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()])


def wiremask_greedy_pass(
    ev: IncrementalProxyEvaluator,
    benchmark,
    movable_idx: List[int],
    *,
    n_per_axis: int = 9,
    local_radius_frac: Optional[float] = 0.30,
    time_budget_s: Optional[float] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    """One WireMask pass: each movable macro is greedily re-placed at
    its lowest-proxy grid candidate (subject to no overlap).

    Args:
        ev: live IncrementalProxyEvaluator
        benchmark: bench
        movable_idx: macros to consider
        n_per_axis: grid resolution (n_per_axis^2 candidates per macro)
        local_radius_frac: if not None, restrict search to ±frac*canvas
            around current position (e.g. 0.3 = 30% of canvas).
            If None, search full canvas.
        time_budget_s: optional hard cap.
    """
    if log is None:
        log = lambda s: None
    t0 = time.perf_counter()
    deadline = (t0 + time_budget_s) if time_budget_s else None

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    diag = math.sqrt(cw * cw + ch * ch)
    sizes_np = benchmark.macro_sizes.cpu().numpy()
    fixed_np = benchmark.macro_fixed.cpu().numpy()
    local_radius = (
        local_radius_frac * diag if local_radius_frac is not None else None
    )

    order = _criticality_order(ev, movable_idx)
    init_proxy = float(ev.current_cost()["proxy"])

    n_moves = 0
    n_overlap_rejects = 0
    n_proxy_rejects = 0
    n_macros_visited = 0
    total_candidates = 0
    total_delta = 0.0

    for m in order:
        if fixed_np[m]:
            continue
        if deadline is not None and time.perf_counter() > deadline:
            break
        n_macros_visited += 1
        cur_xy = (float(ev.placement[m, 0]), float(ev.placement[m, 1]))
        cur_proxy = float(ev.current_cost()["proxy"])
        half_w = float(sizes_np[m, 0]) / 2
        half_h = float(sizes_np[m, 1]) / 2

        candidates = _generate_grid_candidates(
            cur_xy, half_w, half_h, cw, ch, n_per_axis, local_radius,
        )
        if len(candidates) == 0:
            continue
        total_candidates += len(candidates)

        best_xy = cur_xy
        best_proxy = cur_proxy
        for (cx, cy) in candidates:
            d = float(ev.delta_cost(m, (cx, cy))["proxy"])
            if d < best_proxy - 1e-7:
                best_proxy = d
                best_xy = (cx, cy)

        if best_xy == cur_xy:
            continue

        ev.move(m, best_xy)
        ovl = int(
            compute_overlap_metrics(
                ev.placement.detach().clone().to(torch.float32), benchmark
            )["overlap_count"]
        )
        if ovl > 0:
            ev.revert()
            n_overlap_rejects += 1
            continue
        # Sanity: actual proxy should match delta_cost prediction within
        # numeric noise.
        actual_proxy = float(ev.current_cost()["proxy"])
        if actual_proxy >= cur_proxy - 1e-7:
            ev.revert()
            n_proxy_rejects += 1
            continue
        n_moves += 1
        total_delta += (actual_proxy - cur_proxy)

    wall = time.perf_counter() - t0
    final_proxy = float(ev.current_cost()["proxy"])
    log(f"  wiremask pass: visits={n_macros_visited} moves={n_moves} "
        f"ovl_rej={n_overlap_rejects} proxy_rej={n_proxy_rejects} "
        f"cands={total_candidates} Δ={final_proxy - init_proxy:+.5f} "
        f"wall={wall:.0f}s")
    return {
        "n_moves": n_moves,
        "n_overlap_rejects": n_overlap_rejects,
        "n_proxy_rejects": n_proxy_rejects,
        "n_macros_visited": n_macros_visited,
        "init_proxy": init_proxy,
        "final_proxy": final_proxy,
        "delta_proxy": final_proxy - init_proxy,
        "wall_seconds": wall,
    }


def wiremask_polish(
    ev: IncrementalProxyEvaluator,
    benchmark,
    movable_idx: List[int],
    *,
    n_per_axis: int = 9,
    local_radius_frac: Optional[float] = 0.30,
    max_passes: int = 5,
    min_improvement: float = 1e-4,
    time_budget_s: Optional[float] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    """Apply WireMask passes until plateau or budget exhausted."""
    if log is None:
        log = lambda s: None
    t0 = time.perf_counter()
    init_proxy = float(ev.current_cost()["proxy"])
    log(f"wiremask_polish: init={init_proxy:.5f} max_passes={max_passes} "
        f"n_per_axis={n_per_axis} local_radius_frac={local_radius_frac} "
        f"budget={time_budget_s}")

    pass_logs = []
    for p in range(max_passes):
        elapsed = time.perf_counter() - t0
        remaining = (time_budget_s - elapsed) if time_budget_s else None
        if remaining is not None and remaining < 30.0:
            log(f"  pass {p+1}: budget exhausted (remaining={remaining:.0f}s)")
            break
        log(f"  pass {p+1}/{max_passes} ...")
        prev_proxy = float(ev.current_cost()["proxy"])
        stats = wiremask_greedy_pass(
            ev, benchmark, movable_idx,
            n_per_axis=n_per_axis,
            local_radius_frac=local_radius_frac,
            time_budget_s=remaining,
            log=log,
        )
        pass_logs.append(stats)
        new_proxy = stats["final_proxy"]
        if prev_proxy - new_proxy < min_improvement:
            log(f"  plateau (Δ={prev_proxy - new_proxy:.5f} "
                f"< {min_improvement}); stopping")
            break

    final_proxy = float(ev.current_cost()["proxy"])
    wall = time.perf_counter() - t0
    log(f"wiremask_polish done: init={init_proxy:.5f} final={final_proxy:.5f} "
        f"Δ={final_proxy - init_proxy:+.5f} ({100*(final_proxy-init_proxy)/init_proxy:+.3f}%) "
        f"passes={len(pass_logs)} wall={wall:.0f}s")
    return {
        "init_proxy": init_proxy,
        "final_proxy": final_proxy,
        "delta_proxy": final_proxy - init_proxy,
        "delta_pct": 100 * (final_proxy - init_proxy) / init_proxy,
        "n_passes": len(pass_logs),
        "pass_logs": pass_logs,
        "wall_seconds": wall,
    }
