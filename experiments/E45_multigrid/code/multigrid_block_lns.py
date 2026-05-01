"""E45 multigrid — Phase 2 (revised): super-macro block LNS.

Multi-macro move type that exploits the super-macro structure from
Phase 1. For each super-macro (a cluster of 30-100 hard macros from
pymetis partitioning), try translating ALL constituents as a block to
various target centroids. Accept if proxy improves; revert otherwise.

This is the post-smoother that K=3 K-joint cannot reach: single-, 2-,
and 3-macro mechanisms are exhausted at E41's saturation floor (1.0848
on --all). Block moves of 30-100 macros simultaneously explore a
fundamentally new reachable set — moving aggregate mass to escape the
basin choice that DPO/SDF init imposed.

Public API:
- `run_super_block_lns(evaluator, benchmark, plc, super_members,
   time_budget_s, top_M, seed, log_fn) -> dict`
   Operates in-place on `evaluator`. Returns stats.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics


def _block_overlap_with_outside(
    candidate_positions: List[Tuple[float, float]],
    member_idxs: List[int],
    placement: torch.Tensor,
    macro_sizes_np: np.ndarray,
    n_hard: int,
    eps: float = 1e-9,
) -> bool:
    """Check whether the proposed block placement (each member at its
    candidate position) overlaps any macro OUTSIDE the block.

    Same eps direction as E39's post-fix (eps=1e-9 strict separation).
    Inside-block overlaps are NOT checked here — they're the same as
    the original SDF/post-pipeline placement, which was overlap-free
    by construction. The block move is a rigid translation, so
    intra-block geometry is preserved.
    """
    member_set = set(member_idxs)
    pos_np = placement[:n_hard].cpu().numpy().astype(np.float64)
    sz = macro_sizes_np[:n_hard]
    for i_local, m in enumerate(member_idxs):
        if m >= n_hard:
            continue
        nx, ny = candidate_positions[i_local]
        half_w = sz[m, 0] / 2.0
        half_h = sz[m, 1] / 2.0
        for j in range(n_hard):
            if j in member_set or j == m:
                continue
            half_w_j = sz[j, 0] / 2.0
            half_h_j = sz[j, 1] / 2.0
            if (
                abs(nx - pos_np[j, 0]) < (half_w + half_w_j) + eps
                and abs(ny - pos_np[j, 1]) < (half_h + half_h_j) + eps
            ):
                return True
    return False


def run_super_block_lns(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    super_members: List[List[int]],
    time_budget_s: float,
    top_M: int = 8,
    seed: int = 42,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Super-macro block-LNS phase. Iterates through non-empty super-macros;
    for each one, tries M×M target centroids on a canvas grid; commits
    the best-improving rigid translation of all constituents.

    Stops when budget is exhausted OR a full pass produces no commit.
    """
    rng = np.random.default_rng(seed=seed)
    macro_sizes_np = evaluator.macro_sizes.cpu().numpy()
    n_hard = benchmark.num_hard_macros
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    fixed_np = benchmark.macro_fixed.cpu().numpy()

    # Filter super-macros: only non-empty, only those with at least one
    # movable (non-fixed) constituent.
    candidate_supers: List[Tuple[int, List[int]]] = []
    for super_idx, members in enumerate(super_members):
        if not members:
            continue
        movable = [m for m in members if not fixed_np[m] and m < n_hard]
        if not movable:
            continue
        candidate_supers.append((super_idx, movable))
    if log_fn is not None:
        log_fn(
            f"  super-block-LNS budget={time_budget_s:.0f}s, top_M={top_M}, "
            f"|super_macros|={len(candidate_supers)}, seed={seed}"
        )

    # Per-super-macro candidate grids are computed inside the loop —
    # centered on current centroid with range bounded by feasibility
    # (so the block never leaves the canvas). An absolute grid over
    # [0, canvas] would reject most candidates as out-of-bounds.

    t_start = time.perf_counter()
    blocks_tried = 0
    blocks_committed = 0
    total_improvement = 0.0
    pass_idx = 0

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:
            break
        pass_idx += 1
        pass_t0 = time.perf_counter()
        pass_committed = 0
        pass_delta = 0.0
        # Random shuffle of super-macros each pass for fairness.
        order = list(range(len(candidate_supers)))
        rng.shuffle(order)

        for ord_i in order:
            if time.perf_counter() - t_start >= time_budget_s:
                break
            super_idx, movable = candidate_supers[ord_i]
            blocks_tried += 1

            # Save baseline positions of the block.
            saved_xy = [
                (
                    float(evaluator.placement[m, 0]),
                    float(evaluator.placement[m, 1]),
                )
                for m in movable
            ]
            baseline_proxy = evaluator.current_cost()["proxy"]
            baseline_centroid_x = float(np.mean([sxy[0] for sxy in saved_xy]))
            baseline_centroid_y = float(np.mean([sxy[1] for sxy in saved_xy]))

            best_target: Optional[Tuple[float, float]] = None
            best_proxy = baseline_proxy

            # Compute per-block feasible translation range.
            block_xs = [s[0] for s in saved_xy]
            block_ys = [s[1] for s in saved_xy]
            block_half_w = [float(macro_sizes_np[m, 0]) / 2.0 for m in movable]
            block_half_h = [float(macro_sizes_np[m, 1]) / 2.0 for m in movable]
            # Per-member feasible Δ: [hw - x, canvas_w - hw - x] for x-axis,
            # similarly for y. Block-wide feasible Δ is the intersection.
            dx_min = max(
                bhw - bx for bx, bhw in zip(block_xs, block_half_w)
            )
            dx_max = min(
                canvas_w - bhw - bx for bx, bhw in zip(block_xs, block_half_w)
            )
            dy_min = max(
                bhh - by for by, bhh in zip(block_ys, block_half_h)
            )
            dy_max = min(
                canvas_h - bhh - by for by, bhh in zip(block_ys, block_half_h)
            )
            # If the block is already at canvas boundary on every side,
            # dx_min > dx_max — no movement possible.
            if dx_min > dx_max or dy_min > dy_max:
                continue

            grid_dx = np.linspace(dx_min, dx_max, top_M)
            grid_dy = np.linspace(dy_min, dy_max, top_M)
            grid_x = baseline_centroid_x + grid_dx
            grid_y = baseline_centroid_y + grid_dy

            for tx in grid_x:
                for ty in grid_y:
                    if time.perf_counter() - t_start >= time_budget_s:
                        break
                    dx = float(tx) - baseline_centroid_x
                    dy = float(ty) - baseline_centroid_y
                    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                        continue
                    # Compute proposed positions; check in-canvas.
                    proposed_xy: List[Tuple[float, float]] = []
                    in_bounds = True
                    for i_local, m in enumerate(movable):
                        nx = saved_xy[i_local][0] + dx
                        ny = saved_xy[i_local][1] + dy
                        half_w = float(macro_sizes_np[m, 0]) / 2.0
                        half_h = float(macro_sizes_np[m, 1]) / 2.0
                        if (
                            nx < half_w
                            or nx > canvas_w - half_w
                            or ny < half_h
                            or ny > canvas_h - half_h
                        ):
                            in_bounds = False
                            break
                        proposed_xy.append((nx, ny))
                    if not in_bounds:
                        continue

                    # Check overlap with non-cluster macros.
                    if _block_overlap_with_outside(
                        proposed_xy,
                        movable,
                        evaluator.placement,
                        macro_sizes_np,
                        n_hard,
                    ):
                        continue

                    # Apply moves; evaluate; revert.
                    applied: List[int] = []
                    apply_ok = True
                    for i_local, m in enumerate(movable):
                        try:
                            evaluator.move(m, proposed_xy[i_local])
                            applied.append(m)
                        except Exception:
                            apply_ok = False
                            break
                    if apply_ok:
                        new_proxy = evaluator.current_cost()["proxy"]
                        if new_proxy < best_proxy - 1e-9:
                            best_proxy = new_proxy
                            best_target = (float(tx), float(ty))
                    # Revert in reverse order.
                    for j in range(len(applied) - 1, -1, -1):
                        m = applied[j]
                        sx, sy = saved_xy[movable.index(m)]
                        cx = float(evaluator.placement[m, 0])
                        cy = float(evaluator.placement[m, 1])
                        if abs(sx - cx) > 1e-9 or abs(sy - cy) > 1e-9:
                            evaluator.move(m, (sx, sy))

            # Commit the best target if it improves.
            if best_target is not None and best_proxy < baseline_proxy - 1e-7:
                tx, ty = best_target
                dx = tx - baseline_centroid_x
                dy = ty - baseline_centroid_y
                committed_moves: List[Tuple[int, Tuple[float, float]]] = []
                for i_local, m in enumerate(movable):
                    nx = saved_xy[i_local][0] + dx
                    ny = saved_xy[i_local][1] + dy
                    cx = float(evaluator.placement[m, 0])
                    cy = float(evaluator.placement[m, 1])
                    if abs(nx - cx) > 1e-9 or abs(ny - cy) > 1e-9:
                        committed_moves.append((m, (cx, cy)))
                        evaluator.move(m, (nx, ny))

                # Defensive overlap validation; revert if any positive
                # overlap leaked through.
                ov = compute_overlap_metrics(evaluator.placement, benchmark)
                if ov["overlap_count"] > 0:
                    if log_fn is not None:
                        log_fn(
                            f"  super-block-LNS: REVERT super_idx={super_idx} — "
                            f"compute_overlap_metrics found {ov['overlap_count']} "
                            f"overlap(s); reverting {len(committed_moves)} moves"
                        )
                    for j in range(len(committed_moves) - 1, -1, -1):
                        m, (sx, sy) = committed_moves[j]
                        evaluator.move(m, (sx, sy))
                else:
                    delta = best_proxy - baseline_proxy
                    pass_delta += delta
                    total_improvement += delta
                    pass_committed += 1
                    blocks_committed += 1

        pass_wall = time.perf_counter() - pass_t0
        cur_proxy = evaluator.current_cost()["proxy"]
        if log_fn is not None:
            log_fn(
                f"  super-block-LNS pass {pass_idx}: blocks_tried={len(order)}, "
                f"committed={pass_committed}, Δ={pass_delta:+.5f}, "
                f"proxy={cur_proxy:.5f}, pass_wall={pass_wall:.1f}s, "
                f"elapsed={time.perf_counter() - t_start:.1f}s"
            )

        if pass_committed == 0:
            if log_fn is not None:
                log_fn(
                    f"  super-block-LNS converged at pass {pass_idx} (no improvement)"
                )
            break

    return {
        "blocks_tried": blocks_tried,
        "blocks_committed": blocks_committed,
        "passes": pass_idx,
        "total_improvement": total_improvement,
        "wall_total_s": time.perf_counter() - t_start,
    }
