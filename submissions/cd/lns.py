"""
LNS rip-up-and-reinsert module — experiment E3.

CDOnly plateaus on hard benchmarks (ibm14/12/17/18 stuck around 1.24-1.39)
because it moves one macro at a time. A tight cluster of macros that is
collectively in a bad region cannot escape: each one's neighbors block it.
LNS attacks this by ripping up *k* macros and reinserting them, one at a
time but in cost-descending order, doing a full-canvas grid search per
re-inserted macro to find the proxy-minimizing legal slot.

Two primitives, one driver:

  - ``select_destroy_set(evaluator, benchmark, k, strategy)``
        Pick the *k* macro indices to rip up. Movable only — never returns
        a fixed macro. Default strategy 'cost' uses a cheap proxy (sum of
        incident-net WL bbox spans + grid-cell density at the macro's
        current location). 'spatial' picks all movable macros in the most
        congested grid cell and its 8-neighbors, capped at k.

  - ``reinsert_one(evaluator, benchmark, plc, macro_idx, grid_lines_x,
                    grid_lines_y)``
        For one macro, sweep every grid bin center as a candidate position.
        Skip illegal candidates via ``legal_axis_range`` per axis. Probe
        cost via ``evaluator.move(...)/.revert()``. Commit the best. The
        evaluator is left holding the best-found position (or current
        position, if nothing improved).

  - ``run_lns(evaluator, benchmark, plc, movable, time_budget_s, k_schedule)``
        Outer loop: snapshot current cost + every macro's position, pick
        k from schedule, select destroy set, reinsert each in cost-
        descending order. After all reinsertions, accept iff total proxy
        improves; else rewind every moved macro to its pre-iteration
        position. Loop until time budget exhausted.

Critical correctness:
  - Fixed macros never moved (filtered from movable + destroy_set).
  - Reinsert candidates checked for legality on BOTH axes via
    ``legal_axis_range``. Illegal candidates skipped.
  - Revert path uses direct ``evaluator.move(idx, original_xy)`` for each
    macro. The evaluator's snapshot stack is single-step and we move many
    macros, so we cannot use ``revert()``.

Performance:
  - One reinsert = grid_col × grid_row candidates, each ~6.5ms on ibm10.
    For a 32x32 grid that's ~6.7s per macro reinsertion — fine for
    k=5..30 with a 600s budget.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.incremental_evaluator import IncrementalProxyEvaluator

# The evaluate harness loads placers via importlib.spec_from_file_location, so
# the repo root is not on sys.path. Add it so `submissions.cd.cd_only_placer`
# resolves as an implicit namespace package.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Re-import the diagnostic so callers needn't bother. cd_only_placer also
# re-uses these — same module instance via sys.modules cache.
from submissions.cd.cd_only_placer import legal_axis_range  # noqa: F401,E402


# ── Destroy-set selection ────────────────────────────────────────────────────


def _macro_cost_score(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    macro_idx: int,
) -> float:
    """Cheap per-macro cost-contribution proxy.

    Sum of incident-net WL bbox spans (weighted by net weight) + a density
    contribution proxy based on the macro's current grid cell occupancy.

    This is intentionally NOT a true cost-delta: probing every macro's true
    delta would be O(M * O(degree) * 2) ≈ a full sweep, which defeats the
    point of a *fast* destroy selector. The score is monotone-ish with the
    delta so the ranking is good enough for a destroy heuristic.
    """
    nets = evaluator.macro_to_nets[macro_idx]
    wl_score = 0.0
    for n in nets.tolist():
        span = (
            float(evaluator.net_max_x[n] - evaluator.net_min_x[n])
            + float(evaluator.net_max_y[n] - evaluator.net_min_y[n])
        )
        w = float(evaluator.net_weight[n])
        wl_score += w * span

    # Density score: the macro's own area divided across its current cells.
    # Use the cached macro_density_contrib dict.
    contrib = evaluator.macro_density_contrib[macro_idx] or {}
    if contrib:
        # Penalize macros sitting in cells that are already over-occupied
        # (i.e. where total occupied / cell area is high).
        cell_area = evaluator.grid_area
        density_score = 0.0
        for cell, area in contrib.items():
            occ = float(evaluator.grid_occupied[cell])
            density_score += area * (occ / cell_area)
    else:
        density_score = 0.0

    # Weighted sum — wl-dominated (matches proxy weights wl=1.0, density=0.5).
    return wl_score + 0.5 * density_score


def _spatial_destroy_set(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    movable: Sequence[int],
    k: int,
) -> List[int]:
    """Spatial cluster: pick movable macros in most-congested cell + 8 neighbors."""
    # Find most-congested cell by V_net + H_net (raw, un-smoothed; cheap)
    combined = (evaluator.V_net_cong + evaluator.H_net_cong).cpu().numpy()
    if combined.size == 0:
        return list(movable)[:k]
    flat_idx = int(np.argmax(combined))
    grid_col = evaluator.grid_col
    grid_row = evaluator.grid_row
    r = flat_idx // grid_col
    c = flat_idx % grid_col

    gw = evaluator.grid_width
    gh = evaluator.grid_height

    # Cell window [r-1..r+1] x [c-1..c+1]
    r_lo = max(0, r - 1)
    r_hi = min(grid_row - 1, r + 1)
    c_lo = max(0, c - 1)
    c_hi = min(grid_col - 1, c + 1)
    x_lo = c_lo * gw
    x_hi = (c_hi + 1) * gw
    y_lo = r_lo * gh
    y_hi = (r_hi + 1) * gh

    movable_set = set(movable)
    in_window: List[Tuple[int, float]] = []
    for i in movable_set:
        xi = float(evaluator.placement[i, 0])
        yi = float(evaluator.placement[i, 1])
        if x_lo <= xi <= x_hi and y_lo <= yi <= y_hi:
            score = _macro_cost_score(evaluator, benchmark, i)
            in_window.append((i, score))

    if not in_window:
        # Fall back to global cost ranking
        scores = [
            (i, _macro_cost_score(evaluator, benchmark, i)) for i in movable
        ]
        scores.sort(key=lambda t: -t[1])
        return [i for i, _ in scores[:k]]

    in_window.sort(key=lambda t: -t[1])
    return [i for i, _ in in_window[:k]]


def select_destroy_set(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    k: int,
    strategy: str = "cost",
) -> List[int]:
    """Return up to ``k`` movable-macro indices to rip up.

    Args:
        evaluator: live IncrementalProxyEvaluator.
        benchmark: Benchmark — used for ``macro_fixed`` filter.
        k: max destroy-set size.
        strategy: 'cost' (default) — rank movable macros by per-macro cost
            contribution. 'spatial' — pick movable macros in most-congested
            grid cell + 8-neighbor window, capped at k.
    """
    fixed = benchmark.macro_fixed
    movable = [i for i in range(benchmark.num_macros) if not bool(fixed[i])]
    if k <= 0 or not movable:
        return []

    if strategy == "cost":
        scored = [
            (i, _macro_cost_score(evaluator, benchmark, i)) for i in movable
        ]
        scored.sort(key=lambda t: -t[1])
        return [i for i, _ in scored[:k]]
    elif strategy == "spatial":
        return _spatial_destroy_set(evaluator, benchmark, movable, k)
    else:
        raise ValueError(
            f"Unknown destroy strategy '{strategy}' "
            f"(supported: 'cost', 'spatial')"
        )


# ── Reinsert ────────────────────────────────────────────────────────────────


def reinsert_one(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    macro_idx: int,
    grid_lines_x: np.ndarray,
    grid_lines_y: np.ndarray,
    window_radius: int = 5,
) -> Tuple[float, bool]:
    """Search every legal grid-bin center for the best position of one macro.

    Walks a (2·window_radius+1) x (2·window_radius+1) window of grid bin
    centers around the macro's current grid cell — clamped to the canvas.
    For each candidate, checks legality via ``legal_axis_range`` on each
    axis: the candidate is legal iff the candidate x is inside the macro's
    legal x range AND the candidate y is inside the macro's legal y range,
    both computed at the candidate (other-axis) coordinate.

    For each legal candidate, ``evaluator.move(...)`` is called, the proxy
    cost is read, and ``evaluator.revert()`` rolls back. The single best
    position is committed at the end via ``evaluator.move(...)``.

    The local window (default ±5 cells = 11x11 = up to 121 candidates,
    clamped at edges) replaces the previous full-canvas grid sweep, which
    was O(grid_col * grid_row) per reinsert and unworkable on dense grids
    (e.g. 51x44=2244 candidates on ibm17). Long-range relocations are left
    to the CD sweeps.

    Returns (best_proxy, moved) where ``moved=True`` iff the best candidate
    differs from the macro's current position.
    """
    # Grid bin centers
    grid_col = int(plc.grid_col)
    grid_row = int(plc.grid_row)
    gw = float(plc.width / plc.grid_col)
    gh = float(plc.height / plc.grid_row)

    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    half_w = float(evaluator.macro_sizes[macro_idx, 0]) / 2
    half_h = float(evaluator.macro_sizes[macro_idx, 1]) / 2

    cur_x = float(evaluator.placement[macro_idx, 0])
    cur_y = float(evaluator.placement[macro_idx, 1])
    cur_cost = evaluator.current_cost()["proxy"]
    best_cost = cur_cost
    best_xy: Optional[Tuple[float, float]] = None

    # Local window around the macro's current grid cell. Clamp to grid bounds
    # so edge macros simply get a smaller (asymmetric) window — the legality
    # probe still handles canvas-edge cases correctly via legal_axis_range.
    cur_col = int(np.clip(cur_x / gw, 0, grid_col - 1))
    cur_row = int(np.clip(cur_y / gh, 0, grid_row - 1))

    col_lo = max(0, cur_col - window_radius)
    col_hi = min(grid_col - 1, cur_col + window_radius)
    row_lo = max(0, cur_row - window_radius)
    row_hi = min(grid_row - 1, cur_row + window_radius)

    # Precompute an array of bin-center coordinates inside the window.
    cx_arr = (np.arange(col_lo, col_hi + 1, dtype=np.float64) + 0.5) * gw
    cy_arr = (np.arange(row_lo, row_hi + 1, dtype=np.float64) + 0.5) * gh

    for cy in cy_arr:
        # Y-bound prefilter (canvas)
        if cy < half_h or cy > ch - half_h:
            continue
        # Probe legal x range at this y. We need to TEMPORARILY set the
        # macro's y to cy so legal_axis_range uses the right perp slice.
        # Simplest: just check x-range at y=cy by spoofing: legal_axis_range
        # reads placement[macro_idx, other_axis], so we have to update it
        # transiently. To avoid evaluator state churn, we save+restore raw
        # placement[].
        saved_y = float(evaluator.placement[macro_idx, 1])
        evaluator.placement[macro_idx, 1] = cy
        try:
            lo_x, hi_x = legal_axis_range(
                macro_idx, evaluator.placement, evaluator.macro_sizes,
                benchmark.macro_fixed, n_hard, axis=0,
                canvas_w=cw, canvas_h=ch,
            )
        finally:
            evaluator.placement[macro_idx, 1] = saved_y
        if hi_x - lo_x < 1e-7:
            continue
        for cx in cx_arr:
            if cx < lo_x or cx > hi_x:
                continue
            # Check Y-axis legality at candidate x as well (asymmetric
            # geometry: the y-range depends on the x we're probing).
            saved_x = float(evaluator.placement[macro_idx, 0])
            evaluator.placement[macro_idx, 0] = cx
            try:
                lo_y, hi_y = legal_axis_range(
                    macro_idx, evaluator.placement, evaluator.macro_sizes,
                    benchmark.macro_fixed, n_hard, axis=1,
                    canvas_w=cw, canvas_h=ch,
                )
            finally:
                evaluator.placement[macro_idx, 0] = saved_x
            if cy < lo_y - 1e-7 or cy > hi_y + 1e-7:
                continue
            # Legal — probe the proxy cost via move/revert
            evaluator.move(macro_idx, (float(cx), float(cy)))
            c = evaluator.current_cost()["proxy"]
            evaluator.revert()
            if c < best_cost - 1e-9:
                best_cost = c
                best_xy = (float(cx), float(cy))

    # Commit: move to best_xy if it improved; else stay (no-op).
    if best_xy is not None and (
        abs(best_xy[0] - cur_x) > 1e-7 or abs(best_xy[1] - cur_y) > 1e-7
    ):
        evaluator.move(macro_idx, best_xy)
        return best_cost, True
    return cur_cost, False


# ── Outer LNS driver ────────────────────────────────────────────────────────


def run_lns(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    movable: List[int],
    time_budget_s: float,
    k_schedule: Sequence[int] = (5, 10, 20, 30),
    log_fn: Optional[Callable[[str], None]] = None,
    strategy: str = "cost",
    window_radius: int = 5,
) -> Dict:
    """Outer LNS loop: rip up *k* macros, reinsert in cost-descending order,
    accept iff total proxy improves; else revert all moves.

    Args:
        evaluator: live IncrementalProxyEvaluator (already CD-converged).
        benchmark: benchmark dataclass (canvas, fixed mask).
        plc: PlacementCost (grid geometry).
        movable: pre-filtered list of movable macro indices.
        time_budget_s: wall clock budget.
        k_schedule: cycle of destroy-set sizes (default (5, 10, 20, 30)).
        log_fn: optional print-like callable; called once per iteration.
        strategy: destroy-strategy passed to ``select_destroy_set``.
        window_radius: local-window radius (in grid cells) passed to
            ``reinsert_one``. Default 5 → up to 11x11 = 121 candidates per
            reinsert (clamped at canvas edges).

    Returns:
        dict with iterations, accepts, total_moves, total_reinserts, wall_total_s.
    """
    if not k_schedule:
        raise ValueError("k_schedule must be non-empty")
    if not movable:
        return {
            "iterations": 0,
            "accepts": 0,
            "rejects": 0,
            "total_moves": 0,
            "total_reinserts": 0,
            "wall_total_s": 0.0,
        }

    # Precompute grid lines (used as input to reinsert_one for parity with
    # diagnostic helpers; reinsert_one itself walks bin centers, not lines).
    gw = float(plc.width / plc.grid_col)
    gh = float(plc.height / plc.grid_row)
    grid_lines_x = np.arange(plc.grid_col + 1, dtype=np.float64) * gw
    grid_lines_y = np.arange(plc.grid_row + 1, dtype=np.float64) * gh

    iter_idx = 0
    accepts = 0
    rejects = 0
    total_moves = 0
    total_reinserts = 0
    schedule_pos = 0

    t_start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:
            break
        iter_idx += 1
        k = int(k_schedule[schedule_pos % len(k_schedule)])
        schedule_pos += 1

        # Snapshot every macro's pre-iteration position (for revert path).
        # We snapshot ALL macros (cheap: O(num_macros) floats) so the revert
        # path is uniform and bug-resistant — we don't need to remember which
        # macros were touched if we just restore everything.
        snap_positions = evaluator.placement.detach().clone()
        snap_cost = evaluator.current_cost()["proxy"]

        # Pick destroy set
        destroy_set = select_destroy_set(
            evaluator, benchmark, k=k, strategy=strategy
        )
        if not destroy_set:
            break

        # Process in cost-descending order. The selector for 'cost' already
        # returned them in that order; for 'spatial' it also sorts within the
        # window. Re-sort defensively.
        scored = [
            (i, _macro_cost_score(evaluator, benchmark, i))
            for i in destroy_set
        ]
        scored.sort(key=lambda t: -t[1])
        ordered = [i for i, _ in scored]

        iter_t0 = time.perf_counter()
        iter_moves = 0
        iter_reinserts = 0
        for macro_idx in ordered:
            if time.perf_counter() - t_start >= time_budget_s:
                break
            _best_cost, moved = reinsert_one(
                evaluator, benchmark, plc, macro_idx,
                grid_lines_x, grid_lines_y,
                window_radius=window_radius,
            )
            iter_reinserts += 1
            if moved:
                iter_moves += 1

        # Compare final cost to snapshot. Accept iff strictly improved.
        new_cost = evaluator.current_cost()["proxy"]
        accepted = new_cost < snap_cost - 1e-9
        if accepted:
            accepts += 1
            total_moves += iter_moves
        else:
            rejects += 1
            # Revert every macro to its pre-iteration position. We use direct
            # evaluator.move(idx, old_xy) per macro — the snapshot is a single-
            # step undo, but we may have done many moves. Iterate ALL macros
            # whose snapshot positions differ from current (movable only —
            # fixed should be identical by construction).
            cur_positions = evaluator.placement
            for i in movable:
                old_x = float(snap_positions[i, 0])
                old_y = float(snap_positions[i, 1])
                cx = float(cur_positions[i, 0])
                cy = float(cur_positions[i, 1])
                if abs(old_x - cx) > 1e-9 or abs(old_y - cy) > 1e-9:
                    evaluator.move(i, (old_x, old_y))

        total_reinserts += iter_reinserts

        iter_wall = time.perf_counter() - iter_t0
        elapsed = time.perf_counter() - t_start
        if log_fn is not None:
            cost_break = evaluator.current_cost()
            log_fn(
                f"  lns iter {iter_idx:3d}  k={k:3d}  "
                f"elapsed={elapsed:7.1f}s  "
                f"iter_t={iter_wall:6.1f}s  "
                f"snap={snap_cost:.5f} -> new={new_cost:.5f}  "
                f"{'ACCEPT' if accepted else 'reject'}  "
                f"reinserts={iter_reinserts}  moves={iter_moves}  "
                f"[wl={cost_break['wl']:.4f} d={cost_break['density']:.4f} "
                f"c={cost_break['congestion']:.4f}]"
            )

    return {
        "iterations": iter_idx,
        "accepts": accepts,
        "rejects": rejects,
        "total_moves": total_moves,
        "total_reinserts": total_reinserts,
        "wall_total_s": time.perf_counter() - t_start,
    }


# ── Public wrapper class ────────────────────────────────────────────────────


class LNSDestroyAndReinsert:
    """Stateless façade over the LNS primitives.

    Held by ``CDLNSPlacer``; mainly for API hygiene and logging glue. The
    actual algorithm lives in the module-level functions above so they can
    be tested + reused independently.

    Usage::

        lns = LNSDestroyAndReinsert(k_schedule=(5, 10, 20, 30))
        stats = lns.run(evaluator, benchmark, plc, movable,
                        time_budget_s=600.0, log_fn=print)
    """

    def __init__(
        self,
        k_schedule: Sequence[int] = (5, 10, 20, 30),
        strategy: str = "cost",
        window_radius: int = 5,
    ) -> None:
        if not k_schedule:
            raise ValueError("k_schedule must be non-empty")
        self.k_schedule: Tuple[int, ...] = tuple(int(k) for k in k_schedule)
        if any(k <= 0 for k in self.k_schedule):
            raise ValueError(
                f"k_schedule must have all-positive entries (got {self.k_schedule})"
            )
        self.strategy = strategy
        if window_radius < 1:
            raise ValueError(
                f"window_radius must be >= 1 (got {window_radius})"
            )
        self.window_radius = int(window_radius)

    # Forward to the module-level functions for symmetry / discoverability.

    def select_destroy_set(
        self,
        evaluator: IncrementalProxyEvaluator,
        benchmark: Benchmark,
        k: int,
        strategy: Optional[str] = None,
    ) -> List[int]:
        return select_destroy_set(
            evaluator, benchmark, k=k, strategy=strategy or self.strategy,
        )

    def reinsert_one(
        self,
        evaluator: IncrementalProxyEvaluator,
        benchmark: Benchmark,
        plc,
        macro_idx: int,
        grid_lines_x: np.ndarray,
        grid_lines_y: np.ndarray,
    ) -> Tuple[float, bool]:
        return reinsert_one(
            evaluator, benchmark, plc, macro_idx,
            grid_lines_x, grid_lines_y,
            window_radius=self.window_radius,
        )

    def run(
        self,
        evaluator: IncrementalProxyEvaluator,
        benchmark: Benchmark,
        plc,
        movable: List[int],
        time_budget_s: float,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> Dict:
        return run_lns(
            evaluator, benchmark, plc, movable,
            time_budget_s=time_budget_s,
            k_schedule=self.k_schedule,
            log_fn=log_fn,
            strategy=self.strategy,
            window_radius=self.window_radius,
        )
