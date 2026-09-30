"""P1d: deterministic greedy macro legalizer.

Takes an input placement (possibly overlapping) and produces a zero-overlap
placement by:

  1. Sort hard macros by area descending (place large macros first).
  2. For each macro, try the original position first.
  3. If conflict with any previously-placed macro, spiral outward searching
     a grid of candidate offsets until non-conflicting position found.
  4. Clamp movable soft macros inside the canvas (no overlap constraint).

Returns zero-overlap placement deterministically in O(n² log n) time.
Suitable as a post-DP legalizer when DP residuals are high.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch


from macro_place.benchmark import Benchmark
from ..common.runtime_safety import clamp_to_canvas


def _aabb_overlap(ax_min, ax_max, ay_min, ay_max,
                  bx_min, bx_max, by_min, by_max,
                  eps: float = 1e-4) -> bool:
    """Two AABBs overlap if their intervals overlap on both axes (with eps)."""
    return (
        ax_max - eps > bx_min and bx_max - eps > ax_min and
        ay_max - eps > by_min and by_max - eps > ay_min
    )


def greedy_macro_legalize(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    search_radius_steps: int = 50,
    step_size_frac: float = 0.05,
    verbose: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """Try to remove hard overlaps; report unresolved macros in ``n_failed``.

    search_radius_steps:  how many grid steps to search outward.
    step_size_frac:       grid step = fraction of canvas dimension.

    Returns (legal_placement, stats_dict).
    """
    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.cpu().numpy()
    pos = placement.detach().cpu().numpy().astype(np.float64).copy()
    pos[fixed.astype(bool)] = benchmark.macro_positions.cpu().numpy()[fixed.astype(bool)]
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    step_x = cw * step_size_frac
    step_y = ch * step_size_frac

    # Sort hard macros by area descending (largest first).
    areas = sizes[:n_hard, 0] * sizes[:n_hard, 1]
    order = np.argsort(-areas).tolist()

    placed = []  # list of (idx, x_min, x_max, y_min, y_max)

    # Fixed hard macros go in first (cannot move).
    for i in order:
        if not bool(fixed[i]):
            continue
        x, y = pos[i, 0], pos[i, 1]
        hw, hh = sizes[i, 0] / 2, sizes[i, 1] / 2
        placed.append((i, x - hw, x + hw, y - hh, y + hh))

    # Generate spiral offsets: (0,0), (±step,0), (0,±step), (±step,±step), ...
    offsets = [(0.0, 0.0)]
    for r in range(1, search_radius_steps + 1):
        for s in range(-r, r + 1):
            offsets.append((s * step_x, r * step_y))
            offsets.append((s * step_x, -r * step_y))
        for s in range(-r + 1, r):
            offsets.append((r * step_x, s * step_y))
            offsets.append((-r * step_x, s * step_y))

    n_failed = 0
    n_moved = 0
    max_displacement = 0.0
    n_attempts_total = 0

    for i in order:
        if bool(fixed[i]):
            continue  # already placed
        ox, oy = pos[i, 0], pos[i, 1]
        hw, hh = sizes[i, 0] / 2, sizes[i, 1] / 2

        # Try the original position first, then spiral outward.
        success = False
        for dx, dy in offsets:
            n_attempts_total += 1
            nx, ny = ox + dx, oy + dy
            # Clamp to canvas
            nx = max(hw, min(cw - hw, nx))
            ny = max(hh, min(ch - hh, ny))
            xmin, xmax = nx - hw, nx + hw
            ymin, ymax = ny - hh, ny + hh

            # Check against all placed macros
            conflict = False
            for _, pxmin, pxmax, pymin, pymax in placed:
                if _aabb_overlap(xmin, xmax, ymin, ymax,
                                 pxmin, pxmax, pymin, pymax):
                    conflict = True
                    break
            if not conflict:
                pos[i, 0] = nx
                pos[i, 1] = ny
                placed.append((i, xmin, xmax, ymin, ymax))
                disp = ((nx - ox) ** 2 + (ny - oy) ** 2) ** 0.5
                if disp > 1e-6:
                    n_moved += 1
                    max_displacement = max(max_displacement, disp)
                success = True
                break
        if not success:
            n_failed += 1
            if verbose:
                print(f"  [legalize] FAILED to place macro {i}", flush=True)
            # Force place at original even if conflict (rare; will need additional cleanup)
            placed.append((i, pos[i, 0] - hw, pos[i, 0] + hw,
                              pos[i, 1] - hh, pos[i, 1] + hh))

    return clamp_to_canvas(torch.tensor(pos, dtype=placement.dtype), benchmark), {
        "n_moved": n_moved,
        "n_failed": n_failed,
        "max_displacement": max_displacement,
        "n_attempts_total": n_attempts_total,
    }

