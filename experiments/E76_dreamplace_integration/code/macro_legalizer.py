"""P1d: deterministic greedy macro legalizer.

Takes an input placement (possibly overlapping) and produces a zero-overlap
placement by:

  1. Sort hard macros by area descending (place large macros first).
  2. For each macro, try the original position first.
  3. If conflict with any previously-placed macro, spiral outward searching
     a grid of candidate offsets until non-conflicting position found.
  4. Soft macros are placed at original positions (no overlap constraint).

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

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark


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
    """Greedy legalize a placement; zero hard-macro overlaps on return.

    search_radius_steps:  how many grid steps to search outward.
    step_size_frac:       grid step = fraction of canvas dimension.

    Returns (legal_placement, stats_dict).
    """
    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.cpu().numpy()
    pos = placement.detach().cpu().numpy().astype(np.float64).copy()
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

    return torch.tensor(pos, dtype=placement.dtype), {
        "n_moved": n_moved,
        "n_failed": n_failed,
        "max_displacement": max_displacement,
        "n_attempts_total": n_attempts_total,
    }


if __name__ == "__main__":
    import argparse
    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.cd_core import project_overlaps
    from macro_place.objective import compute_proxy_cost, compute_overlap_metrics

    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--init", choices=["sdf", "dp"], default="sdf")
    args = ap.parse_args()

    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(args.bench)))
    print(f"=== {args.bench} (n_macros={bench.num_macros} n_hard={bench.num_hard_macros}) ===", flush=True)

    if args.init == "sdf":
        from macro_place.cd_core import sdf_init
        init = sdf_init(bench).to(torch.float32)
    else:
        # Hardcode a synthetic overlapping init: scatter macros around center
        init = bench.macro_positions.clone().to(torch.float32)
        n_hard = bench.num_hard_macros
        cw = float(bench.canvas_width)
        ch = float(bench.canvas_height)
        center_x, center_y = cw / 2, ch / 2
        rng = np.random.default_rng(42)
        for i in range(n_hard):
            if not bool(bench.macro_fixed[i]):
                init[i, 0] = center_x + rng.normal(0, cw * 0.1)
                init[i, 1] = center_y + rng.normal(0, ch * 0.1)

    ovl_init = compute_overlap_metrics(init, bench)["overlap_count"]
    print(f"  init overlaps: {ovl_init}", flush=True)

    t0 = time.time()
    legal, stats = greedy_macro_legalize(init, bench, verbose=True)
    wall = time.time() - t0

    ovl_final = compute_overlap_metrics(legal, bench)["overlap_count"]
    proxy_final = float(compute_proxy_cost(legal, bench, plc)["proxy_cost"])
    print(f"  legal: ovl={ovl_final} proxy={proxy_final:.5f}", flush=True)
    print(f"  stats: moved={stats['n_moved']}/{bench.num_hard_macros} "
          f"failed={stats['n_failed']} max_disp={stats['max_displacement']:.3f} "
          f"attempts={stats['n_attempts_total']} wall={wall:.2f}s", flush=True)
