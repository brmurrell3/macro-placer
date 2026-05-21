"""E156: Force-directed (FD) macro legalizer.

Drop-in replacement for the E76 greedy spiral-search `macro_legalizer`.
Mechanism: treat hard macros as repulsive nodes; for each overlapping pair
apply a repulsive force proportional to penetration depth (the Minimum
Translation Vector / MTV approach); iterate until no overlaps (or max_iters).

Differences from greedy:
  - All macros move simultaneously (not one-at-a-time as in greedy).
  - Each pair contributes equal-and-opposite shifts along the MTV axis.
  - Multiple overlapping neighbours' contributions sum per-macro.
  - Fixed macros never move; movable macros in contact with a fixed macro
    absorb the full per-pair displacement (not split).
  - Soft macros are ignored entirely (no constraint in our problem).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark


def _count_overlaps_np(
    xc: np.ndarray,
    yc: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    eps: float,
) -> int:
    """Count strictly-overlapping hard-macro pairs (upper-triangle)."""
    dx = np.abs(xc[:, None] - xc[None, :])
    dy = np.abs(yc[:, None] - yc[None, :])
    min_dx = half_w[:, None] + half_w[None, :]
    min_dy = half_h[:, None] + half_h[None, :]
    ovl = (dx + eps < min_dx) & (dy + eps < min_dy)
    np.fill_diagonal(ovl, False)
    return int(np.triu(ovl, k=1).sum())


def fd_legalize(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    max_iters: int = 200,
    strength: float = 2.0,
    decay: float = 1.0,
    jitter_seed: int = 42,
    overlap_eps: float = 1e-4,
    verbose: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """Force-directed legalization via Minimum Translation Vector.

    Per overlapping pair (i,j) — i,j hard macros — compute the smaller of
    `pen_x = min_dx - dx` and `pen_y = min_dy - dy`. Push both macros
    apart by `(pen + safety) * strength / 2` along that MTV axis (full
    `pen + safety` for the movable macro of a movable/fixed pair).

    `strength`: scaling on each iteration's push. Total displacement per
        macro per iter = sum over overlapping neighbours of the per-pair
        contribution. Strength=1.0 means each pair tries to fully resolve
        in one shot; if a macro has many overlaps, summed strength=1.0 will
        overshoot. Strength=0.5-0.8 typically converges in 20-60 iters.

    `decay`: multiplicative damping per iter (1.0 = none). When summed
        contributions exceed needed resolution the system oscillates;
        decay damps that. Default 1.0 (no decay) until we see oscillation.

    `jitter_seed`: rng for breaking ties (perfectly-aligned center pairs).

    Returns (legal_placement, stats_dict).
    """
    sizes_np = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)
    fixed_np = benchmark.macro_fixed.detach().cpu().numpy().astype(bool)
    pos_np = placement.detach().cpu().numpy().astype(np.float64).copy()
    n_hard = int(benchmark.num_hard_macros)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    half_w = sizes_np[:n_hard, 0] / 2.0
    half_h = sizes_np[:n_hard, 1] / 2.0
    movable = ~fixed_np[:n_hard]

    xc = pos_np[:n_hard, 0].copy()
    yc = pos_np[:n_hard, 1].copy()

    # safety margin: push slightly past "just touching" so canonical
    # overlap detection (strict `dx < min_dx`) sees zero overlaps.
    safety_margin = max(1e-3, overlap_eps * 10)

    rng = np.random.default_rng(jitter_seed)

    init_overlaps = _count_overlaps_np(xc, yc, half_w, half_h, overlap_eps)
    n_iters = 0
    last_overlaps = init_overlaps
    cur_strength = float(strength)
    max_disp = 0.0

    if init_overlaps == 0:
        pos_np[:n_hard, 0] = xc
        pos_np[:n_hard, 1] = yc
        return torch.tensor(pos_np, dtype=placement.dtype), {
            "init_overlaps": init_overlaps,
            "final_overlaps": 0,
            "n_iters": 0,
            "max_displacement": 0.0,
            "wall": 0.0,
        }

    t0 = time.time()
    # movable[i] and movable[j] -> share = 0.5 for each.
    # movable[i] and fixed[j]   -> share = 1.0 for i (j cannot move).
    # fixed[i] and fixed[j]     -> share = 0 (cannot move; accept overlap).
    movable_i = movable[:, None]   # [n, 1]
    movable_j = movable[None, :]   # [1, n]
    both_movable = movable_i & movable_j
    only_i_movable = movable_i & ~movable_j
    share_i = np.where(both_movable, 0.5, np.where(only_i_movable, 1.0, 0.0))

    min_dx_const = half_w[:, None] + half_w[None, :]
    min_dy_const = half_h[:, None] + half_h[None, :]

    for it in range(max_iters):
        # Vectorized AABB overlap detection (full symmetric matrix)
        dx_signed = xc[:, None] - xc[None, :]
        dy_signed = yc[:, None] - yc[None, :]
        dx = np.abs(dx_signed)
        dy = np.abs(dy_signed)
        min_dx = min_dx_const
        min_dy = min_dy_const
        ovl = (dx + overlap_eps < min_dx) & (dy + overlap_eps < min_dy)
        np.fill_diagonal(ovl, False)

        n_pairs = int(np.triu(ovl, k=1).sum())
        if n_pairs == 0:
            last_overlaps = 0
            n_iters = it
            break

        # Penetration depths (only meaningful where ovl=True)
        pen_x = (min_dx - dx) + safety_margin
        pen_y = (min_dy - dy) + safety_margin
        # MTV axis: choose the smaller penetration (less disruptive)
        push_along_x = pen_x <= pen_y  # [n, n]

        # Sign: macro i moves in the direction AWAY from macro j.
        # sign(xi - xj) = +1 if xi > xj. For ties on OVERLAPPING pairs, we
        # apply a small random tie-break (not on full N×N to keep this fast).
        sx = np.sign(dx_signed)
        sy = np.sign(dy_signed)
        # Find overlapping pairs with ties (rare), break them per-pair only.
        tie_x_ovl = ovl & (dx_signed == 0)
        tie_y_ovl = ovl & (dy_signed == 0)
        n_tie_x = int(tie_x_ovl.sum())
        n_tie_y = int(tie_y_ovl.sum())
        if n_tie_x > 0:
            sx[tie_x_ovl] = rng.choice([-1.0, 1.0], size=n_tie_x)
        if n_tie_y > 0:
            sy[tie_y_ovl] = rng.choice([-1.0, 1.0], size=n_tie_y)
        # Replace remaining zeros (non-overlapping diagonals) with +1
        sx = np.where(sx == 0, 1.0, sx)
        sy = np.where(sy == 0, 1.0, sy)

        # Per-pair (i,j) displacement of macro i along chosen axis
        x_shift_ij = np.where(
            ovl & push_along_x, sx * pen_x * share_i * cur_strength, 0.0
        )
        y_shift_ij = np.where(
            ovl & ~push_along_x, sy * pen_y * share_i * cur_strength, 0.0
        )

        disp_x = x_shift_ij.sum(axis=1)
        disp_y = y_shift_ij.sum(axis=1)

        # Apply
        xc += disp_x
        yc += disp_y

        # Clamp to canvas
        xc = np.clip(xc, half_w, cw - half_w)
        yc = np.clip(yc, half_h, ch - half_h)

        step_disp = float(np.max(np.hypot(disp_x, disp_y)))
        max_disp = max(max_disp, step_disp)

        # Recount overlaps AFTER the clamp (clamping can re-introduce overlaps
        # if macros piled at canvas boundary).
        ovl_post = (
            (np.abs(xc[:, None] - xc[None, :]) + overlap_eps < min_dx_const)
            & (np.abs(yc[:, None] - yc[None, :]) + overlap_eps < min_dy_const)
        )
        np.fill_diagonal(ovl_post, False)
        n_pairs_post = int(np.triu(ovl_post, k=1).sum())

        n_iters = it + 1
        last_overlaps = n_pairs_post
        if n_pairs_post == 0:
            break

        if verbose and (it + 1) % 20 == 0:
            print(
                f"  [fd_legalize] iter={it+1:3d} pairs={n_pairs:5d} "
                f"step_disp={step_disp:.4f} strength={cur_strength:.4f}",
                flush=True,
            )

        cur_strength = max(0.05, cur_strength * decay)

    final_overlaps = _count_overlaps_np(xc, yc, half_w, half_h, overlap_eps)

    pos_np[:n_hard, 0] = xc
    pos_np[:n_hard, 1] = yc

    wall = time.time() - t0

    if verbose:
        print(
            f"  [fd_legalize] init_overlaps={init_overlaps} "
            f"final_overlaps={final_overlaps} iters={n_iters} "
            f"max_disp={max_disp:.4f} wall={wall:.2f}s",
            flush=True,
        )

    return torch.tensor(pos_np, dtype=placement.dtype), {
        "init_overlaps": init_overlaps,
        "final_overlaps": final_overlaps,
        "n_iters": n_iters,
        "max_displacement": max_disp,
        "wall": wall,
    }


if __name__ == "__main__":
    import argparse

    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--strength", type=float, default=0.7)
    ap.add_argument("--decay", type=float, default=1.0)
    args = ap.parse_args()

    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(args.bench)))
    print(
        f"=== {args.bench} (n_macros={bench.num_macros} "
        f"n_hard={bench.num_hard_macros}) ===",
        flush=True,
    )

    # Build an intentionally overlapping init: center scatter.
    rng = np.random.default_rng(42)
    init = bench.macro_positions.clone().to(torch.float32)
    cw = float(bench.canvas_width)
    ch = float(bench.canvas_height)
    for i in range(bench.num_hard_macros):
        if not bool(bench.macro_fixed[i]):
            init[i, 0] = cw / 2 + rng.normal(0, cw * 0.1)
            init[i, 1] = ch / 2 + rng.normal(0, ch * 0.1)

    init_ovl = compute_overlap_metrics(init, bench)["overlap_count"]
    print(f"  init overlaps: {init_ovl}", flush=True)

    t0 = time.time()
    legal, stats = fd_legalize(
        init, bench,
        max_iters=args.iters,
        strength=args.strength,
        decay=args.decay,
        verbose=True,
    )
    wall = time.time() - t0

    final_ovl = compute_overlap_metrics(legal, bench)["overlap_count"]
    proxy = float(compute_proxy_cost(legal, bench, plc)["proxy_cost"])
    print(
        f"  fd_legalize: ovl={final_ovl} proxy={proxy:.5f} "
        f"iters={stats['n_iters']} max_disp={stats['max_displacement']:.3f} "
        f"wall={wall:.2f}s",
        flush=True,
    )
