"""Pairwise legalisation: remove hard-macro overlaps left by the gradient stages.

Every overlapping pair of hard macros is pushed apart along the axis of
smaller penetration (the minimum translation vector), all pairs at once,
until no pair overlaps. Moves are small and local, so the floorplan the
descent found survives. If pushing alone leaves overlaps, the runtime's
iterative projection is tried, then a stronger push to a slightly wider
clearance.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_overlap_metrics
from thinkorplace.common.cd_core import project_overlaps
from thinkorplace.common.runtime_safety import clamp_to_canvas, valid_placement


def _overlapping(x: np.ndarray, y: np.ndarray, min_dx: np.ndarray, min_dy: np.ndarray, eps: float) -> np.ndarray:
    overlap = (np.abs(x[:, None] - x[None, :]) + eps < min_dx) & (np.abs(y[:, None] - y[None, :]) + eps < min_dy)
    np.fill_diagonal(overlap, False)
    return overlap


def push_apart(placement: torch.Tensor, bench: Benchmark, *, strength: float, overlap_eps: float = 1e-4,
               max_iterations: int = 200, tie_seed: int = 42) -> torch.Tensor:
    """Push overlapping hard-macro pairs apart until none overlap or `max_iterations` pass.

    Two movable macros share each push equally; a movable macro touching a
    fixed one takes all of it. The push goes slightly past touching (a margin
    of 10 x `overlap_eps`, at least 1e-3 um) so the strict overlap test of the
    scorer sees none.
    """
    sizes = bench.macro_sizes.detach().cpu().numpy().astype(np.float64)
    fixed = bench.macro_fixed.detach().cpu().numpy().astype(bool)
    positions = placement.detach().cpu().numpy().astype(np.float64).copy()
    n_hard = int(bench.num_hard_macros)
    width, height = float(bench.canvas_width), float(bench.canvas_height)
    half_w, half_h = sizes[:n_hard, 0] / 2.0, sizes[:n_hard, 1] / 2.0
    x, y = positions[:n_hard, 0].copy(), positions[:n_hard, 1].copy()
    margin = max(1e-3, overlap_eps * 10)
    rng = np.random.default_rng(tie_seed)
    min_dx = half_w[:, None] + half_w[None, :]
    min_dy = half_h[:, None] + half_h[None, :]
    movable = ~fixed[:n_hard]
    share = np.where(movable[:, None] & movable[None, :], 0.5, np.where(movable[:, None] & ~movable[None, :], 1.0, 0.0))

    if _overlapping(x, y, min_dx, min_dy, overlap_eps).any():
        for _ in range(max_iterations):
            dx_signed, dy_signed = x[:, None] - x[None, :], y[:, None] - y[None, :]
            overlap = _overlapping(x, y, min_dx, min_dy, overlap_eps)
            if not np.triu(overlap, k=1).any():
                break
            pen_x = (min_dx - np.abs(dx_signed)) + margin
            pen_y = (min_dy - np.abs(dy_signed)) + margin
            along_x = pen_x <= pen_y
            # each macro moves away from the other; coincident centres get a random direction
            sx, sy = np.sign(dx_signed), np.sign(dy_signed)
            tie_x, tie_y = overlap & (dx_signed == 0), overlap & (dy_signed == 0)
            if tie_x.any():
                sx[tie_x] = rng.choice([-1.0, 1.0], size=int(tie_x.sum()))
            if tie_y.any():
                sy[tie_y] = rng.choice([-1.0, 1.0], size=int(tie_y.sum()))
            sx, sy = np.where(sx == 0, 1.0, sx), np.where(sy == 0, 1.0, sy)
            x = x + np.where(overlap & along_x, sx * pen_x * share * strength, 0.0).sum(axis=1)
            y = y + np.where(overlap & ~along_x, sy * pen_y * share * strength, 0.0).sum(axis=1)
            x, y = np.clip(x, half_w, width - half_w), np.clip(y, half_h, height - half_h)
            if not np.triu(_overlapping(x, y, min_dx, min_dy, overlap_eps), k=1).any():
                break
    positions[:n_hard, 0], positions[:n_hard, 1] = x, y
    return torch.tensor(positions, dtype=placement.dtype)


def is_legal(positions: torch.Tensor, bench: Benchmark) -> bool:
    """Inside the canvas, fixed objects in place, and no hard-macro overlap by the scorer's own count."""
    return valid_placement(positions, bench)[0] and int(compute_overlap_metrics(positions, bench)["overlap_count"]) == 0


def legalize(raw: torch.Tensor, bench: Benchmark) -> Optional[torch.Tensor]:
    """Return a legal placement near `raw`, or None if all three attempts leave an overlap."""
    pushed = clamp_to_canvas(push_apart(clamp_to_canvas(raw, bench), bench, strength=1.0), bench)
    if is_legal(pushed, bench):
        return pushed
    projected = clamp_to_canvas(project_overlaps(pushed, bench)[0], bench)
    if is_legal(projected, bench):
        return projected
    cleared = clamp_to_canvas(push_apart(pushed, bench, strength=2.0, overlap_eps=-1e-5), bench)
    return cleared if is_legal(cleared, bench) else None
