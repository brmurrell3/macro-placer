"""Spreading refinement: turn a rough start into an even, wire-aware seed.

Twenty-eight rounds alternate two steps. An electrostatic model (a Poisson
solve on a bin grid) computes where each object would move to even out
density. Then one sparse linear solve per axis balances net springs against
a tether to that spread target. The tether starts weak and grows 150-fold,
and the bin grid goes from 8 x 8 to 32 x 32, so early rounds settle the
global arrangement and late rounds the local spacing.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.fft import dctn, idctn
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from macro_place.benchmark import Benchmark
from thinkorplace.common.runtime_safety import clamp_to_canvas
from thinkorplace.common.smooth_primitives import _extract_net_data

ROUND_BINS = (8,) * 8 + (16,) * 10 + (32,) * 10


def spread_target(points: np.ndarray, widths: np.ndarray, movable: np.ndarray, bounds: np.ndarray,
                  bins: int, iterations: int) -> np.ndarray:
    """Move objects down the gradient of an electrostatic potential of their own density.

    Each rectangle deposits its exact area into the bins; the potential comes
    from a cosine-transform Poisson solve, whose basis gives no flow through
    the die edges; the force on an object is averaged over its footprint.
    """
    edges = [np.linspace(0, side, bins + 1) for side in bounds]
    cell = bounds / bins
    frequencies = 2 * np.sin(np.pi * np.arange(bins) / (2 * bins))
    eigen = (frequencies[:, None] / cell[1]) ** 2 + (frequencies[None, :] / cell[0]) ** 2
    eigen[0, 0] = 1.0
    area = widths.prod(axis=1).clip(min=1e-14)
    mean_density = area.sum() / bounds.prod()
    result = points.copy()
    for _ in range(iterations):
        overlaps = []
        for axis in range(2):
            half = widths[:, axis, None] / 2
            low = np.maximum(result[:, axis, None] - half, edges[axis][:-1])
            high = np.minimum(result[:, axis, None] + half, edges[axis][1:])
            overlaps.append(np.maximum(0, high - low))
        dx, dy = overlaps
        charge = dy.T @ dx / cell.prod()
        modes = dctn(charge, norm="ortho")
        modes[0, 0] = 0
        potential = idctn(modes / eigen, norm="ortho")
        grad_y, grad_x = np.gradient(potential, cell[1], cell[0])
        motion = np.column_stack([(dy * (dx @ g.T)).sum(axis=1) for g in (grad_x, grad_y)])
        motion /= (mean_density * area[:, None]).clip(min=1e-14)
        result[movable] -= np.clip(motion[movable], -0.8 * cell, 0.8 * cell)
        result[movable] = np.clip(result[movable], widths[movable] / 2, bounds - widths[movable] / 2)
    return result


def wire_solve(axis: int, points: np.ndarray, target: np.ndarray, nets: tuple, strength: float, smoothing: float,
               movable: np.ndarray, full_to_free: np.ndarray, area_factor: np.ndarray) -> np.ndarray:
    """One axis of the spring-versus-tether solve.

    Springs join each interior pin to its net's two extreme pins and the
    extremes to each other (bound-to-bound model), weighted by the inverse of
    their current length, so the quadratic solve approximates linear wirelength.
    """
    net, owner, offset, weight, starts = nets
    pin_xy = points[owner, axis] + offset[:, axis]
    pin = np.arange(len(net))
    minimum = np.minimum.reduceat(pin_xy, starts)
    maximum = np.maximum.reduceat(pin_xy, starts)
    low = np.minimum.reduceat(np.where(pin_xy == minimum[net], pin, len(pin)), starts)[net]
    high = np.maximum.reduceat(np.where(pin_xy == maximum[net], pin, -1), starts)[net]
    interior = (pin != low) & (pin != high)
    left = np.r_[low[pin != low], high[interior]]
    right = np.r_[pin[pin != low], pin[interior]]
    cross_owner = owner[left] != owner[right]
    left, right = left[cross_owner], right[cross_owner]
    w = weight[net[left]] / np.maximum(np.abs(pin_xy[left] - pin_xy[right]), smoothing)
    a, b = owner[left], owner[right]
    ia, ib = full_to_free[a], full_to_free[b]
    free_a, free_b = ia >= 0, ib >= 0
    both = free_a & free_b
    count = len(movable)
    mass = (np.bincount(ia[free_a], weights=w[free_a], minlength=count)
            + np.bincount(ib[free_b], weights=w[free_b], minlength=count))
    reference_mass = np.median(mass[mass > 0]) if np.any(mass > 0) else 1.0
    # the tether is scaled to the median spring mass so `strength` means the same on every design
    tether = strength * reference_mass * area_factor + 1e-5 * reference_mass
    # fixed owners move to the right-hand side; pin offsets enter as constant terms
    offset_difference = offset[left, axis] - offset[right, axis]
    target_a = np.where(free_b, 0, points[b, axis]) - offset_difference
    target_b = np.where(free_a, 0, points[a, axis]) + offset_difference
    rhs = (tether * target[movable, axis]
           + np.bincount(ia[free_a], weights=(w * target_a)[free_a], minlength=count)
           + np.bincount(ib[free_b], weights=(w * target_b)[free_b], minlength=count))
    off_diagonal = coo_matrix((-w[both], (ia[both], ib[both])), shape=(count, count)).tocsr()
    system = diags(mass + tether) + off_diagonal + off_diagonal.T
    return spsolve(system.tocsc(), rhs)


def refine_spreading(start: torch.Tensor, bench: Benchmark, plc) -> torch.Tensor:
    """Return the refined seed, in die coordinates and the dtype of `start`."""
    movable = np.flatnonzero(~bench.macro_fixed.cpu().numpy().astype(bool))
    data = _extract_net_data(bench, plc)
    net, pin = np.nonzero(data.mask.numpy())
    if not len(movable) or not len(net):
        return start
    scale = max(bench.canvas_width, bench.canvas_height)
    bounds = np.array([bench.canvas_width, bench.canvas_height]) / scale
    widths = bench.macro_sizes.cpu().numpy().astype(float) / scale
    points = np.vstack((start.cpu().numpy() / scale, [0., 0.]))  # last row: the common owner of ports
    owners = data.pin_macro_idx.numpy()[net, pin]
    offsets = data.pin_offsets.numpy()[net, pin].astype(float) / scale
    degree = np.bincount(net)
    weight = data.weights.numpy() * 2.0 / (degree - 1)
    nets = (net, owners, offsets, weight, np.r_[0, np.cumsum(degree)[:-1]])
    full_to_free = np.full(len(points), -1, dtype=int)
    full_to_free[movable] = np.arange(len(movable))
    areas = widths[movable].prod(axis=1)
    area_factor = np.sqrt(np.maximum(areas, 1e-14) / np.maximum(np.median(areas), 1e-14))
    for stage, bins in enumerate(ROUND_BINS):
        progress = stage / (len(ROUND_BINS) - 1)
        target = spread_target(points[:-1], widths, movable, bounds, bins, iterations=6)
        strength = 0.04 * (150 ** progress)
        smoothing = 0.04 * (0.08 ** progress)
        candidate = np.column_stack([
            wire_solve(axis, points, target, nets, strength, smoothing, movable, full_to_free, area_factor)
            for axis in range(2)
        ])
        points[movable] = np.clip(candidate, widths[movable] / 2, bounds - widths[movable] / 2)
    return clamp_to_canvas(torch.as_tensor(points[:-1] * scale, dtype=start.dtype), bench)
