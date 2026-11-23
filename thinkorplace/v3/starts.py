"""Cold starting placements built from the netlist alone.

A cold start ignores the supplied positions of movable objects. It comes from
two steps: a quadratic wirelength placement pulled by the fixed I/O pins, then
recursive two-way partitioning of the netlist (KaHyPar), which assigns each
group of tightly connected objects its own region of the die. Different
partitioner seeds give different, equally plausible floorplans, which is what
the seed pool needs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.sparse import bmat, coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.stats import qmc

from macro_place.benchmark import Benchmark
from thinkorplace.common.runtime_safety import clamp_to_canvas
from thinkorplace.common.smooth_primitives import _extract_net_data

KAHYPAR_CONFIG = Path(__file__).with_name("kahypar_cut.ini")


def area_spread(centers: np.ndarray, areas: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Spread points over a region by recursive area-balanced cuts.

    Each cut splits the points at the area median along the region's longer
    side, so the relative order of the points survives while every object
    gets room in proportion to its area.
    """
    spread = np.empty_like(centers)
    pending = [(np.arange(len(centers)), np.zeros(2), region)]
    while pending:
        group, low, high = pending.pop()
        if len(group) == 1:
            spread[group] = (low + high) / 2
            continue
        axis = np.argmax(high - low)
        ordered = group[np.argsort(centers[group, axis], kind="stable")]
        cumulative = np.cumsum(areas[ordered])
        count = 1 + np.argmin(np.abs(cumulative[:-1] - 0.5 * cumulative[-1]))
        cut = low[axis] + (high[axis] - low[axis]) * cumulative[count - 1] / cumulative[-1]
        first_high, second_low = high.copy(), low.copy()
        first_high[axis] = second_low[axis] = cut
        pending.extend([(ordered[:count], low, first_high), (ordered[count:], second_low, high)])
    return spread


def quadratic_start(bench: Benchmark, plc) -> torch.Tensor:
    """Minimise squared pin-to-net-centre distance, then spread by area.

    Each pin is a spring to a free point for its net (a star model), and fixed
    objects and ports enter the right-hand side. A weak tether to a Halton
    point gives disconnected objects a definite position.
    """
    n = bench.num_macros
    canvas = np.array([bench.canvas_width, bench.canvas_height], dtype=np.float64)
    fixed = bench.macro_fixed.cpu().numpy().astype(bool)
    movable = np.flatnonzero(~fixed)
    anchors = np.zeros((n + 1, 2))  # the extra owner stands for ports, whose offsets are absolute
    anchors[np.flatnonzero(fixed)] = bench.macro_positions[torch.from_numpy(fixed)].cpu().numpy()
    result = anchors[:n].copy()
    if not len(movable):
        return torch.as_tensor(result, dtype=bench.macro_sizes.dtype)

    prior = qmc.Halton(2, scramble=False).random(len(movable)) * canvas
    data = _extract_net_data(bench, plc)
    net, pin = np.nonzero(data.mask.numpy())
    centers = prior
    if len(net):
        owner = data.pin_macro_idx.numpy()[net, pin]
        offset = data.pin_offsets.numpy()[net, pin].astype(np.float64)
        weight = data.weights.numpy()[net] / (np.bincount(net)[net] - 1)
        variable = np.full(n + 1, -1, dtype=int)
        variable[movable] = np.arange(len(movable))
        free = variable[owner] >= 0
        net_count = len(data.weights)
        incidence = coo_matrix((weight[free], (variable[owner[free]], net[free])),
                               shape=(len(movable), net_count)).tocsr()
        mass = np.asarray(incidence.sum(axis=1)).ravel()
        net_mass = np.bincount(net, weights=weight, minlength=net_count)
        tether = 1e-3 * np.maximum(mass, 1.0)
        matrix = bmat([[diags(mass + tether), -incidence],
                       [-incidence.T, diags(np.where(net_mass > 0, net_mass, 1.0))]], format="csc")
        rhs = np.vstack((tether[:, None] * prior, np.zeros((net_count, 2))))
        np.add.at(rhs, variable[owner[free]], -weight[free, None] * offset[free])
        np.add.at(rhs, len(movable) + net, weight[:, None] * (offset + anchors[owner]))
        centers = spsolve(matrix, rhs)[:len(movable)]

    sizes = bench.macro_sizes.cpu().numpy()[movable]
    areas = np.maximum(sizes[:, 0] * sizes[:, 1], np.finfo(float).eps)
    result[movable] = area_spread(centers, areas, canvas)
    return clamp_to_canvas(torch.as_tensor(result, dtype=bench.macro_sizes.dtype), bench)


def _bipartition(kahypar, edges: dict, areas: np.ndarray, seed: int) -> np.ndarray:
    """Split a region's objects in two, area-balanced, cutting as little net weight as possible.

    Two extra vertices, fixed one to each side, stand for the pins beyond each
    half of the region, so nets that leave the region pull their objects
    towards the correct side.
    """
    size = len(areas)
    boundaries, flattened, edge_weights = [0], [], []
    for pins, weight in edges.items():
        flattened.extend(pins)
        boundaries.append(len(flattened))
        edge_weights.append(weight)
    object_weights = np.maximum(1, np.rint(areas / areas.sum() * 1e6)).astype(int)
    vertex_weights = np.r_[object_weights, 1, 1]
    graph = kahypar.Hypergraph(size + 2, len(edges), boundaries, flattened, 2, edge_weights, vertex_weights.tolist())
    graph.fixNodeToBlock(size, 0)
    graph.fixNodeToBlock(size + 1, 1)
    context = kahypar.Context()
    context.loadINIconfiguration(str(KAHYPAR_CONFIG))
    context.setK(2)
    context.setSeed(seed)
    # the imbalance allowance must let the largest single object fit in one half
    context.setEpsilon(max(0.14, 2 * vertex_weights.max() / vertex_weights.sum() - 1 + 0.01))
    context.suppressOutput(True)
    kahypar.partition(graph, context)
    return np.array([graph.blockID(i) for i in range(size)])


def partition_start(bench: Benchmark, plc, seed: int, depth: int = 6) -> torch.Tensor:
    """Cold start from `depth` levels of recursive bisection anchored on the I/O pins.

    Each level cuts every region with more than 16 objects across its longer
    side, at the position that matches the area of the two halves, and places
    each half's objects by area-spreading their quadratic positions inside it.
    """
    import kahypar

    prior = quadratic_start(bench, plc).numpy()
    movable = np.flatnonzero(~bench.macro_fixed.cpu().numpy().astype(bool))
    if not len(movable) or depth <= 0:
        return torch.from_numpy(prior)

    positions = prior.copy()
    areas = bench.macro_sizes.cpu().numpy().prod(axis=1)
    areas = np.maximum(areas, np.finfo(areas.dtype).tiny)
    die = np.array([bench.canvas_width, bench.canvas_height])
    nets = _extract_net_data(bench, plc)
    net_ids, pin_ids = np.nonzero(nets.mask.numpy())
    owners = nets.pin_macro_idx.numpy()[net_ids, pin_ids]
    offsets = nets.pin_offsets.numpy()[net_ids, pin_ids]
    weights = nets.weights.numpy()
    net_starts = np.r_[0, np.cumsum(np.bincount(net_ids, minlength=len(weights)))]

    regions = [(movable, np.zeros(2), die)]
    for level in range(depth):
        children = []
        for nodes, lower, upper in regions:
            if len(nodes) <= 16:
                children.append((nodes, lower, upper))
                continue
            axis = np.argmax(upper - lower)
            midpoint = (upper[axis] + lower[axis]) * 0.5
            local = np.full(bench.num_macros + 1, -1)
            local[nodes] = np.arange(len(nodes))
            # a pin outside the region becomes the terminal of the half it lies in
            pin_coord = np.r_[positions[:, axis], 0][owners] + offsets[:, axis]
            local_owner = local[owners]
            encoded = np.where(local_owner < 0, len(nodes) + (pin_coord >= midpoint), local_owner)
            edges: dict = {}
            for net in np.unique(net_ids[local_owner >= 0]):
                pins = tuple(sorted(set(encoded[net_starts[net]:net_starts[net + 1]].tolist())))
                # a net that touches both terminals is cut whatever the assignment
                if len(pins) > 1 and pins[-2:] != (len(nodes), len(nodes) + 1):
                    edges[pins] = edges.get(pins, 0) + max(1, round(weights[net] * 16))
            if not edges:
                children.append((nodes, lower, upper))
                continue
            labels = _bipartition(kahypar, edges, areas[nodes], seed + level)
            left, right = nodes[labels == 0], nodes[labels == 1]
            if not len(left) or not len(right):
                children.append((nodes, lower, upper))
                continue
            cut = lower[axis] + (upper[axis] - lower[axis]) * areas[left].sum() / areas[nodes].sum()
            left_upper, right_lower = upper.copy(), lower.copy()
            left_upper[axis] = right_lower[axis] = cut
            children.extend([(left, lower, left_upper), (right, right_lower, upper)])
        regions = children
        for nodes, lower, upper in regions:
            positions[nodes] = lower + area_spread(prior[nodes], areas[nodes], upper - lower)
    return clamp_to_canvas(torch.from_numpy(positions), bench)
