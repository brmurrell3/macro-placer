"""The seed pool and the span score that orders it.

Most of the final score is decided by the starting placement: the gradient
stages and the coordinate-descent search improve a floorplan locally but
rarely change its arrangement. So each chain first makes many cheap seeds and
descends the promising ones first.

"Promising" is judged by a span score: the weighted sum of net bounding-box
widths and heights at the seed, with horizontal span counted 1.6216 times. On
these designs a micron of horizontal wire uses that much more of its routing
capacity than a micron of vertical wire (ibm07: 106.96 vertical against 65.96
horizontal routing tracks per micron), so the weight makes the score track
congestion as well as wirelength. It costs one pass over the nets, against
tens of seconds for a descent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from thinkorplace.common.cd_core import sdf_init
from thinkorplace.common.runtime_safety import clamp_to_canvas
from thinkorplace.common.smooth_primitives import _extract_net_data
from thinkorplace.v3.spreading import refine_spreading
from thinkorplace.v3.starts import partition_start

HORIZONTAL_WEIGHT = 1.6216
PARTITION_DEPTH = 6


@dataclass
class Seed:
    label: Union[int, str]
    positions: torch.Tensor
    span_score: float


def net_spans(positions: torch.Tensor, bench: Benchmark, plc) -> tuple:
    """Weighted sums of net bounding-box width and height (um), pins at object centre plus offset."""
    nets = _extract_net_data(bench, plc)
    mask = nets.mask.numpy()
    points = np.vstack((positions.double().numpy(), [0., 0.]))  # last row: the common owner of ports
    xy = points[nets.pin_macro_idx.numpy()] + nets.pin_offsets.numpy()
    spans = []
    for axis in range(2):
        values = xy[..., axis]
        high = np.where(mask, values, -np.inf).max(axis=1)
        low = np.where(mask, values, np.inf).min(axis=1)
        spans.append(float((nets.weights.numpy() * (high - low)).sum()))
    return spans[0], spans[1]


def span_score(positions: torch.Tensor, bench: Benchmark, plc) -> float:
    x_span, y_span = net_spans(positions, bench, plc)
    return HORIZONTAL_WEIGHT * x_span + y_span


def cold_seed(number: int, bench: Benchmark, plc) -> Seed:
    """Seed `number`: a partition start (partitioner seed 7 + number) after spreading refinement."""
    positions = refine_spreading(partition_start(bench, plc, seed=7 + number, depth=PARTITION_DEPTH), bench, plc)
    return Seed(number, positions, span_score(positions, bench, plc))


def warm_seeds(bench: Benchmark, plc) -> List[Seed]:
    """Four seeds from the supplied placement, descended in this fixed order.

    The supplied placement often encodes a good arrangement, so the warm
    chain starts from it two ways: through the signed-distance-field
    initialiser of the ranked configuration, and as given; each with and
    without the spreading refinement.
    """
    supplied = bench.macro_positions.clone()
    sdf = sdf_init(bench)
    seeds = []
    for label, start, refine in (("sdf-refined", sdf, True), ("supplied-refined", supplied, True),
                                 ("sdf", sdf, False), ("supplied", supplied, False)):
        start = clamp_to_canvas(torch.as_tensor(start, dtype=bench.macro_sizes.dtype), bench)
        positions = refine_spreading(start, bench, plc) if refine else start
        seeds.append(Seed(label, positions, span_score(positions, bench, plc)))
    return seeds
