"""Coordinate descent on the proxy score, in segments of sweeps.

A sweep visits every movable object in a shuffled order and, for each axis
in turn, searches the object's legal range for the position with the lowest
proxy score as the runtime's incremental evaluator computes it (float64
arithmetic; thinkorplace.common, which also supplies the sweep itself). A segment is the ranked configuration's schedule:
sweeps 1 and 2 move soft clusters only, so hard macros settle against a
fitted background; after sweep 4 the evaluator is rebuilt from the float32
placement, which clears accumulated float64 drift; sweep n draws its visit
order from seed n.

The first segment of each start can score congestion on a widened tail:
the proxy's congestion term averages the top 5% of cell-direction loads, so
a move that relieves a cell just below that cut scores nothing. The widened
score charges the top 25% at the same price per cell for sweeps 1 to 4,
which lets the search clear the whole hot region before sweeps 5 and 6 go
back to the stock score.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from thinkorplace.common.cd_core import _grid_lines, _sweep_macros
from thinkorplace.common.incremental_evaluator import IncrementalProxyEvaluator
from thinkorplace.common.runtime_safety import clamp_to_canvas

SOFT_ONLY_SWEEPS = (1, 2)
REBUILD_AFTER_SWEEP = 4
STOCK_TAIL, WIDE_TAIL = 0.05, 0.25


class WideTailEvaluator(IncrementalProxyEvaluator):
    """The runtime evaluator with congestion = sum of the top 25% of loads / (5% of their count).

    Every load in the wider tail costs what a top-5% load costs in the stock
    score, so the term is never below the stock one. Wirelength, density and
    the incremental bookkeeping are inherited unchanged.
    """

    def __init__(self, bench: Benchmark, plc, placement: torch.Tensor):
        super().__init__(bench, plc, placement)
        total = 2 * self.num_cells
        self.k_stock = int(math.floor(total * STOCK_TAIL))
        self.k_wide = int(math.floor(total * WIDE_TAIL))

    def _loads(self, h_net, v_net, h_macro, v_macro, smooth):
        # the stock term's normalisation, smoothing and vertical-then-horizontal order
        v_total = smooth(v_net / self.grid_v_routes, vertical=True) + v_macro / self.grid_v_routes
        h_total = smooth(h_net / self.grid_h_routes, vertical=False) + h_macro / self.grid_h_routes
        return torch.cat([v_total, h_total], dim=-1)

    def _wide_tail(self, loads: torch.Tensor) -> torch.Tensor:
        return torch.topk(loads, self.k_wide, dim=-1).values.sum(dim=-1) / self.k_stock

    def _congestion_cost_of(self, h_net, v_net, h_macro, v_macro):
        return float(self._wide_tail(self._loads(h_net, v_net, h_macro, v_macro, self._smooth)))

    def _congestion_cost_of_batched(self, h_net, v_net, h_macro, v_macro):
        return self._wide_tail(self._loads(h_net, v_net, h_macro, v_macro, self._smooth_batched))


@dataclass
class SegmentResult:
    positions: torch.Tensor
    sweeps: int = 0
    complete_sweep_seconds: List[float] = field(default_factory=list)


class CoordinateDescent:
    def __init__(self, bench: Benchmark, plc):
        self.bench, self.plc = bench, plc
        self.n_hard = bench.num_hard_macros
        self.movable = torch.where(~bench.macro_fixed.bool())[0].tolist()
        self.grid_x, self.grid_y = _grid_lines(plc)

    def _evaluator(self, positions: torch.Tensor, widened: bool, next_sweep: int) -> IncrementalProxyEvaluator:
        if widened and next_sweep <= REBUILD_AFTER_SWEEP:
            return WideTailEvaluator(self.bench, self.plc, positions)
        return IncrementalProxyEvaluator(self.bench, self.plc, positions)

    def segment(self, positions: torch.Tensor, widened: bool, more: Callable[[int], bool],
                stop_at: Callable[[], float]) -> SegmentResult:
        """Sweep while `more(sweeps done)` holds and the clock is before `stop_at()`.

        A sweep cut short by the clock ends the segment. Returns the float32
        placement after the last sweep.
        """
        evaluator = self._evaluator(positions, widened, 1)
        current = evaluator.current_cost()["proxy"]
        result = SegmentResult(positions)
        while more(result.sweeps) and time.time() < stop_at():
            result.sweeps += 1
            sweep = result.sweeps
            started = time.time()
            order = self.movable.copy()
            np.random.default_rng(sweep).shuffle(order)
            if sweep in SOFT_ONLY_SWEEPS:
                order = [index for index in order if index >= self.n_hard]
            current, _, probes, _ = _sweep_macros(evaluator, self.bench, self.grid_x, self.grid_y, order, current,
                                                  self.n_hard, deadline_check=lambda: time.time() > stop_at())
            result.positions = clamp_to_canvas(evaluator.placement.to(torch.float32), self.bench)
            if probes != 2 * len(order):  # one search per axis per object, unless the clock cut the sweep short
                break
            result.complete_sweep_seconds.append(time.time() - started)
            if sweep == REBUILD_AFTER_SWEEP:
                evaluator = self._evaluator(result.positions, widened, sweep + 1)
                current = evaluator.current_cost()["proxy"]
        return result
