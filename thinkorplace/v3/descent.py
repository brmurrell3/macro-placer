"""The two gradient stages and the soft-cluster interlude.

A seed is descended in two stages of Adam on smooth stand-ins for the proxy
score (wirelength + 0.5 density + 0.5 congestion):

  coarse  96 steps on a cheap model with a routing grid of at most 24 x 24,
          L-shaped soft routes and a growing hard-macro overlap penalty. It
          moves objects far, fast.
  fine    404 steps on the ranked configuration's smooth proxy (thinkorplace.v2:
          log-sum-exp wirelength, Gaussian density, per-net trace congestion),
          with a sharpening wirelength model and a tightening density footprint.

Both stages price congestion with a `RoutingLoss`: the proxy's own term is the
mean of the most loaded 5% of grid cell-directions, which says nothing about
load below that cut. The descents add half the mean load over all
cell-directions, so total routing demand is priced too.

The interlude reuses the late part of the fine schedule at a tenth of the
learning rate, on soft clusters only, between coordinate-descent segments.
"""
from __future__ import annotations

import math
import types
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from macro_place.benchmark import Benchmark
from thinkorplace.common.runtime_safety import clamp_to_canvas
from thinkorplace.common.smooth_primitives import _extract_net_data
from thinkorplace.v2.fast_proxy import fast_loss_with_penalty
from thinkorplace.v2.v4_gaussian_proxy import FastDiffProxyGaussian

COARSE_STEPS, FINE_STEPS = 96, 404
COARSE_GRID, COARSE_LEARNING_RATE = 24, 0.008
# the ranked configuration's fine-stage settings (thinkorplace.v2.placer.Placer defaults)
FINE_LEARNING_RATE = 0.005  # times the canvas width
GAMMA_START, GAMMA_END = 5e-3, 5e-5
OVERLAP_WEIGHT, OVERLAP_RAMP = 10.0, 0.7
BOUNDARY_WEIGHT = 50.0
STOCK_TAIL = 0.05
DENSITY_WEIGHT, CONGESTION_WEIGHT = 0.5, 0.5  # the proxy score's own weights
# coarse model: the log-sum-exp wirelength smoothing falls from 0.02 to 0.001 of the die over the stage; the
# density Gaussians start at least one bin wide and shrink by 65% by three quarters of the way
WL_GAMMA_START, WL_GAMMA_END = 0.02, 0.001
DENSITY_SPREAD_SHRINK, DENSITY_SPREAD_UNTIL = 0.65, 0.75
COARSE_DENSITY_TAIL = 0.10  # the proxy's density term: mean of the fullest 10% of bins
ROUTE_WIDTH_CELLS = 0.60  # standard deviation of a soft route's track, in routing cells
ROUTE_EDGE_SHARPNESS = 3  # slope of a soft route's ends, per routing cell
# fine stage: the density footprint tightens between these fractions of the schedule
FOOTPRINT_START, FOOTPRINT_END = 0.20, 0.70


@dataclass(frozen=True)
class RoutingLoss:
    """Mean of the top `tail` fraction of cell-direction loads, plus `mean_weight` times the mean of all."""
    tail: float = STOCK_TAIL
    mean_weight: float = 0.0


DESCENT_LOSS = RoutingLoss(mean_weight=0.5)


@dataclass(frozen=True)
class Interlude:
    loss: RoutingLoss
    noise_um: float = 0.0  # seeded Gaussian kick before the descent, so repeated interludes differ
    steps: int = 150
    learning_scale: float = 0.1
    progress_from: float = 0.6


# the three interludes of the stacked search, in order
INTERLUDES = (
    Interlude(RoutingLoss(mean_weight=1.0)),
    Interlude(RoutingLoss(tail=0.25), noise_um=0.15),
    Interlude(RoutingLoss(mean_weight=1.0), noise_um=0.15),
)


class CoarseModel:
    """Smooth wirelength, density, L-route congestion and hard overlap on die-normalised centres.

    Positions are a batch of one, [1, N, 2], as in the batched model this
    copies; the shapes are kept so the arithmetic is the same.
    """

    def __init__(self, bench: Benchmark, plc, loss: RoutingLoss):
        self.loss = loss
        self.canvas = torch.tensor([bench.canvas_width, bench.canvas_height])
        self.sizes = bench.macro_sizes.to(dtype=torch.float32) / self.canvas
        self.half = self.sizes / 2
        self.area = self.sizes.prod(dim=-1)
        self.n_hard = bench.num_hard_macros
        self.pairs = torch.triu_indices(self.n_hard, self.n_hard, offset=1)
        left, right = self.pairs.unbind(0)
        self.overlap_width = self.half[left] + self.half[right]

        scale = min(1.0, COARSE_GRID / max(bench.grid_rows, bench.grid_cols))
        self.rows = max(1, round(bench.grid_rows * scale))
        self.cols = max(1, round(bench.grid_cols * scale))
        self.x_edges = torch.linspace(0, 1, self.cols + 1)
        self.y_edges = torch.linspace(0, 1, self.rows + 1)
        self.x_grid = (self.x_edges[1:] + self.x_edges[:-1]) / 2
        self.y_grid = (self.y_edges[1:] + self.y_edges[:-1]) / 2
        self.h_capacity = max(float(plc.hroutes_per_micron) * bench.canvas_height / self.rows, 1e-8)
        self.v_capacity = max(float(plc.vroutes_per_micron) * bench.canvas_width / self.cols, 1e-8)
        self.h_blockage = float(plc.hrouting_alloc) / max(float(plc.hroutes_per_micron), 1e-8)
        self.v_blockage = float(plc.vrouting_alloc) / max(float(plc.vroutes_per_micron), 1e-8)

        # nets grouped by degree rounded up to a power of two, so a few huge nets do not pad every small one
        data = _extract_net_data(bench, plc)
        self.wl_scale = self.canvas / (self.canvas.sum() * max(float(plc.net_cnt), 1.0))
        degree = data.mask.sum(dim=1)
        self.nets = []
        width = 2
        while degree.numel() and width // 2 < int(degree.max()):
            keep = (degree > width // 2) & (degree <= width)
            if keep.any():
                self.nets.append((data.pin_macro_idx[keep, :width], data.pin_offsets[keep, :width] / self.canvas,
                                  data.mask[keep, :width], data.weights[keep]))
            width *= 2

        # every net is routed as source-to-sink two-pin connections
        net, sink = data.mask[:, 1:].nonzero(as_tuple=True)
        self.src = data.pin_macro_idx[net, 0]
        self.sink = data.pin_macro_idx[net, sink + 1]
        self.src_offset = data.pin_offsets[net, 0] / self.canvas
        self.sink_offset = data.pin_offsets[net, sink + 1] / self.canvas
        self.route_weight = data.weights[net].view(1, -1, 1)

    def wirelength(self, all_pos: torch.Tensor, gamma: float) -> torch.Tensor:
        cost = all_pos.new_zeros(all_pos.shape[0])
        for owners, offsets, mask, weights in self.nets:
            pins = all_pos.index_select(1, owners.flatten()).reshape(all_pos.shape[0], *offsets.shape) + offsets
            scaled = pins / gamma
            hidden = ~mask[..., None]
            high = torch.logsumexp(scaled.masked_fill(hidden, -1e8), dim=2)
            neg_low = torch.logsumexp((-scaled).masked_fill(hidden, -1e8), dim=2)
            spans = (high + neg_low) * gamma
            cost = cost + ((spans * self.wl_scale).sum(dim=-1) * weights).sum(dim=1)
        return cost

    @staticmethod
    def bin_mass(position: torch.Tensor, sigma: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        cdf = torch.erf((edges - position.unsqueeze(-1)) / (math.sqrt(2.0) * sigma[:, None]))
        mass = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0)
        return mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # an object is always wholly on the die

    def density(self, pos: torch.Tensor, spread: float) -> torch.Tensor:
        # width / sqrt(12) gives the Gaussian the variance of the rectangle; `spread` keeps it wide early on
        sx = (self.sizes[:, 0] / math.sqrt(12)).clamp_min(spread / self.cols)
        sy = (self.sizes[:, 1] / math.sqrt(12)).clamp_min(spread / self.rows)
        mx = self.bin_mass(pos[..., 0], sx, self.x_edges)
        my = self.bin_mass(pos[..., 1], sy, self.y_edges)
        grid = (my * self.area[:, None]).transpose(1, 2).bmm(mx) * (self.cols * self.rows)
        flat = grid.flatten(start_dim=1)
        count = max(1, int(COARSE_DENSITY_TAIL * flat.shape[1]))
        return 0.5 * flat.sort(dim=1, descending=True).values[:, :count].mean(dim=1)

    @staticmethod
    def occupancy(center: torch.Tensor, half: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        lower = torch.maximum((center - half).unsqueeze(-1), edges[:-1])
        upper = torch.minimum((center + half).unsqueeze(-1), edges[1:])
        return (upper - lower).clamp_min(0)

    def routing(self, pos: torch.Tensor, all_pos: torch.Tensor) -> torch.Tensor:
        source = all_pos.index_select(1, self.src) + self.src_offset
        sink = all_pos.index_select(1, self.sink) + self.sink_offset
        low, high = torch.minimum(source, sink), torch.maximum(source, sink)
        # horizontal first at the source's y, then vertical at the sink's x, as separable soft masks
        row = F.softmax(-0.5 * ((source[..., 1, None] - self.y_grid) * self.rows / ROUTE_WIDTH_CELLS).square(), dim=-1)
        col = F.softmax(-0.5 * ((sink[..., 0, None] - self.x_grid) * self.cols / ROUTE_WIDTH_CELLS).square(), dim=-1)
        hx = torch.sigmoid((self.x_grid - low[..., 0, None]) * (ROUTE_EDGE_SHARPNESS * self.cols))
        hx = hx - torch.sigmoid((self.x_grid - high[..., 0, None]) * (ROUTE_EDGE_SHARPNESS * self.cols))
        vy = torch.sigmoid((self.y_grid - low[..., 1, None]) * (ROUTE_EDGE_SHARPNESS * self.rows))
        vy = vy - torch.sigmoid((self.y_grid - high[..., 1, None]) * (ROUTE_EDGE_SHARPNESS * self.rows))
        h_route = (row * self.route_weight).transpose(1, 2).bmm(hx) / self.h_capacity
        v_route = (vy * self.route_weight).transpose(1, 2).bmm(col) / self.v_capacity
        hard = pos[:, :self.n_hard]
        x_cover = self.occupancy(hard[..., 0], self.half[:self.n_hard, 0], self.x_edges)
        y_cover = self.occupancy(hard[..., 1], self.half[:self.n_hard, 1], self.y_edges)
        blockage = y_cover.transpose(1, 2).bmm(x_cover) * (self.rows * self.cols)
        loads = torch.cat((h_route + self.h_blockage * blockage, v_route + self.v_blockage * blockage), dim=1)
        flat = loads.flatten(start_dim=1)
        count = max(1, int(self.loss.tail * flat.shape[1]))
        cost = flat.sort(dim=1, descending=True).values[:, :count].mean(dim=1)
        if self.loss.mean_weight:
            cost = cost + self.loss.mean_weight * flat.mean(dim=1)
        return cost

    def terms(self, pos: torch.Tensor, progress: float) -> tuple:
        all_pos = torch.cat((pos, pos.new_zeros((pos.shape[0], 1, 2))), dim=1)  # last row: the owner of ports
        gamma = WL_GAMMA_START * (WL_GAMMA_END / WL_GAMMA_START) ** progress
        spread = 1.0 - DENSITY_SPREAD_SHRINK * min(1.0, progress / DENSITY_SPREAD_UNTIL)
        left, right = self.pairs.unbind(0)
        separation = (pos.index_select(1, left) - pos.index_select(1, right)).abs()
        overlap = (self.overlap_width - separation).clamp_min(0).prod(dim=-1).sum(dim=1)
        return self.wirelength(all_pos, gamma), self.density(pos, spread), self.routing(pos, all_pos), overlap


def coarse_descent(seed: torch.Tensor, bench: Benchmark, plc, loss: RoutingLoss = DESCENT_LOSS) -> torch.Tensor:
    model = CoarseModel(bench, plc, loss)
    fixed = bench.macro_fixed.cpu().bool()
    fixed_reference = torch.zeros_like(bench.macro_sizes, device="cpu")
    fixed_reference[fixed] = bench.macro_positions.cpu()[fixed]
    reference = fixed_reference / model.canvas
    movable = (~fixed).view(1, -1, 1)
    high = 1.0 - model.half
    pos = seed.detach()[None].to(dtype=torch.float32) / model.canvas
    pos = torch.where(movable, torch.maximum(model.half, torch.minimum(high, pos)), reference).requires_grad_()
    optimizer = torch.optim.Adam([pos], lr=COARSE_LEARNING_RATE, foreach=False)
    for step in range(COARSE_STEPS):
        progress = step / max(1, COARSE_STEPS - 1)
        wl, density, routing, overlap = model.terms(pos, progress)
        overlap_weight = OVERLAP_WEIGHT * min(1.0, progress / OVERLAP_RAMP)
        loss_value = wl + DENSITY_WEIGHT * density + CONGESTION_WEIGHT * routing + overlap_weight * overlap
        optimizer.zero_grad(set_to_none=True)
        loss_value.sum().backward()
        pos.grad.mul_(movable)
        optimizer.step()
        with torch.no_grad():
            pos.copy_(torch.where(movable, torch.maximum(model.half, torch.minimum(high, pos)), reference))
    return clamp_to_canvas((pos.detach() * model.canvas)[0], bench)


def _shaped_congestion(trace, positions: torch.Tensor) -> torch.Tensor:
    """The v2 trace model's congestion with its final top-5% mean replaced by `trace.routing_loss`."""
    v_route, h_route = trace._trace_route_congestion(positions)
    v_route = v_route / max(trace.grid_v_routes, 1e-9)
    h_route = h_route / max(trace.grid_h_routes, 1e-9)
    v_macro, h_macro = trace._macro_route_congestion(positions)
    v_macro = v_macro / max(trace.grid_v_routes, 1e-9)
    h_macro = h_macro / max(trace.grid_h_routes, 1e-9)
    v_route = trace._box_smooth(v_route, axis="V")
    h_route = trace._box_smooth(h_route, axis="H")
    loads = torch.cat([(v_route + v_macro).flatten(), (h_route + h_macro).flatten()])
    loss = trace.routing_loss
    cost = torch.topk(loads, max(1, int(loss.tail * loads.numel())))[0].mean()
    if loss.mean_weight:
        # the mean is taken over a batch of one, as in the measured code, so the float32 sum order is the same
        cost = cost + loss.mean_weight * loads[None].mean(dim=1)[0]
    return cost


def _update_footprint(proxy: FastDiffProxyGaussian, progress: float) -> None:
    """Between 20% and 70% of the schedule, narrow each density Gaussian to the variance of its rectangle."""
    blend = min(1.0, max(0.0, (progress - FOOTPRINT_START) / (FOOTPRINT_END - FOOTPRINT_START)))
    proxy.sigma_scale = 1.0 + blend * (1.0 / math.sqrt(3.0) - 1.0)
    floor_cells = 0.5 - 0.25 * blend
    proxy.sigma_floor_x = (proxy.cw / proxy.grid_cols) * floor_cells
    proxy.sigma_floor_y = (proxy.ch / proxy.grid_rows) * floor_cells


def smooth_descent(start: torch.Tensor, bench: Benchmark, plc, loss: RoutingLoss, *, steps: int,
                   learning_scale: float = 1.0, progress_from: float = 0.0, ramp_overlap: bool = True,
                   freeze_hard: bool = False) -> torch.Tensor:
    """Adam on the v2 smooth proxy over the part of the fine schedule from `progress_from` to 1."""
    proxy = FastDiffProxyGaussian(bench, plc, device="cpu", gamma_frac=GAMMA_START)
    proxy.trace.routing_loss = loss
    proxy.trace.compute_congestion = types.MethodType(_shaped_congestion, proxy.trace)
    pos = start.detach().to(torch.float32).clone().requires_grad_()
    optimizer = torch.optim.Adam([pos], lr=learning_scale * FINE_LEARNING_RATE * float(bench.canvas_width))
    frozen = bench.macro_fixed.to(dtype=torch.bool).clone()
    if freeze_hard:
        frozen[:bench.num_hard_macros] = True
    for i in range(steps):
        progress = progress_from + (1.0 - progress_from) * i / max(1, steps - 1)
        proxy.set_gamma_frac(GAMMA_START + progress * (GAMMA_END - GAMMA_START))
        _update_footprint(proxy, progress)
        ramp = min(1.0, i / max(1, steps * OVERLAP_RAMP)) if ramp_overlap else 1.0
        optimizer.zero_grad()
        loss_value, _ = fast_loss_with_penalty(proxy, pos, ramp * OVERLAP_WEIGHT, include_congestion=True,
                                               boundary_lambda=BOUNDARY_WEIGHT)
        loss_value.backward()
        pos.grad[frozen] = 0
        optimizer.step()
    return pos.detach().clone()


def fine_descent(raw: torch.Tensor, bench: Benchmark, plc, loss: RoutingLoss = DESCENT_LOSS) -> torch.Tensor:
    return smooth_descent(raw, bench, plc, loss, steps=FINE_STEPS)


def interlude(start: torch.Tensor, bench: Benchmark, plc, spec: Interlude, seed: int) -> torch.Tensor:
    """Gentle late-schedule descent of the soft clusters from a legal placement; hard macros stay put."""
    if spec.noise_um:
        generator = torch.Generator().manual_seed(1000003 * seed + 17)  # the measured runs' seeding of the kick
        noise = torch.randn(start.shape, generator=generator, dtype=torch.float32) * spec.noise_um
        still = bench.macro_fixed.to(dtype=torch.bool).clone()
        still[:bench.num_hard_macros] = True
        noise[still] = 0
        start = clamp_to_canvas(start.to(torch.float32) + noise, bench)
    return smooth_descent(start, bench, plc, spec.loss, steps=spec.steps, learning_scale=spec.learning_scale,
                          progress_from=spec.progress_from, ramp_overlap=False, freeze_hard=True)
