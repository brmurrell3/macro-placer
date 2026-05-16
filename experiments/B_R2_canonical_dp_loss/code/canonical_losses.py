"""Differentiable canonical density + congestion losses for DREAMPlace.

Stock DP `obj_fn` = wirelength + density_weight * eDensity. The canonical
proxy uses top-K density (sum of K highest bin densities) and RUDY
congestion (per-net bbox wire spread, top-K), not electrostatic spread.

This module provides torch-native diff losses:

- `topk_density_loss(positions, sizes, canvas_bounds, bins, K_frac, tau)`
- `rudy_congestion_loss(positions, sizes, flat_netpin, netpin_start,
                        net_weights, canvas_bounds, bins, ...)`

Both are autograd-friendly. Designed to be added to DP's obj_fn as:

    obj = wl + density_weight * eDensity
        + lambda_topk * topk_density_loss(...)
        + lambda_rudy * rudy_congestion_loss(...)

Calibration of lambda values is done by matching canonical proxy
deltas on a fixed placement (see calibrate.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _soft_bin_membership(coord_lo, coord_hi, bin_centers, bin_w):
    """Soft membership of segment [coord_lo, coord_hi] over bins.

    Returns [N_bins] tensor of overlap fractions per bin.
    Differentiable via clamp/relu operations.

    Args:
      coord_lo, coord_hi: scalars (or matching tensors)
      bin_centers: [N_bins] tensor of bin center coords
      bin_w: scalar bin width

    Returns: [N_bins] overlap fractions in [0, 1].
    """
    bin_lo = bin_centers - 0.5 * bin_w
    bin_hi = bin_centers + 0.5 * bin_w
    overlap = (torch.minimum(coord_hi, bin_hi) - torch.maximum(coord_lo, bin_lo)).clamp(min=0.0)
    return overlap / bin_w


def topk_density_loss(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    xl: float, yl: float, xh: float, yh: float,
    num_bins_x: int = 32, num_bins_y: int = 32,
    K_frac: float = 0.05,
    tau: float = 0.05,
    return_map: bool = False,
):
    """Differentiable top-K density loss.

    Computes a 2D occupancy map (cell-area per bin / bin-area = fractional
    density). Reduces via soft top-K: `softmax(map/tau) * map` summed
    over the K_frac highest bins.

    Args:
      positions: [N, 2] cell centers (autograd leaf)
      sizes:     [N, 2] cell sizes (fixed)
      xl, yl, xh, yh: canvas bounds
      num_bins_x, num_bins_y: grid resolution (default 32×32 ≈ TILOS gridsize)
      K_frac: fraction of top bins to sum (canonical uses top-5%)
      tau: softmax temperature; smaller = harder top-K
      return_map: if True also return the [Bx, By] density map

    Returns: scalar loss (or (loss, map) if return_map).
    """
    device = positions.device
    dtype = positions.dtype
    bin_w = (xh - xl) / num_bins_x
    bin_h = (yh - yl) / num_bins_y
    bin_area = bin_w * bin_h
    # bin centers
    bx = xl + bin_w * (0.5 + torch.arange(num_bins_x, dtype=dtype, device=device))
    by = yl + bin_h * (0.5 + torch.arange(num_bins_y, dtype=dtype, device=device))
    # Cell extents
    cx_lo = positions[:, 0] - 0.5 * sizes[:, 0]
    cx_hi = positions[:, 0] + 0.5 * sizes[:, 0]
    cy_lo = positions[:, 1] - 0.5 * sizes[:, 1]
    cy_hi = positions[:, 1] + 0.5 * sizes[:, 1]
    # Soft bin membership per cell, per axis.
    # [N, Bx]: overlap_x[i, j] = (min(cx_hi[i], bx[j]+w/2) - max(cx_lo[i], bx[j]-w/2)).clamp(min=0)
    bx_lo = bx - 0.5 * bin_w
    bx_hi = bx + 0.5 * bin_w
    by_lo = by - 0.5 * bin_h
    by_hi = by + 0.5 * bin_h
    overlap_x = (torch.minimum(cx_hi.unsqueeze(1), bx_hi.unsqueeze(0))
                 - torch.maximum(cx_lo.unsqueeze(1), bx_lo.unsqueeze(0))).clamp(min=0.0)
    overlap_y = (torch.minimum(cy_hi.unsqueeze(1), by_hi.unsqueeze(0))
                 - torch.maximum(cy_lo.unsqueeze(1), by_lo.unsqueeze(0))).clamp(min=0.0)
    # Density map: [Bx, By] = sum over cells of overlap_x[i, bx] * overlap_y[i, by]
    # einsum: 'ij,ik->jk' but we need to multiply outer-product-style.
    # cell_contrib[i, bx, by] = overlap_x[i, bx] * overlap_y[i, by]; sum over i.
    density_map = torch.einsum('ij,ik->jk', overlap_x, overlap_y) / bin_area
    # Top-K reduction
    flat = density_map.flatten()
    K = max(1, int(round(K_frac * flat.numel())))
    sorted_vals, _ = torch.sort(flat, descending=True)
    topk = sorted_vals[:K]
    # Soft top-K: weighted by softmax over the top-K values
    weights = F.softmax(topk / tau, dim=0)
    loss = (weights * topk).sum() * K
    if return_map:
        return loss, density_map
    return loss


def rudy_congestion_loss(
    positions: torch.Tensor,
    netpin_start: torch.Tensor,
    flat_netpin: torch.Tensor,
    net_weights: torch.Tensor,
    xl: float, yl: float, xh: float, yh: float,
    num_bins_x: int = 32, num_bins_y: int = 32,
    K_frac: float = 0.05,
    tau: float = 0.05,
    return_map: bool = False,
):
    """Differentiable RUDY congestion loss.

    For each net, bbox wire demand spreads uniformly over bins covered by
    the bbox. Demand per bin = (bbox_w * net_weight) / num_bins_covered
    for horizontal direction, analogous for vertical. Reduced via soft
    top-K (same as density).

    Args:
      positions: [N_pins, 2] pin positions (since macros have pins at center,
        for our use this is N_macros, 2 — pin = macro center)
      netpin_start: [N_nets+1] CSR pointers
      flat_netpin: [N_total_pins] flat pin indices grouped by net
      net_weights: [N_nets] per-net weight
      xl,yl,xh,yh: canvas bounds
      num_bins_x, num_bins_y: routing grid
      K_frac, tau: top-K reduction params

    Returns: scalar loss (or (loss, h_map, v_map))
    """
    device = positions.device
    dtype = positions.dtype
    n_nets = netpin_start.shape[0] - 1
    bin_w = (xh - xl) / num_bins_x
    bin_h = (yh - yl) / num_bins_y
    bin_area = bin_w * bin_h
    # bin edges for soft membership
    bx = xl + bin_w * (0.5 + torch.arange(num_bins_x, dtype=dtype, device=device))
    by = yl + bin_h * (0.5 + torch.arange(num_bins_y, dtype=dtype, device=device))
    bx_lo = bx - 0.5 * bin_w
    bx_hi = bx + 0.5 * bin_w
    by_lo = by - 0.5 * bin_h
    by_hi = by + 0.5 * bin_h

    h_map = torch.zeros((num_bins_x, num_bins_y), dtype=dtype, device=device)
    v_map = torch.zeros_like(h_map)

    # Vectorized: compute per-net bbox via segment_reduce-style indexing.
    # For each net: x_lo = min(pin_x), x_hi = max(pin_x), similarly y.
    # We iterate nets (typically << 1e5) — Python loop OK for our scale.
    for i in range(n_nets):
        start = int(netpin_start[i].item())
        end = int(netpin_start[i + 1].item())
        if end <= start + 1:
            continue
        pins = positions[flat_netpin[start:end]]  # [k, 2]
        x_lo, x_hi = pins[:, 0].min(), pins[:, 0].max()
        y_lo, y_hi = pins[:, 1].min(), pins[:, 1].max()
        # Soft bin membership over the bbox in each axis.
        # [Bx]: how much of [x_lo, x_hi] overlaps bin j
        over_x = (torch.minimum(x_hi, bx_hi) - torch.maximum(x_lo, bx_lo)).clamp(min=0.0) / bin_w
        over_y = (torch.minimum(y_hi, by_hi) - torch.maximum(y_lo, by_lo)).clamp(min=0.0) / bin_h
        # Total bbox area in routing-grid-units (soft count of bins covered)
        nx = over_x.sum().clamp(min=1.0)
        ny = over_y.sum().clamp(min=1.0)
        bbox_w = (x_hi - x_lo).clamp(min=1e-3)
        bbox_h = (y_hi - y_lo).clamp(min=1e-3)
        w_demand = net_weights[i] * bbox_w / (nx * ny)
        v_demand = net_weights[i] * bbox_h / (nx * ny)
        # Outer product gives per-bin demand contribution
        # [Bx, By] += over_x[:, None] * over_y[None, :] * demand
        h_map = h_map + (over_x.unsqueeze(1) * over_y.unsqueeze(0)) * w_demand
        v_map = v_map + (over_x.unsqueeze(1) * over_y.unsqueeze(0)) * v_demand

    combined = (h_map + v_map).flatten()
    K = max(1, int(round(K_frac * combined.numel())))
    sorted_vals, _ = torch.sort(combined, descending=True)
    topk = sorted_vals[:K]
    weights = F.softmax(topk / tau, dim=0)
    loss = (weights * topk).sum() * K
    if return_map:
        return loss, h_map, v_map
    return loss


def calibrate_lambdas(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    netpin_start: torch.Tensor,
    flat_netpin: torch.Tensor,
    net_weights: torch.Tensor,
    xl: float, yl: float, xh: float, yh: float,
    wirelength: float,  # reference DP wirelength scale
    num_bins_x: int = 32, num_bins_y: int = 32,
):
    """Compute lambda_topk and lambda_rudy such that each loss term
    contributes ~equal magnitude to total objective at the given placement.

    Targets:
      lambda_topk * topk_loss ≈ wirelength
      lambda_rudy * rudy_loss ≈ 0.5 * wirelength  (half weight for congestion)
    """
    with torch.no_grad():
        topk = topk_density_loss(positions, sizes, xl, yl, xh, yh,
                                  num_bins_x, num_bins_y)
        rudy = rudy_congestion_loss(positions, netpin_start, flat_netpin,
                                     net_weights, xl, yl, xh, yh,
                                     num_bins_x, num_bins_y)
    lambda_topk = float(wirelength) / (float(topk) + 1e-9)
    lambda_rudy = 0.5 * float(wirelength) / (float(rudy) + 1e-9)
    return lambda_topk, lambda_rudy
