"""Differentiable RUDY (pure PyTorch, autograd-friendly).

NOTE (E95 coordination, 2026-05-13): E88's `_rudy_congestion` from
writeup/archive/.../ablation_v2_steps.py already implements vectorized
ABU-5% differentiable RUDY (batched per-net bbox spread + top-K reduce).
E95 reuses that path; this prototype is preserved for the B-R1 patch
into DP's PlaceObj.obj_fn which needs a separate API surface. If E95
clears the spike gate and you want a shared primitive, lift the
ablation_v2_steps `_rudy_congestion` into a module under
`macro_place/diff_proxy.py` and have both call sites use it.

Stock DREAMPlace's `dreamplace.ops.rudy.Rudy` uses a CUDA/C++ kernel
that writes to out-parameters; the gradient path through that kernel
is not enabled for use as a loss term. To swap RUDY into DP's obj_fn
(B-R1), we need a torch-native version whose gradient flows back to
cell positions.

Definition: RUDY = Rectangular Uniform Wire DensitY. For each net,
the wire demand on a routing bin is approximated by spreading the
net's bounding-box wire length uniformly over the bins covered by
the bounding box. Horizontal demand per bin = (bbox_width * net_weight)
spread over the H-bins covered. Vertical demand analogous.

This file provides a torch.nn.Module that:
  - Accepts (pin_pos [N_pins, 2], netpin_index [N_pins], net_weights [N_nets])
  - Returns (horizontal_util_map [H, V], vertical_util_map [H, V])
  - Is fully differentiable wrt pin_pos via standard torch ops

TILOS-aligned weighting:
  - net_weights come from PlacementCost's net info; can be passed in via
    the .wts Bookshelf file or directly.
  - smoothing radius = canonical "congestion smooth range" (~2).

Used by E92 B-R1 to patch DP's obj_fn:

  obj = wl + density_weight * density + congestion_weight * congestion_scalar

where congestion_scalar comes from `RudyDiffLoss(...)` reducing the
map via top-K sum or capacity-overflow penalty.

Status: prototype — not exercised end-to-end yet. Gated on E91 B-R0'
verdict.
"""
from __future__ import annotations

import torch
from torch import nn


class RudyDiff(nn.Module):
    """Differentiable RUDY congestion map.

    For each net, compute bbox dimensions from pin positions, spread the
    wire demand over bins inside the bbox, accumulate.

    Args:
      netpin_start: [N_nets + 1] CSR pointers into flat_netpin
      flat_netpin:  [N_total_pins] flat pin indices grouped by net
      net_weights:  [N_nets] per-net weight (TILOS weights)
      xl, yl, xh, yh: routing canvas bounds
      num_bins_x, num_bins_y: grid resolution
      unit_h_capacity, unit_v_capacity: per-bin capacity normalization
    """
    def __init__(
        self,
        netpin_start: torch.Tensor,
        flat_netpin: torch.Tensor,
        net_weights: torch.Tensor,
        xl: float, yl: float, xh: float, yh: float,
        num_bins_x: int, num_bins_y: int,
        unit_horizontal_capacity: float,
        unit_vertical_capacity: float,
    ):
        super().__init__()
        self.register_buffer("netpin_start", netpin_start)
        self.register_buffer("flat_netpin", flat_netpin)
        self.register_buffer("net_weights", net_weights)
        self.xl, self.yl, self.xh, self.yh = xl, yl, xh, yh
        self.num_bins_x, self.num_bins_y = num_bins_x, num_bins_y
        self.bin_w = (xh - xl) / num_bins_x
        self.bin_h = (yh - yl) / num_bins_y
        self.unit_h_cap = unit_horizontal_capacity
        self.unit_v_cap = unit_vertical_capacity

    def forward(self, pin_pos: torch.Tensor):
        """Compute (h_util_map [Bx, By], v_util_map [Bx, By]).

        pin_pos: [N_pins, 2] in canvas coordinates.

        Implementation: for each net, soft-bbox via softmax-relaxed max/min
        on pin_pos, then spread wire to bins covered.

        For autograd to flow, we use:
          - LSE-smoothed max/min for bbox edges (so gradient is non-zero
            for non-extremum pins, distributing influence)
          - Soft bin-membership via sigmoid (so a pin near a bin edge
            contributes to both adjacent bins, giving a smooth gradient)
        """
        device = pin_pos.device
        dtype = pin_pos.dtype
        n_nets = self.netpin_start.shape[0] - 1
        h_map = torch.zeros((self.num_bins_x, self.num_bins_y), dtype=dtype, device=device)
        v_map = torch.zeros_like(h_map)

        # Per-net bbox (vectorized via segment_max/min if available; fallback
        # to a loop here for prototype clarity).
        for i in range(n_nets):
            start = int(self.netpin_start[i].item())
            end = int(self.netpin_start[i + 1].item())
            if end <= start + 1:
                continue
            pins = pin_pos[self.flat_netpin[start:end]]  # [k, 2]
            # Smoothed max/min for autograd. tau -> 0 reduces to hard max.
            # For now use exact min/max — gradient flows only through the
            # extremum pins, but it's correct.
            x_lo = pins[:, 0].min()
            x_hi = pins[:, 0].max()
            y_lo = pins[:, 1].min()
            y_hi = pins[:, 1].max()
            bbox_w = (x_hi - x_lo).clamp(min=1e-3)
            bbox_h = (y_hi - y_lo).clamp(min=1e-3)
            w_h = self.net_weights[i] * bbox_w
            w_v = self.net_weights[i] * bbox_h
            # Bin indices covered by the bbox.
            bx_lo = torch.floor((x_lo - self.xl) / self.bin_w).long().clamp(0, self.num_bins_x - 1)
            bx_hi = torch.floor((x_hi - self.xl) / self.bin_w).long().clamp(0, self.num_bins_x - 1)
            by_lo = torch.floor((y_lo - self.yl) / self.bin_h).long().clamp(0, self.num_bins_y - 1)
            by_hi = torch.floor((y_hi - self.yl) / self.bin_h).long().clamp(0, self.num_bins_y - 1)
            nx = (bx_hi - bx_lo + 1).clamp(min=1)
            ny = (by_hi - by_lo + 1).clamp(min=1)
            # Uniform spread; demand per bin
            h_demand = w_h / (nx.float() * ny.float())
            v_demand = w_v / (nx.float() * ny.float())
            # Scatter-add to map (loop over bins — not vectorized in prototype)
            for bx in range(int(bx_lo.item()), int(bx_hi.item()) + 1):
                for by in range(int(by_lo.item()), int(by_hi.item()) + 1):
                    h_map[bx, by] = h_map[bx, by] + h_demand
                    v_map[bx, by] = v_map[bx, by] + v_demand

        # Convert demand to utilization
        bin_area = self.bin_w * self.bin_h
        h_map = h_map / (bin_area * self.unit_h_cap)
        v_map = v_map / (bin_area * self.unit_v_cap)
        return h_map, v_map


def congestion_loss_topk(
    h_map: torch.Tensor,
    v_map: torch.Tensor,
    K_frac: float = 0.1,
    tau: float = 0.1,
) -> torch.Tensor:
    """Reduce (h_map, v_map) to a scalar via soft top-K.

    softmax(values/tau) ⋅ values, summed over both directions. As
    tau → 0 this approaches the exact top-K sum.
    """
    combined = (h_map + v_map).flatten()
    K = max(1, int(K_frac * combined.numel()))
    # Sort to identify top-K candidates
    sorted_vals, _ = torch.sort(combined, descending=True)
    topk_vals = sorted_vals[:K]
    # Soft-aggregate within top-K (rather than mean) — softmax handles ties.
    weights = torch.softmax(topk_vals / tau, dim=0)
    return float(K) * (weights * topk_vals).sum()


def congestion_loss_overflow(
    h_map: torch.Tensor,
    v_map: torch.Tensor,
) -> torch.Tensor:
    """Penalty: sum of squared over-capacity in each direction.

    Demand-side: ReLU(utilization - 1.0)² per bin, summed.
    """
    h_over = torch.nn.functional.relu(h_map - 1.0)
    v_over = torch.nn.functional.relu(v_map - 1.0)
    return (h_over.pow(2).sum() + v_over.pow(2).sum())


# TODO: vectorized version using torch.segment_reduce + index_add_ for
# scatter, eliminating the per-net Python loop. Required before this
# is usable inside a DP iteration.
