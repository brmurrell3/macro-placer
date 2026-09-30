"""Active smooth-proxy primitives extracted without algorithm changes."""
from dataclasses import dataclass
from typing import List

import torch

from macro_place.benchmark import Benchmark


@dataclass
class NetData:
    """Padded net data for vectorized HPWL/congestion computation."""
    pin_macro_idx: torch.Tensor   # [num_nets, max_pins] long
    pin_offsets: torch.Tensor     # [num_nets, max_pins, 2]
    mask: torch.Tensor            # [num_nets, max_pins] bool
    weights: torch.Tensor         # [num_nets]
    num_macros: int
    total_net_count: int          # for WL normalization (len(plc.nets))


def _extract_net_data(benchmark: Benchmark, plc) -> NetData:
    """Extract pin-level net connectivity from PlacementCost into padded tensors."""
    # Map PlacementCost module index -> benchmark tensor index
    plc_to_tensor = {}
    for t_idx, p_idx in enumerate(benchmark.hard_macro_indices):
        plc_to_tensor[p_idx] = t_idx
    for i, p_idx in enumerate(benchmark.soft_macro_indices):
        plc_to_tensor[p_idx] = benchmark.num_hard_macros + i

    port_idx = benchmark.num_macros  # dummy macro at origin for ports

    nets: List[List[tuple]] = []
    net_weights: List[float] = []

    for driver_name, sink_names in plc.nets.items():
        d_idx = plc.mod_name_to_indices.get(driver_name)
        if d_idx is None:
            continue
        driver_pin = plc.modules_w_pins[d_idx]
        weight = driver_pin.get_weight()

        pins = []
        for pin_name in [driver_name] + sink_names:
            pidx = plc.mod_name_to_indices.get(pin_name)
            if pidx is None:
                continue
            pin = plc.modules_w_pins[pidx]
            ptype = pin.get_type()

            if ptype in ("MACRO_PIN", "SOFT_MACRO_PIN"):
                # PLC ownership preserves hierarchical macro names.
                parent_name = pin.get_macro_name()
                parent_pidx = plc.mod_name_to_indices.get(parent_name)
                if parent_pidx is None or parent_pidx not in plc_to_tensor:
                    continue
                t_idx = plc_to_tensor[parent_pidx]
                xo, yo = pin.get_offset()
                pins.append((t_idx, xo, yo))
            elif ptype == "PORT":
                x, y = pin.get_pos()
                pins.append((port_idx, x, y))

        if len(pins) >= 2:
            nets.append(pins)
            net_weights.append(weight)

    num_nets = len(nets)
    total_net_count = len(plc.nets)

    if num_nets == 0:
        return NetData(
            pin_macro_idx=torch.zeros(0, 1, dtype=torch.long),
            pin_offsets=torch.zeros(0, 1, 2),
            mask=torch.zeros(0, 1, dtype=torch.bool),
            weights=torch.zeros(0),
            num_macros=benchmark.num_macros,
            total_net_count=max(total_net_count, 1),
        )

    max_pins = max(len(n) for n in nets)

    pin_macro_idx = torch.zeros(num_nets, max_pins, dtype=torch.long)
    pin_offsets = torch.zeros(num_nets, max_pins, 2)
    mask = torch.zeros(num_nets, max_pins, dtype=torch.bool)

    for j, net in enumerate(nets):
        for k, (midx, xo, yo) in enumerate(net):
            pin_macro_idx[j, k] = midx
            pin_offsets[j, k, 0] = xo
            pin_offsets[j, k, 1] = yo
            mask[j, k] = True

    return NetData(
        pin_macro_idx=pin_macro_idx,
        pin_offsets=pin_offsets,
        mask=mask,
        weights=torch.tensor(net_weights, dtype=torch.float32),
        num_macros=benchmark.num_macros,
        total_net_count=max(total_net_count, 1),
    )


def _lse_hpwl(positions, net_data: NetData, port_base, gamma):
    """Vectorized log-sum-exp HPWL over all nets."""
    if len(net_data.weights) == 0:
        return torch.tensor(0.0)

    # Append port base for port pins
    all_pos = torch.cat([positions, port_base], dim=0)  # [N+1, 2]

    # Gather pin positions: [num_nets, max_pins, 2]
    pin_pos = all_pos[net_data.pin_macro_idx] + net_data.pin_offsets

    pin_x = pin_pos[:, :, 0]  # [num_nets, max_pins]
    pin_y = pin_pos[:, :, 1]

    mask = net_data.mask  # [num_nets, max_pins]
    big = 1e10

    # LSE max_x: mask invalid pins with -inf
    x_for_max = pin_x.clone()
    x_for_max[~mask] = -big
    lse_max_x = gamma * torch.logsumexp(x_for_max / gamma, dim=1)

    # LSE min_x: mask invalid pins with +inf
    x_for_min = pin_x.clone()
    x_for_min[~mask] = big
    lse_min_x = -gamma * torch.logsumexp(-x_for_min / gamma, dim=1)

    # Same for y
    y_for_max = pin_y.clone()
    y_for_max[~mask] = -big
    lse_max_y = gamma * torch.logsumexp(y_for_max / gamma, dim=1)

    y_for_min = pin_y.clone()
    y_for_min[~mask] = big
    lse_min_y = -gamma * torch.logsumexp(-y_for_min / gamma, dim=1)

    # HPWL per net
    hpwl = (lse_max_x - lse_min_x) + (lse_max_y - lse_min_y)  # [num_nets]

    return (net_data.weights * hpwl).sum()


def _grid_density(positions, sizes,
                  cell_x_min, cell_x_max, cell_y_min, cell_y_max,
                  cell_area, grid_rows, grid_cols):
    """Differentiable grid density with top-10%, matching PlacementCost.

    Returns 0.5 * mean(top_10%) to match get_density_cost().
    """
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    # Macro edges: [N]
    mx_min = positions[:, 0] - half_w
    mx_max = positions[:, 0] + half_w
    my_min = positions[:, 1] - half_h
    my_max = positions[:, 1] + half_h

    # X overlap: [N, C]
    x_overlap = torch.clamp(
        torch.min(mx_max.unsqueeze(1), cell_x_max.unsqueeze(0))
        - torch.max(mx_min.unsqueeze(1), cell_x_min.unsqueeze(0)),
        min=0,
    )
    # Y overlap: [N, R]
    y_overlap = torch.clamp(
        torch.min(my_max.unsqueeze(1), cell_y_max.unsqueeze(0))
        - torch.max(my_min.unsqueeze(1), cell_y_min.unsqueeze(0)),
        min=0,
    )

    # Overlap area per macro per cell: [N, R, C]
    overlap_area = y_overlap.unsqueeze(2) * x_overlap.unsqueeze(1)

    # Density per cell
    density_grid = overlap_area.sum(dim=0) / cell_area  # [R, C]

    # Top-10% aggregation
    flat = density_grid.flatten()
    k = max(1, int(0.10 * len(flat)))
    top_k, _ = torch.topk(flat, k)

    return 0.5 * top_k.mean()


def _rudy_congestion(positions, net_data: NetData, port_base, gamma,
                     cell_x_min, cell_x_max, cell_y_min, cell_y_max,
                     grid_h_routes, grid_v_routes,
                     grid_rows, grid_cols):
    """Differentiable RUDY congestion with ABU-5%.

    Distributes routing demand uniformly across net bounding boxes.
    Batched to control memory for large net counts.
    """
    num_nets = len(net_data.weights)
    if num_nets == 0 or grid_h_routes == 0 or grid_v_routes == 0:
        return torch.tensor(0.0)

    # Pin positions
    all_pos = torch.cat([positions, port_base], dim=0)
    pin_pos = all_pos[net_data.pin_macro_idx] + net_data.pin_offsets

    pin_x = pin_pos[:, :, 0]
    pin_y = pin_pos[:, :, 1]
    mask = net_data.mask
    big = 1e10

    # Smooth bounding boxes for ALL nets (vectorized)
    x_for_max = pin_x.clone(); x_for_max[~mask] = -big
    x_for_min = pin_x.clone(); x_for_min[~mask] = big
    y_for_max = pin_y.clone(); y_for_max[~mask] = -big
    y_for_min = pin_y.clone(); y_for_min[~mask] = big

    bbox_x_max = gamma * torch.logsumexp(x_for_max / gamma, dim=1)
    bbox_x_min = -gamma * torch.logsumexp(-x_for_min / gamma, dim=1)
    bbox_y_max = gamma * torch.logsumexp(y_for_max / gamma, dim=1)
    bbox_y_min = -gamma * torch.logsumexp(-y_for_min / gamma, dim=1)

    bbox_w = (bbox_x_max - bbox_x_min).clamp(min=1e-6)
    bbox_h = (bbox_y_max - bbox_y_min).clamp(min=1e-6)

    # RUDY demand coefficients: [J]
    w = net_data.weights
    h_coeff = w / (bbox_w * grid_h_routes)
    v_coeff = w / (bbox_h * grid_v_routes)

    # Accumulate congestion in batches to control memory
    h_cong = torch.zeros(grid_rows, grid_cols)
    v_cong = torch.zeros(grid_rows, grid_cols)
    batch_size = 2000

    for b_start in range(0, num_nets, batch_size):
        b_end = min(b_start + batch_size, num_nets)
        b = slice(b_start, b_end)

        x_ol = torch.clamp(
            torch.min(bbox_x_max[b].unsqueeze(1), cell_x_max.unsqueeze(0))
            - torch.max(bbox_x_min[b].unsqueeze(1), cell_x_min.unsqueeze(0)),
            min=0,
        )  # [batch, C]
        y_ol = torch.clamp(
            torch.min(bbox_y_max[b].unsqueeze(1), cell_y_max.unsqueeze(0))
            - torch.max(bbox_y_min[b].unsqueeze(1), cell_y_min.unsqueeze(0)),
            min=0,
        )  # [batch, R]

        ol_area = y_ol.unsqueeze(2) * x_ol.unsqueeze(1)  # [batch, R, C]
        h_cong = h_cong + (h_coeff[b].unsqueeze(1).unsqueeze(2) * ol_area).sum(0)
        v_cong = v_cong + (v_coeff[b].unsqueeze(1).unsqueeze(2) * ol_area).sum(0)

    combined = h_cong + v_cong

    flat = combined.flatten()
    k = max(1, int(0.05 * len(flat)))
    top_k, _ = torch.topk(flat, k)
    return top_k.mean()

