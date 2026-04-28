# STATUS: SUPERSEDED 2026-04-27. DPOv2StepsPlacer was the v2-restore variant
# (1.3888 --all alone, 1.3834 inside best_of_v2). Replaced by CDAdaptive (1.1055).
# Imported by other ablations via subclassing — kept as the reference DPO body.
"""
DPO Placer — v2 step count ablation

Restores higher step counts (less aggressive reduction for large benchmarks).
The v3 step_scale aggressively cuts steps for high-complexity benchmarks
(0.25 for >1e9 complexity), which hurts quality. This variant applies a
floor of 0.6 to step_scale, restoring behavior closer to v2 which achieved
1.4107 avg proxy cost.

Only change vs placer.py: step_scale = max(0.6, computed_scale)
"""

import torch
import numpy as np
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List

from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# PlacementCost loading
# ---------------------------------------------------------------------------

def _load_plc(name):
    from macro_place.loader import load_benchmark_from_dir
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    return None


# ---------------------------------------------------------------------------
# Net data extraction
# ---------------------------------------------------------------------------

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
                parent_name = pin_name.split("/")[0]
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


# ---------------------------------------------------------------------------
# Legalization (overlap resolution)
# ---------------------------------------------------------------------------

def _legalize(pos_np, sizes_np, movable, n_hard, canvas_w, canvas_h,
              max_rounds=20, eps=0.002):
    """Iterative overlap repair for hard macros."""
    pos = pos_np.copy()
    half_w = sizes_np[:n_hard, 0] / 2
    half_h = sizes_np[:n_hard, 1] / 2

    for _ in range(max_rounds):
        dx_mat = np.abs(pos[:n_hard, 0:1] - pos[:n_hard, 0:1].T)
        dy_mat = np.abs(pos[:n_hard, 1:2] - pos[:n_hard, 1:2].T)
        min_dx = half_w[:, None] + half_w[None, :] + eps
        min_dy = half_h[:, None] + half_h[None, :] + eps
        overlap = (dx_mat < min_dx) & (dy_mat < min_dy)
        np.fill_diagonal(overlap, False)

        pairs = np.argwhere(np.triu(overlap))
        if len(pairs) == 0:
            break

        for a, b in pairs:
            mov_a = movable[a]
            mov_b = movable[b]
            if not (mov_a or mov_b):
                continue
            viol_x = min_dx[a, b] - dx_mat[a, b]
            viol_y = min_dy[a, b] - dy_mat[a, b]
            if viol_x < viol_y:
                sign = 1.0 if pos[a, 0] < pos[b, 0] else -1.0
                if mov_a and mov_b:
                    pos[a, 0] -= sign * viol_x / 2
                    pos[b, 0] += sign * viol_x / 2
                elif mov_a:
                    pos[a, 0] -= sign * viol_x
                else:
                    pos[b, 0] += sign * viol_x
            else:
                sign = 1.0 if pos[a, 1] < pos[b, 1] else -1.0
                if mov_a and mov_b:
                    pos[a, 1] -= sign * viol_y / 2
                    pos[b, 1] += sign * viol_y / 2
                elif mov_a:
                    pos[a, 1] -= sign * viol_y
                else:
                    pos[b, 1] += sign * viol_y

        # Clamp to canvas
        for i in range(n_hard):
            if movable[i]:
                pos[i, 0] = np.clip(pos[i, 0], half_w[i] + eps,
                                    canvas_w - half_w[i] - eps)
                pos[i, 1] = np.clip(pos[i, 1], half_h[i] + eps,
                                    canvas_h - half_h[i] - eps)
    return pos


# ---------------------------------------------------------------------------
# DPO v2-Steps Placer
# ---------------------------------------------------------------------------

class DPOv2StepsPlacer:
    """DPO placer with restored v2 step counts (less aggressive reduction)."""

    def __init__(self, seed=42, use_sdf_init=True):
        self.seed = seed
        self.use_sdf_init = use_sdf_init

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Load PlacementCost for pin/net data
        plc = _load_plc(benchmark.name)
        if plc is None:
            return benchmark.macro_positions.clone()

        # Extract net data
        net_data = _extract_net_data(benchmark, plc)

        # Initial placement
        if self.use_sdf_init:
            init_pos = self._sdf_init(benchmark)
        else:
            init_pos = benchmark.macro_positions.clone()

        # Run DPO optimization
        optimized = self._optimize(init_pos, benchmark, net_data)

        # Legalize (resolve hard macro overlaps)
        result = self._do_legalize(optimized, benchmark)
        return result

    def _sdf_init(self, benchmark):
        """Run SDF placer for initial positions."""
        import importlib.util
        sdf_path = Path(__file__).parent.parent / "polyhedra" / "init" / "sdf.py"
        spec = importlib.util.spec_from_file_location("sdf", str(sdf_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.SDFPlacer(seed=self.seed).place(benchmark)

    # -----------------------------------------------------------------------
    # Core optimization loop
    # -----------------------------------------------------------------------

    def _optimize(self, init_pos, benchmark, net_data):
        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes = benchmark.macro_sizes
        half_sizes = sizes / 2
        movable = benchmark.get_movable_mask()
        fixed_mask = ~movable

        # WL normalization: (W+H) * total_net_count
        wl_norm = (cw + ch) * net_data.total_net_count

        # Grid setup
        grid_rows = benchmark.grid_rows
        grid_cols = benchmark.grid_cols
        cell_w = cw / grid_cols
        cell_h = ch / grid_rows
        cell_area = cell_w * cell_h
        cell_x_min = torch.arange(grid_cols, dtype=torch.float32) * cell_w
        cell_x_max = cell_x_min + cell_w
        cell_y_min = torch.arange(grid_rows, dtype=torch.float32) * cell_h
        cell_y_max = cell_y_min + cell_h

        # Routing capacity
        grid_h_routes = cell_h * benchmark.hroutes_per_micron
        grid_v_routes = cell_w * benchmark.vroutes_per_micron

        # Port base (fixed at origin for port pin position computation)
        port_base = torch.zeros(1, 2)

        # Benchmark size metrics for adaptive parameters
        n_movable = int(movable.sum().item())
        num_nets = len(net_data.weights)
        cached_cong = 0.0

        # Congestion gradient frequency: compute full congestion backward pass
        # only every K steps to save compute on large benchmarks
        cong_grad_freq = 1 if num_nets < 3000 else (3 if num_nets < 8000 else 5)

        # Optimization variable
        pos = init_pos.clone().detach().requires_grad_(True)
        complexity = n_movable * num_nets * grid_rows * grid_cols

        # v2-steps restoration: compute step_scale but apply a floor of 0.6
        # so large benchmarks still get sufficient optimization steps
        if complexity > 1e9:
            step_scale = 0.25
        elif complexity > 5e8:
            step_scale = 0.45
        elif complexity > 1e8:
            step_scale = 0.65
        else:
            step_scale = 1.3  # Extra steps for small/medium benchmarks

        # KEY CHANGE: Apply floor of 0.6 to prevent over-aggressive step reduction
        step_scale = max(0.6, step_scale)

        def s(n): return max(40, int(n * step_scale))

        # Phase schedule: (steps, gamma_frac_of_W, overlap_lambda, lr)
        phases = [
            (s(250), 0.01,   1.0,     0.5),   # Phase 1: Exploration
            (s(300), 0.002,  50.0,    0.2),   # Phase 2: Refinement
            (s(150), 0.0005, 500.0,   0.1),   # Phase 3: Sharpening
        ]

        best_pos = init_pos.clone()
        best_proxy = float("inf")

        for phase_idx, (n_steps, gamma_frac, lam, lr) in enumerate(phases):
            gamma = gamma_frac * cw

            optimizer = torch.optim.Adam([pos], lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, n_steps, eta_min=lr * 0.05
            )

            for step in range(n_steps):
                optimizer.zero_grad()

                # Enforce fixed macros
                with torch.no_grad():
                    pos.data[fixed_mask] = init_pos[fixed_mask]

                # Clamp to canvas bounds
                clamped = torch.stack([
                    pos[:, 0].clamp(half_sizes[:, 0], cw - half_sizes[:, 0]),
                    pos[:, 1].clamp(half_sizes[:, 1], ch - half_sizes[:, 1]),
                ], dim=1)

                # --- Differentiable proxy cost components ---
                wl = _lse_hpwl(clamped, net_data, port_base, gamma) / wl_norm

                density = _grid_density(clamped, sizes,
                                        cell_x_min, cell_x_max,
                                        cell_y_min, cell_y_max,
                                        cell_area, grid_rows, grid_cols)

                if step % cong_grad_freq == 0:
                    congestion = _rudy_congestion(clamped, net_data, port_base,
                                                  gamma,
                                                  cell_x_min, cell_x_max,
                                                  cell_y_min, cell_y_max,
                                                  grid_h_routes, grid_v_routes,
                                                  grid_rows, grid_cols)
                    cached_cong = congestion.item()
                else:
                    congestion = torch.tensor(cached_cong)

                overlap = _overlap_penalty(clamped[:n_hard], half_sizes[:n_hard])
                overlap_norm = overlap / (cw * ch)

                # Composite loss
                proxy = wl + 0.5 * density + 0.5 * congestion
                loss = proxy + lam * overlap_norm

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_([pos], max_norm=20.0)

                optimizer.step()
                scheduler.step()

                # Track best
                with torch.no_grad():
                    proxy_val = proxy.item()
                    ov_val = overlap_norm.item()
                    # Weight overlap in score — heavier in later phases
                    score = proxy_val + max(1.0, lam * 0.1) * ov_val
                    if score < best_proxy:
                        best_proxy = score
                        bp = pos.data.clone()
                        bp[fixed_mask] = init_pos[fixed_mask]
                        bp[:, 0].clamp_(half_sizes[:, 0], cw - half_sizes[:, 0])
                        bp[:, 1].clamp_(half_sizes[:, 1], ch - half_sizes[:, 1])
                        best_pos = bp

                # Diagnostics (first step of each phase only)
                if step == 0:
                    with torch.no_grad():
                        print(f"    P{phase_idx+1}: wl={wl.item():.3f} "
                              f"den={density.item():.3f} "
                              f"cong={congestion.item():.3f}")

            # Phase summary
            with torch.no_grad():
                ov_count = _count_overlaps(best_pos[:n_hard], half_sizes[:n_hard])
                print(f"  DPO phase {phase_idx+1} done: best_score={best_proxy:.4f}  "
                      f"overlaps={ov_count}")

        return best_pos

    # -----------------------------------------------------------------------
    # Legalization
    # -----------------------------------------------------------------------

    def _do_legalize(self, positions, benchmark):
        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes = benchmark.macro_sizes.numpy().astype(np.float64)
        movable = benchmark.get_movable_mask().numpy()

        pos_np = positions.detach().numpy().astype(np.float64)
        legal = _legalize(pos_np, sizes, movable, n_hard, cw, ch)

        result = positions.clone()
        result[:n_hard] = torch.tensor(legal[:n_hard], dtype=torch.float32)
        return result


# ---------------------------------------------------------------------------
# Differentiable components
# ---------------------------------------------------------------------------

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


def _overlap_penalty(positions, half_sizes):
    """Pairwise overlap penalty for hard macros. Fully differentiable."""
    n = positions.shape[0]
    if n < 2:
        return torch.tensor(0.0)

    dx = torch.abs(positions[:, 0].unsqueeze(1) - positions[:, 0].unsqueeze(0))
    dy = torch.abs(positions[:, 1].unsqueeze(1) - positions[:, 1].unsqueeze(0))
    sep_x = half_sizes[:, 0].unsqueeze(1) + half_sizes[:, 0].unsqueeze(0)
    sep_y = half_sizes[:, 1].unsqueeze(1) + half_sizes[:, 1].unsqueeze(0)
    overlap_area = torch.relu(sep_x - dx) * torch.relu(sep_y - dy)

    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    return (overlap_area * mask).sum()


def _count_overlaps(positions, half_sizes, eps=0.01):
    """Count overlapping hard macro pairs (non-differentiable)."""
    n = positions.shape[0]
    if n < 2:
        return 0
    with torch.no_grad():
        dx = torch.abs(positions[:, 0].unsqueeze(1) - positions[:, 0].unsqueeze(0))
        dy = torch.abs(positions[:, 1].unsqueeze(1) - positions[:, 1].unsqueeze(0))
        sep_x = half_sizes[:, 0].unsqueeze(1) + half_sizes[:, 0].unsqueeze(0)
        sep_y = half_sizes[:, 1].unsqueeze(1) + half_sizes[:, 1].unsqueeze(0)
        overlap = (dx < sep_x - eps) & (dy < sep_y - eps)
        overlap.fill_diagonal_(False)
        return int(overlap.triu().sum().item())
