"""
Optimal Transport Placer — v2

Switch to a custom sliced Wasserstein distance for density spreading.
Much faster than Sinkhorn (O(N log N) per slice) while still providing
OT-quality spreading gradients. Also use grid-based density like SDF v5
as a fallback comparison — hybridize OT spreading with SDF-style density.
"""

import torch
import numpy as np
import math
import random
from pathlib import Path
from macro_place.benchmark import Benchmark


def _load_plc(name):
    from macro_place.loader import load_benchmark_from_dir
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    return None


def _extract_edges(benchmark, plc):
    name_to_bidx = {}
    for bidx, idx in enumerate(benchmark.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx
    edge_dict = {}
    for driver, sinks in plc.nets.items():
        macros = set()
        for pin in [driver] + sinks:
            parent = pin.split("/")[0]
            if parent in name_to_bidx:
                macros.add(name_to_bidx[parent])
        if len(macros) >= 2:
            ml = sorted(macros)
            w = 1.0 / (len(ml) - 1)
            for i in range(len(ml)):
                for j in range(i + 1, len(ml)):
                    pair = (ml[i], ml[j])
                    edge_dict[pair] = edge_dict.get(pair, 0) + w
    if not edge_dict:
        return torch.zeros(0, 2, dtype=torch.long), torch.zeros(0)
    edges = list(edge_dict.keys())
    return (torch.tensor(edges, dtype=torch.long),
            torch.tensor([edge_dict[e] for e in edges], dtype=torch.float32))


def sdf_density_grid(pos, half_sizes, grid_x, grid_y, tau):
    N = pos.shape[0]
    gx = grid_x.unsqueeze(0)
    gy = grid_y.unsqueeze(1)
    density = torch.zeros(len(grid_y), len(grid_x), device=pos.device)
    chunk_size = 128
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        cx = pos[start:end, 0].reshape(-1, 1, 1)
        cy = pos[start:end, 1].reshape(-1, 1, 1)
        hw = half_sizes[start:end, 0].reshape(-1, 1, 1)
        hh = half_sizes[start:end, 1].reshape(-1, 1, 1)
        dx = torch.abs(gx.unsqueeze(0) - cx) - hw
        dy = torch.abs(gy.unsqueeze(0) - cy) - hh
        occ = torch.sigmoid(-dx / tau) * torch.sigmoid(-dy / tau)
        density = density + occ.sum(dim=0)
    return density


def sliced_wasserstein_2d(source_pos, source_weights, target_pos, target_weights, n_slices=32):
    """Compute sliced Wasserstein-2 distance between weighted 2D point clouds.

    Projects onto random 1D directions and computes 1D OT (closed-form via sorting).
    Differentiable through source_pos.
    """
    # Generate random directions on unit circle
    angles = torch.linspace(0, math.pi, n_slices + 1)[:-1]  # deterministic for reproducibility
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)  # [S, 2]

    total_cost = torch.tensor(0.0, device=source_pos.device)

    for d in range(n_slices):
        direction = directions[d]  # [2]

        # Project points onto direction
        src_proj = (source_pos * direction).sum(dim=1)  # [N_src]
        tgt_proj = (target_pos * direction).sum(dim=1)  # [N_tgt]

        # Sort source projections (with weights)
        src_sorted_idx = torch.argsort(src_proj)
        src_sorted = src_proj[src_sorted_idx]
        src_w_sorted = source_weights[src_sorted_idx]

        # Sort target projections (with weights)
        tgt_sorted_idx = torch.argsort(tgt_proj)
        tgt_sorted = tgt_proj[tgt_sorted_idx]
        tgt_w_sorted = target_weights[tgt_sorted_idx]

        # Compute 1D Wasserstein via quantile matching
        # For unequal sizes, interpolate CDFs
        src_cdf = torch.cumsum(src_w_sorted, dim=0)
        tgt_cdf = torch.cumsum(tgt_w_sorted, dim=0)

        # Sample at source CDF values — find matching target quantiles
        # For each source point, find where its CDF value falls in target CDF
        n_src = len(src_sorted)
        for i in range(n_src):
            # Binary search in target CDF for src_cdf[i]
            cdf_val = src_cdf[i]
            # Find matching target position by interpolation
            idx = torch.searchsorted(tgt_cdf, cdf_val.unsqueeze(0)).item()
            idx = min(idx, len(tgt_sorted) - 1)
            total_cost = total_cost + src_w_sorted[i] * (src_sorted[i] - tgt_sorted[idx]) ** 2

    return total_cost / n_slices


def sliced_wasserstein_fast(source_pos, source_weights, target_pos, target_weights, n_slices=16):
    """Fast vectorized sliced Wasserstein using quantile interpolation."""
    # Deterministic directions
    angles = torch.linspace(0, math.pi, n_slices + 1, device=source_pos.device)[:-1]
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)  # [S, 2]

    # Project all points at once: [S, N]
    src_proj = source_pos @ directions.T  # [N_src, S]
    tgt_proj = target_pos @ directions.T  # [N_tgt, S]

    total_cost = torch.tensor(0.0, device=source_pos.device)

    # Use uniform quantile matching: resample both distributions onto M quantile points
    M = 64  # number of quantile points
    quantiles = torch.linspace(0, 1, M + 2, device=source_pos.device)[1:-1]  # exclude 0,1

    for s in range(n_slices):
        # Sort source
        src_s = src_proj[:, s]
        src_idx = torch.argsort(src_s)
        src_sorted = src_s[src_idx]
        src_w = source_weights[src_idx]
        src_cdf = torch.cumsum(src_w, dim=0)

        # Sort target
        tgt_s = tgt_proj[:, s]
        tgt_idx = torch.argsort(tgt_s)
        tgt_sorted = tgt_s[tgt_idx]
        tgt_w = target_weights[tgt_idx]
        tgt_cdf = torch.cumsum(tgt_w, dim=0)

        # Interpolate quantile positions for both distributions
        # Find positions at each quantile level
        src_q_idx = torch.searchsorted(src_cdf, quantiles)
        src_q_idx = src_q_idx.clamp(0, len(src_sorted) - 1)
        src_q_vals = src_sorted[src_q_idx]

        tgt_q_idx = torch.searchsorted(tgt_cdf, quantiles)
        tgt_q_idx = tgt_q_idx.clamp(0, len(tgt_sorted) - 1)
        tgt_q_vals = tgt_sorted[tgt_q_idx]

        # W2 distance for this slice
        total_cost = total_cost + ((src_q_vals - tgt_q_vals) ** 2).mean()

    return total_cost / n_slices


def edge_wl(pos, edges, edge_weights):
    if len(edges) == 0:
        return torch.tensor(0.0, device=pos.device)
    src = pos[edges[:, 0]]
    dst = pos[edges[:, 1]]
    dist = torch.abs(src - dst).sum(dim=1)
    return (edge_weights * dist).sum()


def overlap_penalty(pos, half_sizes):
    N = pos.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=pos.device)
    dx = torch.abs(pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0))
    dy = torch.abs(pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0))
    sep_x = half_sizes[:, 0].unsqueeze(1) + half_sizes[:, 0].unsqueeze(0)
    sep_y = half_sizes[:, 1].unsqueeze(1) + half_sizes[:, 1].unsqueeze(0)
    overlap_area = torch.relu(sep_x - dx) * torch.relu(sep_y - dy)
    mask = torch.triu(torch.ones(N, N, device=pos.device, dtype=torch.bool), diagonal=1)
    return (overlap_area * mask).sum()


def greedy_legalize(pos_np, sizes_np, movable, canvas_w, canvas_h):
    n = len(pos_np)
    half_w = sizes_np[:, 0] / 2
    half_h = sizes_np[:, 1] / 2
    sep_x = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2
    sep_y = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2
    order = sorted(range(n), key=lambda i: -sizes_np[i, 0] * sizes_np[i, 1])
    placed = np.zeros(n, dtype=bool)
    legal = pos_np.copy()
    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue
        if placed.any():
            dx_arr = np.abs(legal[idx, 0] - legal[:, 0])
            dy_arr = np.abs(legal[idx, 1] - legal[:, 1])
            c = (dx_arr < sep_x[idx] + 0.05) & (dy_arr < sep_y[idx] + 0.05) & placed
            c[idx] = False
            if not c.any():
                placed[idx] = True
                continue
        step = max(sizes_np[idx, 0], sizes_np[idx, 1]) * 0.2
        best_p = legal[idx].copy()
        best_d = float('inf')
        for r in range(1, 250):
            found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = np.clip(pos_np[idx, 0] + dxm * step, half_w[idx], canvas_w - half_w[idx])
                    cy = np.clip(pos_np[idx, 1] + dym * step, half_h[idx], canvas_h - half_h[idx])
                    if placed.any():
                        dx_arr = np.abs(cx - legal[:, 0])
                        dy_arr = np.abs(cy - legal[:, 1])
                        c = (dx_arr < sep_x[idx] + 0.05) & (dy_arr < sep_y[idx] + 0.05) & placed
                        c[idx] = False
                        if c.any():
                            continue
                    d = (cx - pos_np[idx, 0]) ** 2 + (cy - pos_np[idx, 1]) ** 2
                    if d < best_d:
                        best_d = d
                        best_p = np.array([cx, cy])
                        found = True
                    if found:
                        break
                if found:
                    break
            if found:
                break
        legal[idx] = best_p
        placed[idx] = True
    return legal


class OTPlacer:
    """Hybrid OT + SDF density placer v2 — fast sliced Wasserstein."""

    def __init__(self, seed=42, n_iters=500, lr=3.0,
                 density_weight=0.4, overlap_weight=80.0,
                 grid_resolution=32, ot_weight=0.3, n_slices=16):
        self.seed = seed
        self.n_iters = n_iters
        self.lr = lr
        self.density_weight = density_weight
        self.overlap_weight = overlap_weight
        self.grid_resolution = grid_resolution
        self.ot_weight = ot_weight
        self.n_slices = n_slices

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes = benchmark.macro_sizes[:n_hard].clone()
        half_sizes = sizes / 2
        movable = benchmark.get_movable_mask()[:n_hard]

        plc = _load_plc(benchmark.name)
        if plc is not None:
            edges, edge_weights = _extract_edges(benchmark, plc)
        else:
            edges = torch.zeros(0, 2, dtype=torch.long)
            edge_weights = torch.zeros(0)

        canvas_diag = math.sqrt(cw ** 2 + ch ** 2)
        tau_start = canvas_diag * 0.05
        tau_end = canvas_diag * 0.002

        grid_res = self.grid_resolution
        grid_x = torch.linspace(0, cw, grid_res)
        grid_y = torch.linspace(0, ch, grid_res)

        # Create uniform target for OT
        n_target = max(n_hard * 2, 100)
        side = int(math.ceil(math.sqrt(n_target)))
        txs = torch.linspace(0, cw, side + 2)[1:-1]
        tys = torch.linspace(0, ch, side + 2)[1:-1]
        target_pts = torch.stack(torch.meshgrid(txs, tys, indexing='xy'), dim=-1).reshape(-1, 2)
        target_weights = torch.ones(len(target_pts)) / len(target_pts)

        # Macro area weights for OT
        areas = sizes[:, 0] * sizes[:, 1]
        macro_weights = areas / areas.sum()

        init_pos = benchmark.macro_positions[:n_hard].clone()
        init_pos[:, 0] = init_pos[:, 0].clamp(half_sizes[:, 0], cw - half_sizes[:, 0])
        init_pos[:, 1] = init_pos[:, 1].clamp(half_sizes[:, 1], ch - half_sizes[:, 1])

        with torch.no_grad():
            wl0 = edge_wl(init_pos, edges, edge_weights)
            wl_norm = max(wl0.item(), 1e-6)

        pos = init_pos.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([pos], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.n_iters, eta_min=self.lr * 0.03)

        best_pos = init_pos.clone()
        best_cost = float('inf')

        for step in range(self.n_iters):
            optimizer.zero_grad()
            frac = step / max(self.n_iters - 1, 1)
            tau = tau_start * (tau_end / tau_start) ** frac

            current_pos = torch.where(movable.unsqueeze(1).expand_as(pos), pos, init_pos)
            clamped = torch.stack([
                current_pos[:, 0].clamp(half_sizes[:, 0], cw - half_sizes[:, 0]),
                current_pos[:, 1].clamp(half_sizes[:, 1], ch - half_sizes[:, 1]),
            ], dim=1)

            # SDF density (top-10% penalty like SDF v5)
            density_grid = sdf_density_grid(clamped, half_sizes, grid_x, grid_y, tau)
            flat_density = density_grid.flatten()
            k_density = max(1, int(0.10 * len(flat_density)))
            top_density, _ = torch.topk(flat_density, k_density)
            density_loss = top_density.mean()

            # Sliced Wasserstein OT spreading
            ot_loss = sliced_wasserstein_fast(
                clamped, macro_weights, target_pts, target_weights, self.n_slices)

            # Wirelength
            wl_loss = edge_wl(clamped, edges, edge_weights) / wl_norm

            # Overlap
            ov_w = self.overlap_weight * (0.05 + 0.95 * frac ** 0.5)
            ov_loss = overlap_penalty(clamped, half_sizes) / (cw * ch)

            # Density weight schedule
            dw = self.density_weight * (1.5 - 0.8 * frac)
            # OT weight: strong early (global spreading), fades later
            ot_w = self.ot_weight * max(0, 1.0 - frac * 1.5)

            loss = wl_loss + dw * density_loss + ot_w * ot_loss / (canvas_diag ** 2) + ov_w * ov_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                if step % 20 == 0 or step == self.n_iters - 1:
                    ov_val = overlap_penalty(clamped, half_sizes).item()
                    cost = wl_loss.item() + 0.1 * ov_val / (cw * ch)
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = clamped.detach().clone()

        with torch.no_grad():
            final_pos = torch.where(movable.unsqueeze(1).expand_as(best_pos), best_pos, init_pos)
            final_pos = torch.stack([
                final_pos[:, 0].clamp(half_sizes[:, 0], cw - half_sizes[:, 0]),
                final_pos[:, 1].clamp(half_sizes[:, 1], ch - half_sizes[:, 1]),
            ], dim=1)

        pos_np = final_pos.numpy().astype(np.float64)
        sizes_np = sizes.numpy().astype(np.float64)
        legal_pos = greedy_legalize(pos_np, sizes_np, movable.numpy(), cw, ch)

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(legal_pos, dtype=torch.float32)
        return full_pos
