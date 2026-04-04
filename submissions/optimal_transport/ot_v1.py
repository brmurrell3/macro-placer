"""
Optimal Transport Placer — v1

Replace SDF density spreading with Wasserstein-2 optimal transport via Sinkhorn divergence.
OT minimizes transport cost to reach uniform density, which should correlate with
lower congestion since it distributes routing demand more evenly.

Based on SDF v5 structure (best so far at 1.5002), swapping density term.
"""

import torch
import numpy as np
import math
import random
from pathlib import Path
from geomloss import SamplesLoss
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


def create_weighted_points(pos, half_sizes, canvas_w, canvas_h):
    """Create area-weighted point cloud from macro positions for OT.

    Each macro is represented by points proportional to its area,
    so larger macros contribute more to the density distribution.
    """
    areas = (2 * half_sizes[:, 0]) * (2 * half_sizes[:, 1])
    total_area = areas.sum()
    # Normalize weights to sum to 1
    weights = areas / total_area
    return pos, weights


def create_uniform_target(n_points, canvas_w, canvas_h):
    """Create uniformly distributed target point cloud on the canvas."""
    # Grid of uniformly spaced points
    side = int(math.ceil(math.sqrt(n_points)))
    xs = torch.linspace(0, canvas_w, side + 2)[1:-1]
    ys = torch.linspace(0, canvas_h, side + 2)[1:-1]
    grid = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), dim=-1).reshape(-1, 2)
    # Take exactly n_points
    grid = grid[:n_points]
    weights = torch.ones(len(grid)) / len(grid)
    return grid, weights


class OTPlacer:
    """Optimal Transport density spreading placer v1."""

    def __init__(self, seed=42, n_iters=500, lr=3.0,
                 ot_weight=0.4, overlap_weight=80.0,
                 blur=0.05, scaling=0.9):
        self.seed = seed
        self.n_iters = n_iters
        self.lr = lr
        self.ot_weight = ot_weight
        self.overlap_weight = overlap_weight
        self.blur = blur  # Sinkhorn blur (epsilon regularization)
        self.scaling = scaling  # Sinkhorn multiscale scaling

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

        # Normalize positions to [0, 1] for Sinkhorn stability
        scale = torch.tensor([cw, ch])

        # Create uniform target distribution (more points = finer resolution)
        n_target = max(n_hard * 4, 200)
        target_pts, target_weights = create_uniform_target(n_target, cw, ch)

        # Sinkhorn divergence loss — blur controls regularization
        # Use area-weighted macro positions vs uniform target
        ot_loss_fn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=self.blur * canvas_diag,
            scaling=self.scaling,
            debias=True,
        )

        with torch.no_grad():
            init_pos = benchmark.macro_positions[:n_hard].clone()
            init_pos[:, 0] = init_pos[:, 0].clamp(half_sizes[:, 0], cw - half_sizes[:, 0])
            init_pos[:, 1] = init_pos[:, 1].clamp(half_sizes[:, 1], ch - half_sizes[:, 1])
            wl0 = edge_wl(init_pos, edges, edge_weights)
            wl_norm = max(wl0.item(), 1e-6)

        pos = init_pos.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([pos], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.n_iters, eta_min=self.lr * 0.03)

        best_pos = init_pos.clone()
        best_cost = float('inf')

        # Compute area weights (constant)
        areas = sizes[:, 0] * sizes[:, 1]
        macro_weights = areas / areas.sum()

        for step in range(self.n_iters):
            optimizer.zero_grad()
            frac = step / max(self.n_iters - 1, 1)

            current_pos = torch.where(movable.unsqueeze(1).expand_as(pos), pos, init_pos)
            clamped = torch.stack([
                current_pos[:, 0].clamp(half_sizes[:, 0], cw - half_sizes[:, 0]),
                current_pos[:, 1].clamp(half_sizes[:, 1], ch - half_sizes[:, 1]),
            ], dim=1)

            # OT density: Sinkhorn divergence between macro distribution and uniform
            ot_loss = ot_loss_fn(macro_weights, clamped, target_weights, target_pts)

            # Wirelength
            wl_loss = edge_wl(clamped, edges, edge_weights) / wl_norm

            # Overlap (ramp up)
            ov_w = self.overlap_weight * (0.05 + 0.95 * frac ** 0.5)
            ov_loss = overlap_penalty(clamped, half_sizes) / (cw * ch)

            # OT weight: higher early for spreading, lower late for WL refinement
            ot_w = self.ot_weight * (1.5 - 0.8 * frac)

            loss = wl_loss + ot_w * ot_loss / (canvas_diag ** 2) + ov_w * ov_loss
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
