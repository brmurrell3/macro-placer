"""
Optimal Transport Placer — v12

True OT placement: start from RANDOM positions, use OT spreading to
achieve uniform density, combined with WL to respect netlist.
Then legalize. This tests the actual OT hypothesis properly.

Phase 1 (spreading): high OT weight, low WL → spread macros uniformly
Phase 2 (refinement): reduce OT, increase WL → optimize connectivity
Phase 3: legalize
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
        legal[idx] = best_p
        placed[idx] = True
    return legal


class OTPlacer:
    """True OT placer v12 — from random init, OT spreading + WL, then legalize."""

    def __init__(self, seed=42, n_iters=800, lr=2.0,
                 density_weight=0.5, ot_weight=200.0,
                 overlap_weight=100.0,
                 grid_resolution=32, blur=0.02, scaling=0.7):
        self.seed = seed
        self.n_iters = n_iters
        self.lr = lr
        self.density_weight = density_weight
        self.ot_weight = ot_weight
        self.overlap_weight = overlap_weight
        self.grid_resolution = grid_resolution
        self.blur = blur
        self.scaling = scaling

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
        tau_start = canvas_diag * 0.08
        tau_end = canvas_diag * 0.002

        grid_res = self.grid_resolution
        grid_x = torch.linspace(0, cw, grid_res)
        grid_y = torch.linspace(0, ch, grid_res)

        # OT setup
        n_target = max(n_hard * 3, 150)
        side = int(math.ceil(math.sqrt(n_target)))
        txs = torch.linspace(0, 1, side + 2)[1:-1]
        tys = torch.linspace(0, 1, side + 2)[1:-1]
        target_pts = torch.stack(torch.meshgrid(txs, tys, indexing='xy'), dim=-1).reshape(-1, 2)
        n_target = len(target_pts)
        target_weights = torch.ones(n_target) / n_target

        areas = sizes[:, 0] * sizes[:, 1]
        macro_weights = areas / areas.sum()

        ot_loss_fn = SamplesLoss(
            loss="sinkhorn", p=2, blur=self.blur,
            scaling=self.scaling, debias=True)

        scale_inv = torch.tensor([1.0 / cw, 1.0 / ch])

        # Start from INITIAL POSITIONS (not random — they carry netlist structure)
        # but with small perturbation to escape the local minimum
        init_pos = benchmark.macro_positions[:n_hard].clone()
        init_pos[:, 0] = init_pos[:, 0].clamp(half_sizes[:, 0], cw - half_sizes[:, 0])
        init_pos[:, 1] = init_pos[:, 1].clamp(half_sizes[:, 1], ch - half_sizes[:, 1])
        fixed_pos = init_pos.clone()  # save for fixed macros

        with torch.no_grad():
            wl0 = edge_wl(init_pos, edges, edge_weights)
            wl_norm = max(wl0.item(), 1e-6)

        pos = init_pos.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([pos], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.n_iters, eta_min=self.lr * 0.02)

        best_pos = init_pos.clone()
        best_cost = float('inf')

        for step in range(self.n_iters):
            optimizer.zero_grad()
            frac = step / max(self.n_iters - 1, 1)
            tau = tau_start * (tau_end / tau_start) ** frac

            current_pos = torch.where(movable.unsqueeze(1).expand_as(pos), pos, fixed_pos)
            clamped = torch.stack([
                current_pos[:, 0].clamp(half_sizes[:, 0], cw - half_sizes[:, 0]),
                current_pos[:, 1].clamp(half_sizes[:, 1], ch - half_sizes[:, 1]),
            ], dim=1)

            # SDF density (top-10%)
            density_grid = sdf_density_grid(clamped, half_sizes, grid_x, grid_y, tau)
            flat_density = density_grid.flatten()
            k_density = max(1, int(0.10 * len(flat_density)))
            top_density, _ = torch.topk(flat_density, k_density)
            density_loss = top_density.mean()

            # OT spreading in [0,1] space
            normalized_pos = clamped * scale_inv
            ot_loss = ot_loss_fn(macro_weights, normalized_pos, target_weights, target_pts)

            # WL
            wl_loss = edge_wl(clamped, edges, edge_weights) / wl_norm

            # Overlap — start strong and stay strong
            ov_loss = overlap_penalty(clamped, half_sizes) / (cw * ch)

            # Phase schedule:
            # Phase 1 (0-40%): strong OT + density, moderate WL
            # Phase 2 (40-100%): WL dominates, OT fades
            if frac < 0.4:
                phase_frac = frac / 0.4
                dw = self.density_weight * (1.5 - 0.5 * phase_frac)
                ot_w = self.ot_weight * (1.0 - 0.3 * phase_frac)
                wl_w = 0.3 + 0.7 * phase_frac
            else:
                phase_frac = (frac - 0.4) / 0.6
                dw = self.density_weight * (1.0 - 0.5 * phase_frac)
                ot_w = self.ot_weight * (0.7 - 0.6 * phase_frac)
                wl_w = 1.0

            loss = wl_w * wl_loss + dw * density_loss + ot_w * ot_loss + self.overlap_weight * ov_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                if step % 10 == 0 or step == self.n_iters - 1:
                    ov_val = overlap_penalty(clamped, half_sizes).item()
                    # Selection: balance WL and density
                    cost = wl_loss.item() + 0.5 * density_loss.item() + 100 * ov_val / (cw * ch)
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = clamped.detach().clone()

        with torch.no_grad():
            final_pos = torch.where(movable.unsqueeze(1).expand_as(best_pos), best_pos, fixed_pos)
            final_pos = torch.stack([
                final_pos[:, 0].clamp(half_sizes[:, 0], cw - half_sizes[:, 0]),
                final_pos[:, 1].clamp(half_sizes[:, 1], ch - half_sizes[:, 1]),
            ], dim=1)

        dist = ((final_pos - init_pos) ** 2).sum(dim=1).sqrt()
        print(f"  [diag] avg_move={dist.mean().item():.2f}μm max={dist.max().item():.2f}μm")

        # Legalize
        pos_np = final_pos.numpy().astype(np.float64)
        sizes_np = sizes.numpy().astype(np.float64)
        legal_pos = greedy_legalize(pos_np, sizes_np, movable.numpy(), cw, ch)

        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(legal_pos, dtype=torch.float32)
        return full_pos
