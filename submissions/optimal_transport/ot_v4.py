"""
Optimal Transport Placer — v4

Pure OT approach: NO SDF density term. Only sliced Wasserstein + WL + overlap.
This tests whether the OT gradient is actually working and providing useful signal.
If results differ from v5 SDF, we know OT gradients flow. Then we can tune the blend.
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


def sliced_wasserstein(source_pts, target_pts, n_slices=32):
    """Differentiable sliced W2 via sort-and-match."""
    N = min(source_pts.shape[0], target_pts.shape[0])
    src = source_pts[:N]
    tgt = target_pts[:N]

    angles = torch.linspace(0, math.pi, n_slices + 1, device=src.device)[:-1]
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    src_proj = src @ directions.T  # [N, S]
    tgt_proj = tgt @ directions.T  # [N, S]

    # Sort along each slice
    src_sorted = torch.sort(src_proj, dim=0).values  # [N, S]
    tgt_sorted = torch.sort(tgt_proj, dim=0).values  # [N, S]

    # Vectorized: squared diff of sorted projections
    cost = ((src_sorted - tgt_sorted) ** 2).mean()
    return cost


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
    """Pure OT placer v4 — sliced Wasserstein only, no SDF density."""

    def __init__(self, seed=42, n_iters=500, lr=3.0,
                 ot_weight=1.0, overlap_weight=80.0,
                 n_slices=32, ot_points=256):
        self.seed = seed
        self.n_iters = n_iters
        self.lr = lr
        self.ot_weight = ot_weight
        self.overlap_weight = overlap_weight
        self.n_slices = n_slices
        self.ot_points = ot_points

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

        # Create uniform target points for OT
        n_pts = self.ot_points
        side = int(math.ceil(math.sqrt(n_pts)))
        txs = torch.linspace(0, cw, side + 2)[1:-1]
        tys = torch.linspace(0, ch, side + 2)[1:-1]
        target_pts = torch.stack(torch.meshgrid(txs, tys, indexing='xy'), dim=-1).reshape(-1, 2)
        target_pts = target_pts[:n_pts]

        # Precompute expansion counts (area-proportional)
        areas = sizes[:, 0] * sizes[:, 1]
        fracs = areas / areas.sum()
        counts = (fracs * n_pts).clamp(min=1).long()
        diff = n_pts - counts.sum().item()
        if diff > 0:
            _, topk_idx = torch.topk(fracs, min(diff, len(fracs)))
            counts[topk_idx] += 1
        elif diff < 0:
            _, topk_idx = torch.topk(fracs, min(-diff, len(fracs)), largest=False)
            counts[topk_idx] = (counts[topk_idx] - 1).clamp(min=1)

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

        # Debug: check if OT gradients flow
        debug_step = 0

        for step in range(self.n_iters):
            optimizer.zero_grad()
            frac = step / max(self.n_iters - 1, 1)

            current_pos = torch.where(movable.unsqueeze(1).expand_as(pos), pos, init_pos)
            clamped = torch.stack([
                current_pos[:, 0].clamp(half_sizes[:, 0], cw - half_sizes[:, 0]),
                current_pos[:, 1].clamp(half_sizes[:, 1], ch - half_sizes[:, 1]),
            ], dim=1)

            # OT spreading: expand macros to area-weighted points
            expanded = torch.repeat_interleave(clamped, counts, dim=0)
            ot_loss = sliced_wasserstein(expanded, target_pts, self.n_slices)

            # Wirelength
            wl_loss = edge_wl(clamped, edges, edge_weights) / wl_norm

            # Overlap
            ov_w = self.overlap_weight * (0.05 + 0.95 * frac ** 0.5)
            ov_loss = overlap_penalty(clamped, half_sizes) / (cw * ch)

            # OT weight schedule: strong early, reduce later
            ot_w = self.ot_weight * (1.5 - frac)

            # Normalize OT loss to be comparable to WL (both ~1.0 scale)
            loss = wl_loss + ot_w * ot_loss / (canvas_diag ** 2 / 4) + ov_w * ov_loss
            loss.backward()

            if step == 0:
                grad_norm = pos.grad.norm().item() if pos.grad is not None else 0
                ot_val = ot_loss.item()
                wl_val = wl_loss.item()
                print(f"  [debug] step=0 wl={wl_val:.4f} ot={ot_val:.1f} "
                      f"ot_norm={ot_val/(canvas_diag**2/4):.4f} grad_norm={grad_norm:.4f}")

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
