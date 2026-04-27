"""
Batched-Seeds DPO Placer (E5 from docs/closing_the_gap.md)

Vectorizes the DPO gradient loop across B seeds simultaneously by adding a
leading batch axis to every tensor (positions, losses, gradients). Runs on
M3 MPS so the per-iteration speedup is real GPU batching, not a serial loop.

Best-of-N selection: after legalization we evaluate the proxy cost for each
slot and return the placement of the best one.

Same `place(self, benchmark) -> Tensor` interface as the other DPO placers.
Default B=64 — pass `batch_size` to the constructor to sweep.

Notes on MPS:
  * Fall back to CPU automatically if MPS is unavailable or if a critical op
    raises NotImplementedError. Common gotchas: scatter_reduce on certain
    dtypes, topk on extremely large flat tensors, certain gather index
    layouts.
  * Per-slot RNG seeding uses CPU torch.Generator → CPU init noise → cast to
    target device, because MPS does not yet support per-tensor generators.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


# ---------------------------------------------------------------------------
# Helpers (PlacementCost / net extraction)
# ---------------------------------------------------------------------------

def _load_plc(name):
    from macro_place.loader import load_benchmark_from_dir
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    return None


class NetData:
    """Padded net data for vectorized HPWL/congestion computation."""
    __slots__ = ("pin_macro_idx", "pin_offsets", "mask", "weights",
                 "num_macros", "total_net_count")

    def __init__(self, pin_macro_idx, pin_offsets, mask, weights,
                 num_macros, total_net_count):
        self.pin_macro_idx = pin_macro_idx
        self.pin_offsets = pin_offsets
        self.mask = mask
        self.weights = weights
        self.num_macros = num_macros
        self.total_net_count = total_net_count


def _extract_net_data(benchmark: Benchmark, plc) -> NetData:
    plc_to_tensor = {}
    for t_idx, p_idx in enumerate(benchmark.hard_macro_indices):
        plc_to_tensor[p_idx] = t_idx
    for i, p_idx in enumerate(benchmark.soft_macro_indices):
        plc_to_tensor[p_idx] = benchmark.num_hard_macros + i

    port_idx = benchmark.num_macros  # dummy port-base macro at origin

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
    total_net_count = max(len(plc.nets), 1)
    if num_nets == 0:
        return NetData(
            pin_macro_idx=torch.zeros(0, 1, dtype=torch.long),
            pin_offsets=torch.zeros(0, 1, 2),
            mask=torch.zeros(0, 1, dtype=torch.bool),
            weights=torch.zeros(0),
            num_macros=benchmark.num_macros,
            total_net_count=total_net_count,
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
        total_net_count=total_net_count,
    )


# ---------------------------------------------------------------------------
# Batched differentiable cost components: leading axis is B (slots)
# ---------------------------------------------------------------------------

def _batched_lse_hpwl(positions, net_data: NetData, port_base, gamma):
    """positions: [B, N, 2]. Returns [B] HPWL totals (weighted)."""
    if len(net_data.weights) == 0:
        return torch.zeros(positions.shape[0], device=positions.device)

    # Append port base across the batch
    B = positions.shape[0]
    pb = port_base.expand(B, -1, -1)         # [B, 1, 2]
    all_pos = torch.cat([positions, pb], dim=1)  # [B, N+1, 2]

    # Index expand: [num_nets, max_pins] -> gather from [B, N+1, 2]
    # Use advanced indexing: all_pos[:, idx] -> [B, num_nets, max_pins, 2]
    pin_pos = all_pos[:, net_data.pin_macro_idx, :] + net_data.pin_offsets
    pin_x = pin_pos[..., 0]   # [B, J, P]
    pin_y = pin_pos[..., 1]

    mask = net_data.mask      # [J, P]
    big = 1e10

    x_for_max = torch.where(mask, pin_x, torch.full_like(pin_x, -big))
    x_for_min = torch.where(mask, pin_x, torch.full_like(pin_x,  big))
    y_for_max = torch.where(mask, pin_y, torch.full_like(pin_y, -big))
    y_for_min = torch.where(mask, pin_y, torch.full_like(pin_y,  big))

    lse_xmax =  gamma * torch.logsumexp( x_for_max / gamma, dim=2)
    lse_xmin = -gamma * torch.logsumexp(-x_for_min / gamma, dim=2)
    lse_ymax =  gamma * torch.logsumexp( y_for_max / gamma, dim=2)
    lse_ymin = -gamma * torch.logsumexp(-y_for_min / gamma, dim=2)

    hpwl = (lse_xmax - lse_xmin) + (lse_ymax - lse_ymin)  # [B, J]
    return (net_data.weights.unsqueeze(0) * hpwl).sum(dim=1)  # [B]


def _batched_grid_density(positions, sizes,
                          cell_x_min, cell_x_max,
                          cell_y_min, cell_y_max,
                          cell_area, grid_rows, grid_cols):
    """positions: [B, N, 2]. Returns [B] density score."""
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    mx_min = positions[..., 0] - half_w   # [B, N]
    mx_max = positions[..., 0] + half_w
    my_min = positions[..., 1] - half_h
    my_max = positions[..., 1] + half_h

    # x_overlap: [B, N, C]
    x_overlap = torch.clamp(
        torch.minimum(mx_max.unsqueeze(-1), cell_x_max)
        - torch.maximum(mx_min.unsqueeze(-1), cell_x_min),
        min=0,
    )
    # y_overlap: [B, N, R]
    y_overlap = torch.clamp(
        torch.minimum(my_max.unsqueeze(-1), cell_y_max)
        - torch.maximum(my_min.unsqueeze(-1), cell_y_min),
        min=0,
    )
    # [B, N, R, C]
    overlap_area = y_overlap.unsqueeze(-1) * x_overlap.unsqueeze(-2)
    density_grid = overlap_area.sum(dim=1) / cell_area  # [B, R, C]
    flat = density_grid.reshape(positions.shape[0], -1)  # [B, R*C]
    k = max(1, int(0.10 * flat.shape[1]))
    top_k, _ = torch.topk(flat, k, dim=1)
    return 0.5 * top_k.mean(dim=1)


def _batched_rudy_congestion(positions, net_data: NetData, port_base, gamma,
                             cell_x_min, cell_x_max,
                             cell_y_min, cell_y_max,
                             grid_h_routes, grid_v_routes,
                             grid_rows, grid_cols, batch_nets=2000):
    """positions: [B, N, 2]. Returns [B] congestion score."""
    B = positions.shape[0]
    num_nets = len(net_data.weights)
    if num_nets == 0 or grid_h_routes == 0 or grid_v_routes == 0:
        return torch.zeros(B, device=positions.device)

    pb = port_base.expand(B, -1, -1)
    all_pos = torch.cat([positions, pb], dim=1)
    pin_pos = all_pos[:, net_data.pin_macro_idx, :] + net_data.pin_offsets
    pin_x = pin_pos[..., 0]
    pin_y = pin_pos[..., 1]
    mask = net_data.mask
    big = 1e10

    x_for_max = torch.where(mask, pin_x, torch.full_like(pin_x, -big))
    x_for_min = torch.where(mask, pin_x, torch.full_like(pin_x,  big))
    y_for_max = torch.where(mask, pin_y, torch.full_like(pin_y, -big))
    y_for_min = torch.where(mask, pin_y, torch.full_like(pin_y,  big))

    bbox_x_max =  gamma * torch.logsumexp( x_for_max / gamma, dim=2)  # [B, J]
    bbox_x_min = -gamma * torch.logsumexp(-x_for_min / gamma, dim=2)
    bbox_y_max =  gamma * torch.logsumexp( y_for_max / gamma, dim=2)
    bbox_y_min = -gamma * torch.logsumexp(-y_for_min / gamma, dim=2)

    bbox_w = (bbox_x_max - bbox_x_min).clamp(min=1e-6)
    bbox_h = (bbox_y_max - bbox_y_min).clamp(min=1e-6)

    w = net_data.weights.unsqueeze(0)  # [1, J]
    h_coeff = w / (bbox_w * grid_h_routes)
    v_coeff = w / (bbox_h * grid_v_routes)

    h_cong = torch.zeros(B, grid_rows, grid_cols, device=positions.device)
    v_cong = torch.zeros(B, grid_rows, grid_cols, device=positions.device)

    for b_start in range(0, num_nets, batch_nets):
        b_end = min(b_start + batch_nets, num_nets)
        sl = slice(b_start, b_end)

        x_ol = torch.clamp(
            torch.minimum(bbox_x_max[:, sl, None], cell_x_max)
            - torch.maximum(bbox_x_min[:, sl, None], cell_x_min),
            min=0,
        )  # [B, b, C]
        y_ol = torch.clamp(
            torch.minimum(bbox_y_max[:, sl, None], cell_y_max)
            - torch.maximum(bbox_y_min[:, sl, None], cell_y_min),
            min=0,
        )  # [B, b, R]

        ol_area = y_ol.unsqueeze(-1) * x_ol.unsqueeze(-2)  # [B, b, R, C]
        h_cong = h_cong + (h_coeff[:, sl, None, None] * ol_area).sum(dim=1)
        v_cong = v_cong + (v_coeff[:, sl, None, None] * ol_area).sum(dim=1)

    combined = (h_cong + v_cong).reshape(B, -1)
    k = max(1, int(0.05 * combined.shape[1]))
    top_k, _ = torch.topk(combined, k, dim=1)
    return top_k.mean(dim=1)


def _batched_overlap_penalty(positions, half_sizes):
    """positions: [B, n_hard, 2]. Returns [B]."""
    B, n, _ = positions.shape
    if n < 2:
        return torch.zeros(B, device=positions.device)
    dx = torch.abs(positions[..., 0:1] - positions[..., 0:1].transpose(1, 2))
    dy = torch.abs(positions[..., 1:2] - positions[..., 1:2].transpose(1, 2))
    sep_x = half_sizes[:, 0:1] + half_sizes[:, 0:1].T
    sep_y = half_sizes[:, 1:2] + half_sizes[:, 1:2].T
    overlap_area = torch.relu(sep_x - dx) * torch.relu(sep_y - dy)
    triu = torch.triu(torch.ones(n, n, dtype=torch.bool,
                                 device=positions.device), diagonal=1)
    return (overlap_area * triu).sum(dim=(1, 2))


# ---------------------------------------------------------------------------
# Legalization (numpy, per-slot)
# ---------------------------------------------------------------------------

def _legalize(pos_np, sizes_np, movable, n_hard, canvas_w, canvas_h,
              max_rounds=20, eps=0.002):
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
            mov_a = movable[a]; mov_b = movable[b]
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
        for i in range(n_hard):
            if movable[i]:
                pos[i, 0] = np.clip(pos[i, 0], half_w[i] + eps,
                                    canvas_w - half_w[i] - eps)
                pos[i, 1] = np.clip(pos[i, 1], half_h[i] + eps,
                                    canvas_h - half_h[i] - eps)
    return pos


# ---------------------------------------------------------------------------
# Device pick / fallback
# ---------------------------------------------------------------------------

def _pick_device(prefer_mps=True):
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# SDF init helper (re-uses existing SDF placer for one base placement;
# we then perturb per slot)
# ---------------------------------------------------------------------------

def _sdf_init(benchmark, seed):
    import importlib.util
    sdf_path = Path(__file__).parent.parent / "polyhedra" / "init" / "sdf.py"
    spec = importlib.util.spec_from_file_location("sdf", str(sdf_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SDFPlacer(seed=seed).place(benchmark)


# ---------------------------------------------------------------------------
# Batched DPO Placer
# ---------------------------------------------------------------------------

class BatchedSeedsPlacer:
    """Vectorize DPO across B seeds on MPS, return best slot's placement."""

    def __init__(self, batch_size: int = 64, seed: int = 42,
                 device: str = "auto", verbose: bool = True,
                 eval_top_k: int = 8):
        """
        eval_top_k: number of best-scoring (per the cheap differentiable
        proxy) slots to evaluate with the expensive PlacementCost-based
        proxy. Setting this to a small constant keeps wall clock from
        being dominated by O(B) eval overhead (each PlacementCost eval is
        ~1.5s on ibm01).
        """
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.verbose = verbose
        self.eval_top_k = int(eval_top_k)
        if device == "auto":
            self.device = _pick_device()
        else:
            self.device = torch.device(device)

    # -----------------------------------------------------------------------
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        plc = _load_plc(benchmark.name)
        if plc is None:
            return benchmark.macro_positions.clone()

        net_data = _extract_net_data(benchmark, plc)

        # SDF init once on CPU (cheap relative to DPO loop)
        base_init = _sdf_init(benchmark, self.seed).detach().cpu()

        # Build B-batched init: SDF base + per-slot Gaussian perturbation on
        # movable hard macros only. The first slot is the unperturbed SDF
        # init so B=1 reproduces the unperturbed baseline within seed noise.
        B = self.batch_size
        N = benchmark.num_macros
        n_hard = benchmark.num_hard_macros
        movable_cpu = benchmark.get_movable_mask().cpu()
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        # Per-slot Gaussian perturbation around the SDF init. Slot 0 is the
        # unperturbed baseline. Slots 1..B-1 use a geometric spread of
        # sigmas so we cover both tight (likely converges to same basin)
        # and looser (escapes to a different basin) initializations.
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        if B == 1:
            sigma_per_slot = torch.zeros(B)
        else:
            sigma_per_slot = torch.empty(B)
            sigma_per_slot[0] = 0.0
            sigma_per_slot[1:] = torch.exp(
                torch.linspace(math.log(0.3), math.log(2.0), B - 1)
            )
        sigma_base = 0.04 * min(cw, ch)
        sigmas = sigma_per_slot.view(B, 1, 1) * sigma_base
        noise = torch.randn(B, N, 2, generator=gen) * sigmas  # CPU
        noise[0] = 0.0
        noise[:, ~movable_cpu, :] = 0.0
        if n_hard < N:
            noise[:, n_hard:, :] = 0.0

        init_b = base_init.unsqueeze(0).expand(B, -1, -1).clone() + noise


        # Move data to device. Wrap in try/except to detect MPS breakage.
        device = self.device
        mps_failed = False
        try:
            optimized_b, best_score_b = self._optimize_batched(
                init_b, base_init, benchmark, net_data, device
            )
        except (NotImplementedError, RuntimeError) as e:
            if device.type == "mps":
                if self.verbose:
                    print(f"  MPS error during DPO: {e!r} — falling back to CPU")
                device = torch.device("cpu")
                mps_failed = True
                self.device = device
                optimized_b, best_score_b = self._optimize_batched(
                    init_b, base_init, benchmark, net_data, device
                )
            else:
                raise

        # Pick top-K candidates by the cheap differentiable proxy, only
        # then evaluate them with the (expensive) PlacementCost proxy.
        K = max(1, min(self.eval_top_k, B))
        topk_scores, topk_idx = torch.topk(best_score_b, K, largest=False)
        candidate_idx = topk_idx.cpu().tolist()

        sizes_np = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
        movable_np = movable_cpu.numpy()
        opt_np = optimized_b.detach().cpu().numpy().astype(np.float64)

        best_cost = float("inf")
        best_idx = -1
        best_legal = None
        eval_costs = []
        for s in candidate_idx:
            legal = _legalize(opt_np[s], sizes_np, movable_np,
                              n_hard, cw, ch)
            cand = torch.tensor(legal, dtype=torch.float32)
            res = compute_proxy_cost(cand, benchmark, plc)
            cost = res["proxy_cost"]
            eval_costs.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_idx = s
                best_legal = legal

        if self.verbose:
            arr = np.array(eval_costs)
            ds = best_score_b.detach().cpu().numpy()
            print(f"  >> batched B={B} dev={self.device.type} "
                  f"top{K}_best={best_cost:.4f} top{K}_mean={arr.mean():.4f} "
                  f"diff_proxy_min={ds.min():.4f} diff_proxy_max={ds.max():.4f} "
                  f"best_slot={best_idx}"
                  + ("  [mps_fallback]" if mps_failed else ""))

        if best_legal is None:
            best_legal = _legalize(opt_np[0], sizes_np, movable_np,
                                   n_hard, cw, ch)
        return torch.tensor(best_legal, dtype=torch.float32)

    # -----------------------------------------------------------------------
    def _optimize_batched(self, init_b, base_init, benchmark, net_data, device):
        B, N, _ = init_b.shape
        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        sizes = benchmark.macro_sizes.to(device)
        half_sizes = sizes / 2
        movable = benchmark.get_movable_mask().to(device)
        fixed_mask = ~movable

        wl_norm = (cw + ch) * net_data.total_net_count

        grid_rows = benchmark.grid_rows
        grid_cols = benchmark.grid_cols
        cell_w = cw / grid_cols
        cell_h = ch / grid_rows
        cell_area = cell_w * cell_h

        cell_x_min = torch.arange(grid_cols, dtype=torch.float32,
                                  device=device) * cell_w
        cell_x_max = cell_x_min + cell_w
        cell_y_min = torch.arange(grid_rows, dtype=torch.float32,
                                  device=device) * cell_h
        cell_y_max = cell_y_min + cell_h

        grid_h_routes = cell_h * benchmark.hroutes_per_micron
        grid_v_routes = cell_w * benchmark.vroutes_per_micron

        port_base = torch.zeros(1, 1, 2, device=device)

        # Move net_data to device
        pin_macro_idx = net_data.pin_macro_idx.to(device)
        pin_offsets = net_data.pin_offsets.to(device)
        net_mask = net_data.mask.to(device)
        net_w = net_data.weights.to(device)
        nd_dev = NetData(
            pin_macro_idx=pin_macro_idx,
            pin_offsets=pin_offsets,
            mask=net_mask,
            weights=net_w,
            num_macros=net_data.num_macros,
            total_net_count=net_data.total_net_count,
        )

        # Move init to device, set requires_grad
        pos = init_b.to(device).clone().detach().requires_grad_(True)
        base_dev = base_init.to(device)

        # Step schedule (mirrors v2-steps; same complexity heuristic)
        n_movable = int(movable.sum().item())
        num_nets = len(net_data.weights)
        complexity = n_movable * num_nets * grid_rows * grid_cols
        if complexity > 1e9:
            step_scale = 0.25
        elif complexity > 5e8:
            step_scale = 0.45
        elif complexity > 1e8:
            step_scale = 0.65
        else:
            step_scale = 1.3
        step_scale = max(0.6, step_scale)
        def s(n): return max(40, int(n * step_scale))

        # In batched mode the congestion kernel scales sub-linearly in B but
        # is still the wall-clock bottleneck at large B. Increase the
        # gradient frequency to amortize.
        base_freq = 1 if num_nets < 3000 else (3 if num_nets < 8000 else 5)
        if B >= 32:
            base_freq *= 4
        elif B >= 8:
            base_freq *= 2
        cong_grad_freq = max(base_freq, 1)

        phases = [
            (s(250), 0.01,   1.0,    0.5),
            (s(300), 0.002,  50.0,   0.2),
            (s(150), 0.0005, 500.0,  0.1),
        ]

        # Track best per slot (CPU side bookkeeping is fine)
        best_pos_b = pos.detach().clone()
        best_score_b = torch.full((B,), float("inf"), device=device)
        cached_cong = torch.zeros(B, device=device)

        for pi, (n_steps, gamma_frac, lam, lr) in enumerate(phases):
            gamma = gamma_frac * cw
            optim = torch.optim.Adam([pos], lr=lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, n_steps, eta_min=lr * 0.05
            )

            for step in range(n_steps):
                optim.zero_grad()

                # Enforce fixed macros from base_init (broadcast across B)
                with torch.no_grad():
                    pos.data[:, fixed_mask, :] = base_dev[fixed_mask, :]

                # Clamp to canvas
                clamped = torch.stack([
                    pos[..., 0].clamp(half_sizes[:, 0], cw - half_sizes[:, 0]),
                    pos[..., 1].clamp(half_sizes[:, 1], ch - half_sizes[:, 1]),
                ], dim=-1)

                wl = _batched_lse_hpwl(clamped, nd_dev, port_base, gamma) / wl_norm

                density = _batched_grid_density(
                    clamped, sizes,
                    cell_x_min, cell_x_max, cell_y_min, cell_y_max,
                    cell_area, grid_rows, grid_cols,
                )

                if step % cong_grad_freq == 0:
                    # Cap memory: total = B * batch_nets * R * C floats.
                    # Aim for <= ~256MB per chunk -> 64M floats -> batch_nets ~= 64M/(B*R*C)
                    grid_cells = grid_rows * grid_cols
                    target_floats = 64 * 1024 * 1024  # 64M floats = 256MB
                    bn = max(64, min(2000, target_floats // (max(B,1) * max(grid_cells,1))))
                    congestion = _batched_rudy_congestion(
                        clamped, nd_dev, port_base, gamma,
                        cell_x_min, cell_x_max, cell_y_min, cell_y_max,
                        grid_h_routes, grid_v_routes, grid_rows, grid_cols,
                        batch_nets=int(bn),
                    )
                    cached_cong = congestion.detach()
                else:
                    congestion = cached_cong

                overlap = _batched_overlap_penalty(
                    clamped[:, :n_hard, :], half_sizes[:n_hard]
                )
                overlap_norm = overlap / (cw * ch)

                proxy = wl + 0.5 * density + 0.5 * congestion           # [B]
                loss_b = proxy + lam * overlap_norm                      # [B]
                # Sum over B → grads flow correctly to each slot
                loss = loss_b.sum()
                loss.backward()

                # Per-slot grad clipping. torch.nn.utils.clip_grad_norm_
                # clips global norm of the parameter, which collapses across
                # the batch dim; that's fine for a stability cap.
                torch.nn.utils.clip_grad_norm_([pos], max_norm=20.0 * math.sqrt(B))
                optim.step()
                sched.step()

                with torch.no_grad():
                    score_b = proxy.detach() + max(1.0, lam * 0.1) * overlap_norm.detach()
                    better = score_b < best_score_b
                    if better.any():
                        # snapshot the better slots
                        snap = pos.detach().clone()
                        snap[:, fixed_mask, :] = base_dev[fixed_mask, :]
                        snap[..., 0].clamp_(half_sizes[:, 0], cw - half_sizes[:, 0])
                        snap[..., 1].clamp_(half_sizes[:, 1], ch - half_sizes[:, 1])
                        best_pos_b = torch.where(
                            better.view(B, 1, 1), snap, best_pos_b
                        )
                        best_score_b = torch.where(better, score_b, best_score_b)

                if step == 0 and self.verbose:
                    with torch.no_grad():
                        print(f"    P{pi+1}: wl={wl.mean().item():.3f} "
                              f"den={density.mean().item():.3f} "
                              f"cong={congestion.mean().item():.3f}")

            if self.verbose:
                with torch.no_grad():
                    print(f"  DPO phase {pi+1} done: "
                          f"best_mean={best_score_b.mean().item():.4f}  "
                          f"best_min={best_score_b.min().item():.4f}")

        return best_pos_b, best_score_b
