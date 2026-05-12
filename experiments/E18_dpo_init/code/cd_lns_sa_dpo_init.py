"""E18 — CDLNSSAPlacer with DPO best_of_v2 init in place of SDF init.

Test if DPO best_of_v2 (avg --all 1.3834 alone, archived) converges to a
basin different from SDF's. If DPO-init -> CD -> LNS -> SA-v2 beats E25 on
any benchmark, that's a multi-basin signal feeding E27 diagnostic.

Pipeline (per benchmark, total ≤ 3600 s legal cap):

  1. DPO best_of_v2 init: run SDFPlacer + DPOv2StepsPlacer, pick the one
     with lower compute_proxy_cost (matches archived BestOfV2Placer).
  2. Project overlaps (in case DPO output has residual overlaps).
  3. Build IncrementalProxyEvaluator.
  4. CD phase (≤ 2400 s).
  5. LNS phase (≤ 600 s, grid-bin).
  6. SA-v2 phase (≤ 600 s, breakpoint Metropolis with best-so-far).
  7. Validate (zero overlaps), preserve fixed macros, return.

Reference:
- E25 — pipeline parent (SDF init); avg --all 1.0954 candidate.
- E11 — basin sensitivity to init.
- writeup/archive/submissions/dpo/best_of_v2_placer.py — archived DPO source.
- writeup/archive/submissions/dpo/ablation_v2_steps.py — archived DPO body.
- macro_place/sdf_init.py — live SDFPlacer (replaces archived polyhedra/init/sdf.py).
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

# The eval harness loads placers via importlib.spec_from_file_location, which
# does NOT add the repo root to sys.path. We need it on sys.path so the
# `macro_place.*` imports resolve.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import (
    _grid_lines,
    axis_breakpoints,
    legal_axis_range,
    project_overlaps,
    run_cd_adaptive,
)
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost
from macro_place.sdf_init import SDFPlacer


# ── DPO v2-steps body (inlined from writeup/archive/submissions/dpo/ablation_v2_steps.py) ──


class _NetData:
    """Padded net data for vectorized HPWL/congestion computation.

    Plain class (not @dataclass) — @dataclass triggers a Python 3.9 +
    importlib.spec_from_file_location bug because the module isn't yet
    registered in sys.modules during exec_module, so dataclasses' type
    introspection fails on cls.__module__. Plain __init__ avoids that.
    """

    def __init__(self, pin_macro_idx, pin_offsets, mask, weights,
                 num_macros, total_net_count):
        self.pin_macro_idx = pin_macro_idx   # [num_nets, max_pins] long
        self.pin_offsets = pin_offsets       # [num_nets, max_pins, 2]
        self.mask = mask                     # [num_nets, max_pins] bool
        self.weights = weights               # [num_nets]
        self.num_macros = num_macros
        self.total_net_count = total_net_count


def _extract_net_data(benchmark: Benchmark, plc) -> _NetData:
    plc_to_tensor = {}
    for t_idx, p_idx in enumerate(benchmark.hard_macro_indices):
        plc_to_tensor[p_idx] = t_idx
    for i, p_idx in enumerate(benchmark.soft_macro_indices):
        plc_to_tensor[p_idx] = benchmark.num_hard_macros + i

    port_idx = benchmark.num_macros

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
        return _NetData(
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

    return _NetData(
        pin_macro_idx=pin_macro_idx,
        pin_offsets=pin_offsets,
        mask=mask,
        weights=torch.tensor(net_weights, dtype=torch.float32),
        num_macros=benchmark.num_macros,
        total_net_count=max(total_net_count, 1),
    )


def _legalize_dpo(pos_np, sizes_np, movable, n_hard, canvas_w, canvas_h,
                  max_rounds=20, eps=0.002):
    """Iterative overlap repair for hard macros (from DPO archive)."""
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

        for i in range(n_hard):
            if movable[i]:
                pos[i, 0] = np.clip(pos[i, 0], half_w[i] + eps,
                                    canvas_w - half_w[i] - eps)
                pos[i, 1] = np.clip(pos[i, 1], half_h[i] + eps,
                                    canvas_h - half_h[i] - eps)
    return pos


def _lse_hpwl(positions, net_data: _NetData, port_base, gamma):
    if len(net_data.weights) == 0:
        return torch.tensor(0.0)
    all_pos = torch.cat([positions, port_base], dim=0)
    pin_pos = all_pos[net_data.pin_macro_idx] + net_data.pin_offsets
    pin_x = pin_pos[:, :, 0]
    pin_y = pin_pos[:, :, 1]
    mask = net_data.mask
    big = 1e10
    x_for_max = pin_x.clone(); x_for_max[~mask] = -big
    lse_max_x = gamma * torch.logsumexp(x_for_max / gamma, dim=1)
    x_for_min = pin_x.clone(); x_for_min[~mask] = big
    lse_min_x = -gamma * torch.logsumexp(-x_for_min / gamma, dim=1)
    y_for_max = pin_y.clone(); y_for_max[~mask] = -big
    lse_max_y = gamma * torch.logsumexp(y_for_max / gamma, dim=1)
    y_for_min = pin_y.clone(); y_for_min[~mask] = big
    lse_min_y = -gamma * torch.logsumexp(-y_for_min / gamma, dim=1)
    hpwl = (lse_max_x - lse_min_x) + (lse_max_y - lse_min_y)
    return (net_data.weights * hpwl).sum()


def _grid_density(positions, sizes,
                  cell_x_min, cell_x_max, cell_y_min, cell_y_max,
                  cell_area, grid_rows, grid_cols):
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    mx_min = positions[:, 0] - half_w
    mx_max = positions[:, 0] + half_w
    my_min = positions[:, 1] - half_h
    my_max = positions[:, 1] + half_h
    x_overlap = torch.clamp(
        torch.min(mx_max.unsqueeze(1), cell_x_max.unsqueeze(0))
        - torch.max(mx_min.unsqueeze(1), cell_x_min.unsqueeze(0)),
        min=0,
    )
    y_overlap = torch.clamp(
        torch.min(my_max.unsqueeze(1), cell_y_max.unsqueeze(0))
        - torch.max(my_min.unsqueeze(1), cell_y_min.unsqueeze(0)),
        min=0,
    )
    overlap_area = y_overlap.unsqueeze(2) * x_overlap.unsqueeze(1)
    density_grid = overlap_area.sum(dim=0) / cell_area
    flat = density_grid.flatten()
    k = max(1, int(0.10 * len(flat)))
    top_k, _ = torch.topk(flat, k)
    return 0.5 * top_k.mean()


def _rudy_congestion(positions, net_data: _NetData, port_base, gamma,
                     cell_x_min, cell_x_max, cell_y_min, cell_y_max,
                     grid_h_routes, grid_v_routes, grid_rows, grid_cols):
    num_nets = len(net_data.weights)
    if num_nets == 0 or grid_h_routes == 0 or grid_v_routes == 0:
        return torch.tensor(0.0)
    all_pos = torch.cat([positions, port_base], dim=0)
    pin_pos = all_pos[net_data.pin_macro_idx] + net_data.pin_offsets
    pin_x = pin_pos[:, :, 0]
    pin_y = pin_pos[:, :, 1]
    mask = net_data.mask
    big = 1e10
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
    w = net_data.weights
    h_coeff = w / (bbox_w * grid_h_routes)
    v_coeff = w / (bbox_h * grid_v_routes)
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
        )
        y_ol = torch.clamp(
            torch.min(bbox_y_max[b].unsqueeze(1), cell_y_max.unsqueeze(0))
            - torch.max(bbox_y_min[b].unsqueeze(1), cell_y_min.unsqueeze(0)),
            min=0,
        )
        ol_area = y_ol.unsqueeze(2) * x_ol.unsqueeze(1)
        h_cong = h_cong + (h_coeff[b].unsqueeze(1).unsqueeze(2) * ol_area).sum(0)
        v_cong = v_cong + (v_coeff[b].unsqueeze(1).unsqueeze(2) * ol_area).sum(0)
    combined = h_cong + v_cong
    flat = combined.flatten()
    k = max(1, int(0.05 * len(flat)))
    top_k, _ = torch.topk(flat, k)
    return top_k.mean()


def _overlap_penalty(positions, half_sizes):
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


def _dpo_v2_optimize(init_pos, benchmark, net_data):
    """DPO v2-steps optimization (from archive)."""
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes
    half_sizes = sizes / 2
    movable = benchmark.get_movable_mask()
    fixed_mask = ~movable

    wl_norm = (cw + ch) * net_data.total_net_count

    grid_rows = benchmark.grid_rows
    grid_cols = benchmark.grid_cols
    cell_w = cw / grid_cols
    cell_h = ch / grid_rows
    cell_area = cell_w * cell_h
    cell_x_min = torch.arange(grid_cols, dtype=torch.float32) * cell_w
    cell_x_max = cell_x_min + cell_w
    cell_y_min = torch.arange(grid_rows, dtype=torch.float32) * cell_h
    cell_y_max = cell_y_min + cell_h

    grid_h_routes = cell_h * benchmark.hroutes_per_micron
    grid_v_routes = cell_w * benchmark.vroutes_per_micron

    port_base = torch.zeros(1, 2)

    n_movable = int(movable.sum().item())
    num_nets = len(net_data.weights)
    cached_cong = 0.0
    cong_grad_freq = 1 if num_nets < 3000 else (3 if num_nets < 8000 else 5)

    pos = init_pos.clone().detach().requires_grad_(True)
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

    phases = [
        (s(250), 0.01,   1.0,     0.5),
        (s(300), 0.002,  50.0,    0.2),
        (s(150), 0.0005, 500.0,   0.1),
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
            with torch.no_grad():
                pos.data[fixed_mask] = init_pos[fixed_mask]
            clamped = torch.stack([
                pos[:, 0].clamp(half_sizes[:, 0], cw - half_sizes[:, 0]),
                pos[:, 1].clamp(half_sizes[:, 1], ch - half_sizes[:, 1]),
            ], dim=1)

            wl = _lse_hpwl(clamped, net_data, port_base, gamma) / wl_norm

            density = _grid_density(clamped, sizes,
                                    cell_x_min, cell_x_max,
                                    cell_y_min, cell_y_max,
                                    cell_area, grid_rows, grid_cols)

            if step % cong_grad_freq == 0:
                congestion = _rudy_congestion(clamped, net_data, port_base, gamma,
                                              cell_x_min, cell_x_max,
                                              cell_y_min, cell_y_max,
                                              grid_h_routes, grid_v_routes,
                                              grid_rows, grid_cols)
                cached_cong = congestion.item()
            else:
                congestion = torch.tensor(cached_cong)

            overlap = _overlap_penalty(clamped[:n_hard], half_sizes[:n_hard])
            overlap_norm = overlap / (cw * ch)

            proxy = wl + 0.5 * density + 0.5 * congestion
            loss = proxy + lam * overlap_norm
            loss.backward()
            torch.nn.utils.clip_grad_norm_([pos], max_norm=20.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                proxy_val = proxy.item()
                ov_val = overlap_norm.item()
                score = proxy_val + max(1.0, lam * 0.1) * ov_val
                if score < best_proxy:
                    best_proxy = score
                    bp = pos.data.clone()
                    bp[fixed_mask] = init_pos[fixed_mask]
                    bp[:, 0].clamp_(half_sizes[:, 0], cw - half_sizes[:, 0])
                    bp[:, 1].clamp_(half_sizes[:, 1], ch - half_sizes[:, 1])
                    best_pos = bp

            if step == 0:
                with torch.no_grad():
                    print(f"    P{phase_idx+1}: wl={wl.item():.3f} "
                          f"den={density.item():.3f} "
                          f"cong={congestion.item():.3f}", flush=True)

        with torch.no_grad():
            ov_count = _count_overlaps(best_pos[:n_hard], half_sizes[:n_hard])
            print(f"  DPO phase {phase_idx+1} done: best_score={best_proxy:.4f}  "
                  f"overlaps={ov_count}", flush=True)

    return best_pos


def _dpo_v2_place(benchmark, plc, seed, sdf_init_pos):
    """DPO v2-steps placer using a provided SDF init (full [num_macros, 2])."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # _dpo_v2_optimize expects [num_macros, 2] init (it indexes by full
    # fixed_mask). The archived DPO computed its own SDF init internally via
    # SDFPlacer.place() which returns a full [num_macros, 2] tensor.
    init_pos_full = sdf_init_pos.clone()
    if init_pos_full.shape[0] != benchmark.num_macros:
        full = benchmark.macro_positions.clone()
        full[:init_pos_full.shape[0]] = init_pos_full
        init_pos_full = full

    net_data = _extract_net_data(benchmark, plc)
    optimized = _dpo_v2_optimize(init_pos_full, benchmark, net_data)

    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.numpy().astype(np.float64)
    movable = benchmark.get_movable_mask().numpy()
    pos_np = optimized.detach().numpy().astype(np.float64)
    legal = _legalize_dpo(pos_np, sizes, movable, n_hard, cw, ch)
    result = optimized.clone()
    result[:n_hard] = torch.tensor(legal[:n_hard], dtype=torch.float32)
    return result


def _best_of_v2_init(benchmark: Benchmark, plc, seed: int = 42,
                     log_fn: Optional[Callable[[str], None]] = None) -> torch.Tensor:
    """Replicates archived BestOfV2Placer: run SDF + DPO-v2, return whichever
    has lower compute_proxy_cost. Uses the live macro_place.sdf_init.SDFPlacer
    (the archived polyhedra/init/sdf.py path no longer exists). Both
    placements are full [num_macros, 2] tensors."""
    sdf_placer = SDFPlacer(seed=seed)
    sdf_positions = sdf_placer.place(benchmark)
    # SDFPlacer returns full [num_macros, 2]; pass through to DPO as init.
    dpo_positions = _dpo_v2_place(benchmark, plc, seed=seed,
                                  sdf_init_pos=sdf_positions)

    sdf_cost = compute_proxy_cost(sdf_positions, benchmark, plc)["proxy_cost"]
    dpo_cost = compute_proxy_cost(dpo_positions, benchmark, plc)["proxy_cost"]
    winner = "SDF" if sdf_cost < dpo_cost else "DPO-v2"
    if log_fn is not None:
        log_fn(f"  best_of_v2: SDF={sdf_cost:.4f}  DPO-v2={dpo_cost:.4f}  -> {winner}")
    return sdf_positions if sdf_cost < dpo_cost else dpo_positions


# ── Grid-bin LNS primitives (mirror E25 exactly) ───────────────────────────


def _cost_aware_destroy(
    evaluator: IncrementalProxyEvaluator,
    hard_movable: List[int],
    K: int,
) -> List[int]:
    cw = evaluator.width / 2.0
    ch = evaluator.height / 2.0
    baseline_p = evaluator.current_cost()["proxy"]
    scores: List[tuple] = []
    for idx in hard_movable:
        try:
            # Pure probe — delta_cost peeks without mutation (~2x faster).
            p = evaluator.delta_cost(idx, (cw, ch))["proxy"]
            scores.append((idx, p - baseline_p))
        except Exception:
            scores.append((idx, 0.0))
    scores.sort(key=lambda s: s[1])
    return [s[0] for s in scores[:K]]


def _is_legal_2d(idx, x, y, placement, macro_sizes_np, n_hard, eps=1e-4):
    if idx >= n_hard:
        return True
    half_w = float(macro_sizes_np[idx, 0]) / 2.0
    half_h = float(macro_sizes_np[idx, 1]) / 2.0
    pos = placement[:n_hard].cpu().numpy().astype(np.float64)
    sz = macro_sizes_np[:n_hard]
    dx = np.abs(pos[:, 0] - x)
    dy = np.abs(pos[:, 1] - y)
    min_dx = half_w + sz[:, 0] / 2.0
    min_dy = half_h + sz[:, 1] / 2.0
    blockers = (dx < min_dx - eps) & (dy < min_dy - eps)
    blockers[idx] = False
    return not bool(np.any(blockers))


def _gridbin_reinsert(evaluator, macro_idx, plc, n_hard, macro_sizes_np,
                      deadline_s, t_start):
    cw = float(plc.width)
    ch = float(plc.height)
    grid_w = cw / plc.grid_col
    grid_h = ch / plc.grid_row
    half_w = float(macro_sizes_np[macro_idx, 0]) / 2.0
    half_h = float(macro_sizes_np[macro_idx, 1]) / 2.0

    cur_cost = evaluator.current_cost()["proxy"]
    cur_xy = (
        float(evaluator.placement[macro_idx, 0]),
        float(evaluator.placement[macro_idx, 1]),
    )
    best_cost = cur_cost
    best_xy = cur_xy

    for col in range(int(plc.grid_col)):
        if time.perf_counter() - t_start >= deadline_s:
            break
        cx = (col + 0.5) * grid_w
        if cx < half_w or cx > cw - half_w:
            continue
        for row in range(int(plc.grid_row)):
            if time.perf_counter() - t_start >= deadline_s:
                break
            cy = (row + 0.5) * grid_h
            if cy < half_h or cy > ch - half_h:
                continue
            if not _is_legal_2d(macro_idx, cx, cy, evaluator.placement,
                                macro_sizes_np, n_hard):
                continue
            # Pure probe — delta_cost peeks without mutation.
            cost = evaluator.delta_cost(macro_idx, (cx, cy))["proxy"]
            if cost < best_cost - 1e-9:
                best_cost = cost
                best_xy = (cx, cy)

    if best_cost < cur_cost - 1e-9 and (
        abs(best_xy[0] - cur_xy[0]) > 1e-7 or abs(best_xy[1] - cur_xy[1]) > 1e-7
    ):
        evaluator.move(macro_idx, best_xy)
        return best_cost - cur_cost
    return 0.0


def _run_lns_gridbin(evaluator, benchmark, plc, hard_movable, time_budget_s,
                    destroy_frac=0.05, destroy_cap=30, seed=42, log_fn=None):
    K = max(1, min(destroy_cap, int(destroy_frac * len(hard_movable))))
    macro_sizes_np = evaluator.macro_sizes.cpu().numpy()
    n_hard = benchmark.num_hard_macros

    if log_fn is not None:
        log_fn(
            f"  LNS budget={time_budget_s:.0f}s, destroy K={K} "
            f"(={destroy_frac*100:.1f}% of {len(hard_movable)} hard movables, "
            f"capped at {destroy_cap}), strategy=cost_aware"
        )

    sample = 0
    total_improvement = 0.0
    t_start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:
            break
        sample += 1
        sample_t0 = time.perf_counter()

        destroy = _cost_aware_destroy(evaluator, hard_movable, K)

        sample_delta = 0.0
        moves_this_sample = 0
        for idx in destroy:
            if time.perf_counter() - t_start >= time_budget_s:
                break
            delta = _gridbin_reinsert(evaluator, idx, plc, n_hard, macro_sizes_np,
                                      deadline_s=time_budget_s, t_start=t_start)
            sample_delta += delta
            if delta < 0:
                moves_this_sample += 1

        total_improvement += sample_delta
        sample_wall = time.perf_counter() - sample_t0
        elapsed = time.perf_counter() - t_start
        cur_proxy = evaluator.current_cost()["proxy"]
        if log_fn is not None:
            log_fn(
                f"  LNS sample {sample}: K={K}, Δ={sample_delta:+.5f} "
                f"(moves={moves_this_sample}/{K}), proxy={cur_proxy:.5f}, "
                f"sample_wall={sample_wall:.1f}s, total elapsed={elapsed:.1f}s"
            )

        if abs(sample_delta) < 1e-7:
            if log_fn is not None:
                log_fn(f"  LNS converged at sample {sample} (no improvement)")
            break

    return {
        "samples": sample,
        "total_improvement": total_improvement,
        "wall_total_s": time.perf_counter() - t_start,
    }


# ── SA polish v2 (mirror E25 exactly) ──────────────────────────────────────


def _run_sa_polish_v2(evaluator, benchmark, plc, hard_movable, time_budget_s,
                     T0=5e-4, Tf=1e-6, seed=42, breakpoint_budget=12, log_fn=None):
    rng = np.random.default_rng(seed=seed)
    grid_lines_x, grid_lines_y = _grid_lines(plc)
    n_hard = benchmark.num_hard_macros

    if not hard_movable:
        if log_fn is not None:
            log_fn("  SA: no hard movable macros; skipping")
        return {
            "proposed": 0, "accepted_better": 0, "accepted_worse": 0,
            "rejected": 0, "skipped": 0,
            "init_proxy": float("nan"), "best_proxy": float("nan"),
            "final_proxy": float("nan"), "improvement_vs_init": 0.0,
            "best_found_at_t": 0.0, "wall_total_s": 0.0,
            "best_restored": False,
        }

    if log_fn is not None:
        log_fn(
            f"  SA v2 budget={time_budget_s:.0f}s, T0={T0:.2e}, Tf={Tf:.2e}, "
            f"|H|={len(hard_movable)}, seed={seed} (best-so-far tracking ON)"
        )

    proposed = 0
    accepted_better = 0
    accepted_worse = 0
    rejected = 0
    skipped = 0
    init_proxy = evaluator.current_cost()["proxy"]
    cur_proxy = init_proxy
    best_proxy = init_proxy
    best_placement = evaluator.placement.detach().clone()
    best_found_at_t = 0.0

    log_ratio = math.log(Tf / T0)
    t_start = time.perf_counter()
    last_log_t = t_start

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:
            break

        frac = elapsed / time_budget_s
        T = T0 * math.exp(log_ratio * frac)

        idx = int(hard_movable[rng.integers(0, len(hard_movable))])
        axis = int(rng.integers(0, 2))

        lo, hi = legal_axis_range(
            idx, evaluator.placement, evaluator.macro_sizes,
            benchmark.macro_fixed, n_hard, axis=axis,
            canvas_w=benchmark.canvas_width,
            canvas_h=benchmark.canvas_height,
        )
        if hi - lo < 1e-5:
            skipped += 1
            continue

        cur_axis_val = float(evaluator.placement[idx, axis])
        grid_lines = grid_lines_x if axis == 0 else grid_lines_y
        cands = axis_breakpoints(
            idx, axis, evaluator, grid_lines, lo, hi,
            max_breakpoints=breakpoint_budget, cur_axis=cur_axis_val,
        )
        if len(cands) > 0:
            mask = np.abs(cands - cur_axis_val) > 1e-6
            cands = cands[mask]
        if len(cands) == 0:
            skipped += 1
            continue

        new_axis_val = float(cands[rng.integers(0, len(cands))])
        cur_xy = (float(evaluator.placement[idx, 0]),
                  float(evaluator.placement[idx, 1]))
        new_xy = list(cur_xy)
        new_xy[axis] = new_axis_val

        proposed += 1
        new_proxy = evaluator.move(idx, tuple(new_xy))["proxy"]
        delta = new_proxy - cur_proxy
        accepted = False

        if delta <= 0.0:
            cur_proxy = new_proxy
            accepted_better += 1
            accepted = True
        else:
            accept_p = math.exp(-delta / T) if T > 0 else 0.0
            if rng.random() < accept_p:
                cur_proxy = new_proxy
                accepted_worse += 1
                accepted = True
            else:
                evaluator.revert()
                rejected += 1

        if accepted and cur_proxy < best_proxy - 1e-12:
            best_proxy = cur_proxy
            best_placement = evaluator.placement.detach().clone()
            best_found_at_t = time.perf_counter() - t_start

        now = time.perf_counter()
        if log_fn is not None and (now - last_log_t) >= 30.0:
            last_log_t = now
            log_fn(
                f"  SA v2 t={now - t_start:6.1f}s T={T:.2e} "
                f"proposed={proposed} better={accepted_better} "
                f"worse={accepted_worse} rejected={rejected} "
                f"skipped={skipped} cur={cur_proxy:.5f} best={best_proxy:.5f}"
            )

    wall = time.perf_counter() - t_start
    final_proxy_chain = evaluator.current_cost()["proxy"]

    best_restored = False
    if best_proxy < final_proxy_chain - 1e-12:
        n_macros = int(evaluator.placement.shape[0])
        for i in range(n_macros):
            tx = float(best_placement[i, 0])
            ty = float(best_placement[i, 1])
            cx = float(evaluator.placement[i, 0])
            cy = float(evaluator.placement[i, 1])
            if abs(tx - cx) > 1e-9 or abs(ty - cy) > 1e-9:
                evaluator.move(i, (tx, ty))
        best_restored = True

    final_proxy = evaluator.current_cost()["proxy"]
    if log_fn is not None:
        log_fn(
            f"  SA v2 done: proposed={proposed}, better={accepted_better}, "
            f"worse={accepted_worse}, rejected={rejected}, skipped={skipped}, "
            f"init={init_proxy:.5f}, best={best_proxy:.5f} "
            f"(found at t={best_found_at_t:.1f}s), "
            f"chain-final={final_proxy_chain:.5f}, "
            f"restored={best_restored}, evaluator-final={final_proxy:.5f}"
        )

    return {
        "proposed": proposed,
        "accepted_better": accepted_better,
        "accepted_worse": accepted_worse,
        "rejected": rejected,
        "skipped": skipped,
        "init_proxy": init_proxy,
        "best_proxy": best_proxy,
        "final_proxy": final_proxy,
        "improvement_vs_init": init_proxy - best_proxy,
        "best_found_at_t": best_found_at_t,
        "wall_total_s": wall,
        "best_restored": best_restored,
    }


# ── Placer ─────────────────────────────────────────────────────────────────


class CDLNSSADPOInitPlacer:
    """E18 placer: DPO best_of_v2 init -> CD -> LNS -> SA-v2.

    Same time budgets and hyperparams as E25 (CDLNSSAPlacer); only the
    init step changes. DPO ~30 min/bench is *additional* wall on top of
    E25's 60 min/bench (3600 s total budget); each phase still runs to
    completion within its own clock.

    All hyperparameters global. No per-benchmark tuning.
    """

    def __init__(
        self,
        cd_hard_cap_s: float = 2400.0,
        cd_min_time_s: float = 300.0,
        cd_patience: int = 3,
        cd_plateau_threshold: float = 0.001,
        lns_budget_s: float = 600.0,
        lns_destroy_frac: float = 0.05,
        lns_destroy_cap: int = 30,
        lns_seed: int = 42,
        sa_budget_s: float = 600.0,
        sa_T0: float = 5e-4,
        sa_Tf: float = 1e-6,
        sa_seed: int = 42,
        sa_breakpoint_budget: int = 12,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.cd_hard_cap_s = float(cd_hard_cap_s)
        self.cd_min_time_s = float(cd_min_time_s)
        self.cd_patience = int(cd_patience)
        self.cd_plateau_threshold = float(cd_plateau_threshold)
        self.lns_budget_s = float(lns_budget_s)
        self.lns_destroy_frac = float(lns_destroy_frac)
        self.lns_destroy_cap = int(lns_destroy_cap)
        self.lns_seed = int(lns_seed)
        self.sa_budget_s = float(sa_budget_s)
        self.sa_T0 = float(sa_T0)
        self.sa_Tf = float(sa_Tf)
        self.sa_seed = int(sa_seed)
        self.sa_breakpoint_budget = int(sa_breakpoint_budget)
        self.seed = int(seed)
        self.verbose = bool(verbose)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _load_plc_for(self, benchmark: Benchmark):
        bench_dir = find_benchmark_dir(benchmark.name)
        return load_benchmark_from_dir(str(bench_dir))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_total0 = time.perf_counter()
        self._log(
            f"=== CDLNSSADPOInitPlacer ({benchmark.name}): "
            f"DPO init -> CD cap={self.cd_hard_cap_s:.0f}s -> "
            f"LNS={self.lns_budget_s:.0f}s -> SA={self.sa_budget_s:.0f}s ==="
        )
        self._log(
            f"  num_macros={benchmark.num_macros}, "
            f"n_hard={benchmark.num_hard_macros}, "
            f"canvas={benchmark.canvas_width:.1f}x{benchmark.canvas_height:.1f}"
        )

        # 1. DPO best_of_v2 init (replaces SDF init)
        _, plc = self._load_plc_for(benchmark)
        t_init0 = time.perf_counter()
        placement = _best_of_v2_init(
            benchmark, plc, seed=self.seed,
            log_fn=self._log if self.verbose else None,
        )
        self._log(f"  DPO init wall = {time.perf_counter() - t_init0:.1f} s")

        # 2. Project overlaps
        placement, proj_iters = project_overlaps(placement, benchmark)
        init_overlaps = compute_overlap_metrics(placement, benchmark)
        self._log(
            f"  overlap projection: {proj_iters} iters, "
            f"residual count={init_overlaps['overlap_count']}"
        )

        # 3. Build evaluator
        placement_f64 = placement.detach().clone().to(torch.float64)
        evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
        init_cost = evaluator.current_cost()
        self._log(
            f"  init proxy={init_cost['proxy']:.5f} "
            f"[wl={init_cost['wl']:.4f} d={init_cost['density']:.4f} "
            f"c={init_cost['congestion']:.4f}]"
        )

        # 4. CD phase
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        hard_movable = [
            i for i in range(benchmark.num_hard_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        self._log(f"  starting CD phase (cap={self.cd_hard_cap_s:.0f}s)")
        cd_stats = run_cd_adaptive(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            movable=movable,
            min_time_s=self.cd_min_time_s,
            hard_cap_s=self.cd_hard_cap_s,
            patience=self.cd_patience,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=self._log if self.verbose else None,
        )
        cd_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  CD done: exit={cd_stats['exit_reason']}, "
            f"sweeps={cd_stats['sweeps']}, accepts={cd_stats['total_moves']}, "
            f"wall={cd_stats['wall_total_s']:.1f}s, proxy={cd_proxy:.5f}"
        )

        # 5. LNS phase
        self._log(f"  starting LNS phase (budget={self.lns_budget_s:.0f}s)")
        lns_stats = _run_lns_gridbin(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.lns_budget_s,
            destroy_frac=self.lns_destroy_frac,
            destroy_cap=self.lns_destroy_cap,
            seed=self.lns_seed,
            log_fn=self._log if self.verbose else None,
        )
        lns_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  LNS done: samples={lns_stats['samples']}, "
            f"Δ={lns_stats['total_improvement']:+.5f}, "
            f"wall={lns_stats['wall_total_s']:.1f}s, proxy={lns_proxy:.5f}"
        )

        # 6. SA-v2 phase
        self._log(f"  starting SA-v2 phase (budget={self.sa_budget_s:.0f}s)")
        sa_stats = _run_sa_polish_v2(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.sa_budget_s,
            T0=self.sa_T0,
            Tf=self.sa_Tf,
            seed=self.sa_seed,
            breakpoint_budget=self.sa_breakpoint_budget,
            log_fn=self._log if self.verbose else None,
        )
        final_cost = evaluator.current_cost()
        self._log(
            f"  SA-v2 done: lift_vs_init={sa_stats['improvement_vs_init']:+.5f}, "
            f"best={sa_stats['best_proxy']:.5f}, "
            f"final proxy={final_cost['proxy']:.5f}"
        )
        self._log(
            f"  TOTAL lifts: CD={init_cost['proxy'] - cd_proxy:+.5f}, "
            f"LNS={cd_proxy - lns_proxy:+.5f}, "
            f"SA={lns_proxy - final_cost['proxy']:+.5f}"
        )

        # 7. Pull placement back; preserve fixed macros
        final_placement_f64 = evaluator.placement.detach().clone().cpu()
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            final_placement_f64[fixed_mask] = original_positions[fixed_mask]
        final_placement = final_placement_f64.to(torch.float32)

        overlaps = compute_overlap_metrics(final_placement, benchmark)
        if overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"CDLNSSADPOInitPlacer produced {overlaps['overlap_count']} "
                f"overlaps (area {overlaps['total_overlap_area']:.4f}) on "
                f"'{benchmark.name}'"
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(DPO + project + CD + LNS + SA + validate)"
        )
        return final_placement
