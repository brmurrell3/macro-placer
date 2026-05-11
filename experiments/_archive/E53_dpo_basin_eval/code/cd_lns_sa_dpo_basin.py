"""E53 — CDLNSSA + DPO best_of_v2 init + GPU DPO basin-evaluator LNS.

Replaces E41's K-macro joint LNS step with a basin-hopping loop whose
basin-evaluator is short-burst GPU DPO (Adam on the smooth proxy) on
the destroyed macros. The hypothesis is that the M3 Max MPS device
settles a destroyed-macros basin in ~10-50 ms, vs ~5 s/K-tuple for
K-joint's exact `top_N^K=125` enumeration. That is 100×+ more basin
samples per second in the same budget, plus headroom for multi-seed
verification (the missing defense against verification-time score
regression).

Pipeline (per benchmark):
  1. DPO best_of_v2 init (E18).
  2. project_overlaps to clear DPO residuals.
  3. Build IncrementalProxyEvaluator.
  4. CD adaptive phase (≤ 2400 s).
  5. Grid-bin LNS phase (≤ 600 s).
  6. SA-v2 phase (≤ 600 s, T₀=5e-4).
  7. **GPU DPO basin LNS (≤ 600 s).** Destroy K=3 → scatter → Adam
     refine destroyed-only on MPS (≤80 steps) → CPU project + per-macro
     legality + exact-proxy accept/revert. (Replaces E41's K-joint.)
  8. Validate, preserve fixed macros, return.

Reference:
- E41 — parent (replaces K-joint with this DPO-basin step).
- E18 — DPO init + smooth-proxy primitives (`_lse_hpwl`, `_grid_density`,
  `_rudy_congestion`, `_overlap_penalty`, `_extract_net_data`,
  `_NetData`, `_best_of_v2_init`).
- E39 — LNS-gridbin, SA-v2, `_cost_aware_destroy`, `_is_legal_2d`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics

from experiments.E18_dpo_init.code.cd_lns_sa_dpo_init import (
    _NetData,
    _best_of_v2_init,
    _extract_net_data,
)
from experiments.E39_kmacro_joint_lns.code.cd_lns_sa_kjoint import (
    _cost_aware_destroy,
    _is_legal_2d,
    run_lns_gridbin,
    run_sa_polish_v2,
)


# ── Device selection ───────────────────────────────────────────────────────


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Device-aware smooth-proxy primitives ───────────────────────────────────
# Forks of E18's CPU functions with explicit device-aware allocations.


def _lse_hpwl_dev(positions, net_data, port_base, gamma):
    if len(net_data.weights) == 0:
        return torch.tensor(0.0, device=positions.device)
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


def _grid_density_dev(positions, sizes,
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


def _rudy_congestion_dev(positions, net_data, port_base, gamma,
                         cell_x_min, cell_x_max, cell_y_min, cell_y_max,
                         grid_h_routes, grid_v_routes, grid_rows, grid_cols):
    num_nets = len(net_data.weights)
    device = positions.device
    if num_nets == 0 or grid_h_routes == 0 or grid_v_routes == 0:
        return torch.tensor(0.0, device=device)
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
    h_cong = torch.zeros(grid_rows, grid_cols, device=device)
    v_cong = torch.zeros(grid_rows, grid_cols, device=device)
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


def _overlap_penalty_dev(positions, half_sizes):
    n = positions.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=positions.device)
    dx = torch.abs(positions[:, 0].unsqueeze(1) - positions[:, 0].unsqueeze(0))
    dy = torch.abs(positions[:, 1].unsqueeze(1) - positions[:, 1].unsqueeze(0))
    sep_x = half_sizes[:, 0].unsqueeze(1) + half_sizes[:, 0].unsqueeze(0)
    sep_y = half_sizes[:, 1].unsqueeze(1) + half_sizes[:, 1].unsqueeze(0)
    overlap_area = torch.relu(sep_x - dx) * torch.relu(sep_y - dy)
    mask = torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=positions.device), diagonal=1
    )
    return (overlap_area * mask).sum()


def _net_data_to_device(net_data: _NetData, device: torch.device) -> _NetData:
    return _NetData(
        pin_macro_idx=net_data.pin_macro_idx.to(device),
        pin_offsets=net_data.pin_offsets.to(device),
        mask=net_data.mask.to(device),
        weights=net_data.weights.to(device),
        num_macros=net_data.num_macros,
        total_net_count=net_data.total_net_count,
    )


# ── DPO basin LNS phase ────────────────────────────────────────────────────


def _dpo_settle_full(
    init_pos_dev: torch.Tensor,
    fixed_mask_dev: torch.Tensor,
    init_pinned_dev: torch.Tensor,
    benchmark: Benchmark,
    net_data_dev: _NetData,
    half_sizes_dev: torch.Tensor,
    sizes_dev: torch.Tensor,
    cell_x_min, cell_x_max, cell_y_min, cell_y_max,
    cell_area, grid_rows, grid_cols, grid_h_routes, grid_v_routes,
    port_base_dev, wl_norm: float,
    n_steps: int, lr: float, gamma: float, overlap_lambda: float,
) -> torch.Tensor:
    """Adam on smooth proxy + overlap penalty, optimizing ALL positions
    (fixed macros are re-pinned each step). Returns updated full-pose
    tensor on the same device as `init_pos_dev`.
    """
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    n_hard = benchmark.num_hard_macros

    pos = init_pos_dev.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([pos], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        with torch.no_grad():
            pos.data[fixed_mask_dev] = init_pinned_dev[fixed_mask_dev]
        clamped = torch.stack([
            pos[:, 0].clamp(half_sizes_dev[:, 0], cw - half_sizes_dev[:, 0]),
            pos[:, 1].clamp(half_sizes_dev[:, 1], ch - half_sizes_dev[:, 1]),
        ], dim=1)
        wl = _lse_hpwl_dev(clamped, net_data_dev, port_base_dev, gamma) / wl_norm
        density = _grid_density_dev(
            clamped, sizes_dev,
            cell_x_min, cell_x_max, cell_y_min, cell_y_max,
            cell_area, grid_rows, grid_cols,
        )
        congestion = _rudy_congestion_dev(
            clamped, net_data_dev, port_base_dev, gamma,
            cell_x_min, cell_x_max, cell_y_min, cell_y_max,
            grid_h_routes, grid_v_routes, grid_rows, grid_cols,
        )
        overlap = _overlap_penalty_dev(clamped[:n_hard], half_sizes_dev[:n_hard])
        overlap_norm = overlap / (cw * ch)
        proxy = wl + 0.5 * density + 0.5 * congestion
        loss = proxy + overlap_lambda * overlap_norm
        loss.backward()
        torch.nn.utils.clip_grad_norm_([pos], max_norm=20.0)
        optimizer.step()

    with torch.no_grad():
        pos.data[fixed_mask_dev] = init_pinned_dev[fixed_mask_dev]
        pos.data[:, 0] = pos.data[:, 0].clamp(
            half_sizes_dev[:, 0], cw - half_sizes_dev[:, 0]
        )
        pos.data[:, 1] = pos.data[:, 1].clamp(
            half_sizes_dev[:, 1], ch - half_sizes_dev[:, 1]
        )
    return pos.detach()


def _sync_evaluator_to(
    evaluator: IncrementalProxyEvaluator,
    target_placement: torch.Tensor,
    hard_movable: List[int],
) -> int:
    """Apply per-macro moves so evaluator's placement matches target.
    Only walks hard_movable (soft macros' positions don't affect overlap
    metrics; their positions in target are honored too via direct write
    to evaluator.placement for soft indices). Returns number of moves
    applied via evaluator.move(...)."""
    n_moves = 0
    target_f64 = target_placement.detach().cpu().to(torch.float64)
    for m in hard_movable:
        nx = float(target_f64[m, 0])
        ny = float(target_f64[m, 1])
        cx = float(evaluator.placement[m, 0])
        cy = float(evaluator.placement[m, 1])
        if abs(nx - cx) > 1e-9 or abs(ny - cy) > 1e-9:
            evaluator.move(m, (nx, ny))
            n_moves += 1
    # Soft macros: write directly (they don't affect overlap; their cost
    # contribution flows through HPWL but the evaluator state for them
    # is fully captured in self.placement + self._update_macro_pin_positions).
    n_soft_writes = 0
    for m in range(evaluator.benchmark.num_hard_macros, evaluator.benchmark.num_macros):
        if bool(evaluator.benchmark.macro_fixed[m]):
            continue
        nx = float(target_f64[m, 0])
        ny = float(target_f64[m, 1])
        cx = float(evaluator.placement[m, 0])
        cy = float(evaluator.placement[m, 1])
        if abs(nx - cx) > 1e-9 or abs(ny - cy) > 1e-9:
            evaluator.move(m, (nx, ny))
            n_soft_writes += 1
    return n_moves + n_soft_writes


def run_dpo_basin_lns(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    hard_movable: List[int],
    time_budget_s: float,
    n_steps: int = 120,
    lr: float = 0.02,
    gamma_frac: float = 0.001,
    overlap_lambda: float = 500.0,
    noise_std_frac_schedule: tuple = (0.0, 0.005, 0.015, 0.030, 0.060),
    seed: int = 42,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Multi-restart full-pose GPU DPO basin sampler.

    Each restart:
      1. snapshot current best-known placement (init_pos for restart 0;
         best for subsequent).
      2. perturb ALL hard-movable positions with Gaussian noise scaled
         by `noise_std_frac_schedule[restart_idx % len]` × canvas mean
         dimension. Restart 0 has noise=0 (pure polish); subsequent
         restarts cycle through the schedule for basin diversity.
      3. send to device; run `n_steps` Adam steps on smooth proxy +
         overlap penalty (all-pose, fixed macros re-pinned each step).
      4. project_overlaps to clean residual overlaps.
      5. compute exact proxy via `compute_proxy_cost`.
      6. accept iff lower than best by > 1e-7. Defer evaluator state
         update to end-of-budget.

    The evaluator's state is updated ONCE at the end via
    `_sync_evaluator_to(best)`, avoiding per-restart incremental churn.
    This matches the K-joint pattern of accept-after-batch.
    """
    from macro_place.objective import compute_proxy_cost  # avoid heavy top-of-file import

    rng = np.random.default_rng(seed=seed)
    device = _select_device()

    if not hard_movable:
        cur = evaluator.current_cost()["proxy"]
        return {
            "restarts": 0, "accepts": 0, "best_proxy": cur,
            "init_proxy": cur, "final_proxy": cur,
            "total_improvement": 0.0,
            "wall_total_s": 0.0, "dpo_wall_s": 0.0, "project_wall_s": 0.0,
            "exact_eval_wall_s": 0.0, "device": str(device),
        }

    # Constants and device tensors built once.
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    n_hard = benchmark.num_hard_macros
    canvas_mean = 0.5 * (cw + ch)
    sizes_cpu = benchmark.macro_sizes.to(torch.float32)
    half_sizes_cpu = sizes_cpu / 2

    sizes_dev = sizes_cpu.to(device)
    half_sizes_dev = half_sizes_cpu.to(device)
    fixed_mask_dev = benchmark.macro_fixed.to(device)
    grid_rows = benchmark.grid_rows
    grid_cols = benchmark.grid_cols
    cell_w = cw / grid_cols
    cell_h = ch / grid_rows
    cell_area = cell_w * cell_h
    cell_x_min = (torch.arange(grid_cols, dtype=torch.float32) * cell_w).to(device)
    cell_x_max = cell_x_min + cell_w
    cell_y_min = (torch.arange(grid_rows, dtype=torch.float32) * cell_h).to(device)
    cell_y_max = cell_y_min + cell_h
    grid_h_routes = float(cell_h * benchmark.hroutes_per_micron)
    grid_v_routes = float(cell_w * benchmark.vroutes_per_micron)
    port_base_dev = torch.zeros(1, 2, device=device)

    net_data_cpu = _extract_net_data(benchmark, plc)
    net_data_dev = _net_data_to_device(net_data_cpu, device)
    wl_norm = float((cw + ch) * net_data_cpu.total_net_count)
    gamma = float(gamma_frac * cw)

    if log_fn is not None:
        log_fn(
            f"  DPO-basin: device={device}, n_steps={n_steps}, lr={lr}, "
            f"gamma={gamma:.3f} (={gamma_frac}*cw), overlap_λ={overlap_lambda}, "
            f"noise_schedule={noise_std_frac_schedule}, "
            f"budget={time_budget_s:.0f}s, |H|={len(hard_movable)}, "
            f"num_nets={len(net_data_cpu.weights)}"
        )

    init_proxy = float(evaluator.current_cost()["proxy"])
    init_placement = evaluator.placement.detach().clone()
    best_proxy = init_proxy
    best_placement = init_placement.clone()

    # Pinned reference for fixed macros (so they re-pin at start positions).
    init_pinned_dev = init_placement.to(torch.float32).to(device)

    hard_movable_idx = np.array(hard_movable, dtype=np.int64)
    t_start = time.perf_counter()
    t_dpo = 0.0
    t_project = 0.0
    t_exact = 0.0
    restarts = 0
    accepts = 0
    skipped_overlap = 0

    while time.perf_counter() - t_start < time_budget_s:
        # 1-2. Take BEST seed and perturb hard_movable.
        candidate_cpu = best_placement.detach().clone().to(torch.float32)
        noise_std_frac = noise_std_frac_schedule[restarts % len(noise_std_frac_schedule)]
        noise_std = float(noise_std_frac * canvas_mean)
        if noise_std > 0:
            noise = rng.normal(0.0, noise_std, size=(len(hard_movable_idx), 2))
            for i_, m in enumerate(hard_movable_idx):
                hx = float(half_sizes_cpu[m, 0])
                hy = float(half_sizes_cpu[m, 1])
                nx = float(candidate_cpu[m, 0]) + float(noise[i_, 0])
                ny = float(candidate_cpu[m, 1]) + float(noise[i_, 1])
                candidate_cpu[m, 0] = max(hx, min(cw - hx, nx))
                candidate_cpu[m, 1] = max(hy, min(ch - hy, ny))

        # 3. Send to device + GPU settle.
        candidate_dev = candidate_cpu.to(device)
        t0 = time.perf_counter()
        settled_dev = _dpo_settle_full(
            init_pos_dev=candidate_dev,
            fixed_mask_dev=fixed_mask_dev,
            init_pinned_dev=init_pinned_dev,
            benchmark=benchmark, net_data_dev=net_data_dev,
            half_sizes_dev=half_sizes_dev, sizes_dev=sizes_dev,
            cell_x_min=cell_x_min, cell_x_max=cell_x_max,
            cell_y_min=cell_y_min, cell_y_max=cell_y_max,
            cell_area=cell_area, grid_rows=grid_rows, grid_cols=grid_cols,
            grid_h_routes=grid_h_routes, grid_v_routes=grid_v_routes,
            port_base_dev=port_base_dev, wl_norm=wl_norm,
            n_steps=n_steps, lr=lr, gamma=gamma,
            overlap_lambda=overlap_lambda,
        )
        if device.type == "mps":
            torch.mps.synchronize()
        t_dpo += time.perf_counter() - t0

        # 4. Pull to CPU; project overlaps full-placement.
        settled_cpu = settled_dev.detach().cpu().to(torch.float32)
        # Re-pin fixed macros (defensive — _dpo_settle_full already does
        # this but float32 round-trip can drift).
        fixed_mask_cpu = benchmark.macro_fixed
        if fixed_mask_cpu.any():
            settled_cpu[fixed_mask_cpu] = init_placement.to(torch.float32)[fixed_mask_cpu]

        t0 = time.perf_counter()
        projected, n_proj = project_overlaps(settled_cpu, benchmark)
        t_project += time.perf_counter() - t0

        # 5. Verify zero overlaps (project_overlaps caps at 50 iters; if
        # it didn't converge, skip this candidate).
        ov = compute_overlap_metrics(projected, benchmark)
        if ov["overlap_count"] > 0:
            skipped_overlap += 1
            restarts += 1
            continue

        # 6. Exact proxy via compute_proxy_cost.
        t0 = time.perf_counter()
        cost = compute_proxy_cost(projected, benchmark, plc)
        t_exact += time.perf_counter() - t0
        cand_proxy = float(cost["proxy_cost"])

        if cand_proxy < best_proxy - 1e-7:
            best_proxy = cand_proxy
            best_placement = projected.detach().clone().to(torch.float64)
            accepts += 1
            if log_fn is not None:
                log_fn(
                    f"  DPO-basin restart {restarts}: ACCEPT "
                    f"noise_std_frac={noise_std_frac:.4f} "
                    f"proxy={cand_proxy:.5f} (Δ={best_proxy - init_proxy:+.5f})"
                )

        restarts += 1

    # 7. Sync evaluator state to best_placement (one batched update).
    moves_synced = 0
    if best_proxy < init_proxy - 1e-7:
        moves_synced = _sync_evaluator_to(evaluator, best_placement, hard_movable)
        # Defensive: post-sync overlap check.
        ov = compute_overlap_metrics(evaluator.placement, benchmark)
        if ov["overlap_count"] > 0:
            # Roll back to init_placement.
            if log_fn is not None:
                log_fn(
                    f"  DPO-basin: post-sync overlap detected "
                    f"({ov['overlap_count']} overlap(s)); rolling back."
                )
            _sync_evaluator_to(evaluator, init_placement, hard_movable)

    final_proxy = float(evaluator.current_cost()["proxy"])
    return {
        "restarts": restarts, "accepts": accepts,
        "skipped_overlap": skipped_overlap,
        "moves_synced": moves_synced,
        "init_proxy": init_proxy, "best_proxy": best_proxy,
        "final_proxy": final_proxy,
        "total_improvement": init_proxy - final_proxy,
        "wall_total_s": time.perf_counter() - t_start,
        "dpo_wall_s": t_dpo,
        "project_wall_s": t_project,
        "exact_eval_wall_s": t_exact,
        "device": str(device),
    }


# ── Placer ─────────────────────────────────────────────────────────────────


class CDLNSSADPOBasinPlacer:
    """E53 — DPO init + CD + LNS-gridbin + SA-v2 + GPU DPO basin LNS.

    Same backbone as E41. Step 7 swaps K-joint enumeration for a GPU
    DPO basin-evaluator inside an LNS destroy/refine/accept loop.
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
        dpo_basin_budget_s: float = 600.0,
        dpo_basin_n_steps: int = 120,
        dpo_basin_lr: float = 0.02,
        dpo_basin_gamma_frac: float = 0.001,
        dpo_basin_overlap_lambda: float = 500.0,
        dpo_basin_noise_schedule: tuple = (0.0, 0.005, 0.015, 0.030, 0.060),
        dpo_basin_seed: int = 42,
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
        self.dpo_basin_budget_s = float(dpo_basin_budget_s)
        self.dpo_basin_n_steps = int(dpo_basin_n_steps)
        self.dpo_basin_lr = float(dpo_basin_lr)
        self.dpo_basin_gamma_frac = float(dpo_basin_gamma_frac)
        self.dpo_basin_overlap_lambda = float(dpo_basin_overlap_lambda)
        self.dpo_basin_noise_schedule = tuple(dpo_basin_noise_schedule)
        self.dpo_basin_seed = int(dpo_basin_seed)
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
            f"=== CDLNSSADPOBasinPlacer ({benchmark.name}): "
            f"DPO-init -> CD -> LNS -> SA -> DPO-basin ==="
        )
        self._log(
            f"  num_macros={benchmark.num_macros}, "
            f"n_hard={benchmark.num_hard_macros}, "
            f"canvas={benchmark.canvas_width:.1f}x{benchmark.canvas_height:.1f}, "
            f"device_avail={_select_device()}"
        )

        # 1. DPO best_of_v2 init.
        _, plc = self._load_plc_for(benchmark)
        t_init0 = time.perf_counter()
        placement = _best_of_v2_init(
            benchmark, plc, seed=self.seed,
            log_fn=self._log if self.verbose else None,
        )
        self._log(f"  DPO init wall = {time.perf_counter() - t_init0:.1f} s")

        # 2. Project overlaps.
        placement, proj_iters = project_overlaps(placement, benchmark)
        init_overlaps = compute_overlap_metrics(placement, benchmark)
        self._log(
            f"  overlap projection: {proj_iters} iters, "
            f"residual count={init_overlaps['overlap_count']}"
        )

        # 3. Build evaluator.
        placement_f64 = placement.detach().clone().to(torch.float64)
        evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
        init_cost = evaluator.current_cost()
        self._log(
            f"  init proxy={init_cost['proxy']:.5f} "
            f"[wl={init_cost['wl']:.4f} d={init_cost['density']:.4f} "
            f"c={init_cost['congestion']:.4f}]"
        )

        movable = [i for i in range(benchmark.num_macros)
                   if not bool(benchmark.macro_fixed[i])]
        hard_movable = [i for i in range(benchmark.num_hard_macros)
                        if not bool(benchmark.macro_fixed[i])]

        # 4. CD phase.
        self._log(f"  starting CD phase (cap={self.cd_hard_cap_s:.0f}s)")
        cd_stats = run_cd_adaptive(
            evaluator=evaluator, benchmark=benchmark, plc=plc,
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

        # 5. LNS phase.
        self._log(f"  starting LNS phase (budget={self.lns_budget_s:.0f}s)")
        lns_stats = run_lns_gridbin(
            evaluator=evaluator, benchmark=benchmark, plc=plc,
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

        # 6. SA-v2 phase.
        self._log(f"  starting SA-v2 phase (budget={self.sa_budget_s:.0f}s)")
        sa_stats = run_sa_polish_v2(
            evaluator=evaluator, benchmark=benchmark, plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.sa_budget_s,
            T0=self.sa_T0, Tf=self.sa_Tf,
            seed=self.sa_seed,
            breakpoint_budget=self.sa_breakpoint_budget,
            log_fn=self._log if self.verbose else None,
        )
        sa_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  SA-v2 done: lift_vs_init={sa_stats['improvement_vs_init']:+.5f}, "
            f"best={sa_stats['best_proxy']:.5f}, proxy={sa_proxy:.5f}"
        )

        # 7. Multi-restart full-pose GPU DPO basin phase.
        self._log(
            f"  starting DPO-basin phase (budget={self.dpo_basin_budget_s:.0f}s, "
            f"n_steps={self.dpo_basin_n_steps}, "
            f"noise={self.dpo_basin_noise_schedule})"
        )
        db_stats = run_dpo_basin_lns(
            evaluator=evaluator, benchmark=benchmark, plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.dpo_basin_budget_s,
            n_steps=self.dpo_basin_n_steps,
            lr=self.dpo_basin_lr,
            gamma_frac=self.dpo_basin_gamma_frac,
            overlap_lambda=self.dpo_basin_overlap_lambda,
            noise_std_frac_schedule=self.dpo_basin_noise_schedule,
            seed=self.dpo_basin_seed,
            log_fn=self._log if self.verbose else None,
        )
        final_cost = evaluator.current_cost()
        self._log(
            f"  DPO-basin done: device={db_stats['device']}, "
            f"restarts={db_stats['restarts']}, accepts={db_stats['accepts']}, "
            f"skipped_overlap={db_stats['skipped_overlap']}, "
            f"moves_synced={db_stats['moves_synced']}, "
            f"best={db_stats['best_proxy']:.5f}, "
            f"Δ={db_stats['total_improvement']:+.5f}, "
            f"wall={db_stats['wall_total_s']:.1f}s "
            f"(dpo={db_stats['dpo_wall_s']:.1f}s, "
            f"project={db_stats['project_wall_s']:.1f}s, "
            f"exact={db_stats['exact_eval_wall_s']:.1f}s), "
            f"final proxy={final_cost['proxy']:.5f}"
        )
        self._log(
            f"  TOTAL lifts: CD={init_cost['proxy'] - cd_proxy:+.5f}, "
            f"LNS={cd_proxy - lns_proxy:+.5f}, "
            f"SA={lns_proxy - sa_proxy:+.5f}, "
            f"DPO-basin={sa_proxy - final_cost['proxy']:+.5f}"
        )

        # 8. Pull placement back; preserve fixed macros.
        final_placement_f64 = evaluator.placement.detach().clone().cpu()
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            final_placement_f64[fixed_mask] = original_positions[fixed_mask]
        final_placement = final_placement_f64.to(torch.float32)

        overlaps = compute_overlap_metrics(final_placement, benchmark)
        if overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"CDLNSSADPOBasinPlacer produced {overlaps['overlap_count']} "
                f"overlaps (area {overlaps['total_overlap_area']:.4f}) on "
                f"'{benchmark.name}'"
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(DPO + project + CD + LNS + SA + DPO-basin + validate)"
        )
        return final_placement
