"""K-block SP transplant: cluster-level transplant between basins.

Generalizes E61 V2 (canvas-quadrant, K~150) to finer-than-quadrant
granularity (K=10-50) with topology-targeted cluster centers.

Each transplant attempt:
  1. Pick cluster center (high-disagreement macro or random).
  2. Take K nearest macros to center.position (in current state).
  3. Set those K macros to other_basin's positions.
  4. project_overlaps to absorb local disruption.
  5. If feasible AND proxy improved: accept.

Tested K values: 10, 20, 50, 100. K=2 was E70 (FALSIFIED — too small).
K~150 is E61 V2 quadrant. K=10-50 is the unexplored sweet spot.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_E69 = _ROOT / "experiments" / "E69_sequence_pair_search" / "code"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_E69) not in sys.path:
    sys.path.insert(0, str(_E69))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sp_inverter import encode

from macro_place.benchmark import Benchmark
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def _build_disagreement_density(sp_a, sp_b, n_hard) -> np.ndarray:
    """For each macro, count how many pair disagreements involve it."""
    rp_a = sp_a.rank_plus(); rm_a = sp_a.rank_minus()
    rp_b = sp_b.rank_plus(); rm_b = sp_b.rank_minus()
    density = np.zeros(n_hard, dtype=np.int64)
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            p_a = rp_a[i] < rp_a[j]; m_a = rm_a[i] < rm_a[j]
            p_b = rp_b[i] < rp_b[j]; m_b = rm_b[i] < rm_b[j]
            if (p_a != p_b) or (m_a != m_b):
                density[i] += 1
                density[j] += 1
    return density


def _k_nearest_macros(pos: np.ndarray, n_hard: int, center_idx: int,
                      k: int) -> List[int]:
    """Return indices of k macros nearest (by L2) to center_idx, including center."""
    cx, cy = pos[center_idx, 0], pos[center_idx, 1]
    dists = np.hypot(pos[:n_hard, 0] - cx, pos[:n_hard, 1] - cy)
    order = np.argsort(dists)
    return order[:k].tolist()


def kblock_transplant_search(
    state: torch.Tensor,
    other: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    K_values: Tuple[int, ...] = (10, 20, 50),
    max_trials: int = 500,
    budget_seconds: float = 1800.0,
    log: Optional[Callable[[str], None]] = None,
    seed: int = 42,
    density_centered_frac: float = 0.7,
    extended_proj_iters: int = 100,
) -> Tuple[torch.Tensor, dict]:
    """K-block transplant search.

    For each trial: pick cluster center (density-targeted with prob
    density_centered_frac, else random), pick K from K_values, take
    K nearest macros, transplant their positions from other basin,
    project_overlaps with extended iterations, accept iff feasible
    AND proxy improvement.

    Returns (final_placement, stats_dict).
    """
    if log is None:
        log = lambda s: print(s, flush=True)

    rng = np.random.default_rng(seed)
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes

    current = state.detach().clone()
    other_np = other.detach().cpu().numpy().astype(np.float64)

    log("[kblock] encoding SP for state and other basins...")
    sp_state = encode(state, sizes, n_hard)
    sp_other = encode(other, sizes, n_hard)
    log("[kblock] computing disagreement density...")
    density = _build_disagreement_density(sp_state, sp_other, n_hard)
    log(f"[kblock] density min={density.min()}, mean={density.mean():.1f}, "
        f"max={density.max()}, sum={density.sum()}")

    # Sort macros by disagreement density (descending).
    high_density_order = np.argsort(-density)

    starting_proxy = float(compute_proxy_cost(current, benchmark, plc)["proxy_cost"])
    log(f"[kblock] starting proxy: {starting_proxy:.5f}")

    current_proxy = starting_proxy
    accepted = 0
    feas_failed = 0
    proxy_failed = 0
    proxy_history = [starting_proxy]
    accepts_by_K: dict = {k: 0 for k in K_values}
    t0 = time.time()
    trial_idx = 0

    for trial_idx in range(max_trials):
        if (time.time() - t0) > budget_seconds:
            log(f"[kblock] reached budget {budget_seconds:.0f}s")
            break

        # Pick center.
        if rng.random() < density_centered_frac:
            # Sample from top-K density macros.
            top_n = max(20, n_hard // 4)
            center_idx = int(high_density_order[rng.integers(top_n)])
        else:
            center_idx = int(rng.integers(n_hard))

        # Pick K.
        K = int(K_values[rng.integers(len(K_values))])
        K = min(K, n_hard)

        # Find K nearest macros.
        cur_pos = current[:n_hard].detach().cpu().numpy().astype(np.float64)
        cluster = _k_nearest_macros(cur_pos, n_hard, center_idx, K)

        # Transplant.
        candidate = current.clone()
        for m in cluster:
            candidate[m, 0] = float(other_np[m, 0])
            candidate[m, 1] = float(other_np[m, 1])

        # Legalize with extended iterations. The standard project_overlaps
        # caps at 50 iters; for cluster transplant we may need more.
        # We call it multiple times if needed.
        candidate, n_iter1 = project_overlaps(candidate, benchmark)
        ovl = compute_overlap_metrics(candidate, benchmark)
        # Keep applying project_overlaps if still has overlaps and within budget.
        passes_used = 1
        while ovl["overlap_count"] > 0 and passes_used < extended_proj_iters // 50:
            candidate, _ = project_overlaps(candidate, benchmark)
            ovl = compute_overlap_metrics(candidate, benchmark)
            passes_used += 1

        if ovl["overlap_count"] > 0:
            feas_failed += 1
            continue

        cand_proxy = float(compute_proxy_cost(candidate, benchmark, plc)["proxy_cost"])

        if cand_proxy < current_proxy - 1e-7:
            current = candidate
            current_proxy = cand_proxy
            accepted += 1
            accepts_by_K[K] += 1
            proxy_history.append(current_proxy)
            if accepted <= 5 or accepted % 5 == 0:
                log(f"[kblock] [{trial_idx + 1}/{max_trials}] K={K} center={center_idx} "
                    f"density={density[center_idx]} ACCEPT → proxy {current_proxy:.5f} "
                    f"(Δ={cand_proxy - proxy_history[-2]:+.5f}, accepted={accepted})")
        else:
            proxy_failed += 1

    wall = time.time() - t0
    trials = trial_idx + 1
    log(
        f"[kblock] done: trials={trials} accepted={accepted} feas_failed={feas_failed} "
        f"proxy_failed={proxy_failed} wall={wall:.0f}s"
    )
    log(f"[kblock] accepts by K: {accepts_by_K}")
    log(f"[kblock] final proxy: {current_proxy:.5f} "
        f"(vs starting {starting_proxy:.5f}, Δ={current_proxy - starting_proxy:+.5f})")

    return current, {
        "trials": trials,
        "accepted": accepted,
        "feas_failed": feas_failed,
        "proxy_failed": proxy_failed,
        "accepts_by_K": accepts_by_K,
        "starting_proxy": starting_proxy,
        "final_proxy": current_proxy,
        "improvement": starting_proxy - current_proxy,
        "improvement_frac": (starting_proxy - current_proxy) / starting_proxy
                             if starting_proxy > 0 else 0.0,
        "wall_seconds": wall,
        "proxy_history": proxy_history,
    }


def cd_polish(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    budget_seconds: float = 600.0,
    log: Optional[Callable[[str], None]] = None,
) -> torch.Tensor:
    if log is None:
        log = lambda s: print(s, flush=True)
    log(f"[cd-polish] starting (budget {budget_seconds:.0f}s)...")
    placement = placement.detach().clone()
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement)
    n_hard = benchmark.num_hard_macros
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable = [i for i in range(n_hard) if not bool(fixed[i])]
    info = run_cd_adaptive(
        evaluator, benchmark, plc, movable,
        min_time_s=60.0, hard_cap_s=budget_seconds,
        patience=3, plateau_threshold=0.001,
        log_fn=None,
    )
    log(f"[cd-polish] done: {info.get('exit_reason')} after {info.get('total_moves', '?')} moves")
    return evaluator.placement.detach().clone().to(torch.float32)
