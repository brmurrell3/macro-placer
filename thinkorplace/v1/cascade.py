"""Cascading Hessian saddle escape — iterate E74 until true local min.

For each iteration:
  1. Compute smooth-proxy Hessian at current state via Lanczos.
  2. If smallest eigenvalue is non-negative (within tolerance), stop —
     we're at a true local minimum.
  3. Try ±ε perturbations along the soft mode, polish each via CD.
  4. Accept the best polished result; iterate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch


from ..common.hessian import SmoothProxy, find_softest_eigenvectors

from ..common.bench_paths import find_benchmark_dir
from ..common.runtime_safety import monotonic_seconds
from ..common.cd_core import project_overlaps, run_cd_adaptive
from ..common.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def cascading_saddle_escape(
    init_state: torch.Tensor,
    benchmark,
    plc,
    *,
    max_iters: int = 5,
    eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
    polish_budget: float = 180.0,
    total_budget_s: float = 3600.0,
    min_improvement: float = 1e-5,
    eigval_tolerance: float = 1e-3,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, dict]:
    """Apply Hessian saddle escape iteratively until plateau / true min.

    Stop conditions:
      - smallest eigenvalue >= -eigval_tolerance (true local min within tol)
      - improvement < min_improvement (cascading saturated)
      - iter >= max_iters
      - total wall > total_budget_s
    """
    if log is None:
        log = lambda s: print(s, flush=True)

    n_hard = benchmark.num_hard_macros
    n_macros = init_state.shape[0]
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable_idx = [i for i in range(n_macros) if not bool(fixed[i])]
    movable_np = (~benchmark.macro_fixed.cpu().numpy())

    state = init_state.detach().clone()
    init_proxy = float(compute_proxy_cost(state, benchmark, plc)["proxy_cost"])
    log(f"[cascade] init proxy: {init_proxy:.5f}")
    best_proxy = init_proxy
    best_state = state.detach().clone()

    iter_log = []
    t_start = monotonic_seconds()
    smooth = SmoothProxy(benchmark, plc)
    avg_iter_wall = 0.0  # rolling estimate of per-iter cost

    for it in range(max_iters):
        elapsed = monotonic_seconds() - t_start
        remaining = total_budget_s - elapsed
        # Stop if remaining is less than the predicted next-iter cost. For
        # the first iter, fall back to a 60s safety floor; subsequent iters
        # use 1.2 * avg_iter_wall to leave a safety margin.
        threshold = 60.0 if avg_iter_wall == 0.0 else max(60.0, avg_iter_wall * 1.2)
        if remaining < threshold:
            log(f"[cascade] iter {it}: budget would be exceeded "
                f"(elapsed={elapsed:.0f}s, remaining={remaining:.0f}s, "
                f"threshold={threshold:.0f}s based on avg iter {avg_iter_wall:.0f}s)")
            break

        iter_t_start = monotonic_seconds()
        log(f"[cascade] iter {it+1}/{max_iters} (elapsed={elapsed:.0f}s, "
            f"remaining={remaining:.0f}s, predicted_iter_wall={threshold:.0f}s)")

        # Find soft mode.
        t_eig = monotonic_seconds()
        eigvals, eigvecs = find_softest_eigenvectors(
            smooth, state, movable_np, k=1, log=lambda s: None,
        )
        eig_wall = monotonic_seconds() - t_eig
        if len(eigvals) == 0:
            log(f"  no eigvecs returned; stopping cascade")
            break
        lam_min = float(eigvals[0])
        log(f"  λ_min = {lam_min:.4e} (eig_wall={eig_wall:.1f}s)")

        if lam_min >= -eigval_tolerance:
            log(f"  λ_min >= -tol = -{eigval_tolerance:.0e}; reached true "
                f"local min, stopping cascade")
            iter_log.append({"iter": it, "lam_min": lam_min, "stopped": "true_min"})
            break

        # Rebuild mask_flat for this iteration (find_softest_eigenvectors
        # returns only movable-coord eigvecs).
        n_dim = n_macros * 2
        mask_flat = np.zeros(n_dim, dtype=bool)
        for i, m in enumerate(movable_np):
            if bool(m):
                mask_flat[2 * i] = True
                mask_flat[2 * i + 1] = True

        # Build v_2d and try ±ε perturbations.
        v_full = np.zeros(n_dim, dtype=np.float64)
        v_full[mask_flat] = eigvecs[:, 0]
        v_2d = v_full.reshape(n_macros, 2)
        v_unit = v_2d / (np.linalg.norm(v_2d) + 1e-12)

        state_np = state.detach().cpu().numpy()
        hw_np = (benchmark.macro_sizes[:, 0] / 2).cpu().numpy()
        hh_np = (benchmark.macro_sizes[:, 1] / 2).cpu().numpy()
        cw, ch = smooth.cw, smooth.ch
        prev_iter_proxy = best_proxy

        for sign in [+1.0, -1.0]:
            for eps in eps_values:
                if (monotonic_seconds() - t_start) > total_budget_s - 30:
                    break
                pp = state_np + sign * eps * v_unit
                pp[:, 0] = np.clip(pp[:, 0], hw_np, cw - hw_np)
                pp[:, 1] = np.clip(pp[:, 1], hh_np, ch - hh_np)
                cand = torch.tensor(pp, dtype=torch.float32)
                cand[~torch.tensor(movable_np)] = state[~torch.tensor(movable_np)]
                cand, _ = project_overlaps(cand, benchmark)
                ovl = compute_overlap_metrics(cand, benchmark)["overlap_count"]
                if ovl > 0:
                    continue
                ev = IncrementalProxyEvaluator(benchmark, plc, cand.clone())
                cd_remaining = min(polish_budget, max(0.0, total_budget_s - (monotonic_seconds() - t_start)))
                if cd_remaining <= 0:
                    break
                run_cd_adaptive(
                    ev, benchmark, plc, movable_idx,
                    min_time_s=min(30.0, cd_remaining), hard_cap_s=cd_remaining,
                    patience=3, plateau_threshold=0.001, log_fn=None,
                )
                polished = ev.placement.detach().clone().to(torch.float32)
                pol_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
                pol_ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
                if pol_ovl == 0 and pol_proxy < best_proxy - 1e-7:
                    best_proxy = pol_proxy
                    best_state = polished.detach().clone()
                    state = polished
                    log(f"  iter{it+1} sign={sign:+.0f} eps={eps:.1f}: "
                        f"NEW BEST {pol_proxy:.5f} (Δ vs init={pol_proxy - init_proxy:+.5f})")

        # Update rolling avg iter wall (used to predict next-iter cost)
        iter_wall = monotonic_seconds() - iter_t_start
        if avg_iter_wall == 0.0:
            avg_iter_wall = iter_wall
        else:
            # Exponential moving avg with alpha=0.5 — biases toward recent
            # iters in case contention is changing over time.
            avg_iter_wall = 0.5 * avg_iter_wall + 0.5 * iter_wall

        iter_log.append({
            "iter": it,
            "lam_min": lam_min,
            "proxy_after": best_proxy,
            "improvement_this_iter": prev_iter_proxy - best_proxy,
            "iter_wall": iter_wall,
        })

        if best_proxy >= prev_iter_proxy - min_improvement:
            log(f"  no improvement (< {min_improvement:.0e}); cascade saturated")
            break

    total_wall = monotonic_seconds() - t_start
    log(f"[cascade] done: init={init_proxy:.5f} best={best_proxy:.5f} "
        f"Δ={best_proxy - init_proxy:+.5f} wall={total_wall:.0f}s iters={len(iter_log)}")

    return best_state, {
        "init_proxy": init_proxy,
        "best_proxy": best_proxy,
        "improvement": init_proxy - best_proxy,
        "improvement_frac": (init_proxy - best_proxy) / init_proxy,
        "iters_run": len(iter_log),
        "iter_log": iter_log,
        "wall_seconds": total_wall,
    }

