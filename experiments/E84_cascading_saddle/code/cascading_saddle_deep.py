"""Deeper cascading saddle escape — match vmallela_v7's depth.

Differences from cascading_saddle.py:
  - k=5 eigvecs per iteration (vs k=1)
  - Try ±ε along EACH of 5 eigvecs (vs only softest)
  - max_iters=10 (vs 5)
  - 7 eps values (vs 3)
  - shorter polish per direction (60s vs 180s) to fit time budget

Hypothesis: vmallela_v7 hits 1.011 with deeper Hessian saddle search.
Our k=1, 3 eps, max_iters=5 leaves 6× × 3.3× × 2× = ~40× less search.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_E74_PATH = _HERE.parents[1] / "E74_hessian_saddle" / "code"
sys.path.insert(0, str(_E74_PATH))
from hessian_saddle import SmoothProxy, find_softest_eigenvectors

from macro_place.cd_core import run_cd_adaptive, project_overlaps
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def cascading_saddle_escape_deep(
    init_state: torch.Tensor,
    benchmark,
    plc,
    *,
    max_iters: int = 10,
    n_eigvecs: int = 5,
    eps_values: Tuple[float, ...] = (0.1, 0.3, 0.7, 1.5, 3.0),
    polish_budget: float = 60.0,
    total_budget_s: float = 2400.0,
    min_improvement: float = 1e-5,
    eigval_tolerance: float = 1e-3,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, dict]:
    """Deeper Hessian saddle escape: explore k softest eigvecs × N eps each."""
    if log is None:
        log = lambda s: print(s, flush=True)

    n_hard = benchmark.num_hard_macros
    n_macros = init_state.shape[0]
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable_idx = [i for i in range(n_macros) if not bool(fixed[i])]
    movable_np = (~benchmark.macro_fixed.cpu().numpy())

    state = init_state.detach().clone()
    init_proxy = float(compute_proxy_cost(state, benchmark, plc)["proxy_cost"])
    log(f"[cascade-deep] init proxy: {init_proxy:.5f} (k={n_eigvecs}, "
        f"eps={list(eps_values)}, max_iters={max_iters}, polish={polish_budget}s)")
    best_proxy = init_proxy
    best_state = state.detach().clone()

    iter_log = []
    t_start = time.time()
    smooth = SmoothProxy(benchmark, plc)
    avg_iter_wall = 0.0
    # Per-direction cost estimate (used for inner break)
    avg_dir_wall = polish_budget * 1.3  # initial estimate

    for it in range(max_iters):
        elapsed = time.time() - t_start
        remaining = total_budget_s - elapsed
        threshold = 60.0 if avg_iter_wall == 0.0 else max(60.0, avg_iter_wall * 1.2)
        if remaining < threshold:
            log(f"[cascade-deep] iter {it}: budget would be exceeded "
                f"(remaining={remaining:.0f}s, threshold={threshold:.0f}s)")
            break

        iter_t_start = time.time()
        log(f"[cascade-deep] iter {it+1}/{max_iters} (elapsed={elapsed:.0f}s, "
            f"remaining={remaining:.0f}s)")

        # Find k softest eigvecs
        t_eig = time.time()
        eigvals, eigvecs = find_softest_eigenvectors(
            smooth, state, movable_np, k=n_eigvecs, log=lambda s: None,
        )
        eig_wall = time.time() - t_eig
        if len(eigvals) == 0:
            log(f"  no eigvecs returned; stopping")
            break
        lam_min = float(eigvals[0])
        log(f"  λ[0]={lam_min:.4e} λ[-1]={float(eigvals[-1]):.4e} eig_wall={eig_wall:.1f}s")

        # Stop if all eigvals are positive (true min)
        if lam_min >= -eigval_tolerance:
            log(f"  all eigvals positive (>= -tol); reached true min")
            iter_log.append({"iter": it, "lam_min": lam_min, "stopped": "true_min"})
            break

        prev_iter_proxy = best_proxy
        n_dim = n_macros * 2
        mask_flat = np.zeros(n_dim, dtype=bool)
        for i, m in enumerate(movable_np):
            if bool(m):
                mask_flat[2 * i] = True
                mask_flat[2 * i + 1] = True

        state_np = state.detach().cpu().numpy()
        hw_np = (benchmark.macro_sizes[:, 0] / 2).cpu().numpy()
        hh_np = (benchmark.macro_sizes[:, 1] / 2).cpu().numpy()
        cw, ch = smooth.cw, smooth.ch

        # Try ±ε along each of k eigvecs (only those with negative eigenvalue)
        n_dirs_explored = 0
        n_dirs_improved = 0
        for k_idx in range(n_eigvecs):
            if k_idx >= len(eigvals):
                break
            if float(eigvals[k_idx]) >= -eigval_tolerance:
                # Skip directions with non-negative eigenvalue (no escape needed)
                continue
            v_full = np.zeros(n_dim, dtype=np.float64)
            v_full[mask_flat] = eigvecs[:, k_idx]
            v_2d = v_full.reshape(n_macros, 2)
            v_unit = v_2d / (np.linalg.norm(v_2d) + 1e-12)

            for sign in [+1.0, -1.0]:
                for eps in eps_values:
                    # Budget guard: skip remaining dirs if no time left
                    remaining_now = total_budget_s - (time.time() - t_start) - 30
                    if remaining_now < avg_dir_wall:
                        log(f"  budget exhausted at k={k_idx} sign={sign:+.0f} eps={eps}; "
                            f"({n_dirs_explored} dirs explored, {n_dirs_improved} improved)")
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
                    dir_t = time.time()
                    ev = IncrementalProxyEvaluator(benchmark, plc, cand.clone())
                    run_cd_adaptive(
                        ev, benchmark, plc, movable_idx,
                        min_time_s=15.0, hard_cap_s=polish_budget,
                        patience=3, plateau_threshold=0.001, log_fn=None,
                    )
                    polished = ev.placement.detach().clone().to(torch.float32)
                    pol_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
                    pol_ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
                    dir_wall = time.time() - dir_t
                    avg_dir_wall = 0.5 * avg_dir_wall + 0.5 * dir_wall
                    n_dirs_explored += 1
                    if pol_ovl == 0 and pol_proxy < best_proxy - 1e-7:
                        best_proxy = pol_proxy
                        best_state = polished.detach().clone()
                        state = polished
                        n_dirs_improved += 1
                        log(f"  it{it+1} k={k_idx} sign={sign:+.0f} eps={eps:.2f}: "
                            f"NEW BEST {pol_proxy:.5f}")
                else:
                    continue
                break
            else:
                continue
            break

        iter_wall = time.time() - iter_t_start
        if avg_iter_wall == 0.0:
            avg_iter_wall = iter_wall
        else:
            avg_iter_wall = 0.5 * avg_iter_wall + 0.5 * iter_wall

        iter_log.append({
            "iter": it, "lam_min": lam_min,
            "proxy_after": best_proxy,
            "improvement_this_iter": prev_iter_proxy - best_proxy,
            "iter_wall": iter_wall,
            "dirs_explored": n_dirs_explored,
            "dirs_improved": n_dirs_improved,
        })

        if best_proxy >= prev_iter_proxy - min_improvement:
            log(f"  no improvement this iter; stopping cascade")
            break

    total_wall = time.time() - t_start
    log(f"[cascade-deep] done: init={init_proxy:.5f} best={best_proxy:.5f} "
        f"Δ={best_proxy - init_proxy:+.5f} wall={total_wall:.0f}s iters={len(iter_log)}")

    return best_state, {
        "init_proxy": init_proxy,
        "best_proxy": best_proxy,
        "iters_run": len(iter_log),
        "iter_log": iter_log,
        "total_wall": total_wall,
    }
