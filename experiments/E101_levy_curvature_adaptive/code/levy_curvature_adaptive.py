"""E101 Curvature-adaptive Lévy saddle escape.

Modification of `experiments/E97_levy_saddle/code/levy_saddle.py:levy_saddle_escape`:
the half-Cauchy median (eps_scale) is set per-iter from the current iter's
smallest eigval magnitude:
    σ_iter = beta * |λ_min|^{-0.5}

Theoretical basis: classical Newton step size ~ 1/√(curvature). The smooth
proxy at the cascade plateau has a softest eigenvector with eigval λ_min;
along that direction the local quadratic approximation is f ≈ f₀ + (1/2)·λ_min·t².
Optimal step to escape this quadratic is t* ~ 1/√(λ_min).

We don't actually want a Newton step (that would minimize the quadratic in
the soft direction); we want to ESCAPE to a different basin. So the step
should be a few times the Newton scale. β=2.0-5.0 is a reasonable starting
range; β=3.0 chosen as default.

Same eigvec, same Lévy distribution, same K_eps. Only the per-iter median
changes — λ_min is observable per iter, no per-bench tuning.
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
_E74 = _ROOT / "experiments" / "E74_hessian_saddle" / "code"
_E97 = _ROOT / "experiments" / "E97_levy_saddle" / "code"
for p in (_E74, _E97, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hessian_saddle import SmoothProxy, find_softest_eigenvectors
from levy_saddle import _sample_levy_magnitudes

from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def curvature_adaptive_levy_saddle_escape(
    init_state: torch.Tensor,
    benchmark,
    plc,
    *,
    max_iters: int = 6,
    K_eps: int = 3,
    beta: float = 3.0,   # σ multiplier on Newton scale 1/√|λ|
    eps_scale_floor: float = 0.1,    # don't let σ drop below this
    eps_scale_ceiling: float = 10.0, # or above this
    polish_budget: float = 60.0,
    total_budget_s: float = 3600.0,
    min_improvement: float = 1e-5,
    eigval_tolerance: float = 1e-3,
    rng_seed: int = 42,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, dict]:
    """Cascade saddle: Lévy ε magnitudes with curvature-adaptive median σ."""
    if log is None:
        log = lambda s: print(s, flush=True)
    rng = np.random.default_rng(rng_seed)

    n_macros = init_state.shape[0]
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable_idx = [i for i in range(n_macros) if not bool(fixed[i])]
    movable_np = (~benchmark.macro_fixed.cpu().numpy())

    state = init_state.detach().clone()
    init_proxy = float(compute_proxy_cost(state, benchmark, plc)["proxy_cost"])
    log(f"[curv-levy] init proxy: {init_proxy:.5f} (β={beta})")
    best_proxy = init_proxy
    best_state = state.detach().clone()

    iter_log = []
    t_start = time.time()
    smooth = SmoothProxy(benchmark, plc)
    avg_iter_wall = 0.0

    for it in range(max_iters):
        elapsed = time.time() - t_start
        remaining = total_budget_s - elapsed
        threshold = 60.0 if avg_iter_wall == 0.0 else max(60.0, avg_iter_wall * 1.2)
        if remaining < threshold:
            log(f"[curv-levy] iter {it}: budget would be exceeded; stopping")
            break

        iter_t_start = time.time()

        eigvals, eigvecs = find_softest_eigenvectors(
            smooth, state, movable_np, k=1, log=lambda s: None,
        )
        if len(eigvals) == 0:
            log(f"  no eigvecs returned; stopping")
            break
        lam_min = float(eigvals[0])
        log(f"  λ_min = {lam_min:.4e}")

        if lam_min >= -eigval_tolerance:
            log(f"  λ_min >= -tol; reached local min")
            iter_log.append({"iter": it, "lam_min": lam_min, "stopped": "true_min"})
            break

        # Curvature-adaptive σ: median ~ β / √|λ_min|
        eps_scale_adaptive = beta / (abs(lam_min) ** 0.5)
        eps_scale_adaptive = max(eps_scale_floor, min(eps_scale_ceiling, eps_scale_adaptive))
        eps_grid = _sample_levy_magnitudes(K_eps, eps_scale_adaptive, rng)
        log(f"[curv-levy] iter {it+1}/{max_iters} σ_adaptive={eps_scale_adaptive:.3f} "
            f"eps_grid={['%.3f' % e for e in eps_grid]}")

        n_dim = n_macros * 2
        mask_flat = np.zeros(n_dim, dtype=bool)
        for i, m in enumerate(movable_np):
            if bool(m):
                mask_flat[2 * i] = True
                mask_flat[2 * i + 1] = True

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
            for eps in eps_grid:
                if (time.time() - t_start) > total_budget_s - 30:
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
                run_cd_adaptive(
                    ev, benchmark, plc, movable_idx,
                    min_time_s=20.0, hard_cap_s=polish_budget,
                    patience=3, plateau_threshold=0.001, log_fn=None,
                )
                polished = ev.placement.detach().clone().to(torch.float32)
                pol_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
                pol_ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
                if pol_ovl == 0 and pol_proxy < best_proxy - 1e-7:
                    best_proxy = pol_proxy
                    best_state = polished.detach().clone()
                    state = polished
                    log(f"  iter{it+1} sign={sign:+.0f} eps={eps:.3f}: NEW BEST {pol_proxy:.5f} "
                        f"(Δ vs init={pol_proxy - init_proxy:+.5f})")

        iter_wall = time.time() - iter_t_start
        avg_iter_wall = iter_wall if avg_iter_wall == 0.0 else 0.5 * avg_iter_wall + 0.5 * iter_wall
        iter_log.append({
            "iter": it, "lam_min": lam_min, "eps_scale_adaptive": eps_scale_adaptive,
            "proxy_after": best_proxy,
            "improvement_this_iter": prev_iter_proxy - best_proxy, "iter_wall": iter_wall,
            "eps_grid": eps_grid,
        })

        if best_proxy >= prev_iter_proxy - min_improvement:
            log(f"  no improvement; saturated")
            break

    total_wall = time.time() - t_start
    log(f"[curv-levy] done: init={init_proxy:.5f} best={best_proxy:.5f} "
        f"Δ={best_proxy - init_proxy:+.5f} wall={total_wall:.0f}s iters={len(iter_log)}")

    return best_state, {
        "init_proxy": init_proxy, "best_proxy": best_proxy,
        "improvement": init_proxy - best_proxy,
        "improvement_frac": (init_proxy - best_proxy) / init_proxy,
        "iters_run": len(iter_log), "iter_log": iter_log,
        "wall_seconds": total_wall, "beta": beta,
    }
