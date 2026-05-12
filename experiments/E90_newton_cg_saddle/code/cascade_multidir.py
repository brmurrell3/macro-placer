"""E90 — cascading multi-direction saddle escape.

Like E84's `cascading_saddle_escape` but with `multi_saddle_escape` as the
inner loop instead of E74's single-direction sweep. Iterates until no
improvement, time budget exhausted, or eigvec floor reaches non-negative
(true local min under the smooth proxy).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_E74 = _ROOT / "experiments" / "E74_hessian_saddle" / "code"
if str(_E74) not in sys.path:
    sys.path.insert(0, str(_E74))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hessian_saddle import SmoothProxy, find_softest_eigenvectors  # type: ignore
from multi_saddle import multi_saddle_escape

from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def cascade_multidir_escape(
    init_state: torch.Tensor,
    benchmark,
    plc,
    *,
    K: int = 3,
    eps_values: Tuple[float, ...] = (0.5, 2.0),
    polish_budget_s: float = 60.0,
    max_iters: int = 5,
    total_budget_s: float = 1800.0,
    min_improvement: float = 1e-5,
    eigval_tolerance: float = 1e-3,
    only_rank_at_least: int = 2,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, dict]:
    """Iterate `multi_saddle_escape` until plateau or budget.

    Per iter: find K softest eigvecs, run all rank-≥`only_rank_at_least`
    sign-combinations × eps, accept the best lifted result, repeat.
    """
    if log is None:
        log = lambda s: print(s, flush=True)

    init_proxy = float(compute_proxy_cost(init_state, benchmark, plc)["proxy_cost"])
    log(f"[cascade-md] init proxy: {init_proxy:.5f}")
    best_proxy = init_proxy
    best_state = init_state.detach().clone()

    iter_log = []
    t_start = time.time()
    smooth = SmoothProxy(benchmark, plc)
    movable_np = (~benchmark.macro_fixed.cpu().numpy())

    for it in range(max_iters):
        elapsed = time.time() - t_start
        remaining = total_budget_s - elapsed
        if remaining < polish_budget_s * 2:
            log(f"[cascade-md] iter {it}: budget exhausted (elapsed={elapsed:.0f}s)")
            break

        log(f"[cascade-md] iter {it+1}/{max_iters} (elapsed={elapsed:.0f}s, "
            f"remaining={remaining:.0f}s)")

        # Quick eig check to allow early-stop on true local min.
        eigvals, _ = find_softest_eigenvectors(
            smooth, best_state, movable_np, k=1, log=lambda s: None,
        )
        lam_min = float(eigvals[0]) if len(eigvals) else 0.0
        log(f"  λ_min = {lam_min:.4e}")
        if lam_min >= -eigval_tolerance:
            log(f"  λ_min ≥ -tol = -{eigval_tolerance:.0e}; true local min, stopping")
            iter_log.append({"iter": it, "lam_min": lam_min, "stopped": "true_min"})
            break

        prev_iter_proxy = best_proxy
        iter_budget = min(remaining - 30, total_budget_s / max_iters)
        new_state, stats = multi_saddle_escape(
            best_state, benchmark, plc,
            K=K, eps_values=eps_values,
            polish_budget_s=polish_budget_s,
            only_rank_at_least=only_rank_at_least,
            total_budget_s=iter_budget,
            log=lambda s: log("    " + s),
        )
        iter_best = stats["best_proxy"]
        improvement = prev_iter_proxy - iter_best
        log(f"  iter {it+1} best={iter_best:.5f}  Δ={improvement:+.5f}  "
            f"({100 * improvement / prev_iter_proxy:+.3f}%)")
        iter_log.append({
            "iter": it,
            "lam_min": lam_min,
            "proxy_before": prev_iter_proxy,
            "proxy_after": iter_best,
            "improvement_this_iter": improvement,
            "attempts": len(stats["attempts"]),
        })

        if iter_best < prev_iter_proxy - min_improvement:
            best_proxy = iter_best
            best_state = new_state.detach().clone()
        else:
            log(f"  no improvement (< {min_improvement:.0e}); cascade-md saturated")
            break

    total_wall = time.time() - t_start
    log(f"[cascade-md] done: init={init_proxy:.5f} -> best={best_proxy:.5f} "
        f"(Δ={best_proxy - init_proxy:+.5f} = {100 * (best_proxy - init_proxy) / init_proxy:+.3f}%) "
        f"wall={total_wall:.0f}s iters={len(iter_log)}")

    return best_state, {
        "init_proxy": init_proxy,
        "best_proxy": best_proxy,
        "improvement": init_proxy - best_proxy,
        "improvement_frac": (init_proxy - best_proxy) / init_proxy if init_proxy > 0 else 0.0,
        "iters_run": len(iter_log),
        "iter_log": iter_log,
        "wall_seconds": total_wall,
    }
