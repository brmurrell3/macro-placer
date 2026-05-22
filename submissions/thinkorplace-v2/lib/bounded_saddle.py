"""E138 — bounded-eigsh variant of E84 cascading saddle escape.

Same algorithm as E84 `cascading_saddle_escape`, but the inner Lanczos
call is bounded with `maxiter=50, tol=1e-2` (vs default `maxiter=500,
tol=1e-4`). Motivation: on EPYC, each eigsh iteration was ~380 s vs
budget 240 s, eating CD polish time. Sacrifices eigenvector precision
(saddle direction may be slightly off) but should still escape CD
plateaus.

We copy the cascade loop here instead of patching E84 so that other
callers of E84 are not perturbed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import scipy.sparse.linalg as spla
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for _p in (_ROOT, _HERE):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

from hessian_saddle import SmoothProxy, hessian_vector_product  # noqa: E402

from macro_place.cd_core import project_overlaps, run_cd_adaptive  # noqa: E402
from macro_place.incremental_evaluator import IncrementalProxyEvaluator  # noqa: E402
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost  # noqa: E402


def find_softest_eigenvectors_bounded(
    smooth_proxy: SmoothProxy,
    state: torch.Tensor,
    movable_mask: np.ndarray,
    *,
    k: int = 1,
    tol: float = 1e-2,
    maxiter: int = 50,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bounded variant of E74 find_softest_eigenvectors.

    Caps Lanczos iterations to `maxiter` and uses a looser tolerance
    (`tol=1e-2`). Falls through to the same shifted-power fallback that
    E74 uses if ARPACK fails to converge.
    """
    if log is None:
        log = lambda s: print(s, flush=True)

    n_macros = state.shape[0]
    n_dim = n_macros * 2

    mask_flat = np.zeros(n_dim, dtype=bool)
    for i, m in enumerate(movable_mask):
        if bool(m):
            mask_flat[2 * i] = True
            mask_flat[2 * i + 1] = True
    n_free = int(mask_flat.sum())
    log(f"[bounded] n_macros={n_macros}, n_free={n_free}, "
        f"maxiter={maxiter}, tol={tol:.0e}")

    state_full = state.detach().clone()

    def matvec(v_free: np.ndarray) -> np.ndarray:
        v_full = np.zeros(n_dim, dtype=np.float64)
        v_full[mask_flat] = v_free
        v_tensor = torch.tensor(v_full.reshape(n_macros, 2), dtype=torch.float32)
        hv = hessian_vector_product(smooth_proxy, state_full, v_tensor)
        hv_flat = hv.cpu().numpy().reshape(n_dim)
        return hv_flat[mask_flat]

    op = spla.LinearOperator(
        shape=(n_free, n_free), matvec=matvec, dtype=np.float64,
    )

    t0 = time.time()
    eigvals = []
    eigvecs = np.zeros((n_free, 0))
    try:
        eigvals, eigvecs = spla.eigsh(
            op, k=k, which="SA", tol=tol, maxiter=maxiter,
            ncv=min(2 * k + 20, n_free),
        )
    except spla.ArpackNoConvergence as e:
        log(f"[bounded] 'SA' did not converge after maxiter={maxiter}; "
            f"got {len(e.eigenvalues)} of {k} eigenvalues")
        if len(e.eigenvalues) > 0:
            eigvals = e.eigenvalues
            eigvecs = e.eigenvectors

    if len(eigvals) == 0:
        log(f"[bounded] FALLBACK: shifted power iteration on -H")
        v_random = np.random.randn(n_free).astype(np.float64)
        v_random /= np.linalg.norm(v_random)
        v_est = v_random.copy()
        h_norm_est = 1.0
        for _ in range(5):
            v_new = matvec(v_est)
            h_norm_est = np.linalg.norm(v_new)
            if h_norm_est < 1e-12:
                break
            v_est = v_new / h_norm_est
        alpha = 2.0 * h_norm_est + 1.0
        v = np.random.randn(n_free).astype(np.float64)
        v /= np.linalg.norm(v)
        # Bound the fallback too — was 200 in E74, use 50 here.
        for _ in range(50):
            v_new = alpha * v - matvec(v)
            n_new = np.linalg.norm(v_new)
            if n_new < 1e-12:
                break
            v = v_new / n_new
        hv = matvec(v)
        lam = float(np.dot(v, hv) / (np.dot(v, v) + 1e-12))
        eigvals = np.array([lam])
        eigvecs = v.reshape(-1, 1)
        log(f"[bounded]   power iteration found lambda_min ~ {lam:.4e}")

    log(f"[bounded] Lanczos done in {time.time() - t0:.1f}s; eigvals = {eigvals}")
    return eigvals, eigvecs


def bounded_cascading_saddle_escape(
    init_state: torch.Tensor,
    benchmark,
    plc,
    *,
    max_iters: int = 2,
    eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
    polish_budget: float = 180.0,
    total_budget_s: float = 120.0,
    min_improvement: float = 1e-5,
    eigval_tolerance: float = 1e-3,
    eigsh_maxiter: int = 50,
    eigsh_tol: float = 1e-2,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, dict]:
    """Bounded cascading saddle escape.

    Mirrors E84 cascading_saddle_escape exactly except for the inner
    eigsh call, which is bounded via `eigsh_maxiter` / `eigsh_tol`.
    """
    if log is None:
        log = lambda s: print(s, flush=True)

    n_macros = init_state.shape[0]
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable_idx = [i for i in range(n_macros) if not bool(fixed[i])]
    movable_np = (~benchmark.macro_fixed.cpu().numpy())

    state = init_state.detach().clone()
    init_proxy = float(compute_proxy_cost(state, benchmark, plc)["proxy_cost"])
    log(f"[bounded-cascade] init proxy: {init_proxy:.5f}")
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
            log(f"[bounded-cascade] iter {it}: budget would be exceeded "
                f"(elapsed={elapsed:.0f}s, remaining={remaining:.0f}s, "
                f"threshold={threshold:.0f}s)")
            break

        iter_t_start = time.time()
        log(f"[bounded-cascade] iter {it+1}/{max_iters} "
            f"(elapsed={elapsed:.0f}s, remaining={remaining:.0f}s)")

        t_eig = time.time()
        eigvals, eigvecs = find_softest_eigenvectors_bounded(
            smooth, state, movable_np, k=1,
            tol=eigsh_tol, maxiter=eigsh_maxiter,
            log=lambda s: log(f"  {s}"),
        )
        eig_wall = time.time() - t_eig
        if len(eigvals) == 0:
            log(f"  no eigvecs returned; stopping cascade")
            break
        lam_min = float(eigvals[0])
        log(f"  lambda_min = {lam_min:.4e} (eig_wall={eig_wall:.1f}s)")

        if lam_min >= -eigval_tolerance:
            log(f"  lambda_min >= -tol; reached true local min, stopping")
            iter_log.append({"iter": it, "lam_min": lam_min, "stopped": "true_min",
                             "eig_wall": eig_wall})
            break

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

        accepted_this_iter = False
        for sign in [+1.0, -1.0]:
            for eps in eps_values:
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
                    min_time_s=30.0, hard_cap_s=polish_budget,
                    patience=3, plateau_threshold=0.001, log_fn=None,
                )
                polished = ev.placement.detach().clone().to(torch.float32)
                pol_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
                pol_ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
                if pol_ovl == 0 and pol_proxy < best_proxy - 1e-7:
                    best_proxy = pol_proxy
                    best_state = polished.detach().clone()
                    state = polished
                    accepted_this_iter = True
                    log(f"  iter{it+1} sign={sign:+.0f} eps={eps:.1f}: "
                        f"NEW BEST {pol_proxy:.5f}")

        iter_wall = time.time() - iter_t_start
        if avg_iter_wall == 0.0:
            avg_iter_wall = iter_wall
        else:
            avg_iter_wall = 0.5 * avg_iter_wall + 0.5 * iter_wall

        iter_log.append({
            "iter": it,
            "lam_min": lam_min,
            "proxy_after": best_proxy,
            "improvement_this_iter": prev_iter_proxy - best_proxy,
            "iter_wall": iter_wall,
            "eig_wall": eig_wall,
            "accepted": accepted_this_iter,
        })

        if best_proxy >= prev_iter_proxy - min_improvement:
            log(f"  no improvement (< {min_improvement:.0e}); saturated")
            break

    total_wall = time.time() - t_start
    log(f"[bounded-cascade] done: init={init_proxy:.5f} best={best_proxy:.5f} "
        f"delta={best_proxy - init_proxy:+.5f} wall={total_wall:.0f}s "
        f"iters={len(iter_log)}")

    return best_state, {
        "init_proxy": init_proxy,
        "best_proxy": best_proxy,
        "improvement": init_proxy - best_proxy,
        "improvement_frac": (init_proxy - best_proxy) / init_proxy,
        "iters_run": len(iter_log),
        "iter_log": iter_log,
        "wall_seconds": total_wall,
    }
