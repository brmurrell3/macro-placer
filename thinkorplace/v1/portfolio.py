"""E100 Weight-portfolio cascade saddle — multi-objective Hessian eigvec search.

Novel extension of E84/E97 cascade saddle. The standard smooth proxy is
  P(x) = WL(x) + 0.5·D(x) + 0.5·C(x)
and the cascade saddle finds the softest eigvec of ∇²P at each plateau.

But the softest direction of the WEIGHTED SUM may hide soft directions of
INDIVIDUAL components. Example: if WL has a strong negative mode that's
exactly cancelled by density's positive mode in P's Hessian, the cascade
never explores that direction.

Mechanism: cycle through a small portfolio of weight vectors w =
(w_WL, w_D, w_C) at each cascade iter. For each w, compute eigvec of
  P_w(x) = w_WL·WL + w_D·D + w_C·C
and ε-perturb. Polish each candidate, accept improvements on the CANONICAL
proxy (the true objective).

Portfolio (default):
  [1.0, 0.5, 0.5]  — canonical
  [1.0, 0.0, 1.0]  — congestion focus (drop density)
  [1.0, 1.0, 0.0]  — density focus (drop congestion)
  [0.0, 1.0, 1.0]  — non-WL (find non-WL escapes)
  [1.0, 2.0, 0.0]  — overkill WL+density

Each w is queried independently; eigvecs differ across w. Acceptance is
ALWAYS against the canonical metric.

Composable with E97 Lévy: per-w ε grid uses Lévy magnitudes.

Spike protocol (ibm03 cached cascade plateau):
  Compare:
    A. Pure Lévy (single weight = [1, 0.5, 0.5])
    B. Portfolio Lévy (5 weight vectors)
  Same wall budget. Portfolio gets 1/5 polishes per iter per weight.

If portfolio finds basin invisible to single-weight, lift will appear at
iter 2+ (iter 1 weight=canonical = identical to Lévy).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch


from ..common.hessian import (
    SmoothProxy, find_softest_eigenvectors, _lse_hpwl, _grid_density,
    _rudy_congestion,
)
from .levy import _sample_levy_magnitudes

from ..common.bench_paths import find_benchmark_dir
from ..common.cd_core import project_overlaps, run_cd_adaptive
from ..common.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


class WeightedSmoothProxy(SmoothProxy):
    """SmoothProxy with overridable component weights.

    Parent SmoothProxy.cost computes `wl + 0.5 * density + 0.5 * congestion`.
    This subclass exposes a `cost_weighted(positions, w_wl, w_den, w_cong)`
    method computing `w_wl * wl + w_den * density + w_cong * congestion`,
    used to build per-weight Hessians.

    cost() retains canonical [1, 0.5, 0.5] behavior for backcompat with the
    parent class.
    """

    def cost_weighted(
        self, positions: torch.Tensor,
        w_wl: float, w_den: float, w_cong: float,
    ) -> torch.Tensor:
        clamped = torch.stack([
            positions[:, 0].clamp(self.half_sizes[:, 0], self.cw - self.half_sizes[:, 0]),
            positions[:, 1].clamp(self.half_sizes[:, 1], self.ch - self.half_sizes[:, 1]),
        ], dim=1)

        result = torch.zeros(1, dtype=positions.dtype, device=positions.device).squeeze()
        if w_wl != 0.0:
            wl = _lse_hpwl(clamped, self.net_data, self.port_base, self.gamma) / self.wl_norm
            result = result + w_wl * wl
        if w_den != 0.0:
            density = _grid_density(
                clamped, self.sizes,
                self.cell_x_min, self.cell_x_max,
                self.cell_y_min, self.cell_y_max,
                self.cell_area, self.grid_rows, self.grid_cols,
            )
            result = result + w_den * density
        if w_cong != 0.0:
            congestion = _rudy_congestion(
                clamped, self.net_data, self.port_base, self.gamma,
                self.cell_x_min, self.cell_x_max,
                self.cell_y_min, self.cell_y_max,
                self.grid_h_routes, self.grid_v_routes,
                self.grid_rows, self.grid_cols,
            )
            result = result + w_cong * congestion
        return result


def make_weighted_cost(
    smooth: WeightedSmoothProxy, w_wl: float, w_den: float, w_cong: float,
):
    """Closure: returns a function f(positions) → scalar tensor, usable with hvp."""
    def f(positions: torch.Tensor) -> torch.Tensor:
        return smooth.cost_weighted(positions, w_wl, w_den, w_cong)
    return f


def find_softest_for_weight(
    smooth: WeightedSmoothProxy, state: torch.Tensor, movable_np: np.ndarray,
    w: Tuple[float, float, float],
) -> Tuple[float, np.ndarray]:
    """Build a weighted SmoothProxy on the fly, find its softest eigvec.

    Uses the same Lanczos eigsh path as the parent find_softest_eigenvectors,
    but with a custom cost function.
    """
    # Hack: temporarily monkeypatch smooth.cost to use this weighted cost.
    orig_cost = smooth.cost
    smooth.cost = lambda positions: smooth.cost_weighted(positions, *w)
    try:
        eigvals, eigvecs = find_softest_eigenvectors(
            smooth, state, movable_np, k=1, log=lambda s: None,
        )
    finally:
        smooth.cost = orig_cost
    if len(eigvals) == 0:
        return float("inf"), np.zeros(int(movable_np.sum() * 2))
    return float(eigvals[0]), eigvecs[:, 0]


def portfolio_saddle_escape(
    init_state: torch.Tensor,
    benchmark,
    plc,
    *,
    max_iters: int = 3,
    portfolio: List[Tuple[float, float, float]] = None,
    K_eps: int = 2,
    eps_scale: float = 1.0,
    polish_budget: float = 60.0,
    total_budget_s: float = 1800.0,
    min_improvement: float = 1e-5,
    eigval_tolerance: float = 1e-3,
    rng_seed: int = 42,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, dict]:
    """Cascade saddle with portfolio of weight vectors + Lévy magnitudes."""
    if log is None:
        log = lambda s: print(s, flush=True)
    rng = np.random.default_rng(rng_seed)

    if portfolio is None:
        portfolio = [
            (1.0, 0.5, 0.5),  # canonical
            (1.0, 0.0, 1.0),  # cong focus
            (1.0, 1.0, 0.0),  # density focus
            (0.0, 1.0, 1.0),  # non-WL
        ]

    n_macros = init_state.shape[0]
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable_idx = [i for i in range(n_macros) if not bool(fixed[i])]
    movable_np = (~benchmark.macro_fixed.cpu().numpy())

    state = init_state.detach().clone()
    init_proxy = float(compute_proxy_cost(state, benchmark, plc)["proxy_cost"])
    log(f"[portfolio] init proxy: {init_proxy:.5f} (|portfolio|={len(portfolio)})")
    best_proxy = init_proxy
    best_state = state.detach().clone()

    iter_log = []
    t_start = time.time()
    smooth = WeightedSmoothProxy(benchmark, plc)

    for it in range(max_iters):
        elapsed = time.time() - t_start
        remaining = total_budget_s - elapsed
        if remaining < 90.0:
            log(f"[portfolio] iter {it}: budget would be exceeded; stopping")
            break

        iter_t_start = time.time()
        prev_iter_proxy = best_proxy

        # For each weight vector in portfolio, find its softest eigvec + ε-perturb.
        for w_idx, w in enumerate(portfolio):
            if (time.time() - t_start) > total_budget_s - 60:
                break
            lam, v = find_softest_for_weight(smooth, state, movable_np, w)
            log(f"  iter {it+1} w[{w_idx}]={w} λ_w={lam:.4e}")
            if lam >= -eigval_tolerance:
                log(f"    λ_w >= -tol; skip this weight")
                continue

            # Embed to full vector
            n_dim = n_macros * 2
            mask_flat = np.zeros(n_dim, dtype=bool)
            for i, m in enumerate(movable_np):
                if bool(m):
                    mask_flat[2 * i] = True
                    mask_flat[2 * i + 1] = True
            v_full = np.zeros(n_dim, dtype=np.float64)
            v_full[mask_flat] = v
            v_2d = v_full.reshape(n_macros, 2)
            v_unit = v_2d / (np.linalg.norm(v_2d) + 1e-12)

            state_np = state.detach().cpu().numpy()
            hw_np = (benchmark.macro_sizes[:, 0] / 2).cpu().numpy()
            hh_np = (benchmark.macro_sizes[:, 1] / 2).cpu().numpy()
            cw, ch = smooth.cw, smooth.ch
            eps_grid = _sample_levy_magnitudes(K_eps, eps_scale, rng)

            for sign in [+1.0, -1.0]:
                for eps in eps_grid:
                    if (time.time() - t_start) > total_budget_s - 60:
                        break
                    pp = state_np + sign * eps * v_unit
                    pp[:, 0] = np.clip(pp[:, 0], hw_np, cw - hw_np)
                    pp[:, 1] = np.clip(pp[:, 1], hh_np, ch - hh_np)
                    cand = torch.tensor(pp, dtype=torch.float32)
                    cand[~torch.tensor(movable_np)] = state[~torch.tensor(movable_np)]
                    cand, _ = project_overlaps(cand, benchmark)
                    if compute_overlap_metrics(cand, benchmark)["overlap_count"] > 0:
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
                        log(f"    NEW BEST {pol_proxy:.5f} w={w} sign={sign:+.0f} "
                            f"eps={eps:.3f} (Δ vs init={pol_proxy - init_proxy:+.5f})")

        iter_wall = time.time() - iter_t_start
        iter_log.append({
            "iter": it, "proxy_after": best_proxy,
            "improvement_this_iter": prev_iter_proxy - best_proxy,
            "iter_wall": iter_wall,
        })

        if best_proxy >= prev_iter_proxy - min_improvement:
            log(f"  no improvement this iter; saturated")
            break

    total_wall = time.time() - t_start
    log(f"[portfolio] done: init={init_proxy:.5f} best={best_proxy:.5f} "
        f"Δ={best_proxy - init_proxy:+.5f} wall={total_wall:.0f}s iters={len(iter_log)}")

    return best_state, {
        "init_proxy": init_proxy, "best_proxy": best_proxy,
        "improvement": init_proxy - best_proxy,
        "improvement_frac": (init_proxy - best_proxy) / init_proxy,
        "iters_run": len(iter_log), "iter_log": iter_log,
        "wall_seconds": total_wall, "portfolio": portfolio,
    }

