"""Periphery cascade: add periphery direction as a perturbation candidate
alongside the Hessian eigvec in cascade saddle escape.

At each cascade iter:
  - Compute Hessian smallest eigvec v_eig (as in E97)
  - Compute periphery unit vector v_peri (per-macro nearest-edge direction)
  - Try perturbations along BOTH directions × K_eps Lévy magnitudes
  - Pick best feasible polish

The periphery direction may not be a local-curvature-soft direction, but
it's a STRUCTURAL direction known to improve WNS/TNS (DAC25-ReMaP).
Adds a second escape mode the eigvec-only cascade misses.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]
_E74 = _ROOT / "experiments" / "E74_hessian_saddle" / "code"
_E97 = _ROOT / "experiments" / "E97_levy_saddle" / "code"
for p in (_E74, _E97, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hessian_saddle import SmoothProxy, find_softest_eigenvectors
from levy_saddle import _sample_levy_magnitudes

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def compute_periphery_direction(state: torch.Tensor, benchmark) -> np.ndarray:
    """Unit vector per macro toward its nearest canvas edge."""
    cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
    half_w = (benchmark.macro_sizes[:, 0] / 2).cpu().numpy()
    half_h = (benchmark.macro_sizes[:, 1] / 2).cpu().numpy()
    fixed = benchmark.macro_fixed.cpu().numpy()
    pos = state.cpu().numpy()
    v = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        if fixed[i]:
            continue
        x, y = pos[i, 0], pos[i, 1]
        dx_l = x - half_w[i]; dx_r = (cw - half_w[i]) - x
        dy_b = y - half_h[i]; dy_t = (ch - half_h[i]) - y
        dists = [dx_l, dx_r, dy_b, dy_t]
        nearest = int(np.argmin([abs(d) for d in dists]))
        if nearest == 0:   v[i] = (-1.0, 0.0)
        elif nearest == 1: v[i] = (+1.0, 0.0)
        elif nearest == 2: v[i] = (0.0, -1.0)
        else:              v[i] = (0.0, +1.0)
    # Unit-normalize (per-macro vectors are already unit, but normalize the full vec for ε scaling).
    norm = np.linalg.norm(v) + 1e-12
    return v / norm


def periphery_cascade(
    init_state: torch.Tensor,
    benchmark,
    plc,
    *,
    max_iters: int = 3,
    K_eps: int = 3,
    eps_scale: float = 1.0,
    polish_budget: float = 180.0,
    total_budget_s: float = 1800.0,
    rng_seed: int = 42,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, dict]:
    """Cascade saddle escape with periphery direction as a 2nd candidate."""
    if log is None:
        log = lambda s: print(s, flush=True)
    rng = np.random.default_rng(rng_seed)

    n_macros = init_state.shape[0]
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable_idx = [i for i in range(n_macros) if not bool(fixed[i])]
    movable_np = (~benchmark.macro_fixed.cpu().numpy())

    state = init_state.detach().clone()
    init_proxy = float(compute_proxy_cost(state, benchmark, plc)["proxy_cost"])
    log(f"[periphery_cascade] init proxy: {init_proxy:.5f}")
    best_proxy = init_proxy
    best_state = state.detach().clone()

    t_start = time.time()
    smooth = SmoothProxy(benchmark, plc)

    for it in range(max_iters):
        elapsed = time.time() - t_start
        if elapsed > total_budget_s - 60:
            log(f"[periphery_cascade] iter {it}: budget exhausted")
            break

        eps_grid = _sample_levy_magnitudes(K_eps, eps_scale, rng)
        log(f"[periphery_cascade] iter {it+1}/{max_iters} eps={['%.2f' % e for e in eps_grid]}")

        # Direction 1: Hessian eigvec.
        eigvals, eigvecs = find_softest_eigenvectors(
            smooth, state, movable_np, k=1, log=lambda s: None,
        )
        n_dim = n_macros * 2
        mask_flat = np.zeros(n_dim, dtype=bool)
        for i, m in enumerate(movable_np):
            if bool(m):
                mask_flat[2 * i] = True
                mask_flat[2 * i + 1] = True
        if len(eigvals) > 0:
            v_full = np.zeros(n_dim, dtype=np.float64)
            v_full[mask_flat] = eigvecs[:, 0]
            v_eig = v_full.reshape(n_macros, 2)
            v_eig = v_eig / (np.linalg.norm(v_eig) + 1e-12)
            log(f"  λ_min = {eigvals[0]:.4e}")
        else:
            v_eig = None

        # Direction 2: periphery.
        v_peri = compute_periphery_direction(state, benchmark)

        # Try both directions × ±sign × eps grid.
        candidates = []
        if v_eig is not None:
            candidates.append(("eig", v_eig))
        candidates.append(("peri", v_peri))

        state_np = state.detach().cpu().numpy()
        hw_np = (benchmark.macro_sizes[:, 0] / 2).cpu().numpy()
        hh_np = (benchmark.macro_sizes[:, 1] / 2).cpu().numpy()
        cw, ch = smooth.cw, smooth.ch

        prev_proxy = best_proxy

        for dir_name, v_unit in candidates:
            for sign in ([+1.0, -1.0] if dir_name == "eig" else [+1.0]):
                # Periphery only +1 sign (away from center = toward edge).
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
                        log(f"  {dir_name} sign={sign:+.0f} eps={eps:.2f}: ovl={ovl} after project; trying polish anyway")
                    ev = IncrementalProxyEvaluator(benchmark, plc, cand.clone())
                    run_cd_adaptive(
                        ev, benchmark, plc, movable_idx,
                        min_time_s=30.0, hard_cap_s=polish_budget,
                        patience=3, plateau_threshold=0.001, log_fn=None,
                    )
                    polished = ev.placement.detach().clone().to(torch.float32)
                    pol_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
                    pol_ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
                    tag = "*" if (pol_ovl == 0 and pol_proxy < best_proxy - 1e-7) else " "
                    log(f"  {tag} {dir_name} s={sign:+.0f} ε={eps:.2f} → proxy={pol_proxy:.5f} ovl={pol_ovl}")
                    if pol_ovl == 0 and pol_proxy < best_proxy - 1e-7:
                        best_proxy = pol_proxy
                        best_state = polished.detach().clone()
                        state = polished

        if best_proxy >= prev_proxy - 1e-5:
            log(f"  no improvement; saturated")
            break

    total_wall = time.time() - t_start
    log(f"[periphery_cascade] done: init={init_proxy:.5f} best={best_proxy:.5f} "
        f"Δ={best_proxy - init_proxy:+.5f} wall={total_wall:.0f}s")
    return best_state, {
        "init_proxy": init_proxy, "best_proxy": best_proxy,
        "wall_seconds": total_wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--start-pt", required=True, help="path to cached placement .pt")
    ap.add_argument("--max-iters", type=int, default=2)
    ap.add_argument("--K-eps", type=int, default=3)
    ap.add_argument("--eps-scale", type=float, default=1.0)
    ap.add_argument("--polish-budget", type=float, default=120.0)
    ap.add_argument("--total-budget", type=float, default=1200.0)
    args = ap.parse_args()

    bench_dir = find_benchmark_dir(args.bench)
    bench, plc = load_benchmark_from_dir(str(bench_dir))
    data = torch.load(args.start_pt, map_location="cpu", weights_only=False)
    init = data["placement"].to(torch.float32) if isinstance(data, dict) else data.to(torch.float32)

    final, stats = periphery_cascade(
        init, bench, plc,
        max_iters=args.max_iters, K_eps=args.K_eps,
        eps_scale=args.eps_scale, polish_budget=args.polish_budget,
        total_budget_s=args.total_budget,
    )
    print(f"\nFinal: {stats}")
    out = _ROOT / "experiments" / "E107_periphery_bias" / "results" / f"{args.bench}_periphery_cascade.pt"
    torch.save({"bench": args.bench, "placement": final, "stats": stats}, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
