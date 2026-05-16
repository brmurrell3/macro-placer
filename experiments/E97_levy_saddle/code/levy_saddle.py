"""E97 Lévy-flight cascade saddle escape — heavy-tailed ε magnitudes.

Drop-in variant of `experiments/E84_cascading_saddle/code/cascading_saddle.py`.
Replaces the fixed ε grid `eps_values=(0.3, 1.0, 3.0)` with K random draws from
a heavy-tailed (half-Cauchy) distribution.

Rationale: the fixed Gaussian-scale grid biases toward "typical" perturbation
sizes (~1× cw·sigma). Heavy-tailed magnitudes give:
  - most draws around the typical scale (median ~σ for half-Cauchy)
  - occasional 5-10× larger jumps (heavy tail) — escape distant basins
  - occasional 0.1× small probes — find adjacent valleys

Same eigvec direction; only the magnitude distribution changes. Pure additive
exploration, doesn't break any cascade saddle invariant.

Spike protocol:
  1. Load a cached cascade plateau (or a fresh SDF→CD-adaptive plateau).
  2. Run baseline (Gaussian eps grid) saddle escape.
  3. Run Lévy variant saddle escape from same plateau.
  4. Compare final proxy.

If Lévy finds a strictly lower proxy in <2× wall, integrate into hybrid.
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

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_E74 = _ROOT / "experiments" / "E74_hessian_saddle" / "code"
_E84 = _ROOT / "experiments" / "E84_cascading_saddle" / "code"
for p in (_E74, _E84, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hessian_saddle import SmoothProxy, find_softest_eigenvectors

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def _sample_levy_magnitudes(K: int, scale: float, rng: np.random.Generator) -> List[float]:
    """K positive magnitudes from |Cauchy(0, scale)|.

    Cauchy has heavy tails (no finite variance). Median = scale.
    Typical: 50% draws in [0.41*scale, 2.4*scale], 5% > 12.7*scale.

    Returns ascending-sorted magnitudes so we can break the inner loop
    when budget runs out without missing the smallest probes.
    """
    # Half-Cauchy: |X| where X ~ Cauchy(0, scale)
    u = rng.uniform(0.0, 1.0, size=K)
    # Inverse CDF for half-Cauchy: F^-1(u) = scale * tan(pi/2 * u)
    mags = scale * np.tan(np.pi * 0.5 * u)
    # Cap at 50*scale to avoid wasting evals on absurd jumps
    mags = np.clip(mags, 0.01 * scale, 50.0 * scale)
    return sorted(mags.tolist())


def levy_saddle_escape(
    init_state: torch.Tensor,
    benchmark,
    plc,
    *,
    max_iters: int = 5,
    K_eps: int = 6,
    eps_scale: float = 1.0,  # median of half-Cauchy
    polish_budget: float = 180.0,
    total_budget_s: float = 3600.0,
    min_improvement: float = 1e-5,
    eigval_tolerance: float = 1e-3,
    rng_seed: int = 42,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, dict]:
    """Cascade saddle escape with Lévy-flight ε magnitudes."""
    if log is None:
        log = lambda s: print(s, flush=True)
    rng = np.random.default_rng(rng_seed)

    n_macros = init_state.shape[0]
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable_idx = [i for i in range(n_macros) if not bool(fixed[i])]
    movable_np = (~benchmark.macro_fixed.cpu().numpy())

    state = init_state.detach().clone()
    init_proxy = float(compute_proxy_cost(state, benchmark, plc)["proxy_cost"])
    log(f"[levy] init proxy: {init_proxy:.5f}")
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
            log(f"[levy] iter {it}: budget would be exceeded; stopping")
            break

        iter_t_start = time.time()
        eps_grid = _sample_levy_magnitudes(K_eps, eps_scale, rng)
        log(f"[levy] iter {it+1}/{max_iters} eps_grid={['%.3f' % e for e in eps_grid]}")

        eigvals, eigvecs = find_softest_eigenvectors(
            smooth, state, movable_np, k=1, log=lambda s: None,
        )
        if len(eigvals) == 0:
            log(f"  no eigvecs returned; stopping cascade")
            break
        lam_min = float(eigvals[0])
        log(f"  λ_min = {lam_min:.4e}")

        if lam_min >= -eigval_tolerance:
            log(f"  λ_min >= -tol; reached local min")
            iter_log.append({"iter": it, "lam_min": lam_min, "stopped": "true_min"})
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

        # Same eps grid for both signs (eigvec sign is arbitrary).
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
                    log(f"  iter{it+1} sign={sign:+.0f} eps={eps:.3f}: NEW BEST {pol_proxy:.5f} "
                        f"(Δ vs init={pol_proxy - init_proxy:+.5f})")

        iter_wall = time.time() - iter_t_start
        avg_iter_wall = iter_wall if avg_iter_wall == 0.0 else 0.5 * avg_iter_wall + 0.5 * iter_wall
        iter_log.append({
            "iter": it, "lam_min": lam_min, "proxy_after": best_proxy,
            "improvement_this_iter": prev_iter_proxy - best_proxy, "iter_wall": iter_wall,
            "eps_grid": eps_grid,
        })

        if best_proxy >= prev_iter_proxy - min_improvement:
            log(f"  no improvement; saturated")
            break

    total_wall = time.time() - t_start
    log(f"[levy] done: init={init_proxy:.5f} best={best_proxy:.5f} "
        f"Δ={best_proxy - init_proxy:+.5f} wall={total_wall:.0f}s iters={len(iter_log)}")

    return best_state, {
        "init_proxy": init_proxy, "best_proxy": best_proxy,
        "improvement": init_proxy - best_proxy,
        "improvement_frac": (init_proxy - best_proxy) / init_proxy,
        "iters_run": len(iter_log), "iter_log": iter_log,
        "wall_seconds": total_wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--start-pt", help="path to .pt with 'placement' key")
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--K-eps", type=int, default=6)
    ap.add_argument("--eps-scale", type=float, default=1.0)
    ap.add_argument("--polish-budget", type=float, default=60.0)
    ap.add_argument("--total-budget", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", choices=["levy", "gaussian"], default="levy",
                    help="levy = heavy-tail; gaussian = original fixed [0.3,1,3]")
    args = ap.parse_args()

    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(args.bench)))
    if args.start_pt:
        data = torch.load(args.start_pt, weights_only=False)
        if "placement" in data:
            start = data["placement"]
        elif "polished_placement" in data:
            start = data["polished_placement"]
        else:
            raise SystemExit(f"no known placement key in {args.start_pt}: {list(data.keys())}")
        start = start.to(torch.float32)
    else:
        # Fresh SDF→CD plateau (~30s for ibm01-class)
        from macro_place.sdf_init import SDFPlacer
        from macro_place.cd_core import run_cd_adaptive as _run_cd
        print(f"No --start-pt; building SDF→CD plateau for {args.bench}...")
        sdf = SDFPlacer(seed=42, n_iters=500).place(bench).to(torch.float32)
        sdf, _ = project_overlaps(sdf, bench)
        ev = IncrementalProxyEvaluator(bench, plc, sdf.clone().to(torch.float64))
        n_hard = bench.num_hard_macros
        fixed = bench.macro_fixed.cpu().numpy()
        movable = [i for i in range(bench.num_macros) if not bool(fixed[i])]
        _run_cd(
            evaluator=ev, benchmark=bench, plc=plc, movable=movable,
            min_time_s=30.0, hard_cap_s=120.0,
            patience=3, plateau_threshold=0.001, log_fn=None,
        )
        start = ev.placement.detach().clone().to(torch.float32)
        start, _ = project_overlaps(start, bench)
        init_proxy = float(compute_proxy_cost(start, bench, plc)["proxy_cost"])
        print(f"  plateau proxy: {init_proxy:.5f}")

    if args.variant == "levy":
        result, stats = levy_saddle_escape(
            start, bench, plc,
            max_iters=args.max_iters, K_eps=args.K_eps, eps_scale=args.eps_scale,
            polish_budget=args.polish_budget, total_budget_s=args.total_budget,
            rng_seed=args.seed,
        )
    else:
        from cascading_saddle import cascading_saddle_escape
        result, stats = cascading_saddle_escape(
            start, bench, plc,
            max_iters=args.max_iters, eps_values=(0.3, 1.0, 3.0),
            polish_budget=args.polish_budget, total_budget_s=args.total_budget,
        )

    final_proxy = float(compute_proxy_cost(result, bench, plc)["proxy_cost"])
    final_ovl = compute_overlap_metrics(result, bench)["overlap_count"]
    print(f"\n=== {args.bench} {args.variant.upper()} RESULT ===")
    print(f"  init: {stats['init_proxy']:.5f}")
    print(f"  best: {stats['best_proxy']:.5f}  (Δ={stats['improvement']:+.5f} "
          f"= {stats['improvement_frac']*100:+.3f}%)")
    print(f"  iters: {stats['iters_run']}, ovl: {final_ovl}, wall: {stats['wall_seconds']:.0f}s")

    out_dir = _HERE.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.bench}_{args.variant}_seed{args.seed}.json"
    out_path.write_text(json.dumps({
        "bench": args.bench, "variant": args.variant, "seed": args.seed,
        "K_eps": args.K_eps, "eps_scale": args.eps_scale,
        "init_proxy": stats["init_proxy"], "best_proxy": stats["best_proxy"],
        "improvement": stats["improvement"], "improvement_frac": stats["improvement_frac"],
        "iters_run": stats["iters_run"], "iter_log": stats["iter_log"],
        "final_overlap": int(final_ovl), "wall_seconds": stats["wall_seconds"],
    }, indent=2))
    print(f"  results: {out_path}")


if __name__ == "__main__":
    main()
