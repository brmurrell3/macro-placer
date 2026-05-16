"""E99 Tabu cascade saddle — orthogonalize against previously-tried eigvecs.

Extension of E84 cascade + E97 Lévy. The cascade saddle escape iterates the
Hessian saddle at each plateau; each iter computes the softest eigvec and
perturbs along ±ε. But the cascade may revisit the SAME soft mode after a
small polish swing — wasting iterations on directions already explored.

Tabu rule: maintain a small LRU list of recently-used eigvecs (default
tabu_size=2). When computing the next perturbation direction:
  1. Request top-k smallest-algebraic eigvecs (k = tabu_size + 1) from eigsh.
  2. Score each by (eigval + λ * max_overlap_with_tabu).
  3. Pick the one minimizing score → soft AND orthogonal to history.
  4. Append the chosen direction to tabu list (drop oldest if > tabu_size).

This forces the cascade to explore HIGHER-RANK soft modes after the principal
one saturates — directions invisible to single-vec cascade.

Composable with Lévy: tabu picks the DIRECTION; Lévy picks the MAGNITUDES.
Both layers add diversity to the escape search.

Spike protocol:
  Same ibm01 baseline as E97. Compare:
    A. Pure Lévy (E97 levy_saddle_escape)
    B. Tabu + Lévy (this file)
  If Tabu adds 0.05-0.2% on top of Lévy, integrate into hybrid placer.
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
_E97 = _ROOT / "experiments" / "E97_levy_saddle" / "code"
for p in (_E74, _E84, _E97, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hessian_saddle import SmoothProxy, find_softest_eigenvectors
from levy_saddle import _sample_levy_magnitudes

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def _pick_tabu_direction(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    tabu_dirs: List[np.ndarray],
    overlap_lambda: float = 10.0,
) -> Tuple[int, np.ndarray, float]:
    """Pick the eigvec minimizing (eigval + λ · max_overlap_with_tabu).

    Returns (idx, chosen_eigvec, eigval).
    """
    n_eigs = eigvecs.shape[1]
    best_idx, best_score = 0, float("inf")
    for j in range(n_eigs):
        v_j = eigvecs[:, j]
        if tabu_dirs:
            max_overlap = max(abs(float(np.dot(v_j, t))) for t in tabu_dirs)
        else:
            max_overlap = 0.0
        score = float(eigvals[j]) + overlap_lambda * max_overlap
        if score < best_score:
            best_score = score
            best_idx = j
    return best_idx, eigvecs[:, best_idx], float(eigvals[best_idx])


def tabu_levy_saddle_escape(
    init_state: torch.Tensor,
    benchmark,
    plc,
    *,
    max_iters: int = 5,
    K_eps: int = 3,
    eps_scale: float = 1.0,
    tabu_size: int = 2,
    overlap_lambda: float = 10.0,
    polish_budget: float = 180.0,
    total_budget_s: float = 3600.0,
    min_improvement: float = 1e-5,
    eigval_tolerance: float = 1e-3,
    rng_seed: int = 42,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, dict]:
    """Cascade saddle escape with tabu eigvec selection + Lévy magnitudes."""
    if log is None:
        log = lambda s: print(s, flush=True)
    rng = np.random.default_rng(rng_seed)

    n_macros = init_state.shape[0]
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable_idx = [i for i in range(n_macros) if not bool(fixed[i])]
    movable_np = (~benchmark.macro_fixed.cpu().numpy())

    state = init_state.detach().clone()
    init_proxy = float(compute_proxy_cost(state, benchmark, plc)["proxy_cost"])
    log(f"[tabu-levy] init proxy: {init_proxy:.5f} (tabu_size={tabu_size})")
    best_proxy = init_proxy
    best_state = state.detach().clone()

    tabu_dirs: List[np.ndarray] = []
    iter_log = []
    t_start = time.time()
    smooth = SmoothProxy(benchmark, plc)
    avg_iter_wall = 0.0

    for it in range(max_iters):
        elapsed = time.time() - t_start
        remaining = total_budget_s - elapsed
        threshold = 60.0 if avg_iter_wall == 0.0 else max(60.0, avg_iter_wall * 1.2)
        if remaining < threshold:
            log(f"[tabu-levy] iter {it}: budget would be exceeded; stopping")
            break

        iter_t_start = time.time()
        eps_grid = _sample_levy_magnitudes(K_eps, eps_scale, rng)
        log(f"[tabu-levy] iter {it+1}/{max_iters} eps_grid={['%.3f' % e for e in eps_grid]}")

        k_request = max(1, tabu_size + 1)
        eigvals, eigvecs = find_softest_eigenvectors(
            smooth, state, movable_np, k=k_request, log=lambda s: None,
        )
        if eigvecs.shape[1] == 0:
            log(f"  no eigvecs returned; stopping")
            break

        chosen_idx, v_chosen, lam_chosen = _pick_tabu_direction(
            eigvals, eigvecs, tabu_dirs, overlap_lambda=overlap_lambda,
        )
        log(f"  chosen eigvec idx={chosen_idx}/{eigvecs.shape[1]-1} "
            f"λ={lam_chosen:.4e} (λ_min={eigvals[0]:.4e})")

        # Append to tabu list (LRU)
        tabu_dirs.append(v_chosen.copy())
        if len(tabu_dirs) > tabu_size:
            tabu_dirs.pop(0)

        if lam_chosen >= -eigval_tolerance:
            log(f"  λ_chosen >= -tol; effectively at min along this mode")
            iter_log.append({"iter": it, "lam_chosen": lam_chosen,
                             "chosen_idx": chosen_idx, "stopped": "true_min"})
            break

        n_dim = n_macros * 2
        mask_flat = np.zeros(n_dim, dtype=bool)
        for i, m in enumerate(movable_np):
            if bool(m):
                mask_flat[2 * i] = True
                mask_flat[2 * i + 1] = True
        v_full = np.zeros(n_dim, dtype=np.float64)
        v_full[mask_flat] = v_chosen
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
                    log(f"  iter{it+1} idx={chosen_idx} sign={sign:+.0f} eps={eps:.3f}: "
                        f"NEW BEST {pol_proxy:.5f} (Δ vs init={pol_proxy - init_proxy:+.5f})")

        iter_wall = time.time() - iter_t_start
        avg_iter_wall = iter_wall if avg_iter_wall == 0.0 else 0.5 * avg_iter_wall + 0.5 * iter_wall
        iter_log.append({
            "iter": it, "lam_chosen": lam_chosen, "chosen_idx": chosen_idx,
            "proxy_after": best_proxy,
            "improvement_this_iter": prev_iter_proxy - best_proxy, "iter_wall": iter_wall,
            "eps_grid": eps_grid, "tabu_size_in_use": len(tabu_dirs),
        })

        if best_proxy >= prev_iter_proxy - min_improvement:
            log(f"  no improvement this iter; saturated")
            break

    total_wall = time.time() - t_start
    log(f"[tabu-levy] done: init={init_proxy:.5f} best={best_proxy:.5f} "
        f"Δ={best_proxy - init_proxy:+.5f} wall={total_wall:.0f}s iters={len(iter_log)}")

    return best_state, {
        "init_proxy": init_proxy, "best_proxy": best_proxy,
        "improvement": init_proxy - best_proxy,
        "improvement_frac": (init_proxy - best_proxy) / init_proxy,
        "iters_run": len(iter_log), "iter_log": iter_log,
        "wall_seconds": total_wall, "tabu_size": tabu_size,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--start-pt", help="path to .pt with 'placement' key")
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--K-eps", type=int, default=3)
    ap.add_argument("--eps-scale", type=float, default=1.0)
    ap.add_argument("--tabu-size", type=int, default=2)
    ap.add_argument("--overlap-lambda", type=float, default=10.0)
    ap.add_argument("--polish-budget", type=float, default=60.0)
    ap.add_argument("--total-budget", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(args.bench)))
    if args.start_pt:
        data = torch.load(args.start_pt, weights_only=False)
        if "placement" in data:
            start = data["placement"]
        elif "polished_placement" in data:
            start = data["polished_placement"]
        else:
            raise SystemExit(f"no placement key in {args.start_pt}")
        start = start.to(torch.float32)
    else:
        from macro_place.sdf_init import SDFPlacer
        from macro_place.cd_core import run_cd_adaptive as _run_cd
        print(f"No --start-pt; building SDF→CD plateau for {args.bench}...")
        sdf = SDFPlacer(seed=42, n_iters=500).place(bench).to(torch.float32)
        sdf, _ = project_overlaps(sdf, bench)
        ev = IncrementalProxyEvaluator(bench, plc, sdf.clone().to(torch.float64))
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

    result, stats = tabu_levy_saddle_escape(
        start, bench, plc,
        max_iters=args.max_iters, K_eps=args.K_eps, eps_scale=args.eps_scale,
        tabu_size=args.tabu_size, overlap_lambda=args.overlap_lambda,
        polish_budget=args.polish_budget, total_budget_s=args.total_budget,
        rng_seed=args.seed,
    )

    final_proxy = float(compute_proxy_cost(result, bench, plc)["proxy_cost"])
    final_ovl = compute_overlap_metrics(result, bench)["overlap_count"]
    print(f"\n=== {args.bench} TABU+LEVY RESULT ===")
    print(f"  init: {stats['init_proxy']:.5f}")
    print(f"  best: {stats['best_proxy']:.5f}  (Δ={stats['improvement']:+.5f} "
          f"= {stats['improvement_frac']*100:+.3f}%)")
    print(f"  iters: {stats['iters_run']}, ovl: {final_ovl}, "
          f"wall: {stats['wall_seconds']:.0f}s")

    out_dir = _HERE.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.bench}_tabulevy_seed{args.seed}.json"
    out_path.write_text(json.dumps({
        "bench": args.bench, "variant": "tabu_levy", "seed": args.seed,
        "K_eps": args.K_eps, "eps_scale": args.eps_scale, "tabu_size": args.tabu_size,
        **stats, "final_overlap": int(final_ovl),
    }, indent=2, default=str))
    print(f"  results: {out_path}")


if __name__ == "__main__":
    main()
