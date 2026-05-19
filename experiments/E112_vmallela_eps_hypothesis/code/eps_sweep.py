"""E112 — vmallela ε-magnitude hypothesis spike.

Research claim: vmallela's 28% Hessian saddle lift (vs our 3.5%) comes
from ε scaled to canvas_diagonal/√M, not soft-mode × small ε. Our current
eps_values=(0.3, 1.0, 3.0) is ~100-1500× too small.

This script:
  1. Loads bench (ibm01 default), runs CD-only init to a plateau.
  2. Computes Hessian soft mode eigvec.
  3. Tries ε ∈ {1e-3 to 1e+3} × per-macro-magnitude.
  4. For each ε, polishes the perturbed state via CD-adaptive.
  5. Reports final proxy per ε.

If a much larger ε gives substantially deeper polished proxy than the
default (0.3, 1.0, 3.0), the hypothesis is confirmed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_E74 = _ROOT / "experiments" / "E74_hessian_saddle" / "code"
if str(_E74) not in sys.path:
    sys.path.insert(0, str(_E74))

from hessian_saddle import SmoothProxy, find_softest_eigenvectors
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd, run_cd_adaptive, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="ibm01")
    ap.add_argument("--cd-budget-s", type=float, default=120.0)
    ap.add_argument("--polish-budget-s", type=float, default=90.0)
    args = ap.parse_args()

    torch.set_num_threads(1)
    print(f"E112 — eps magnitude spike on {args.bench}")
    print(f"=" * 70)
    t0 = time.perf_counter()
    bench_dir = find_benchmark_dir(args.bench)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    placement = sdf_init(benchmark)
    ev = IncrementalProxyEvaluator(benchmark, plc, placement)
    init_proxy = float(ev.current_cost()["proxy"])
    cw, ch = benchmark.canvas_width, benchmark.canvas_height
    canvas_diag = float((cw**2 + ch**2) ** 0.5)
    hard_movable = [i for i in range(benchmark.num_hard_macros)
                    if not bool(benchmark.macro_fixed[i])]
    movable_all = hard_movable + list(
        range(benchmark.num_hard_macros, placement.shape[0]))
    n_movable = len(movable_all)
    print(f"  init proxy={init_proxy:.5f} canvas={cw:.0f}×{ch:.0f} "
          f"diag={canvas_diag:.0f} movable={n_movable}")

    print(f"[{time.perf_counter()-t0:5.1f}s] CD polish ({args.cd_budget_s}s)...")
    run_cd(ev, benchmark, plc, movable_all, args.cd_budget_s, log_fn=None)
    plateau_proxy = float(ev.current_cost()["proxy"])
    plateau = ev.placement.detach().clone()
    print(f"[{time.perf_counter()-t0:5.1f}s] plateau proxy={plateau_proxy:.5f}")

    fixed = benchmark.macro_fixed.cpu().numpy()
    movable_np = (~fixed)
    n_macros = plateau.shape[0]
    n_dim = n_macros * 2
    smooth = SmoothProxy(benchmark, plc)

    print(f"[{time.perf_counter()-t0:5.1f}s] Computing soft mode eigvec...")
    eigvals, eigvecs = find_softest_eigenvectors(
        smooth, plateau, movable_np, k=1, log=lambda s: None,
    )
    lam_min = float(eigvals[0])
    print(f"  λ_min={lam_min:.4e}")
    mask_flat = np.zeros(n_dim, dtype=bool)
    for i, m in enumerate(movable_np):
        if bool(m):
            mask_flat[2 * i] = True
            mask_flat[2 * i + 1] = True
    v_full = np.zeros(n_dim, dtype=np.float64)
    v_full[mask_flat] = eigvecs[:, 0]
    v_2d = v_full.reshape(n_macros, 2)
    v_unit = v_2d / (np.linalg.norm(v_2d) + 1e-12)
    eigvec_max_per_macro = float(np.max(np.linalg.norm(v_unit, axis=1)))
    eigvec_mean_per_macro = float(np.mean(np.linalg.norm(v_unit, axis=1)))
    print(f"  eigvec per-macro: max={eigvec_max_per_macro:.4f} "
          f"mean={eigvec_mean_per_macro:.4f} "
          f"(uniform expectation 1/√M = {1/(n_movable**0.5):.4f})")

    # Lit recommendation: per-macro perturb ~ canvas_diag/√M = soft scale.
    # Our default: ε=3 → per-macro = 3 × max_eigvec_per_macro
    # Want: per-macro = canvas_diag/√M
    # → ε = (canvas_diag/√M) / max_eigvec_per_macro
    eps_lit_recommended = canvas_diag / (n_movable ** 0.5) / eigvec_max_per_macro
    print(f"  lit recommendation ε ≈ canvas_diag/√M / max(|v_i|) "
          f"= {eps_lit_recommended:.1f}")

    state_np = plateau.detach().cpu().numpy()
    hw_np = (benchmark.macro_sizes[:, 0] / 2).cpu().numpy()
    hh_np = (benchmark.macro_sizes[:, 1] / 2).cpu().numpy()

    eps_sweep = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0,
                 eps_lit_recommended,
                 max(1.0, eps_lit_recommended / 3.0),
                 eps_lit_recommended * 3.0]
    eps_sweep = sorted(set(round(e, 1) for e in eps_sweep))

    results = []
    movable_t = torch.tensor(movable_np)
    for sign in [+1.0]:  # +1 sign for speed; -1 symmetric in 2D
        for eps in eps_sweep:
            t_eps = time.perf_counter()
            pp = state_np + sign * eps * v_unit
            pp[:, 0] = np.clip(pp[:, 0], hw_np, cw - hw_np)
            pp[:, 1] = np.clip(pp[:, 1], hh_np, ch - hh_np)
            cand = torch.tensor(pp, dtype=torch.float32)
            cand[~movable_t] = plateau[~movable_t].to(torch.float32)
            cand, _ = project_overlaps(cand, benchmark)
            pre_proxy = float(compute_proxy_cost(cand, benchmark, plc)["proxy_cost"])
            pre_ovl = int(compute_overlap_metrics(cand, benchmark)["overlap_count"])
            if pre_ovl > 0:
                results.append({"eps": eps, "sign": sign, "pre_proxy": pre_proxy,
                                "post_proxy": None, "post_ovl": pre_ovl,
                                "delta_pct": None, "wall": 0.0,
                                "status": f"pre_ovl={pre_ovl}"})
                print(f"  ε={eps:>8.1f} pre={pre_proxy:.5f} ovl={pre_ovl} SKIP")
                continue

            ev2 = IncrementalProxyEvaluator(benchmark, plc, cand.clone())
            run_cd_adaptive(
                ev2, benchmark, plc, movable_all,
                min_time_s=30.0, hard_cap_s=args.polish_budget_s,
                patience=3, plateau_threshold=0.001, log_fn=None,
            )
            polished = ev2.placement.detach().clone().to(torch.float32)
            post_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
            post_ovl = int(compute_overlap_metrics(polished, benchmark)["overlap_count"])
            wall = time.perf_counter() - t_eps
            delta_pct = (post_proxy - plateau_proxy) / plateau_proxy * 100.0
            results.append({"eps": eps, "sign": sign, "pre_proxy": pre_proxy,
                            "post_proxy": post_proxy, "post_ovl": post_ovl,
                            "delta_pct": delta_pct, "wall": wall,
                            "status": "ok"})
            print(f"  ε={eps:>8.1f} pre={pre_proxy:.5f} post={post_proxy:.5f} "
                  f"Δ={delta_pct:+.3f}% ovl={post_ovl} wall={wall:.0f}s",
                  flush=True)

    print()
    print(f"=" * 70)
    print(f"E112 summary — bench={args.bench} plateau={plateau_proxy:.5f}")
    print(f"{'eps':>8} {'post':>10} {'delta':>10} {'ovl':>4} {'wall':>6}")
    valid = [r for r in results if r["status"] == "ok"]
    for r in sorted(valid, key=lambda x: x["post_proxy"]):
        winner = " ← BEST" if r == min(valid, key=lambda x: x["post_proxy"]) else ""
        print(f"{r['eps']:>8.1f} {r['post_proxy']:>10.5f} "
              f"{r['delta_pct']:>+9.3f}% {r['post_ovl']:>4d} "
              f"{r['wall']:>5.0f}s{winner}")
    print(f"\nTotal wall: {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
