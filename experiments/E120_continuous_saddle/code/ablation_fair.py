"""E120 — fair ablation: identical CD polish budget, with vs without saddle escape.

Compares:
  A: V3 Adam 300 steps → legalize → CD polish 300s
  B: V3 Adam 300 steps + 1 saddle escape stage → legalize → CD polish 300s

If saddle escape adds value, B should beat A by a measurable margin. If
B ties A, the smooth-basin lift is being washed out by CD polish.

Run with controlled CD budget (not budget_seconds), so both A and B
get the same polish quality. The only variable is the saddle escape.

Run:
  uv run python experiments/E120_continuous_saddle/code/ablation_fair.py BENCH
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, sdf_init, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from continuous_saddle_placer import (
    _adam_descend,
    find_softest_eigenvector_smooth,
    DiffProxyV3,
)
from macro_legalizer import greedy_macro_legalize


def make_proxy(benchmark, plc):
    return DiffProxyV3(benchmark, plc, device="cpu", gamma_frac=5e-3)


def adam_descent(benchmark, plc, num_steps, lr_frac, fixed_mask):
    init_pos = sdf_init(benchmark)
    init_pos, _ = project_overlaps(init_pos, benchmark)
    proxy = make_proxy(benchmark, plc)
    cw = float(benchmark.canvas_width)
    lr = lr_frac * cw
    final, stats = _adam_descend(
        proxy, init_pos, fixed_mask,
        num_steps=num_steps, lr=lr,
        gamma_start_frac=5e-3, gamma_end_frac=5e-5,
        overlap_lambda_start=0.0, overlap_lambda_end=10.0,
        overlap_ramp_pct=0.7, boundary_lambda=50.0,
        log=lambda s: None, log_every=10_000,
        starting_step=0, total_steps_for_anneal=num_steps,
    )
    return final, proxy, stats


def adam_resume(proxy, init_pos, fixed_mask, num_steps, lr_frac, cw,
                starting_step, total_anneal):
    lr = lr_frac * cw
    final, stats = _adam_descend(
        proxy, init_pos, fixed_mask,
        num_steps=num_steps, lr=lr,
        gamma_start_frac=5e-3, gamma_end_frac=5e-5,
        overlap_lambda_start=0.0, overlap_lambda_end=10.0,
        overlap_ramp_pct=0.7, boundary_lambda=50.0,
        log=lambda s: None, log_every=10_000,
        starting_step=starting_step, total_steps_for_anneal=total_anneal,
    )
    return final, stats


def legalize_and_polish(positions, benchmark, plc, cd_budget_s):
    legal, _ = greedy_macro_legalize(positions, benchmark,
                                     search_radius_steps=80,
                                     step_size_frac=0.02, verbose=False)
    ovl = compute_overlap_metrics(legal, benchmark)["overlap_count"]
    if ovl > 0:
        legal, _ = project_overlaps(legal, benchmark)
    ev = IncrementalProxyEvaluator(benchmark, plc, legal)
    movable = [i for i in range(benchmark.num_macros)
               if not bool(benchmark.macro_fixed[i])]
    run_cd_adaptive(ev, benchmark, plc, movable,
                    min_time_s=cd_budget_s * 0.5,
                    hard_cap_s=cd_budget_s,
                    patience=5, plateau_threshold=0.001, log_fn=None)
    polished = ev.placement.detach().clone().to(torch.float32)
    final_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
    final_ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
    return polished, final_proxy, final_ovl


def run_variant_A(benchmark, plc, cd_budget_s, num_steps):
    """A: V3 Adam num_steps → legalize → CD polish."""
    t0 = time.time()
    fixed_mask = benchmark.macro_fixed.bool()
    positions, proxy, stats = adam_descent(
        benchmark, plc, num_steps=num_steps, lr_frac=0.005, fixed_mask=fixed_mask,
    )
    descent_wall = time.time() - t0
    print(f"  [A] V3 descent: smooth={stats['final_smooth']:.5f} "
          f"wall={descent_wall:.0f}s", flush=True)

    polish_t0 = time.time()
    polished, final_proxy, final_ovl = legalize_and_polish(
        positions, benchmark, plc, cd_budget_s,
    )
    print(f"  [A] CD polish ({cd_budget_s:.0f}s budget, wall={time.time()-polish_t0:.0f}s): "
          f"proxy={final_proxy:.5f} ovl={final_ovl}", flush=True)
    return {
        "variant": "A",
        "smooth_basin": stats["final_smooth"],
        "final_proxy": final_proxy,
        "final_ovl": final_ovl,
        "total_wall_s": time.time() - t0,
    }


def run_variant_B(benchmark, plc, cd_budget_s, num_steps, num_resume_steps,
                  eps_values=(0.3, 1.0, 3.0)):
    """B: V3 Adam num_steps + 1 saddle escape → legalize → CD polish.

    Each saddle attempt does num_resume_steps Adam from the perturbed state.
    Pick best smooth basin.
    """
    import numpy as np
    t0 = time.time()
    fixed_mask = benchmark.macro_fixed.bool()
    positions, proxy, stats = adam_descent(
        benchmark, plc, num_steps=num_steps, lr_frac=0.005, fixed_mask=fixed_mask,
    )
    descent_wall = time.time() - t0
    basin_smooth = stats["final_smooth"]
    print(f"  [B] V3 descent: smooth={basin_smooth:.5f} "
          f"wall={descent_wall:.0f}s", flush=True)

    # Saddle escape
    saddle_t0 = time.time()
    movable_np = (~benchmark.macro_fixed.cpu().numpy())
    eigvals, eigvecs = find_softest_eigenvector_smooth(
        proxy, positions.to(proxy.device), movable_np,
        k=1, tol=1e-3, maxiter=200, log=lambda s: None,
    )
    eigsh_wall = time.time() - saddle_t0
    if len(eigvals) == 0:
        print(f"  [B] no eigvec returned; falling back to no-saddle", flush=True)
        best_pos = positions
        best_smooth = basin_smooth
    else:
        lam_min = float(eigvals[0])
        print(f"  [B] λ_min = {lam_min:.4e} (eigsh wall={eigsh_wall:.0f}s)", flush=True)

        n_macros = positions.shape[0]
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

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        gr, gc = benchmark.grid_rows, benchmark.grid_cols
        cell_w = cw / gc
        cell_h = ch / gr
        cell_size = min(cell_w, cell_h)
        per_macro_disp = np.linalg.norm(v_unit, axis=1)
        max_disp = per_macro_disp.max() + 1e-12

        basin_pos_np = positions.detach().cpu().numpy()
        fixed_mask_np = benchmark.macro_fixed.cpu().numpy()
        half_w = (benchmark.macro_sizes[:, 0] / 2).cpu().numpy()
        half_h = (benchmark.macro_sizes[:, 1] / 2).cpu().numpy()

        best_pos = positions
        best_smooth = basin_smooth
        total_anneal = num_steps + num_resume_steps

        for sign in (+1.0, -1.0):
            for eps in eps_values:
                scale = (eps * cell_size) / max_disp
                disp = sign * scale * v_unit
                cand_np = basin_pos_np + disp
                cand_np[:, 0] = np.clip(cand_np[:, 0], half_w, cw - half_w)
                cand_np[:, 1] = np.clip(cand_np[:, 1], half_h, ch - half_h)
                cand_np[fixed_mask_np] = basin_pos_np[fixed_mask_np]
                cand_t = torch.tensor(cand_np, dtype=torch.float32)

                resumed, rs = adam_resume(
                    proxy, cand_t, fixed_mask, num_resume_steps,
                    lr_frac=0.005, cw=cw,
                    starting_step=num_steps, total_anneal=total_anneal,
                )
                with torch.no_grad():
                    sc, _ = proxy.cost(resumed.to(proxy.device), include_congestion=True)
                    resumed_smooth = float(sc.item())

                tag = " NEW BEST" if resumed_smooth < best_smooth - 1e-7 else ""
                print(f"  [B] sign={sign:+.0f} eps={eps:.1f}: "
                      f"smooth={resumed_smooth:.5f} Δ={resumed_smooth - basin_smooth:+.5f}{tag}",
                      flush=True)
                if resumed_smooth < best_smooth - 1e-7:
                    best_smooth = resumed_smooth
                    best_pos = resumed

    saddle_wall = time.time() - saddle_t0
    print(f"  [B] saddle escape done: basin {basin_smooth:.5f} → {best_smooth:.5f} "
          f"(Δ={best_smooth - basin_smooth:+.5f}) wall={saddle_wall:.0f}s", flush=True)

    polish_t0 = time.time()
    polished, final_proxy, final_ovl = legalize_and_polish(
        best_pos, benchmark, plc, cd_budget_s,
    )
    print(f"  [B] CD polish ({cd_budget_s:.0f}s budget, wall={time.time()-polish_t0:.0f}s): "
          f"proxy={final_proxy:.5f} ovl={final_ovl}", flush=True)
    return {
        "variant": "B",
        "smooth_basin": basin_smooth,
        "smooth_after_saddle": best_smooth,
        "final_proxy": final_proxy,
        "final_ovl": final_ovl,
        "total_wall_s": time.time() - t0,
    }


def main():
    bench_name = sys.argv[1] if len(sys.argv) > 1 else "ibm10"
    cd_budget_s = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0
    num_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    num_resume_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 100

    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"Loaded {bench_name}: {benchmark.num_macros} macros "
          f"({benchmark.num_hard_macros} hard), nets={benchmark.num_nets}",
          flush=True)

    results = {}

    # Variant A
    print(f"\n--- Variant A: pure V3 Adam descent ---", flush=True)
    torch.manual_seed(42)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    results["A"] = run_variant_A(benchmark, plc, cd_budget_s, num_steps)

    # Variant B
    print(f"\n--- Variant B: V3 Adam + saddle escape ---", flush=True)
    torch.manual_seed(42)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    results["B"] = run_variant_B(
        benchmark, plc, cd_budget_s, num_steps, num_resume_steps,
    )

    print(f"\n=== ABLATION on {bench_name} (CD budget {cd_budget_s:.0f}s) ===",
          flush=True)
    print(f"  A (no saddle): smooth={results['A']['smooth_basin']:.5f}  "
          f"final={results['A']['final_proxy']:.5f}  wall={results['A']['total_wall_s']:.0f}s",
          flush=True)
    print(f"  B (saddle): smooth={results['B']['smooth_basin']:.5f} → "
          f"{results['B']['smooth_after_saddle']:.5f}  "
          f"final={results['B']['final_proxy']:.5f}  wall={results['B']['total_wall_s']:.0f}s",
          flush=True)
    delta = results["B"]["final_proxy"] - results["A"]["final_proxy"]
    pct = 100.0 * delta / results["A"]["final_proxy"]
    print(f"  Δ(B − A) = {delta:+.5f} ({pct:+.2f}%)", flush=True)

    out_path = _HERE.parent / "results" / f"ablation_fair_{bench_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
