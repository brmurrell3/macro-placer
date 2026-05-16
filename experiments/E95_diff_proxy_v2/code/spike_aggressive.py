"""E95 — more aggressive C1 spike (longer descent, finer perturbation).

If the standard sweep shows no lift on hard benches, try:
  1. Longer L-BFGS (8 stages × 50 iters = 400 iters total)
  2. Finer perturbation seeds (sigma_frac=0.001, 0.0005, 0.0002) — closer to
     the cascade basin so we don't fall off the cliff
  3. Multi-restart: if a stage shows no canonical improvement, restart from
     the best-so-far placement
  4. SGD with smaller peak lr (1e-3 instead of 5e-3) to crawl through the
     smooth landscape

Driven by individual command-line args so we can target specific benches.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
_E88_DIR = _ROOT / "experiments" / "E88_diff_proxy" / "code"
_E76_DIR = _ROOT / "experiments" / "E76_dreamplace_integration" / "code"
for p in (_ROOT, _E88_DIR, _E76_DIR, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from diff_proxy_v2 import DiffProxyV2
from spike_v2 import descend_lbfgs, evaluate_with_legalize
from macro_legalizer import greedy_macro_legalize


def aggressive_descent_with_restart(
    proxy: DiffProxyV2,
    benchmark,
    plc,
    init: torch.Tensor,
    *,
    num_stages: int = 8,
    steps_per_stage: int = 50,
    label: str = "aggressive",
):
    """L-BFGS with restart on stagnation: if a stage's post-canonical
    isn't better than the running best, restart from the running best.
    """
    pos_in = init.clone()
    best_canon = float("inf")
    best_pos = init.clone()
    baseline = compute_proxy_cost(init, benchmark, plc)
    baseline_canon = float(baseline["proxy_cost"])
    if int(baseline["overlap_count"]) == 0:
        best_canon = baseline_canon
        best_pos = init.clone()
    log = []
    t0 = time.time()

    for stage_global in range(num_stages):
        # Geometric γ schedule: start 0.005, end 5e-5
        frac = stage_global / max(1, num_stages - 1)
        log_g_start, log_g_end = math.log(0.005), math.log(5e-5)
        gamma_frac = math.exp(log_g_start + frac * (log_g_end - log_g_start))
        proxy.set_gamma_frac(gamma_frac)
        lam = 500 + frac * 9500

        pos_local = pos_in.clone().detach().requires_grad_(True)
        opt = torch.optim.LBFGS(
            [pos_local], lr=1.0, max_iter=steps_per_stage,
            history_size=20, line_search_fn="strong_wolfe",
            tolerance_grad=1e-9, tolerance_change=1e-12,
        )

        def closure():
            opt.zero_grad()
            cost, parts = proxy.cost(pos_local)
            penalty = proxy.overlap_penalty(pos_local)
            boundary = proxy.out_of_canvas_penalty(pos_local)
            canvas_area = proxy.cw * proxy.ch
            total = cost + lam * penalty / canvas_area + 500.0 * boundary / canvas_area
            total.backward()
            return total

        try:
            opt.step(closure)
        except Exception as exc:
            log.append({"stage": stage_global, "exception": repr(exc)})
            continue

        # Stage-end eval
        with torch.no_grad():
            half = benchmark.macro_sizes / 2.0
            cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
            pos_clamped = pos_local.clone()
            pos_clamped[:, 0].clamp_(half[:, 0], cw - half[:, 0])
            pos_clamped[:, 1].clamp_(half[:, 1], ch - half[:, 1])
        canon = compute_proxy_cost(pos_clamped.detach(), benchmark, plc)
        canon_v = float(canon["proxy_cost"])
        ovl = int(canon["overlap_count"])
        log.append({
            "stage": stage_global,
            "gamma_frac": gamma_frac,
            "lam": lam,
            "canon": canon_v,
            "ovl": ovl,
        })
        print(f"[{label}] stage={stage_global} γfrac={gamma_frac:.5f} λ={lam:.0f} "
              f"canon={canon_v:.4f} ovl={ovl} best_canon={best_canon:.4f}")

        if ovl == 0 and canon_v < best_canon:
            best_canon = canon_v
            best_pos = pos_clamped.detach().clone()
            pos_in = pos_clamped.detach().clone()  # continue from improved
        else:
            # Restart from best-so-far (only if we have a valid best)
            if best_canon < float("inf"):
                pos_in = best_pos.clone()

    wall = time.time() - t0
    return best_pos, best_canon, log, wall


def run(bench_name: str):
    bench_dir = Path(f"external/MacroPlacement/Testcases/ICCAD04/{bench_name}")
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    cached = torch.load(
        f"experiments/E84_cascading_saddle/results/cascade_{bench_name}.pt",
        weights_only=False, map_location="cpu",
    )
    init = cached["placement"].clone().float()
    proxy = DiffProxyV2(benchmark, plc, device="cpu", gamma_frac=0.005)

    baseline = compute_proxy_cost(init, benchmark, plc)
    baseline_canon = float(baseline["proxy_cost"])
    print(f"\n=== {bench_name} aggressive ===")
    print(f"baseline cascade-cached canon={baseline_canon:.5f} "
          f"ovl={int(baseline['overlap_count'])} macros={init.shape[0]}")

    best_pos, best_canon, log, wall = aggressive_descent_with_restart(
        proxy, benchmark, plc, init,
        num_stages=8, steps_per_stage=50,
        label=bench_name,
    )

    ev = evaluate_with_legalize(best_pos, benchmark, plc, label=f"{bench_name}_agg")
    post_canon = ev["post_legalize"]["proxy_cost"]
    lift = (baseline_canon - post_canon) / baseline_canon * 100.0

    summary = {
        "bench": bench_name,
        "num_macros": init.shape[0],
        "baseline_canon": baseline_canon,
        "best_canon_during_descent": best_canon,
        "post_leg_canon": post_canon,
        "post_leg_ovl": ev["post_legalize"]["overlap_count"],
        "lift_pct": lift,
        "wall_s": wall,
        "log": log,
    }
    print(f"[{bench_name}] aggressive: post-leg {post_canon:.4f} "
          f"(baseline {baseline_canon:.4f}, lift {lift:+.2f}%)")

    if ev["post_legalize"]["overlap_count"] == 0:
        torch.save(
            {"placement": ev["placement"], "bench_name": bench_name, "label": "aggressive"},
            f"experiments/E95_diff_proxy_v2/results/{bench_name}_aggressive.pt",
        )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benches", nargs="*", default=["ibm01", "ibm10", "ibm12", "ibm14", "ibm17"])
    args = ap.parse_args()

    all_results = {}
    out_path = Path("experiments/E95_diff_proxy_v2/results/aggressive_summary.json")
    for b in args.benches:
        try:
            all_results[b] = run(b)
        except Exception as exc:
            all_results[b] = {"error": repr(exc), "tb": traceback.format_exc()}
        out_path.write_text(json.dumps(all_results, indent=2, default=str))

    print(f"\n{'bench':<8} {'baseline':>10} {'post_leg':>10} {'lift':>8}")
    for b, r in all_results.items():
        if "error" in r:
            print(f"{b:<8} ERROR: {r['error']}")
            continue
        print(f"{b:<8} {r['baseline_canon']:>10.4f} {r['post_leg_canon']:>10.4f} "
              f"{r['lift_pct']:>7.2f}%")


if __name__ == "__main__":
    main()
