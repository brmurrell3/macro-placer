"""E95 — sweep C1 spike v2 across multiple benches.

For each benchmark in the list:
  - Load cached cascade-uncapped placement
  - Run cascade_lbfgs descent
  - Run a perturb-and-descend variant (basin escape with Gaussian seeds)
  - Report (baseline canon, post-descent canon, lift %)

Picks best variant per bench. Saves placements and rolls up a comparison
table.

Skips overhead by not running SDF init variants — ibm01 spike already
showed those are far from cascade quality. The interesting test is
"can C1 lift cascade input on harder benches?"
"""
from __future__ import annotations

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
from diff_proxy_v2 import DiffProxyV2, loss_with_penalty
from macro_legalizer import greedy_macro_legalize
from spike_v2 import descend_lbfgs, descend_sgd, evaluate_with_legalize, _clamp_to_canvas


def descend_with_perturb_seeds(
    proxy: DiffProxyV2,
    benchmark,
    plc,
    init: torch.Tensor,
    *,
    num_seeds: int = 4,
    perturb_sigma_frac: float = 0.01,  # 1% of canvas
    label: str = "perturb",
):
    """Run multiple LBFGS descents from small Gaussian perturbations of init.

    Tests "basin escape via C1": cascade input perturbed, descended via
    smooth proxy, then legalized. If smooth gradient navigates to a
    different basin, post-leg canonical might land lower than cascade.
    """
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sigma_x = cw * perturb_sigma_frac
    sigma_y = ch * perturb_sigma_frac

    half = benchmark.macro_sizes / 2.0
    results = []
    best_canon = float("inf")
    best_pos = init.clone()

    for seed in range(num_seeds):
        torch.manual_seed(1000 + seed)
        noise = torch.randn_like(init) * torch.tensor([sigma_x, sigma_y])
        perturbed = init + noise
        perturbed[:, 0].clamp_(half[:, 0], cw - half[:, 0])
        perturbed[:, 1].clamp_(half[:, 1], ch - half[:, 1])

        try:
            final_pos, log, wall, var_best_pos, var_best_canon = descend_lbfgs(
                proxy, benchmark, plc, perturbed,
                num_stages=3, steps_per_stage=20,
                label=f"{label}_seed{seed}",
            )
            ev = evaluate_with_legalize(var_best_pos, benchmark, plc,
                                         label=f"{label}_seed{seed}_best")
            canon_v = ev["post_legalize"]["proxy_cost"]
            if canon_v < best_canon and ev["post_legalize"]["overlap_count"] == 0:
                best_canon = canon_v
                best_pos = ev["placement"].clone()
            results.append({
                "seed": seed,
                "canon_post_leg": canon_v,
                "ovl": ev["post_legalize"]["overlap_count"],
                "wall": wall,
            })
        except Exception as exc:
            print(f"[{label}_seed{seed}] FAILED: {exc}")
            results.append({"seed": seed, "error": repr(exc)})

    return best_pos, best_canon, results


def run_bench(bench_name: str, max_macros_for_perturb: int = 2000):
    print(f"\n========== {bench_name} ==========")
    bench_dir = Path(f"external/MacroPlacement/Testcases/ICCAD04/{bench_name}")
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    cached_path = Path(f"experiments/E84_cascading_saddle/results/cascade_{bench_name}.pt")
    if not cached_path.exists():
        return {"bench": bench_name, "error": f"missing cache: {cached_path}"}

    cached = torch.load(str(cached_path), weights_only=False, map_location="cpu")
    init = cached["placement"].clone().float()

    proxy = DiffProxyV2(benchmark, plc, device="cpu", gamma_frac=0.005)

    baseline = compute_proxy_cost(init, benchmark, plc)
    baseline_canon = float(baseline["proxy_cost"])
    print(f"[{bench_name}] cached cascade canon={baseline_canon:.5f} "
          f"ovl={int(baseline['overlap_count'])} num_macros={init.shape[0]}")

    variants = {}

    # --- Variant 1: L-BFGS descent from cascade
    t0 = time.time()
    try:
        pos_final, log, wall, var_best_pos, var_best_canon = descend_lbfgs(
            proxy, benchmark, plc, init,
            num_stages=4, steps_per_stage=25,
            label="lbfgs",
        )
        ev = evaluate_with_legalize(var_best_pos, benchmark, plc, label=f"{bench_name}_lbfgs")
        variants["lbfgs"] = {
            "best_canon_during_descent": var_best_canon,
            "post_leg_canon": ev["post_legalize"]["proxy_cost"],
            "post_leg_ovl": ev["post_legalize"]["overlap_count"],
            "wall": time.time() - t0,
            "descent_log": log,
        }
        if ev["post_legalize"]["overlap_count"] == 0:
            torch.save(
                {"placement": ev["placement"], "bench_name": bench_name, "label": "lbfgs"},
                f"experiments/E95_diff_proxy_v2/results/{bench_name}_lbfgs.pt",
            )
    except Exception as exc:
        variants["lbfgs"] = {"error": repr(exc), "tb": traceback.format_exc()}

    # --- Variant 2: SGD descent from cascade
    t0 = time.time()
    try:
        proxy.set_gamma_frac(0.005)  # reset
        pos_final, log, wall, var_best_pos, var_best_canon = descend_sgd(
            proxy, benchmark, plc, init,
            num_stages=4, steps_per_stage=80,
            label="sgd",
        )
        ev = evaluate_with_legalize(var_best_pos, benchmark, plc, label=f"{bench_name}_sgd")
        variants["sgd"] = {
            "best_canon_during_descent": var_best_canon,
            "post_leg_canon": ev["post_legalize"]["proxy_cost"],
            "post_leg_ovl": ev["post_legalize"]["overlap_count"],
            "wall": time.time() - t0,
            "descent_log": log,
        }
        if ev["post_legalize"]["overlap_count"] == 0:
            torch.save(
                {"placement": ev["placement"], "bench_name": bench_name, "label": "sgd"},
                f"experiments/E95_diff_proxy_v2/results/{bench_name}_sgd.pt",
            )
    except Exception as exc:
        variants["sgd"] = {"error": repr(exc), "tb": traceback.format_exc()}

    # --- Variant 3: perturb-and-descend (basin escape)
    if init.shape[0] <= max_macros_for_perturb:
        t0 = time.time()
        try:
            proxy.set_gamma_frac(0.005)  # reset
            best_pos, best_canon, seed_results = descend_with_perturb_seeds(
                proxy, benchmark, plc, init,
                num_seeds=4, perturb_sigma_frac=0.01,
                label="perturb",
            )
            variants["perturb"] = {
                "best_canon_across_seeds": best_canon,
                "seed_results": seed_results,
                "wall": time.time() - t0,
            }
            if best_canon < float("inf"):
                torch.save(
                    {"placement": best_pos, "bench_name": bench_name, "label": "perturb"},
                    f"experiments/E95_diff_proxy_v2/results/{bench_name}_perturb.pt",
                )
        except Exception as exc:
            variants["perturb"] = {"error": repr(exc), "tb": traceback.format_exc()}
    else:
        variants["perturb"] = {"skipped": f"num_macros={init.shape[0]} > {max_macros_for_perturb}"}

    # Roll up
    candidate_canons = []
    for vname, v in variants.items():
        c = v.get("post_leg_canon") or v.get("best_canon_across_seeds")
        if isinstance(c, (int, float)) and not math.isinf(c):
            candidate_canons.append((vname, c))
    if candidate_canons:
        best_variant, best_canon = min(candidate_canons, key=lambda kv: kv[1])
        lift = (baseline_canon - best_canon) / baseline_canon * 100.0
    else:
        best_variant, best_canon, lift = None, float("inf"), float("nan")

    summary = {
        "bench": bench_name,
        "num_macros": init.shape[0],
        "baseline_canon": baseline_canon,
        "best_variant": best_variant,
        "best_canon": best_canon,
        "lift_pct": lift,
        "variants": variants,
    }
    print(f"[{bench_name}] BEST: {best_variant} → {best_canon:.4f} "
          f"(baseline {baseline_canon:.4f}, lift {lift:+.2f}%)")
    return summary


def main():
    benches = ["ibm01", "ibm10", "ibm12", "ibm14", "ibm17"]
    all_results = {}
    t_global = time.time()
    for b in benches:
        all_results[b] = run_bench(b)
        # Save incremental results in case we crash
        out = Path("experiments/E95_diff_proxy_v2/results/sweep_summary.json")
        out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n=== TOTAL WALL: {time.time() - t_global:.0f}s ===")

    print(f"\n{'bench':<8} {'macros':>7} {'baseline':>10} {'best':>10} {'best_var':>10} {'lift_%':>8}")
    for b, r in all_results.items():
        if "error" in r:
            print(f"{b:<8} ERROR: {r['error']}")
            continue
        print(f"{b:<8} {r['num_macros']:>7} {r['baseline_canon']:>10.4f} "
              f"{r['best_canon']:>10.4f} {str(r['best_variant']):>10} "
              f"{r['lift_pct']:>7.2f}%")


if __name__ == "__main__":
    main()
