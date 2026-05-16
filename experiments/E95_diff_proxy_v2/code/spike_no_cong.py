"""E95 — spike variant: descend on WL + density only (no congestion).

Hypothesis: if the RUDY mismatch is the only thing breaking C1, then
disabling congestion in the smooth proxy should let WL+density gradient
provide useful descent. The smooth proxy without congestion is correct
(WL fix from DiffProxyV2; density matches canonical exactly).

If this still doesn't lift, then there's a deeper issue — maybe smooth
proxy at cascade local min is a true zero-gradient point (no useful
descent direction even with correct WL+density).
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
from diff_proxy_v2 import DiffProxyV2
from macro_legalizer import greedy_macro_legalize


def descent_no_cong(proxy, benchmark, plc, init, num_stages=4, steps_per_stage=25, label="nocong"):
    pos = init.clone().detach().requires_grad_(True)
    best_canon = float("inf")
    best_pos = init.clone()
    log = []
    t0 = time.time()
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    half = benchmark.macro_sizes / 2.0
    canvas_area = cw * ch
    baseline = compute_proxy_cost(init, benchmark, plc)
    if int(baseline["overlap_count"]) == 0:
        best_canon = float(baseline["proxy_cost"])
        best_pos = init.clone()

    for stage in range(num_stages):
        frac = stage / max(1, num_stages - 1)
        log_g_start, log_g_end = math.log(0.005), math.log(1e-4)
        gamma_frac = math.exp(log_g_start + frac * (log_g_end - log_g_start))
        lam = 500 + frac * 9500
        proxy.set_gamma_frac(gamma_frac)

        pos_local = pos.detach().clone().requires_grad_(True)
        opt = torch.optim.LBFGS(
            [pos_local], lr=1.0, max_iter=steps_per_stage,
            history_size=20, line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            cost, _ = proxy.cost(pos_local, include_congestion=False)
            penalty = proxy.overlap_penalty(pos_local)
            boundary = proxy.out_of_canvas_penalty(pos_local)
            total = cost + lam * penalty / canvas_area + 500.0 * boundary / canvas_area
            total.backward()
            return total

        try:
            opt.step(closure)
        except Exception as exc:
            log.append({"stage": stage, "exception": repr(exc)})
            continue

        with torch.no_grad():
            pos_clamped = pos_local.clone()
            pos_clamped[:, 0].clamp_(half[:, 0], cw - half[:, 0])
            pos_clamped[:, 1].clamp_(half[:, 1], ch - half[:, 1])
        c = compute_proxy_cost(pos_clamped.detach(), benchmark, plc)
        canon = float(c["proxy_cost"]); ovl = int(c["overlap_count"])
        log.append({"stage": stage, "gamma_frac": gamma_frac, "lam": lam, "canon": canon, "ovl": ovl})
        print(f"[{label}] stage={stage} γ={gamma_frac:.5f} λ={lam:.0f} canon={canon:.4f} ovl={ovl} best={best_canon:.4f}")
        if ovl == 0 and canon < best_canon:
            best_canon = canon
            best_pos = pos_clamped.detach().clone()
            pos = pos_clamped.detach().clone().requires_grad_(True)

    legal, _ = greedy_macro_legalize(best_pos, benchmark, verbose=False)
    canon_post = compute_proxy_cost(legal, benchmark, plc)
    return float(canon_post["proxy_cost"]), int(canon_post["overlap_count"]), time.time() - t0, log


def main():
    benches = sys.argv[1:] or ["ibm10", "ibm12"]
    results = {}
    for bench in benches:
        print(f"\n========== {bench} (WL+density only, no cong gradient) ==========")
        bench_dir = Path(f"external/MacroPlacement/Testcases/ICCAD04/{bench}")
        benchmark, plc = load_benchmark_from_dir(str(bench_dir))
        cached = torch.load(f"experiments/E84_cascading_saddle/results/cascade_{bench}.pt",
                            weights_only=False, map_location="cpu")
        pos = cached["placement"].clone().float()
        baseline = compute_proxy_cost(pos, benchmark, plc)
        print(f"  baseline canon={baseline['proxy_cost']:.4f} ovl={int(baseline['overlap_count'])}")
        proxy = DiffProxyV2(benchmark, plc, gamma_frac=0.005)
        try:
            post_canon, ovl, wall, log = descent_no_cong(proxy, benchmark, plc, pos, label=bench)
            lift = (float(baseline["proxy_cost"]) - post_canon) / float(baseline["proxy_cost"]) * 100
            results[bench] = {
                "baseline": float(baseline["proxy_cost"]),
                "post_leg_canon": post_canon,
                "ovl": ovl, "lift_pct": lift, "wall_s": wall, "log": log,
            }
            print(f"  → post-leg {post_canon:.4f} ovl={ovl} lift={lift:+.3f}%")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results[bench] = {"error": repr(exc), "tb": traceback.format_exc()}

    out = Path("experiments/E95_diff_proxy_v2/results/no_cong_summary.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n{'bench':<8} {'baseline':>10} {'no_cong':>10} {'lift':>8}")
    for b, r in results.items():
        if "error" in r:
            print(f"{b}: ERROR"); continue
        print(f"{b:<8} {r['baseline']:>10.4f} {r['post_leg_canon']:>10.4f} {r['lift_pct']:>+7.2f}%")


if __name__ == "__main__":
    main()
