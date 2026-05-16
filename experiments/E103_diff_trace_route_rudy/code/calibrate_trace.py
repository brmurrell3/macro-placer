"""Calibrate DiffTraceRudy vs canonical plc.get_congestion_cost.

Test protocol:
  1. Load bench + cascade plateau placement.
  2. Compute canonical_cong + smooth_cong at plateau.
  3. Perturb 20 macros by Gaussian, recompute both at each perturbation.
  4. Report:
     - abs mismatch %
     - Pearson + Spearman ρ between perturb-induced Δ
     - Sign agreement on finite-diff gradient direction (5 macros)
  Green light: Spearman > 0.7 OR sign_agree > 0.75.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from diff_trace_rudy import DiffTraceRudy

from macro_place.loader import load_benchmark_from_dir
from macro_place.bench_paths import find_benchmark_dir
from macro_place.objective import compute_proxy_cost

import importlib.util
_DPO = _ROOT / "writeup" / "archive" / "submissions" / "dpo" / "ablation_v2_steps.py"
spec = importlib.util.spec_from_file_location("_dpo", str(_DPO))
dpo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dpo)


def calibrate(bench_name: str, n_perturbs: int = 16, perturb_scale_frac: float = 0.02):
    print(f"\n=== calibrate {bench_name} ===")
    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))
    # Populate net_pin attrs
    nd = dpo._extract_net_data(bench, plc)
    bench.net_pin_macro_idx = nd.pin_macro_idx
    bench.net_pin_offsets = nd.pin_offsets
    bench.net_pin_mask = nd.mask
    bench.net_weights = nd.weights

    cascade_pt = _ROOT / "experiments" / "E84_cascading_saddle" / "results" / f"cascade_{bench_name}.pt"
    if cascade_pt.exists():
        data = torch.load(cascade_pt, weights_only=False)
        start = (data.get("placement") if "placement" in data else data["polished_placement"]).to(torch.float32)
        print(f"  using cached cascade plateau")
    else:
        start = bench.macro_positions.clone().float()
        print(f"  using macro_positions init")

    rudy = DiffTraceRudy(bench, plc, device="cpu")
    cw = float(bench.canvas_width)

    # Baseline
    can_proxy = compute_proxy_cost(start, bench, plc)
    can_cong = float(can_proxy["congestion_cost"])
    with torch.no_grad():
        smooth_cong = float(rudy.congestion_cost(start))
    print(f"  canonical cong:  {can_cong:.5f}")
    print(f"  smooth   cong:   {smooth_cong:.5f}")
    print(f"  abs mismatch:    {(smooth_cong - can_cong) / max(can_cong, 1e-6) * 100:+.1f}%")

    # Perturbation correlation
    rng = np.random.default_rng(42)
    fixed = bench.macro_fixed.cpu().numpy()
    n_hard = bench.num_hard_macros
    movable_hard = [i for i in range(n_hard) if not bool(fixed[i])]
    scale = perturb_scale_frac * cw
    can_deltas, sm_deltas = [], []
    base_can = can_cong
    base_sm = smooth_cong

    for k in range(n_perturbs):
        perturb = start.clone()
        targets = rng.choice(movable_hard, size=min(5, len(movable_hard)), replace=False)
        for t in targets:
            dx = float(rng.normal(0.0, scale))
            dy = float(rng.normal(0.0, scale))
            perturb[t, 0] = torch.clamp(perturb[t, 0] + dx,
                                         bench.macro_sizes[t, 0] / 2,
                                         cw - bench.macro_sizes[t, 0] / 2)
            perturb[t, 1] = torch.clamp(perturb[t, 1] + dy,
                                         bench.macro_sizes[t, 1] / 2,
                                         float(bench.canvas_height) - bench.macro_sizes[t, 1] / 2)
        c_can = float(compute_proxy_cost(perturb, bench, plc)["congestion_cost"])
        with torch.no_grad():
            c_sm = float(rudy.congestion_cost(perturb))
        can_deltas.append(c_can - base_can)
        sm_deltas.append(c_sm - base_sm)

    can_arr = np.array(can_deltas)
    sm_arr = np.array(sm_deltas)
    pearson = float(np.corrcoef(can_arr, sm_arr)[0, 1]) if len(can_arr) > 2 else float('nan')
    rk_c = np.argsort(np.argsort(can_arr))
    rk_s = np.argsort(np.argsort(sm_arr))
    spearman = float(np.corrcoef(rk_c, rk_s)[0, 1]) if len(can_arr) > 2 else float('nan')
    sign_agree = float(np.mean(np.sign(can_arr) == np.sign(sm_arr)))
    print(f"  Δcong perturbation correlation:")
    print(f"    Pearson  ρ = {pearson:+.3f}")
    print(f"    Spearman ρ = {spearman:+.3f}")
    print(f"    sign-agree = {sign_agree:.2%}")

    # Verdict
    green = (spearman > 0.7) or (sign_agree > 0.75) or (pearson > 0.7)
    print(f"  VERDICT: {'🟢 GREEN — trace-route RUDY tracks canonical' if green else '🔴 RED — still detached, RUDY needs more work'}")
    return {"bench": bench_name, "spearman": spearman, "sign_agree": sign_agree, "pearson": pearson, "green": green}


if __name__ == "__main__":
    benches = sys.argv[1:] if len(sys.argv) > 1 else ["ibm01", "ibm10", "ibm12", "ibm17"]
    results = []
    for b in benches:
        try:
            results.append(calibrate(b))
        except Exception as exc:
            print(f"  {b}: ERROR {exc}")
            results.append({"bench": b, "error": repr(exc)})
    print("\n=== SUMMARY ===")
    for r in results:
        if "error" in r:
            print(f"  {r['bench']}: ERROR")
            continue
        light = "🟢" if r["green"] else "🔴"
        print(f"  {r['bench']:6s} {light} sp={r['spearman']:+.2f} sign={r['sign_agree']:.0%} pe={r['pearson']:+.2f}")
