"""Smoke test for cd_lns_sa_cascade_dp_lane_patched on ibm01.

Targets:
  - Raw smooth basin (SDF init + smooth descent + greedy legalize): < 1.40
  - After 60s CD: < 0.90 (cascade gets 0.85)

Also reports what each lane independently produces. This is the
local-testable equivalent of "patched DREAMPlace" — running smooth-
proxy gradient descent (the surrogate for DP's loss-replaced version).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments/E88_diff_proxy/code"))
sys.path.insert(0, str(_ROOT / "experiments/E95_diff_proxy_v2/code"))
sys.path.insert(0, str(_ROOT / "experiments/E76_dreamplace_integration/code"))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, sdf_init, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost
from smooth_global_placer import SmoothGlobalPlacer


def main():
    bench_dir = find_benchmark_dir("ibm01")
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(
        f"ibm01: {benchmark.num_macros} macros "
        f"({benchmark.num_hard_macros} hard), "
        f"canvas {benchmark.canvas_width:.1f}x{benchmark.canvas_height:.1f}",
        flush=True,
    )

    results = {}

    # Reference: SDF + project_overlaps (no descent, no CD).
    print("\n=== Reference: SDF + project_overlaps ===", flush=True)
    t0 = time.time()
    sdf_pos = sdf_init(benchmark)
    sdf_pos, n_iter = project_overlaps(sdf_pos, benchmark)
    sdf_proxy = float(compute_proxy_cost(sdf_pos, benchmark, plc)["proxy_cost"])
    sdf_ovl = compute_overlap_metrics(sdf_pos, benchmark)["overlap_count"]
    print(
        f"  SDF: proxy={sdf_proxy:.5f} ovl={sdf_ovl} "
        f"wall={time.time()-t0:.1f}s",
        flush=True,
    )
    results["sdf"] = {"proxy": sdf_proxy, "ovl": sdf_ovl}

    # === Variant 1: SmoothGlobalPlacer default config (no CD) ===
    print("\n=== Variant 1: SDF + smooth descent (raw, no CD) ===", flush=True)
    t0 = time.time()
    placer = SmoothGlobalPlacer(
        num_steps=500, lr_frac=0.005,
        gamma_start_frac=5e-3, gamma_end_frac=5e-5,
        overlap_lambda_end=50.0,
        overlap_ramp_pct=0.7,
        legalize_step_frac=0.005,
        legalize_radius_steps=200,
        init="sdf", rng_seed=42, verbose=False,
    )
    raw_pos = placer.place(benchmark)
    raw_proxy = float(compute_proxy_cost(raw_pos, benchmark, plc)["proxy_cost"])
    raw_ovl = compute_overlap_metrics(raw_pos, benchmark)["overlap_count"]
    raw_wall = time.time() - t0
    print(
        f"  raw smooth basin: proxy={raw_proxy:.5f} ovl={raw_ovl} "
        f"wall={raw_wall:.1f}s [target <1.40: {'PASS' if raw_proxy < 1.40 else 'FAIL'}]",
        flush=True,
    )
    results["raw_smooth"] = {
        "proxy": raw_proxy, "ovl": raw_ovl, "wall_s": raw_wall,
    }

    # === Variant 2: + 60s CD ===
    print("\n=== Variant 2: SDF + smooth + 60s CD ===", flush=True)
    t1 = time.time()
    ev = IncrementalProxyEvaluator(benchmark, plc, raw_pos)
    movable = [
        i for i in range(benchmark.num_macros)
        if not bool(benchmark.macro_fixed[i])
    ]
    run_cd_adaptive(
        ev, benchmark, plc, movable,
        min_time_s=60.0, hard_cap_s=60.0,
        patience=3, plateau_threshold=0.001,
    )
    cd_pos = ev.placement.detach().clone().to("cpu").float()
    cd_proxy = float(compute_proxy_cost(cd_pos, benchmark, plc)["proxy_cost"])
    cd_ovl = compute_overlap_metrics(cd_pos, benchmark)["overlap_count"]
    cd_wall = time.time() - t1
    total_wall = raw_wall + cd_wall
    print(
        f"  smooth+CD60s: proxy={cd_proxy:.5f} ovl={cd_ovl} "
        f"CD_wall={cd_wall:.1f}s total={total_wall:.1f}s "
        f"[target <0.90: {'PASS' if cd_proxy < 0.90 else 'FAIL'}]",
        flush=True,
    )
    results["smooth_cd60"] = {
        "proxy": cd_proxy, "ovl": cd_ovl,
        "cd_wall_s": cd_wall, "total_wall_s": total_wall,
    }

    # === Variant 3: + extended CD (240s, what the placer would use) ===
    print("\n=== Variant 3: SDF + smooth + 240s CD (placer setting) ===", flush=True)
    t1 = time.time()
    ev = IncrementalProxyEvaluator(benchmark, plc, raw_pos)
    run_cd_adaptive(
        ev, benchmark, plc, movable,
        min_time_s=60.0, hard_cap_s=240.0,
        patience=3, plateau_threshold=0.001,
    )
    cd_pos2 = ev.placement.detach().clone().to("cpu").float()
    cd_proxy2 = float(compute_proxy_cost(cd_pos2, benchmark, plc)["proxy_cost"])
    cd_ovl2 = compute_overlap_metrics(cd_pos2, benchmark)["overlap_count"]
    cd_wall2 = time.time() - t1
    print(
        f"  smooth+CD240s: proxy={cd_proxy2:.5f} ovl={cd_ovl2} "
        f"CD_wall={cd_wall2:.1f}s",
        flush=True,
    )
    results["smooth_cd240"] = {
        "proxy": cd_proxy2, "ovl": cd_ovl2, "cd_wall_s": cd_wall2,
    }

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    print(f"  SDF reference:           proxy={results['sdf']['proxy']:.5f}", flush=True)
    print(f"  raw smooth basin:        proxy={results['raw_smooth']['proxy']:.5f}", flush=True)
    print(f"  smooth + CD 60s:         proxy={results['smooth_cd60']['proxy']:.5f}", flush=True)
    print(f"  smooth + CD 240s:        proxy={results['smooth_cd240']['proxy']:.5f}", flush=True)
    print(
        f"\nTarget gates:\n"
        f"  raw   < 1.40: {results['raw_smooth']['proxy']:.5f} "
        f"{'PASS' if results['raw_smooth']['proxy'] < 1.40 else 'FAIL'}\n"
        f"  cd60s < 0.90: {results['smooth_cd60']['proxy']:.5f} "
        f"{'PASS' if results['smooth_cd60']['proxy'] < 0.90 else 'FAIL'}\n"
        f"  cd240 < 0.86: {results['smooth_cd240']['proxy']:.5f} "
        f"{'PASS (matches DP+polish 0.862)' if results['smooth_cd240']['proxy'] < 0.86 else 'INFO'}",
        flush=True,
    )

    out_path = _HERE.parent / "results" / "smoke_ibm01_patched.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
