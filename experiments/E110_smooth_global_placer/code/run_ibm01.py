"""Tonight's checkpoint: E110 SmoothGlobalPlacer on ibm01.

Print:
  - Reference SDF + project_overlaps proxy (no descent)
  - For each config: descent → legalize → final proxy + wall

Kill gate: if all configs give final proxy >1.40 OR best wall >15 min,
abandon the gradient-lane direction. If any config <1.10, continue
overnight sweep.
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

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, sdf_init
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer import SmoothGlobalPlacer


CONFIGS = [
    # label, kwargs
    ("default_500",     dict(num_steps=500,  lr_frac=0.005, init="sdf",
                             overlap_lambda_end=50.0)),
    ("longer_1500",     dict(num_steps=1500, lr_frac=0.003, init="sdf",
                             overlap_lambda_end=50.0)),
    ("from_random",     dict(num_steps=1500, lr_frac=0.005, init="random",
                             overlap_lambda_end=50.0)),
    ("strict_overlap",  dict(num_steps=800,  lr_frac=0.004, init="sdf",
                             overlap_lambda_end=200.0, overlap_ramp_pct=0.5)),
]


def main():
    bench_dir = find_benchmark_dir("ibm01")
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"Loaded ibm01: {benchmark.num_macros} macros "
          f"({benchmark.num_hard_macros} hard), "
          f"{benchmark.num_nets} nets, "
          f"canvas {benchmark.canvas_width:.0f}x{benchmark.canvas_height:.0f}",
          flush=True)

    # Reference: SDF + project_overlaps
    print("\n=== Reference: SDF + project_overlaps (no descent) ===", flush=True)
    t0 = time.time()
    sdf_pos = sdf_init(benchmark)
    sdf_pos, n_iter = project_overlaps(sdf_pos, benchmark)
    sdf_proxy = float(compute_proxy_cost(sdf_pos, benchmark, plc)["proxy_cost"])
    sdf_ovl = compute_overlap_metrics(sdf_pos, benchmark)["overlap_count"]
    print(f"  SDF proxy={sdf_proxy:.5f} ovl={sdf_ovl} "
          f"project_iter={n_iter} wall={time.time()-t0:.1f}s\n", flush=True)

    results = {
        "bench": "ibm01",
        "sdf_reference": {
            "proxy": sdf_proxy, "overlaps": sdf_ovl,
            "wall_s": time.time() - t0,
        },
        "configs": [],
    }

    for label, kwargs in CONFIGS:
        print(f"\n=== Config: {label}  kwargs={kwargs} ===", flush=True)
        t0 = time.time()
        try:
            placer = SmoothGlobalPlacer(verbose=True, **kwargs)
            pos = placer.place(benchmark)
            proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
            ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
            wall = time.time() - t0
            print(f"\n>>> [{label}] FINAL proxy={proxy:.5f} ovl={ovl} wall={wall:.1f}s",
                  flush=True)
            results["configs"].append({
                "label": label, "kwargs": kwargs,
                "proxy": proxy, "overlaps": ovl, "wall_s": wall,
                "error": None,
            })
        except Exception as exc:
            wall = time.time() - t0
            print(f"\n>>> [{label}] EXCEPTION: {exc} wall={wall:.1f}s",
                  flush=True)
            import traceback
            traceback.print_exc()
            results["configs"].append({
                "label": label, "kwargs": kwargs,
                "proxy": None, "overlaps": None, "wall_s": wall,
                "error": repr(exc),
            })

    # Summary
    print("\n\n=== SUMMARY ===", flush=True)
    print(f"  SDF reference: proxy={sdf_proxy:.5f}", flush=True)
    for r in results["configs"]:
        status = f"proxy={r['proxy']:.5f}" if r["proxy"] is not None else f"ERR={r['error']}"
        print(f"  {r['label']:24s}: {status:30s} ovl={r['overlaps']} wall={r['wall_s']:.1f}s",
              flush=True)

    out_path = _HERE.parent / "results" / "ibm01_first_pass.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
