"""E123 — Quick smoke: measure smooth-basin proxy only (no CD polish).

Fast iteration script: runs Adam descent + legalize for each (ε, iters)
config and reports the LEGAL post-Adam proxy. Skips the expensive
CD polish. Use this to find promising ε settings, then run test_ibm17.py
for the full pipeline.

~3 mins per config on ibm17. 4 configs = ~12 mins.
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
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v3_sinkhorn import SmoothGlobalPlacerV3Sinkhorn


def main():
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm17"
    bench_dir = find_benchmark_dir(bench)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(
        f"Loaded {bench}: {benchmark.num_macros} macros "
        f"({benchmark.num_hard_macros} hard), nets={benchmark.num_nets}",
        flush=True,
    )

    base_cfg = dict(
        num_steps=500,
        lr_frac=0.005,
        init="sdf",
        overlap_lambda_end=10.0,
        verbose=True,
        log_every=100,
    )

    # ε=0.5 matches canonical scalar within 0.26% on ibm17 (eps_sweep.py).
    # ε=0.3 has slightly less canonical fidelity but sharper top-K signal.
    # Anneal 0.5→0.1: smooth gradient first, sharpen as basin settles.
    configs = [
        ("topk (ablation, == E111 V3)", {"use_sinkhorn": False}),
        ("sinkhorn ε=0.5, iters=100",    {"use_sinkhorn": True, "sinkhorn_eps": 0.5, "sinkhorn_iters": 100}),
        ("sinkhorn ε=0.3, iters=80",     {"use_sinkhorn": True, "sinkhorn_eps": 0.3, "sinkhorn_iters": 80}),
        ("sinkhorn ε anneal 0.5→0.1",    {"use_sinkhorn": True, "sinkhorn_eps": 0.5, "sinkhorn_eps_end": 0.1, "sinkhorn_iters": 100}),
    ]

    results = []
    for label, extra in configs:
        print(f"\n=== {label} ===", flush=True)
        # Fresh benchmark+plc each time (plc state may be mutated)
        benchmark, plc = load_benchmark_from_dir(str(bench_dir))
        placer = SmoothGlobalPlacerV3Sinkhorn(**base_cfg, **extra)
        t0 = time.time()
        pos = placer.place(benchmark)
        wall = time.time() - t0
        proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        print(f"  RESULT: proxy={proxy:.5f}  ovl={ovl}  wall={wall:.1f}s", flush=True)
        results.append({
            "label": label, "config": extra,
            "raw_proxy": proxy, "ovl": ovl, "wall_s": wall,
        })

    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        print(f"  {r['label']:42s}  proxy={r['raw_proxy']:.5f}  ovl={r['ovl']}  wall={r['wall_s']:.0f}s",
              flush=True)

    out = _HERE.parent / "results" / f"quick_smoke_{bench}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
