"""E120 — quick test on ibm01/ibm03/ibm05 to validate lift early.

These benches are small/medium so each comparison takes 2-5 min, but
gives us a 3-bench signal before committing to --fast/--all.
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
    _ROOT / "submissions" / "_archive" / "e111_minimal",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

from continuous_saddle_placer import ContinuousHessianSaddlePlacer


def run_one_bench(bench_name: str, budget_s: float) -> dict:
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(
        f"\n[bench={bench_name}] {benchmark.num_macros} macros "
        f"({benchmark.num_hard_macros} hard), nets={benchmark.num_nets}",
        flush=True,
    )

    result = {"bench": bench_name, "budget_s": budget_s}

    # 1) E120
    torch.manual_seed(42)
    placer = ContinuousHessianSaddlePlacer(
        budget_seconds=budget_s,
        num_steps_phaseA=250,
        num_steps_resume=100,
        max_saddle_stages=2,
        overlap_lambda_end=10.0,
        cd_polish_s=max(60.0, budget_s * 0.4),
        eigsh_maxiter=200,
        verbose=True,
        log_every=200,
    )
    t0 = time.time()
    pos = placer.place(benchmark)
    p = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
    w = time.time() - t0
    print(f"  [E120] proxy={p:.5f} wall={w:.0f}s", flush=True)
    result["e120"] = {"proxy": p, "wall_s": w}

    # Reload plc to reset.
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    # 2) V3Min baseline (matched budget)
    from placer import E111MinimalPlacer
    torch.manual_seed(42)
    placer_b = E111MinimalPlacer(
        budget_seconds=budget_s,
        num_steps=400,
        overlap_lambda_end=10.0,
        cd_polish_s=max(60.0, budget_s * 0.4),
        rng_seed=42,
        verbose=False,
    )
    t0 = time.time()
    pos_b = placer_b.place(benchmark)
    p_b = float(compute_proxy_cost(pos_b, benchmark, plc)["proxy_cost"])
    w_b = time.time() - t0
    print(f"  [V3Min] proxy={p_b:.5f} wall={w_b:.0f}s", flush=True)
    result["v3min"] = {"proxy": p_b, "wall_s": w_b}
    result["delta"] = p - p_b
    result["delta_pct"] = 100.0 * (p - p_b) / p_b
    print(
        f"  [DELTA] {result['delta']:+.5f} ({result['delta_pct']:+.2f}%)",
        flush=True,
    )

    return result


def main():
    # Pick small/medium benches for quick signal.
    BENCHES = [("ibm01", 240), ("ibm03", 300), ("ibm05", 300)]
    results = []
    for bench, budget in BENCHES:
        r = run_one_bench(bench, budget)
        results.append(r)

    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        print(
            f"  {r['bench']:8s} "
            f"E120={r['e120']['proxy']:.5f} "
            f"V3Min={r['v3min']['proxy']:.5f} "
            f"Δ={r['delta']:+.5f} ({r['delta_pct']:+.2f}%) "
            f"wall E120/V3={r['e120']['wall_s']:.0f}s/{r['v3min']['wall_s']:.0f}s",
            flush=True,
        )

    avg_delta = sum(r["delta_pct"] for r in results) / len(results)
    print(f"\n  Avg Δ: {avg_delta:+.2f}%", flush=True)

    out = _HERE.parent / "results" / "small_benches.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
