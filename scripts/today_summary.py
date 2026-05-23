"""Aggregate today's experiment results into a unified per-bench table.

Reads `results/experiment_log.jsonl` + per-run JSON files from today
(2026-05-21), prints:
  1. Per-experiment summary (avg, wall, bench list)
  2. Per-benchmark cross-experiment comparison (which exp is best on each bench)
  3. Projection table for --all if known per-bench numbers hold

Usage:
  uv run python scripts/today_summary.py
  uv run python scripts/today_summary.py --hypothesis-prefix E1
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Bench size data
BENCH_MOVABLES = {
    "ibm01": 246, "ibm02": 271, "ibm03": 290, "ibm04": 295,
    "ibm06": 178, "ibm07": 291, "ibm08": 301, "ibm09": 253,
    "ibm10": 786, "ibm11": 373, "ibm12": 651, "ibm13": 424,
    "ibm14": 614, "ibm15": 393, "ibm16": 458, "ibm17": 760,
    "ibm18": 285,
}
ADAPTIVE_THRESHOLD = 400
FAST_SUBSET = {"ibm01", "ibm04", "ibm09", "ibm13"}


def load_log(path: Path, today_only: bool = True):
    """Read experiment_log.jsonl, return list of result dicts."""
    if not path.exists():
        return []
    results = []
    for line in path.read_text().strip().split("\n"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if today_only and not d.get("timestamp", "").startswith("2026-05-21"):
            continue
        results.append(d)
    return results


def per_bench_proxy(results):
    """Returns dict: bench -> list of (hypothesis, proxy, wall_seconds)."""
    out = defaultdict(list)
    for d in results:
        h = d["hypothesis"]
        pb = d.get("per_benchmark", {})
        for bench, bench_data in pb.items():
            if isinstance(bench_data, dict):
                proxy = bench_data.get("proxy_cost", bench_data.get("avg_proxy_cost"))
                wall = bench_data.get("runtime_seconds")
            else:
                proxy = bench_data
                wall = d.get("runtime_seconds")
            if proxy is not None:
                out[bench].append((h, float(proxy), wall))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-days", action="store_true", help="Include pre-today runs")
    ap.add_argument("--hypothesis-prefix", default="", help="Filter to hypothesis prefix")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    log_path = repo / "results" / "experiment_log.jsonl"

    results = load_log(log_path, today_only=not args.all_days)
    if args.hypothesis_prefix:
        results = [r for r in results if r["hypothesis"].startswith(args.hypothesis_prefix)]
    if not results:
        print("No results match filter.")
        return

    print("=" * 90)
    print(f"PER-EXPERIMENT SUMMARY ({len(results)} runs)")
    print("=" * 90)
    print(f"{'Hypothesis':18s} {'Mode':8s} {'Avg proxy':>10s} {'Wall (s)':>10s} {'Benches':<40s}")
    print("-" * 90)
    for d in sorted(results, key=lambda r: r["hypothesis"]):
        h = d["hypothesis"]
        mode = d.get("mode", "?")
        avg = d.get("avg_proxy_cost", 0.0)
        wall = d.get("runtime_seconds", 0.0)
        pb = d.get("per_benchmark", {})
        benches = list(pb.keys()) if isinstance(pb, dict) else []
        bench_str = ",".join(benches[:6]) + ("..." if len(benches) > 6 else "")
        print(f"{h:18s} {mode:8s} {avg:10.5f} {wall:10.0f} {bench_str:<40s}")

    pb_results = per_bench_proxy(results)
    print()
    print("=" * 90)
    print(f"PER-BENCHMARK CROSS-EXPERIMENT TABLE")
    print("=" * 90)
    print(f"{'Bench':8s} {'Movables':>8s} {'Subset':>8s} {'Best':>9s} {'Best exp':18s} {'All runs':50s}")
    print("-" * 110)

    for bench in sorted(pb_results.keys()):
        runs = pb_results[bench]
        runs_sorted = sorted(runs, key=lambda x: x[1])
        best_h, best_p, _ = runs_sorted[0]
        movables = BENCH_MOVABLES.get(bench, 0)
        subset = "FAST" if bench in FAST_SUBSET else ""
        all_runs = "  ".join(f"{h}={p:.4f}" for h, p, _ in runs_sorted)
        print(f"{bench:8s} {movables:>8d} {subset:>8s} {best_p:9.5f} {best_h:18s} {all_runs[:50]}")

    print()
    print("=" * 90)
    print(f"E150 ADAPTIVE ROUTING PROJECTION")
    print("=" * 90)
    print(f"Threshold: {ADAPTIVE_THRESHOLD} movables")
    print(f"Small (<{ADAPTIVE_THRESHOLD}): use E142/E148 stack (3-iter portfolio)")
    print(f"Large (>={ADAPTIVE_THRESHOLD}): use E143 stack (K-joint+SA)")
    print()
    print(f"{'Bench':8s} {'Movables':>8s} {'Lane':>6s} {'Best known':>11s} {'Source':18s}")
    print("-" * 70)
    sum_proxy = 0.0
    n_known = 0
    for bench in sorted(BENCH_MOVABLES.keys()):
        movables = BENCH_MOVABLES[bench]
        lane = "E143" if movables >= ADAPTIVE_THRESHOLD else "E142+"
        best = "?"
        src = "(not yet)"
        if bench in pb_results:
            runs = sorted(pb_results[bench], key=lambda x: x[1])
            best = f"{runs[0][1]:.5f}"
            src = runs[0][0]
            sum_proxy += runs[0][1]
            n_known += 1
        print(f"{bench:8s} {movables:>8d} {lane:>6s} {best:>11s} {src:18s}")
    if n_known > 0:
        print(f"\nKnown-bench avg ({n_known}/17): {sum_proxy/n_known:.5f}")
        if n_known < 17:
            print(f"(NOT a full --all average; awaits the remaining {17 - n_known} benches)")


if __name__ == "__main__":
    main()
