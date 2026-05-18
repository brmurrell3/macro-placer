"""May 19 kill-gate harness — full --fast (4 benches) × 4 variants comparison.

Runs each variant serially on each bench, writes a single summary JSON
to results/fast_compare_<timestamp>.json. Variants: base / h1 / h2 / h1h2.

Kill gate: best variant's IBM --fast average must beat base by ≥ 0.5%.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _run_one(variant: str, bench: str, budget_s: float, results_dir: Path) -> dict:
    """Spawn a subprocess to avoid module-load contamination between variants."""
    import subprocess

    log_file = results_dir / f"{variant}_{bench}.log"
    cmd = [
        "uv", "run", "python", "-u", "-c",
        f"""
import os, sys, time, json
os.environ['MPC_V2_H1'] = {'1' if 'h1' in variant else '0'!r}
os.environ['MPC_V2_H2'] = {'1' if 'h2' in variant else '0'!r}
sys.path.insert(0, {str(_ROOT)!r})
from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost, compute_overlap_metrics
import importlib.util
if {repr(variant) == "'base'"}:
    spec = importlib.util.spec_from_file_location('p', {str(_ROOT / 'submissions/cd_lns_sa_cascade_stacked_periphery/placer.py')!r})
else:
    spec = importlib.util.spec_from_file_location('p', {str(_ROOT / 'submissions/cd_lns_sa_cascade_v2/placer.py')!r})
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
cls = mod.CDLNSSACascadeStackedPeripheryPlacer if {repr(variant) == "'base'"} else mod.CDLNSSACascadeV2Placer
placer = cls(budget_seconds={budget_s}, verbose=False)
bench_dir = find_benchmark_dir({bench!r})
benchmark, plc = load_benchmark_from_dir(str(bench_dir))
t0 = time.time()
placement = placer.place(benchmark)
wall = time.time() - t0
proxy = float(compute_proxy_cost(placement, benchmark, plc)['proxy_cost'])
ovl = compute_overlap_metrics(placement, benchmark)
result = {{
    'variant': {variant!r}, 'bench': {bench!r},
    'proxy': proxy,
    'overlap_count': int(ovl['overlap_count']),
    'total_overlap_area': float(ovl['total_overlap_area']),
    'wall_s': wall,
    'budget_s': {budget_s},
}}
print('RESULT_JSON', json.dumps(result))
""",
    ]
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        proc.wait()
    text = log_file.read_text()
    for line in text.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON "):])
    return {
        "variant": variant, "bench": bench,
        "proxy": float("nan"), "overlap_count": -1,
        "wall_s": float("nan"), "budget_s": budget_s,
        "error": "no RESULT_JSON in log; see " + str(log_file),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--budget-s", type=float, default=1500.0,
                   help="Per-bench wall budget (default 1500s ≈ 25 min)")
    p.add_argument("--benches", nargs="+",
                   default=["ibm01", "ibm04", "ibm09", "ibm13"])
    p.add_argument("--variants", nargs="+",
                   default=["base", "h2", "h1", "h1h2"])
    args = p.parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).resolve().parent.parent / "results" / f"fast_compare_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"results -> {results_dir}")

    all_results = []
    t_total = time.time()
    for v in args.variants:
        for b in args.benches:
            t0 = time.time()
            print(f"\n>>> {v}/{b} ...", flush=True)
            r = _run_one(v, b, args.budget_s, results_dir)
            r["wall_total_s"] = time.time() - t0
            print(f"   proxy={r.get('proxy', 'NaN'):.5f}  "
                  f"overlaps={r.get('overlap_count', '?')}  "
                  f"wall={r['wall_total_s']:.0f}s", flush=True)
            all_results.append(r)

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {summary_path}")

    # Per-variant aggregate.
    by_variant: dict = {}
    for r in all_results:
        v = r["variant"]
        if v not in by_variant:
            by_variant[v] = {"proxies": [], "overlaps": 0, "wall": 0.0}
        if r.get("proxy") == r.get("proxy"):  # not NaN
            by_variant[v]["proxies"].append(r["proxy"])
        by_variant[v]["overlaps"] += max(0, r.get("overlap_count", 0))
        by_variant[v]["wall"] += r.get("wall_total_s", 0.0)

    print("\n" + "=" * 60)
    print(f"{'variant':<10} {'avg_proxy':>12} {'overlaps':>10} {'wall_s':>10}")
    base_avg = float("nan")
    for v, d in by_variant.items():
        if d["proxies"]:
            avg = sum(d["proxies"]) / len(d["proxies"])
        else:
            avg = float("nan")
        if v == "base":
            base_avg = avg
        print(f"{v:<10} {avg:>12.5f} {d['overlaps']:>10d} {d['wall']:>10.0f}")
    print(f"\nElapsed total: {time.time() - t_total:.0f}s")

    print("\nKill gate (≥0.5% lift on --fast):")
    for v, d in by_variant.items():
        if v == "base" or not d["proxies"]:
            continue
        avg = sum(d["proxies"]) / len(d["proxies"])
        delta = (avg - base_avg) / base_avg * 100.0
        status = "PASS" if delta <= -0.5 else "FAIL"
        print(f"  {v:<10} delta vs base = {delta:+.2f}%  [{status}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
