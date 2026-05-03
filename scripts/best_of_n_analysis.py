"""Compute per-bench best-of-N across hypotheses from results/experiment_log.jsonl.

Usage:
    python scripts/best_of_n_analysis.py --mode all --hypotheses E25 E41 E48_hybrid_all E54_all
    python scripts/best_of_n_analysis.py --mode fast --hypotheses E48_hybrid_fast E52_e41_seed1_fast E54

For each (hypothesis, mode) pair, takes the *most recent* matching log entry
and reports per-bench winners + avg of the per-bench-best.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def load_log(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def latest_entry_for(rows: List[Dict], hypothesis: str, mode: str) -> Optional[Dict]:
    matches = [r for r in rows if r.get("hypothesis") == hypothesis and r.get("mode") == mode]
    if not matches:
        return None
    matches.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default="results/experiment_log.jsonl",
        help="Path to experiment_log.jsonl",
    )
    parser.add_argument("--mode", required=True, choices=["fast", "all", "ng45"])
    parser.add_argument(
        "--hypotheses", nargs="+", required=True,
        help="Hypothesis names to combine (must match --hypothesis tags)",
    )
    args = parser.parse_args()

    rows = load_log(Path(args.log))
    entries: Dict[str, Dict] = {}
    for h in args.hypotheses:
        e = latest_entry_for(rows, h, args.mode)
        if e is None:
            print(f"WARN: no {args.mode} entry for hypothesis '{h}'", file=sys.stderr)
            continue
        entries[h] = e

    if not entries:
        print("ERROR: no matching entries", file=sys.stderr)
        return 1

    # Print individual averages.
    print(f"\n=== Individual {args.mode} averages ===")
    for h, e in entries.items():
        print(f"  {h:40} avg={e['avg_proxy_cost']:.5f}  ts={e.get('timestamp','')}")

    # Determine bench set (intersection of all entries).
    bench_sets = [set(e["per_benchmark"].keys()) for e in entries.values()]
    common = set.intersection(*bench_sets) if bench_sets else set()
    if not common:
        print("ERROR: no common benchmarks across entries", file=sys.stderr)
        return 1
    benches = sorted(common)

    # Per-bench best-of.
    print(f"\n=== Per-bench best-of-{len(entries)} ===")
    print(f"  {'bench':<14} " + "  ".join(f"{h[:18]:<18}" for h in entries) + "  WINNER       VALUE")
    bestofs: Dict[str, float] = {}
    for b in benches:
        per = {h: entries[h]["per_benchmark"][b] for h in entries}
        winner = min(per, key=per.get)
        bv = per[winner]
        bestofs[b] = bv
        cells = "  ".join(f"{per[h]:<18.5f}" for h in entries)
        marker = "*" if all(per[h] == per[winner] for h in entries) else ""
        print(f"  {b:<14} {cells}  {winner[:12]:<12} {bv:.5f}")

    avg_best = sum(bestofs.values()) / len(bestofs)
    print(f"\n  AVG of best-of-{len(entries)}: {avg_best:.5f}")

    # Compare to each individual.
    print(f"\n=== Best-of-{len(entries)} vs individuals ===")
    for h, e in entries.items():
        delta = avg_best - e["avg_proxy_cost"]
        pct = delta / e["avg_proxy_cost"] * 100
        print(f"  vs {h:40} Δ={delta:+.5f} ({pct:+.2f} %)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
