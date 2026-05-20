"""Morning check — summarize overnight E110 results in one screen.

Usage: uv run python experiments/E110_smooth_global_placer/code/check_overnight.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]


def _tail(path: Path, n: int = 5) -> str:
    if not path.exists():
        return f"(missing: {path})"
    with path.open() as f:
        lines = f.readlines()
    return "".join(lines[-n:])


def _job_alive(pid_or_pattern: str) -> bool:
    res = os.popen(f"pgrep -f '{pid_or_pattern}'").read().strip()
    return bool(res)


def summarize_sweep(jsonl_path: Path, summary_path: Path) -> None:
    if not summary_path.exists():
        print(f"  ⚠ summary missing — sweep not finished yet")
        if jsonl_path.exists():
            rows = []
            with jsonl_path.open() as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
            runs = [r for r in rows if r.get("type") == "sweep_run"]
            refs = {r["bench"]: r["cd_proxy"] for r in rows if r.get("type") == "sdf_ref"}
            print(f"  partial: {len(runs)} runs, {len(refs)} refs")
            if runs and refs:
                runs_with_delta = []
                for r in runs:
                    if r["cd_proxy"] is None or r["bench"] not in refs:
                        continue
                    delta_pct = (r["cd_proxy"] - refs[r["bench"]]) / refs[r["bench"]] * 100
                    runs_with_delta.append((r, delta_pct))
                runs_with_delta.sort(key=lambda x: x[1])
                print(f"  best partial runs:")
                for r, d in runs_with_delta[:5]:
                    print(f"    {r['bench']} Δ={d:+.2f}% cfg={r.get('config_label','?')}")
        return

    summary = json.loads(summary_path.read_text())
    rankings = summary.get("rankings", [])
    if not rankings:
        print("  ⚠ no rankings — sweep produced nothing")
        return
    print(f"  Refs: {summary.get('sdf_refs', {})}")
    print(f"  Total configs tested: {summary.get('total_configs_tested', '?')}")
    print(f"  Top 5 by avg Δ%:")
    for r in rankings[:5]:
        cfg_str = ", ".join(f"{k}={v}" for k, v in r["config"].items() if k != "verbose")
        print(f"    Δ_avg={r['avg_delta_pct']:+.2f}% (W{r['wins']}/L{r['losses']}) "
              f"min={r['min_delta_pct']:+.2f}% max={r['max_delta_pct']:+.2f}%")
        print(f"      cfg: {cfg_str}")
        print(f"      per-bench: {r.get('per_bench', {})}")


def summarize_all_eval(json_pattern: str) -> None:
    results_dir = _ROOT / "results"
    files = sorted(results_dir.glob(json_pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        print(f"  ⚠ no result file matching {json_pattern}")
        return
    f = files[-1]
    print(f"  loaded {f.name} (mtime {time.ctime(f.stat().st_mtime)})")
    data = json.loads(f.read_text())
    bench_results = data.get("benchmark_results") or data.get("benchmarks") or {}
    if not bench_results:
        print("  ⚠ no benchmark_results in JSON")
        print(f"  keys: {list(data.keys())}")
        return
    proxies = []
    for bname, br in bench_results.items():
        proxy = br.get("proxy_cost") or br.get("proxy")
        ovl = br.get("overlap_count") or br.get("overlaps")
        valid = br.get("valid", "?")
        proxies.append((bname, proxy, ovl, valid))
    proxies.sort()
    print(f"  {'bench':<8} {'proxy':>10} {'ovl':>5} {'valid':>5}")
    valid_proxies = []
    for n, p, o, v in proxies:
        if p is not None:
            print(f"  {n:<8} {p:>10.5f} {str(o):>5} {str(v):>5}")
            valid_proxies.append(p)
        else:
            print(f"  {n:<8} {'NULL':>10} {str(o):>5} {str(v):>5}")
    if valid_proxies:
        avg = sum(valid_proxies) / len(valid_proxies)
        print(f"  AVG: {avg:.5f}  (n={len(valid_proxies)})")


def main():
    print("=" * 70)
    print("E110 OVERNIGHT CHECK — " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # Process status
    print("\n>>> RUNNING JOBS")
    jobs = [
        ("handpicked sweep", "sweep_harness.py.*handpicked"),
        ("ibm04_diag sweep", "sweep_harness.py.*ibm04_diag"),
        ("ibm04winners on --fast", "sweep_harness.py.*ibm04winners_on_fast"),
        ("Lane-4 (Variant A) --all", "evaluate_parallel.py.*E110_lane4_default"),
        ("Plateau (Variant B) --all", "evaluate_parallel.py.*E110_plateau_default"),
    ]
    for name, pattern in jobs:
        alive = _job_alive(pattern)
        print(f"  [{('ALIVE' if alive else 'DONE')}] {name}")

    # Sweep results
    res_dir = _ROOT / "experiments" / "E110_smooth_global_placer" / "results"

    print("\n>>> HANDPICKED SWEEP (--fast)")
    summarize_sweep(
        res_dir / "handpicked_fast.jsonl",
        res_dir / "handpicked_fast_summary.json",
    )

    print("\n>>> IBM04 DIAGNOSTIC SWEEP")
    summarize_sweep(
        res_dir / "ibm04_diag.jsonl",
        res_dir / "ibm04_diag_summary.json",
    )

    print("\n>>> IBM04-WINNERS-ON-FAST SWEEP (generalization)")
    summarize_sweep(
        res_dir / "ibm04winners_on_fast.jsonl",
        res_dir / "ibm04winners_on_fast_summary.json",
    )

    print("\n>>> VARIANT A (Lane-4) --ALL (default config)")
    summarize_all_eval("*StackedPeripheryE110Placer_*all*.json")

    print("\n>>> VARIANT B (Plateau) --ALL (default config)")
    summarize_all_eval("*StackedPeripheryE110PlateauPlacer_*all*.json")

    # Log tails
    print("\n>>> RECENT LOG ACTIVITY")
    for name, path in [
        ("handpicked", "/tmp/e110_handpicked_sweep.log"),
        ("ibm04_diag", "/tmp/e110_ibm04_diag.log"),
        ("variant_A",  "/tmp/e110_lane4_all_p4.log"),
        ("variant_B",  "/tmp/e110_plateau_all_p2.log"),
    ]:
        print(f"\n--- {name} (tail 5) ---")
        print(_tail(Path(path), n=5))

    print("\n" + "=" * 70)
    print("Next actions:")
    print("  1. Pick winning E110 config from sweep summary")
    print("  2. If Lane-4 default --all avg < 1.0525 → cross-validate on EPYC")
    print("  3. If Lane-4 default --all loses → rebuild w/ sweep winner, re-run --all")
    print("  4. Run NG45 4-bench validation")
    print("=" * 70)


if __name__ == "__main__":
    main()
