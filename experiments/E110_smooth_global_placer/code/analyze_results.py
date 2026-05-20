"""Quick analyzer for all the --all results we have. Returns:
- per-bench numbers
- avg, overlaps, qualified
- vs Option C 1.0575 reference

Usage:
  uv run python experiments/E110_smooth_global_placer/code/analyze_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
RESULTS = _ROOT / "results"

OPTION_C_AVG = 1.0575  # M3-verified yesterday
OPTION_C_NG45 = 0.6893


def load(pattern):
    files = sorted(RESULTS.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def print_all_result(label, json_path):
    if not json_path or not json_path.exists():
        print(f"  [{label}] no file yet ({json_path})")
        return None
    d = json.loads(json_path.read_text())
    avg = d.get("avg_proxy_cost")
    ovl = d.get("total_overlaps")
    qual = d.get("qualified")
    runtime = d.get("total_runtime_seconds", 0)
    print(f"\n=== {label} ===  ({json_path.name})")
    print(f"  avg={avg:.5f}  overlaps={ovl}  qualified={qual}  runtime={runtime:.0f}s")
    if avg is not None:
        delta_pct = (avg - OPTION_C_AVG) / OPTION_C_AVG * 100
        print(f"  vs Option C 1.0575: Δ={avg - OPTION_C_AVG:+.5f} ({delta_pct:+.2f}%)")
    print(f"  {'bench':<8} {'proxy':>9}")
    if d.get("benchmarks"):
        for b in d["benchmarks"]:
            n = b.get("name") or b.get("benchmark") or "?"
            p = b.get("proxy_cost") or b.get("proxy")
            ps = f"{p:.5f}" if isinstance(p, float) else "NULL"
            print(f"  {n:<8} {ps:>9}")
    return avg


def print_ng45_result(label, json_path):
    if not json_path or not json_path.exists():
        print(f"  [{label}] no NG45 file yet ({json_path})")
        return None
    d = json.loads(json_path.read_text())
    avg = d.get("avg_proxy_cost")
    ovl = d.get("total_overlaps")
    qual = d.get("qualified")
    print(f"\n=== {label} (NG45) === ({json_path.name})")
    print(f"  avg={avg:.5f}  ovl={ovl}  qualified={qual}")
    if avg is not None:
        delta_pct = (avg - OPTION_C_NG45) / OPTION_C_NG45 * 100
        print(f"  vs Option C NG45 0.6893: Δ={delta_pct:+.2f}%")
    if d.get("benchmarks"):
        for b in d["benchmarks"]:
            n = b.get("name") or "?"
            p = b.get("proxy_cost") or b.get("proxy")
            ps = f"{p:.5f}" if isinstance(p, float) else "NULL"
            print(f"  {n:<14} {ps:>9}")
    return avg


def main():
    print(f"=== ANALYZE_RESULTS ({Path.cwd()}) ===")
    print(f"Reference: Option C IBM={OPTION_C_AVG:.5f}, NG45={OPTION_C_NG45:.5f}")

    results = {}
    results["Variant A default"] = print_all_result(
        "Variant A default (Lane-4)",
        load("CDLNSSACascadeStackedPeripheryE110Placer_*.json"),
    )
    results["Variant B default"] = print_all_result(
        "Variant B default (Plateau)",
        load("CDLNSSACascadeStackedPeripheryE110PlateauPlacer_*.json"),
    )
    results["Variant A ovl10"] = print_all_result(
        "Variant A ovl10 (Lane-4)",
        load("CDLNSSACascadeStackedPeripheryE110Ovl10Placer_*.json"),
    )
    results["Variant A ovl10 (jobs8)"] = print_all_result(
        "Variant A ovl10 jobs=8 (Lane-4 re-run)",
        load("CDLNSSACascadeStackedPeripheryE110Ovl10Placer_*all*.json"),
    )
    results["Variant A nocong"] = print_all_result(
        "Variant A nocong (Lane-4)",
        load("CDLNSSACascadeStackedPeripheryE110NoCongPlacer_*.json"),
    )
    results["E110Minimal 600s"] = print_all_result(
        "E110Minimal 600s (no cascade)",
        load("E110Minimal600sPlacer_*.json"),
    )
    results["E110Minimal 3300s"] = print_all_result(
        "E110Minimal 3300s (no cascade)",
        load("E110MinimalPlacer_*.json"),
    )
    # NG45 results
    results["NG45 default Lane-4"] = print_ng45_result(
        "NG45 default Lane-4",
        load("CDLNSSACascadeStackedPeripheryE110Placer_*ng45*.json")
        or load("CDLNSSACascadeStackedPeripheryE110Placer_*NG45*.json")
        or load("CDLNSSACascadeStackedPeripheryE110Placer_*.json"),
    )

    # Ranking
    print("\n=== RANKING (lower is better) ===")
    ranked = sorted(
        [(k, v) for k, v in results.items() if v is not None],
        key=lambda kv: kv[1],
    )
    for k, v in ranked:
        delta = (v - OPTION_C_AVG) / OPTION_C_AVG * 100
        marker = "WIN" if delta < -0.5 else ("TIE" if abs(delta) <= 0.5 else "LOSS")
        print(f"  {v:.5f}  {marker:<4}  Δ={delta:+.2f}%  {k}")


if __name__ == "__main__":
    main()
