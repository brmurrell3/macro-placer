"""Head-to-head E111 baseline vs E122 Steiner on ibm17 at 720s budget.

Mirrors submissions/_archive/e111_minimal_ovl10_720s settings (overlap_lambda_end=10,
cd_polish=600s, budget=720s).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def run_e111():
    """Run e111_minimal baseline with ovl_lambda_end=10, cd_polish=600s, budget=720s."""
    # Import the archived base placer directly
    import importlib.util
    base_path = _ROOT / "submissions" / "_archive" / "e111_minimal" / "placer.py"
    spec = importlib.util.spec_from_file_location("_e111_base", str(base_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.E111MinimalPlacer(
        budget_seconds=720.0,
        cd_polish_s=600.0,
        overlap_lambda_end=10.0,
    )


def run_e122():
    """Run E122 Steiner placer."""
    import importlib.util
    base_path = _ROOT / "submissions" / "e111_minimal_steiner3pin" / "placer.py"
    spec = importlib.util.spec_from_file_location("_e122_placer", str(base_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Placer()


def main(bench_name: str = "ibm17"):
    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost, compute_overlap_metrics

    print(f"\n=== Head-to-head on {bench_name} ===", flush=True)
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, _ = load_benchmark_from_dir(str(bench_dir))

    results = {}
    for label, builder in [("E111_baseline", run_e111), ("E122_steiner", run_e122)]:
        print(f"\n--- {label} ---", flush=True)
        t0 = time.time()
        placer = builder()
        pos = placer.place(benchmark)
        wall = time.time() - t0
        # Reload PLC for clean eval
        _, plc = load_benchmark_from_dir(str(bench_dir))
        proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        print(f"  {label}: proxy={proxy:.5f} ovl={ovl} wall={wall:.0f}s", flush=True)
        results[label] = {"proxy": proxy, "ovl": ovl, "wall": wall}

    print(f"\n=== SUMMARY ===", flush=True)
    for label, r in results.items():
        print(f"  {label:18s} proxy={r['proxy']:.5f}  ovl={r['ovl']}  wall={r['wall']:.0f}s", flush=True)

    if "E111_baseline" in results and "E122_steiner" in results:
        diff = results["E122_steiner"]["proxy"] - results["E111_baseline"]["proxy"]
        pct = diff / max(results["E111_baseline"]["proxy"], 1e-6) * 100.0
        print(f"\n  E122 - E111: {diff:+.5f}  ({pct:+.2f}%)", flush=True)
    return results


if __name__ == "__main__":
    b = sys.argv[1] if len(sys.argv) > 1 else "ibm17"
    main(b)
