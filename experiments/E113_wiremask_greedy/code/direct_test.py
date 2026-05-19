"""Direct test harness — bypass evaluate.py to use custom budget.

Tests one or more placers on a list of benches, allowing budget override.
Useful for fast iteration when default budget=3300s/bench is too slow.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def _load_placer_class(path: str):
    """Load placer class from a submission placer.py — finds the
    class via convention 'Placer = ...' alias or the LAST class with a
    'place' method (most-derived placer in the module).
    """
    spec = importlib.util.spec_from_file_location("placer_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "Placer") and isinstance(getattr(mod, "Placer"), type):
        return mod.Placer
    # Find classes DEFINED in this module (not imported).
    candidates = []
    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        if not isinstance(obj, type) or not hasattr(obj, "place"):
            continue
        # Filter to classes defined in this module's file.
        try:
            mod_file = getattr(obj, "__module__", None)
            if mod_file != "placer_under_test":
                continue
        except Exception:
            pass
        candidates.append(obj)
    if not candidates:
        raise RuntimeError(f"No placer class found in {path}")
    # Pick the most-specific class — the one not subclassed by any other.
    for c in candidates:
        if not any((other is not c and issubclass(other, c)) for other in candidates):
            return c
    return candidates[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("placers", nargs="+",
                    help="placer.py paths (submission dirs or specific files)")
    ap.add_argument("--benches", nargs="+", default=["ibm01"])
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    print(f"=" * 78)
    print(f"E113 direct test: budget={args.budget}s benches={args.benches}")
    print(f"=" * 78)

    overall = {}
    for placer_path in args.placers:
        p = Path(placer_path)
        if p.is_dir():
            p = p / "placer.py"
        cls = _load_placer_class(str(p))
        name = p.parent.name
        print(f"\n>>> Placer: {name} ({cls.__name__})")
        overall[name] = {}
        for bench_name in args.benches:
            bench_dir = find_benchmark_dir(bench_name)
            benchmark, plc = load_benchmark_from_dir(str(bench_dir))
            try:
                placer = cls(budget_seconds=args.budget, verbose=False)
            except TypeError:
                placer = cls(budget_seconds=args.budget)
            t_start = time.time()
            placement = placer.place(benchmark)
            wall = time.time() - t_start
            proxy = float(compute_proxy_cost(placement, benchmark, plc)["proxy_cost"])
            ovl = int(compute_overlap_metrics(placement, benchmark)["overlap_count"])
            overall[name][bench_name] = {"proxy": proxy, "overlaps": ovl, "wall": wall}
            print(f"  {bench_name}: proxy={proxy:.5f} ovl={ovl} wall={wall:.0f}s")
        # Per-placer summary
        valid = [v for v in overall[name].values() if v["overlaps"] == 0]
        if valid:
            avg = sum(v["proxy"] for v in valid) / len(valid)
            print(f"  AVG ({len(valid)} benches): {avg:.5f}")

    print("\n" + "=" * 78)
    print("Cross-placer summary (proxy per bench):")
    print(f"{'bench':<12}" + "".join(f"{name[:18]:>20}" for name in overall.keys()))
    for bench_name in args.benches:
        row = f"{bench_name:<12}"
        for name in overall.keys():
            v = overall[name].get(bench_name)
            if v is None:
                row += f"{'—':>20}"
            else:
                row += f"  {v['proxy']:.5f} ovl={v['overlaps']:>2d}"
        print(row)

    # Aggregated row
    row = f"{'AVG':<12}"
    for name in overall.keys():
        valid = [v for v in overall[name].values() if v["overlaps"] == 0]
        if valid:
            avg = sum(v["proxy"] for v in valid) / len(valid)
            row += f"  {avg:.5f}      "
        else:
            row += f"{'N/A':>20}"
    print(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
