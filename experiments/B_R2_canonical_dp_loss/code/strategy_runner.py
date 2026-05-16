"""Run a specified placer on a bench, save placement + proxy to disk.

Used by ensemble_placer.py as the subprocess entry point.

Usage:
  python3 strategy_runner.py <placer_module_path> <placer_class> <bench> <out_path>
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/ubuntu/macro-place-challenge-2026")
sys.path.insert(0, str(ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost, compute_overlap_metrics


def main():
    placer_path = sys.argv[1]
    placer_class = sys.argv[2]
    bench_name = sys.argv[3]
    out_path = Path(sys.argv[4])

    # Load placer module
    spec = importlib.util.spec_from_file_location("strategy_placer", placer_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    PlacerCls = getattr(mod, placer_class)

    # Load bench
    bench_dir = find_benchmark_dir(bench_name)
    bench, plc = load_benchmark_from_dir(str(bench_dir))

    t0 = time.time()
    placer = PlacerCls()
    placement = placer.place(bench)
    wall = time.time() - t0

    # Eval
    proxy_d = compute_proxy_cost(placement, bench, plc)
    ovl = compute_overlap_metrics(placement, bench)['overlap_count']

    # Save placement + stats
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(placement, str(out_path) + ".pt")
    with open(str(out_path) + ".json", "w") as f:
        json.dump({
            "bench": bench_name,
            "proxy": float(proxy_d['proxy_cost']),
            "wirelength": float(proxy_d.get('wirelength_cost', 0)),
            "density": float(proxy_d.get('density_cost', 0)),
            "congestion": float(proxy_d.get('congestion_cost', 0)),
            "overlaps": int(ovl),
            "wall": wall,
        }, f, indent=2)

    print(f"[strategy] {bench_name} proxy={proxy_d['proxy_cost']:.5f} ovl={ovl} wall={wall:.0f}s")


if __name__ == '__main__':
    main()
