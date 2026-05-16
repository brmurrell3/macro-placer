"""Helper: run a single placer lane as a subprocess.

Usage: python _lane_runner.py <bench_name> <placer_relpath> <label> <out_pkl> <budget_s>
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
import time
from pathlib import Path


def main():
    if len(sys.argv) != 6:
        print(f"usage: {sys.argv[0]} <bench> <placer_relpath> <label> <out_pkl> <budget_s>",
              file=sys.stderr)
        sys.exit(2)
    bench_name, placer_relpath, label, out_pkl, budget_s = sys.argv[1:]
    budget = float(budget_s)

    _ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_ROOT))

    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost

    bench_dir = find_benchmark_dir(bench_name)
    benchmark, _plc = load_benchmark_from_dir(str(bench_dir))

    spec = importlib.util.spec_from_file_location(
        f"_lane_{label}", str(_ROOT / placer_relpath)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    placer_cls = None
    for name in dir(mod):
        if name.endswith("Placer") and not name.startswith("_"):
            obj = getattr(mod, name)
            if isinstance(obj, type):
                placer_cls = obj
                break
    if placer_cls is None:
        raise RuntimeError(f"No Placer class found in {placer_relpath}")

    try:
        placer = placer_cls(budget_seconds=budget)
    except TypeError:
        placer = placer_cls()

    t0 = time.time()
    placement = placer.place(benchmark)
    wall = time.time() - t0

    canon = compute_proxy_cost(placement, benchmark, _plc)

    result = {
        "label": label,
        "placement": placement,
        "proxy_cost": float(canon["proxy_cost"]),
        "overlap_count": int(canon["overlap_count"]),
        "wirelength_cost": float(canon["wirelength_cost"]),
        "density_cost": float(canon["density_cost"]),
        "congestion_cost": float(canon["congestion_cost"]),
        "wall_seconds": wall,
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(result, f)
    print(
        f"LANE {label} done: proxy={canon['proxy_cost']:.5f} "
        f"ovl={int(canon['overlap_count'])} wall={wall:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
