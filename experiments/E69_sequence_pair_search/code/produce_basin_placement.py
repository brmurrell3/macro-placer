"""Run E25 or E41 placer on a benchmark and save the placement to .pt.

Usage:
  uv run python experiments/E69_sequence_pair_search/code/produce_basin_placement.py \
      <e25|e41> <bench_name> <out_path>

This is a minimal wrapper: load benchmark + plc, run placer.place(), save
{placement, proxy, overlap_count, bench_name, placer_name, wall_seconds}
to a torch.save dict at out_path. The diagnostic script reads these.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


def _load_placer(placer_name: str):
    if placer_name == "e25":
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location(
            "e25_placer", str(_ROOT / "submissions" / "cd_lns_sa" / "placer.py")
        )
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.CDLNSSAPlacer
    if placer_name == "e41":
        # Dynamic import to keep e25 path-only loading working.
        from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import (
            CDLNSSADPOKJointPlacer,
        )
        return CDLNSSADPOKJointPlacer
    raise ValueError(f"unknown placer: {placer_name}")


def main() -> None:
    if len(sys.argv) != 4:
        print("usage: produce_basin_placement.py <e25|e41> <bench> <out.pt>")
        sys.exit(2)
    placer_name = sys.argv[1]
    bench_name = sys.argv[2]
    out_path = Path(sys.argv[3])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[E69] loading {bench_name}...", flush=True)
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    print(f"[E69] running {placer_name} on {bench_name}...", flush=True)
    placer_cls = _load_placer(placer_name)
    placer = placer_cls()
    t0 = time.time()
    placement = placer.place(benchmark)
    wall = time.time() - t0

    metrics = compute_proxy_cost(placement, benchmark, plc)
    overlap = metrics["overlap_count"]
    proxy = metrics["proxy_cost"]
    print(
        f"[E69] DONE {placer_name}/{bench_name}: proxy={proxy:.5f} "
        f"overlap={overlap} wall={wall:.1f}s", flush=True
    )

    torch.save(
        {
            "placement": placement.detach().cpu(),
            "proxy": float(proxy),
            "overlap_count": int(overlap),
            "wall_seconds": float(wall),
            "bench_name": bench_name,
            "placer_name": placer_name,
            "macro_sizes": benchmark.macro_sizes.detach().cpu(),
            "num_hard_macros": int(benchmark.num_hard_macros),
            "canvas_width": float(benchmark.canvas_width),
            "canvas_height": float(benchmark.canvas_height),
        },
        out_path,
    )
    print(f"[E69] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
