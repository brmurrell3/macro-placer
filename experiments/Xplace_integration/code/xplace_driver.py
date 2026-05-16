"""Driver: run Xplace placement on a TILOS bench, return our placement tensor.

Workflow:
  1. Build design_info .pt (via tilos_to_xplace_pt.build_design_info)
  2. Call Xplace's run_placement_main_nesterov to do GP + DP
  3. Extract final node_pos from PlaceData
  4. Convert back to our (N, 2) center-coordinate tensor (only macros, no ports)
  5. Compute canonical proxy

Usage:
  python xplace_driver.py <bench_name> [<inner_iter=10000>] [<gp_only=False>]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = Path("/home/ubuntu/macro-place-challenge-2026")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, "/home/ubuntu/Xplace")

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost, compute_overlap_metrics

# Build design_info from our benchmark
sys.path.insert(0, str(_HERE))
from tilos_to_xplace_pt import build_design_info


def run_xplace(bench_name: str, inner_iter: int = 10000,
               gp_only: bool = False, num_threads: int = 4,
               seed: int = 42) -> torch.Tensor:
    """Run Xplace on a benchmark, return placement as (N_macros, 2) μm centers."""
    # Load our benchmark
    bench_dir = find_benchmark_dir(bench_name)
    bench, plc = load_benchmark_from_dir(str(bench_dir))
    n_macros = bench.num_macros

    # Build + save design_info if not already cached
    pt_path = Path(f"/home/ubuntu/Xplace/data/cad/ispd2005/{bench_name}.pt")
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    if not pt_path.exists():
        print(f"[driver] building design_info for {bench_name}")
        info = build_design_info(bench, plc)
        torch.save(info, str(pt_path))

    # Construct Xplace args via argparse
    sys.argv = [
        "main.py",
        "--custom_path", f"benchmark:ispd2005,design_name:{bench_name},aux:/tmp/dummy,bookshelf_variety:ispd2005",
        "--load_from_raw", "False",
        "--exp_id", f"driver_{bench_name}",
        "--result_dir", "result",
        "--global_placement", "True",
        "--detail_placement", "False" if gp_only else "True",
        "--final_route_eval", "False",
        "--mixed_size", "True",
        "--gpu", "0",
        "--inner_iter", str(inner_iter),
        "--seed", str(seed),
        "--num_threads", str(num_threads),
        "--write_placement", "False",
        "--write_global_placement", "False",
    ]

    # Ensure /tmp/dummy exists (Xplace checks aux file existence)
    Path("/tmp/dummy").touch()

    # Set CWD to Xplace root for relative paths
    cwd_orig = os.getcwd()
    os.chdir("/home/ubuntu/Xplace")
    try:
        # Reuse Xplace's main but capture data
        from main import get_option
        import logging
        from utils.logger import setup_logger
        args = get_option()

        # Run timestamp + design suffix
        import datetime
        args.exp_id = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + args.exp_id
        args.exp_id = "{}_{}".format(args.exp_id, args.design_name)

        # Get logger
        logger = setup_logger(args, sys.argv)

        # Load dataset via Xplace's load_dataset (uses our .pt)
        from src.database import load_dataset
        from utils.setup_dataset import find_design_params
        params = find_design_params(args, logger)
        data, rawdb, gpdb = load_dataset(args, logger, params)

        # Run placement
        from src.run_placement_nesterov import global_placement_main, detail_placement_main
        t0 = time.time()
        node_pos, iteration, gp_hpwl, overflow, gp_time, gp_per_iter = global_placement_main(
            args, logger, data, args.density_weight if hasattr(args, 'density_weight') else 8e-5,
            args.wa_coeff if hasattr(args, 'wa_coeff') else 4.0,
            None  # init_density_map
        )
        print(f"[driver] GP done: hpwl={gp_hpwl:.3e} overflow={overflow:.3f} time={gp_time:.1f}s")
        if not gp_only:
            try:
                node_pos, dp_hpwl, top5overflow, lg_time, dp_time = detail_placement_main(
                    args, logger, data, gpdb, node_pos, iteration
                )
                print(f"[driver] DP done: hpwl={dp_hpwl:.3e} top5={top5overflow:.3f}")
            except Exception as e:
                print(f"[driver] DP failed (using GP result): {e}")
        total_wall = time.time() - t0
    finally:
        os.chdir(cwd_orig)

    # node_pos is (N, 2) lower-left in scaled (die_scale) coords. Convert to our μm centers.
    # data.die_scale / die_shift give the affine transform.
    node_pos_cpu = node_pos.detach().cpu()
    # Xplace stores positions as (N, 2) normalized to [0,1] × die_scale + die_shift
    # data.node_size has the size in absolute scaled units; cpos = lpos + size/2
    node_size_abs = data.node_size.detach().cpu() * data.die_scale.cpu()
    die_scale = data.die_scale.detach().cpu()
    die_shift = data.die_shift.detach().cpu()
    # node_pos in PlaceData is "node_pos" which after update from optimizer is...
    # Let's read data.write_pl source to confirm: exact_node_pos = node_pos * die_scale + die_shift
    exact_node_pos = node_pos_cpu * die_scale + die_shift  # (N, 2) absolute scaled cpos
    # Convert from scaled units back to μm: divide by microns (=1000)
    centers_um = exact_node_pos / float(data.microns)

    # Take only the macro indices (first n_macros are Mov type per our build)
    macro_centers = centers_um[:n_macros, :]
    return macro_centers, bench, plc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--inner_iter", type=int, default=10000)
    ap.add_argument("--gp_only", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    placement, bench, plc = run_xplace(
        args.bench, inner_iter=args.inner_iter,
        gp_only=args.gp_only, num_threads=args.threads, seed=args.seed
    )
    wall = time.time() - t0

    proxy = compute_proxy_cost(placement, bench, plc)
    ovl = compute_overlap_metrics(placement, bench)["overlap_count"]
    print()
    print(f"=== XPLACE RESULT bench={args.bench} ===")
    print(f"  proxy_cost: {proxy['proxy_cost']:.5f}")
    print(f"  wirelength: {float(proxy.get('wirelength_cost', 0)):.4f}")
    print(f"  density:    {float(proxy.get('density_cost', 0)):.4f}")
    print(f"  congestion: {float(proxy.get('congestion_cost', 0)):.4f}")
    print(f"  overlaps:   {ovl}")
    print(f"  wall:       {wall:.0f}s")


if __name__ == "__main__":
    main()
