"""Simplified Xplace driver: monkey-patch write_placement to dump node_pos.

Then read the dumped tensor, transform to our format, compute canonical proxy.
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
sys.path.insert(0, str(_HERE))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost, compute_overlap_metrics

from tilos_to_xplace_pt import build_design_info


def run_xplace_on_bench(bench_name: str, inner_iter: int = 10000,
                        gp_only: bool = False, num_threads: int = 4,
                        seed: int = 42, dump_path: str = "/tmp/xplace_dump.pt"):
    """Drive Xplace end-to-end on a TILOS bench; return placement as (N_macros, 2)."""
    # Load benchmark
    bench_dir = find_benchmark_dir(bench_name)
    bench, plc = load_benchmark_from_dir(str(bench_dir))
    n_macros = bench.num_macros

    # Build + save .pt
    pt_path = Path(f"/home/ubuntu/Xplace/data/cad/ispd2005/{bench_name}.pt")
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    if not pt_path.exists():
        info = build_design_info(bench, plc)
        torch.save(info, str(pt_path))

    # Ensure dummy aux exists
    Path("/tmp/dummy").touch()

    # Setup CLI args
    sys.argv = [
        "main.py",
        "--custom_path", f"benchmark:ispd2005,design_name:{bench_name},aux:/tmp/dummy,bookshelf_variety:ispd2005",
        "--load_from_raw", "False",
        "--exp_id", f"v2_{bench_name}_{int(time.time())}",
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
    ]

    cwd_orig = os.getcwd()
    os.chdir("/home/ubuntu/Xplace")
    try:
        # Capture GP output (before any legalization) — that's what we want
        from src import detail_placement
        from src import run_placement_nesterov

        captured = {}

        # Patch detail_placement_main to skip entirely + capture incoming GP pos
        def skip_dp(node_pos, gpdb, rawdb, ps, data, args, logger):
            logger.info(f"[skip_dp] capturing pre-DP node_pos shape={node_pos.shape}")
            captured['node_pos'] = node_pos.detach().cpu().clone()
            captured['data'] = data
            return node_pos, 0.0, 0.0, 0.0, 0.0
        detail_placement.detail_placement_main = skip_dp
        run_placement_nesterov.detail_placement_main = skip_dp

        # Patch GP wrapper to also capture (in case DP is off)
        original_gp = run_placement_nesterov.global_placement_main
        def patched_gp(gpdb, rawdb, ps, data, args, logger, params, gputimer=None):
            result = original_gp(gpdb, rawdb, ps, data, args, logger, params, gputimer)
            captured['node_pos'] = result[0].detach().cpu().clone()
            captured['data'] = data
            return result
        run_placement_nesterov.global_placement_main = patched_gp

        from main import get_option
        args = get_option()
        import datetime
        args.exp_id = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + args.exp_id
        args.exp_id = "{}_{}".format(args.exp_id, args.design_name)

        from utils.logger import setup_logger
        logger = setup_logger(args, sys.argv)

        from src.run_placement import run_placement_main
        run_placement_main(args, logger)
    finally:
        os.chdir(cwd_orig)

    if 'node_pos' not in captured:
        raise RuntimeError("Xplace did not produce a node_pos")

    node_pos = captured['node_pos']  # (N, 2) lower-left in scaled coords
    data = captured['data']
    die_scale = data.die_scale.detach().cpu()
    die_shift = data.die_shift.detach().cpu()
    # Xplace's node_pos is normalized to [0,1] (per write_pl); exact = node_pos*scale + shift
    exact_node_pos = node_pos * die_scale + die_shift  # (N, 2) absolute scaled
    centers_um = exact_node_pos / float(data.microns)  # μm centers
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
    placement, bench, plc = run_xplace_on_bench(
        args.bench, inner_iter=args.inner_iter,
        gp_only=args.gp_only, num_threads=args.threads, seed=args.seed,
    )
    wall = time.time() - t0
    proxy = compute_proxy_cost(placement, bench, plc)
    ovl = compute_overlap_metrics(placement, bench)["overlap_count"]
    print()
    print(f"=== XPLACE bench={args.bench} ===")
    print(f"  proxy: {float(proxy['proxy_cost']):.5f} (wl={float(proxy.get('wirelength_cost',0)):.3f} d={float(proxy.get('density_cost',0)):.3f} c={float(proxy.get('congestion_cost',0)):.3f})")
    print(f"  overlaps: {ovl}")
    print(f"  wall: {wall:.0f}s")
    # Save placement
    torch.save(placement, f"/tmp/xplace_{args.bench}.pt")
    print(f"  saved: /tmp/xplace_{args.bench}.pt")


if __name__ == "__main__":
    main()
