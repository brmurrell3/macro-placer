"""Native DREAMPlace runner with B-R2 canonical loss support.

Reads BR2_LAMBDA_TOPK / BR2_LAMBDA_RUDY from env (read by patched PlaceObj.py).
Otherwise behaves like stock DREAMPlace.

Usage:
    python3 native_dp_runner.py <bench_name>

  e.g. BR2_LAMBDA_TOPK=1.0 BR2_LAMBDA_RUDY=0.5 python3 native_dp_runner.py ibm10
"""
from __future__ import annotations

import os
import sys
import time
import json
import shutil
from pathlib import Path

# Ensure macro_place package is importable
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost, compute_overlap_metrics

# E76 utilities for bookshelf conversion and macro legalize
sys.path.insert(0, str(ROOT / "experiments" / "E76_dreamplace_integration" / "code"))
import tilos_to_bookshelf as _t2b
import bookshelf_to_pt as _b2pt
from macro_legalizer import greedy_macro_legalize


def read_placement(pl_path: str, benchmark) -> torch.Tensor:
    d = _b2pt.parse_pl(Path(pl_path))
    SCALE = 1000.0
    pos = torch.zeros(benchmark.num_macros, 2, dtype=torch.float32)
    for i in range(benchmark.num_macros):
        if i in d:
            llx, lly = d[i]
            w, h = float(benchmark.macro_sizes[i, 0]), float(benchmark.macro_sizes[i, 1])
            pos[i, 0] = llx / SCALE + w / 2.0
            pos[i, 1] = lly / SCALE + h / 2.0
        else:
            pos[i] = benchmark.macro_positions[i]
    return pos


def write_bookshelf(benchmark, out_dir):
    _t2b._write_nodes(benchmark, Path(out_dir))
    _t2b._write_pl(benchmark, Path(out_dir))
    _t2b._write_nets(benchmark, Path(out_dir))
    _t2b._write_scl(benchmark, Path(out_dir))
    _t2b._write_wts(benchmark, Path(out_dir))
    _t2b._write_aux(benchmark, Path(out_dir))


def run_native_dp(bench_name: str, target_density: float = 0.85,
                  stop_overflow: float = 0.07, dp_iter: int = 1000,
                  dp_lr: float = 0.01, work_dir: Path = None) -> dict:
    if work_dir is None:
        work_dir = Path(f"/tmp/br2_dp_{bench_name}")
    work_dir.mkdir(parents=True, exist_ok=True)

    bench_dir = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench_name
    bench, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"[runner] {bench_name}: num_hard={bench.num_hard_macros} num_macros={bench.num_macros}")

    init_costs = compute_proxy_cost(bench.macro_positions, bench, plc)
    print(f"[runner] init canonical proxy={init_costs['proxy_cost']:.5f}")

    # Write bookshelf to work_dir
    bookshelf_dir = work_dir / "bookshelf"
    bookshelf_dir.mkdir(exist_ok=True)
    write_bookshelf(bench, bookshelf_dir)
    print(f"[runner] bookshelf written: {bookshelf_dir}")

    # Write DREAMPlace JSON config
    auxfile = bookshelf_dir / f"{bench.name}.aux"
    dp_config = {
        "aux_input": str(auxfile),
        "target_density": float(target_density),
        "density_weight": 8e-5,
        "gpu": 0,
        "num_threads": 4,
        "global_place_stages": [{
            "num_bins_x": 1024,
            "num_bins_y": 1024,
            "iteration": int(dp_iter),
            "learning_rate": float(dp_lr),
            "wirelength": "weighted_average",
            "optimizer": "nesterov",
            "Llambda_density_weight_iteration": 1,
            "Lsub_iteration": 1,
        }],
        "result_dir": str(work_dir / "results"),
        "RePlAce_LOWER_PCOF": 0.95,
        "RePlAce_UPPER_PCOF": 1.05,
        "RePlAce_ref_hpwl": 350000.0,
        "stop_overflow": float(stop_overflow),
        "macro_place_flag": 0,
        "routability_opt_flag": 0,
        "legalize_flag": 0,
        "detailed_place_flag": 0,
        "random_seed": 42,
        "global_place_flag": 1,
        "scale_factor": 1.0,
        "shift_factor": [0.0, 0.0],
        "ignore_net_degree": 100,
    }
    config_file = work_dir / "dp.json"
    with open(config_file, "w") as f:
        json.dump(dp_config, f, indent=2)

    # Construct DP placer manually so we can set br2 lambdas on PlaceObj
    dp_install = Path(os.environ.get("DREAMPLACE_ROOT", "/home/ubuntu/DREAMPlace_cpu/install"))
    sys.path.insert(0, str(dp_install))
    sys.path.insert(0, str(dp_install / "dreamplace"))
    # Set canonical_losses importable
    sys.path.insert(0, str(ROOT / "experiments" / "B_R2_canonical_dp_loss" / "code"))

    import dreamplace.Params as Params
    import dreamplace.PlaceDB as PlaceDB
    import dreamplace.NonLinearPlace as NonLinearPlace

    params = Params.Params()
    params.load(str(config_file))

    placedb = PlaceDB.PlaceDB()
    placedb(params)
    print(f"[runner] PlaceDB ready: {placedb.num_movable_nodes} movable nodes, "
          f"{placedb.num_filler_nodes} fillers")

    t0 = time.time()
    placer = NonLinearPlace.NonLinearPlace(params, placedb, None)
    metrics = placer(params, placedb, params.global_place_stages[0]["learning_rate"])
    dp_wall = time.time() - t0
    print(f"[runner] DP optimization wall: {dp_wall:.1f}s")

    # Read placement from placedb in-memory state (lower-left corner stored as
    # placedb.node_x / placedb.node_y; we need centers).
    placement = torch.zeros(bench.num_macros, 2, dtype=torch.float32)
    # Note: DP scales positions by SCALE=1000 (matches our bookshelf SCALE).
    # And reorders movable→fixed; placedb has a permutation.
    # Simpler: write the .pl file then read.
    out_dir = Path(dp_config["result_dir"]) / bench.name
    out_dir.mkdir(parents=True, exist_ok=True)
    gp_out = out_dir / f"{bench.name}.gp.pl"
    placedb.write(params, str(gp_out))
    print(f"[runner] reading placement: {gp_out}")
    placement = read_placement(str(gp_out), bench)
    raw_proxy = compute_proxy_cost(placement, bench, plc)
    raw_ovl = compute_overlap_metrics(placement, bench)['overlap_count']
    print(f"[runner] raw DP basin: proxy={raw_proxy['proxy_cost']:.5f} ovl={raw_ovl}")

    # Greedy macro legalize
    legal, n_iter = greedy_macro_legalize(placement, bench)
    legal_proxy = compute_proxy_cost(legal, bench, plc)
    legal_ovl = compute_overlap_metrics(legal, bench)['overlap_count']
    print(f"[runner] post-legalize: proxy={legal_proxy['proxy_cost']:.5f} ovl={legal_ovl} (n_iter={n_iter})")

    return {
        'bench': bench_name,
        'init_proxy': float(init_costs['proxy_cost']),
        'dp_basin_proxy': float(raw_proxy['proxy_cost']),
        'dp_basin_ovl': int(raw_ovl),
        'legal_proxy': float(legal_proxy['proxy_cost']),
        'legal_ovl': int(legal_ovl),
        'dp_wall': float(dp_wall),
        'lambda_topk': float(os.environ.get('BR2_LAMBDA_TOPK', 0.0)),
        'lambda_rudy': float(os.environ.get('BR2_LAMBDA_RUDY', 0.0)),
    }


if __name__ == '__main__':
    bench = sys.argv[1] if len(sys.argv) > 1 else 'ibm10'
    result = run_native_dp(bench)
    print(f"\n=== RESULT ===")
    print(json.dumps(result, indent=2))
