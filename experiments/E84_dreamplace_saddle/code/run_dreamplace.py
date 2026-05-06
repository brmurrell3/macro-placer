"""E84 helper: run DREAMPlace via Docker on a Bookshelf input.

Spawns the limbo018/dreamplace:cuda container with the project mounted,
generates a JSON config that points at our converted Bookshelf files, runs
DREAMPlace (GPU), reads back the .pl output, returns macro positions.

Returned positions are in OUR microns (Benchmark coordinate space).

Wall budget on RTX 4070: <30 sec for ibm01-class benchmarks. The Hessian
saddle escape we apply on top is the bulk of the wall. Total target wall
per bench: ~20 min, fits cap with massive headroom.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from macro_place.benchmark import Benchmark

# This file may be imported alongside benchmark_to_bookshelf.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_to_bookshelf import write_bookshelf, read_bookshelf_pl

DREAMPLACE_HOST_DIR = (
    _REPO_ROOT / "experiments" / "E84_dreamplace_saddle" / "DREAMPlace"
)
DREAMPLACE_IMAGE = "macro-placer/dreamplace:built"  # built locally with deps pre-installed


def _make_dp_config(
    bookshelf_aux: Path,
    result_dir: Path,
    work_dir: Path,
    gpu: bool = True,
) -> Path:
    """Generate a DREAMPlace JSON config pointing at our bookshelf input.

    Paths must be CONTAINER paths (under /work/...) since DREAMPlace runs
    inside the container. The host path under work_dir maps to /work/...
    """
    aux_in_container = "/work/" + str(bookshelf_aux.relative_to(work_dir)).replace("\\", "/")
    result_in_container = "/work/" + str(result_dir.relative_to(work_dir)).replace("\\", "/")
    config = {
        "aux_input": aux_in_container,
        "gpu": 1 if gpu else 0,
        "num_threads": 8,
        "deterministic_flag": 1,
        "result_dir": result_in_container,
        "global_place_stages": [
            {
                "num_bins_x": 512,
                "num_bins_y": 512,
                "iteration": 1000,
                "learning_rate": 0.01,
                "wirelength": "weighted_average",
                "optimizer": "nesterov",
            }
        ],
        "target_density": 1.0,
        "density_weight": 8e-5,
        "random_seed": 1000,
        "scale_factor": 0.0,  # auto
        "shift_factor": [0, 0],
        "ignore_net_degree": 100,
        "gp_noise_ratio": 0.025,
        "enable_fillers": 1,
        "global_place_flag": 1,
        "legalize_flag": 1,        # produce a legal-ish placement
        "detailed_place_flag": 0,  # skip detailed (we'll Hessian-polish on top)
        # macro_place_flag: 0 keeps the better basin (proxy 1.05 vs 1.29 in macro mode).
        # We'll fully legalize ourselves via CD.
        "abacus_legalize_flag": 1,
        "stop_overflow": 0.1,
        "dtype": "float32",
        "plot_flag": 0,
        "RePlAce_ref_hpwl": 350000,
        "RePlAce_LOWER_PCOF": 0.95,
        "RePlAce_UPPER_PCOF": 1.05,
        "random_center_init_flag": 1,
        "sort_nets_by_degree": 0,
        "num_threads": 4,
        "dump_global_place_solution_flag": 0,
    }
    cfg_path = result_dir / f"{bookshelf_aux.stem}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    return cfg_path


def run_dreamplace(
    benchmark: Benchmark,
    work_dir: Path,
    scale: int = 1000,
    gpu: bool = True,
    log_fn=None,
) -> torch.Tensor:
    """Run DREAMPlace on `benchmark`; return [num_macros, 2] tensor of placed centers.

    Steps:
      1. Convert benchmark → Bookshelf in `work_dir/<name>/bookshelf/`.
      2. Generate JSON config for DREAMPlace.
      3. Spawn docker container with /DREAMPlace mounted (built install) +
         work_dir mounted, run dreamplace/Placer.py.
      4. Read back the .pl output, convert to centers tensor in microns.
    """
    log = log_fn or (lambda s: print(s, flush=True))
    name = benchmark.name
    bench_dir = work_dir / name
    bookshelf_dir = bench_dir / "bookshelf"
    result_dir = bench_dir / "results"

    # 1. Convert.
    log(f"  [E84] writing bookshelf to {bookshelf_dir}")
    aux_path = write_bookshelf(benchmark, bookshelf_dir, scale=scale)

    # 2. JSON config.
    cfg_path = _make_dp_config(aux_path, result_dir, work_dir, gpu=gpu)
    log(f"  [E84] config: {cfg_path}")

    # 3. Run DREAMPlace inside docker.
    # Volume mounts:
    #   - DREAMPlace install (built C++ ops + Python module): /DREAMPlace
    #   - our work_dir (bookshelf input + result output):     /work
    cfg_in_container = "/work/" + str(cfg_path.relative_to(work_dir)).replace("\\", "/")
    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all" if gpu else "0",
        "-v", f"{DREAMPLACE_HOST_DIR}:/DREAMPlace",
        "-v", f"{work_dir}:/work",
        # Run from install/ — Placer.py expects relative paths starting from
        # the install dir.
        "-w", "/DREAMPlace/install",
        "-e", "PYTHONPATH=/DREAMPlace/install:/DREAMPlace",
        DREAMPLACE_IMAGE,
        "python", "dreamplace/Placer.py", cfg_in_container,
    ]
    env = os.environ.copy()
    env["MSYS_NO_PATHCONV"] = "1"  # Git Bash compatibility
    log(f"  [E84] launching docker run for DREAMPlace ({benchmark.name})")
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    log(f"  [E84] DREAMPlace wall = {wall:.1f}s, exit {proc.returncode}")
    if proc.returncode != 0:
        log("  [E84] STDOUT (last 30 lines):")
        for line in proc.stdout.splitlines()[-30:]:
            log("    " + line)
        log("  [E84] STDERR (last 30 lines):")
        for line in proc.stderr.splitlines()[-30:]:
            log("    " + line)
        raise RuntimeError(f"DREAMPlace failed (exit {proc.returncode})")

    # 4. Read .pl output. DREAMPlace writes <name>.gp.pl (global place) and
    # <name>.lg.pl (legalized) in result_dir/<name>/. Prefer legalized.
    candidates = []
    for pattern in ["**/*.lg.pl", "**/*.gp.pl"]:
        candidates.extend(result_dir.glob(pattern))
    if not candidates:
        raise RuntimeError(
            f"DREAMPlace output .pl not found in {result_dir}; "
            f"contents: {list(result_dir.iterdir())}"
        )
    # Sort lg.pl first (more legalized = preferred).
    candidates.sort(key=lambda p: 0 if ".lg." in p.name else 1)
    output_pl = candidates[0]
    log(f"  [E84] reading {output_pl}")
    placed = read_bookshelf_pl(output_pl, scale=scale)

    # Convert lower-left → centers in microns.
    n = benchmark.num_macros
    out_xy = benchmark.macro_positions.detach().clone()
    sizes = benchmark.macro_sizes
    for i in range(n):
        mname = benchmark.macro_names[i] if i < len(benchmark.macro_names) else f"m{i}"
        mname = mname.replace(" ", "_").replace("\t", "_")
        if mname in placed:
            llx, lly = placed[mname]
            cx = llx + float(sizes[i, 0]) / 2.0
            cy = lly + float(sizes[i, 1]) / 2.0
            out_xy[i, 0] = cx
            out_xy[i, 1] = cy

    return out_xy


if __name__ == "__main__":
    # Smoke-test path: run DREAMPlace on ibm01 and report.
    import macro_place.loader as _loader_mod
    _orig_load_benchmark = _loader_mod.load_benchmark

    def _patched(netlist_file, plc_file=None, name=None):
        netlist_file = str(netlist_file).replace("\\", "/")
        if plc_file is not None:
            plc_file = str(plc_file).replace("\\", "/")
        return _orig_load_benchmark(netlist_file, plc_file, name)

    _loader_mod.load_benchmark = _patched

    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

    bench_name = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
    work_dir = (
        _REPO_ROOT / "experiments" / "E84_dreamplace_saddle" / "dp_runs"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    placed = run_dreamplace(benchmark, work_dir)
    proxy = float(compute_proxy_cost(placed, benchmark, plc)["proxy_cost"])
    ovl = compute_overlap_metrics(placed, benchmark)["overlap_count"]
    print(f"\n=== {bench_name} DREAMPlace result ===")
    print(f"  proxy = {proxy:.5f}")
    print(f"  overlaps = {ovl}")
