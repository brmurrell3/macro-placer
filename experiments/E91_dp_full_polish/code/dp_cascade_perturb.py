"""E91 B-R4: cascade-as-init → DP perturber → cascade polish.

Hypothesis: cascade's plateau (~1.06–0.99) is stuck on local minima of
discrete moves. DP's gradient on smoothed (WL + density) can shake out
non-local without being structurally aligned to canonical. Risk: DP
pulls toward wrong basin and the re-polish can't recover.

Pipeline:
  1. Load cached cascade placement (.pt from E84 results).
  2. Write Bookshelf with cascade positions as the initial .pl.
  3. Run DP for short iteration count (PERTURB_ITERS, default 100) —
     basin-escape perturbation, not full convergence.
  4. greedy_macro_legalize + project_overlaps.
  5. Cascading saddle escape with remaining budget.
  6. Compare final proxy to (a) cascade alone cached, (b) the autopsy
     cascade reference.

Usage:
  cd ~/macro-place-challenge-2026
  OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \\
    DREAMPLACE_ROOT=/home/ubuntu/DREAMPlace_cuda/install \\
    DP_DOCKER_IMAGE=dreamplace:custom \\
    PERTURB_ITERS=100 \\
    python3 experiments/E91_dp_full_polish/code/dp_cascade_perturb.py ibm10

Output: experiments/E91_dp_full_polish/results/<bench>_dp_perturb.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))
_E76 = _ROOT / "experiments" / "E76_dreamplace_integration" / "code"
sys.path.insert(0, str(_E76))
_E84 = _ROOT / "experiments" / "E84_cascading_saddle" / "code"
sys.path.insert(0, str(_E84))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.cd_core import project_overlaps
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

import tilos_to_bookshelf as bk_writer
import bookshelf_to_pt as bk_reader
from macro_legalizer import greedy_macro_legalize

from cascading_saddle import cascading_saddle_escape


def _write_bookshelf_with_init(bench, init_placement, tmp_dir):
    """Write the Bookshelf input files but with init_placement as the .pl
    positions instead of bench.macro_positions.
    """
    SCALE = float(bk_writer.SCALE)
    # Patch bench.macro_positions to the init placement before writing.
    orig_positions = bench.macro_positions.detach().clone()
    bench.macro_positions = init_placement.detach().clone()
    try:
        bk_writer._write_nodes(bench, tmp_dir)
        bk_writer._write_pl(bench, tmp_dir)
        bk_writer._write_nets(bench, tmp_dir)
        bk_writer._write_scl(bench, tmp_dir)
        bk_writer._write_wts(bench, tmp_dir)
        bk_writer._write_aux(bench, tmp_dir)
    finally:
        bench.macro_positions = orig_positions


def run_dp_perturb(bench, init_placement, perturb_iters, log) -> Tuple[Optional[torch.Tensor], float, str]:
    """Run DREAMPlace with init_placement as starting positions, for
    perturb_iters iterations (short, basin-escape mode).
    """
    dp_image = os.environ.get("DP_DOCKER_IMAGE", "dreamplace:custom")
    dp_root_host = os.environ.get("DREAMPLACE_ROOT", "/home/ubuntu/DREAMPlace_cuda/install")
    SCALE = float(bk_writer.SCALE)

    with tempfile.TemporaryDirectory(prefix=f"dpp_{bench.name}_", dir="/tmp") as tmp:
        tmp = Path(tmp)
        _write_bookshelf_with_init(bench, init_placement, tmp)
        cfg = tmp / "dp.json"
        cfg.write_text(json.dumps({
            "aux_input": f"/work/{bench.name}.aux",
            "target_density": 0.85,
            "density_weight": 8e-5,
            "gpu": int(os.environ.get("DP_USE_GPU", "1")),
            "num_threads": int(os.environ.get("DP_NUM_THREADS", "8")),
            "global_place_stages": [{
                "num_bins_x": 1024, "num_bins_y": 1024,
                "iteration": perturb_iters,
                "learning_rate": 0.005,  # smaller LR for perturbation
                "wirelength": "weighted_average", "optimizer": "nesterov",
            }],
            "legalize_flag": 0,
            "detailed_place_flag": 0,
            "stop_overflow": 0.01,  # tighter exit (won't converge fully at low iter)
            "result_dir": "/work",
        }))

        uid, gid = os.getuid(), os.getgid()
        docker_args = ["sudo", "docker", "run", "--rm",
                       "--user", f"{uid}:{gid}",
                       "-v", f"{tmp}:/work",
                       "-v", f"{dp_root_host}:/dp_install:ro",
                       "-w", "/work"]
        if int(os.environ.get("DP_USE_GPU", "1")):
            docker_args.extend(["--gpus", "all"])
        cmd = docker_args + [
            dp_image,
            "python", "/dp_install/dreamplace/Placer.py", "/work/dp.json",
        ]

        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return None, time.time() - t0, "timeout"
        wall = time.time() - t0
        if proc.returncode != 0:
            log(f"  [DP] FAILED exit={proc.returncode}")
            log(f"  [DP] stderr: {(proc.stderr or '')[-300:]}")
            return None, wall, f"dp_exit_{proc.returncode}"

        gp_pl = tmp / f"{bench.name}.gp.pl"
        if not gp_pl.exists():
            for c in tmp.rglob("*.gp.pl"):
                gp_pl = c
                break
            if not gp_pl.exists():
                return None, wall, "no_output"
        pl_map = bk_reader.parse_pl(gp_pl)
        n = bench.num_macros
        sizes = bench.macro_sizes.cpu().numpy()
        placement = bench.macro_positions.clone().detach()
        for i in range(n):
            if i in pl_map:
                llx, lly = pl_map[i]
                placement[i, 0] = llx / SCALE + float(sizes[i, 0]) / 2.0
                placement[i, 1] = lly / SCALE + float(sizes[i, 1]) / 2.0
        return placement.to(torch.float32), wall, "ok"


def main(bench_name: str, budget_s: float = 1800.0):
    log = lambda s: print(s, flush=True)
    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))
    log(f"\n=== E91 B-R4 cascade→DP-perturb→cascade on {bench_name} ===")

    # Load cached cascade
    cache_path = (_ROOT / "experiments" / "E84_cascading_saddle"
                  / "results" / f"cascade_{bench_name}.pt")
    if not cache_path.exists():
        log(f"  ABORT: no cached cascade at {cache_path}")
        return
    cache = torch.load(cache_path, weights_only=False)
    cascade_placement = cache["placement"].to(torch.float32)
    cascade_proxy = float(compute_proxy_cost(cascade_placement, bench, plc)["proxy_cost"])
    log(f"  cached cascade proxy: {cascade_proxy:.5f}")

    perturb_iters = int(os.environ.get("PERTURB_ITERS", "100"))
    log(f"  PERTURB_ITERS = {perturb_iters}")

    t_total0 = time.time()
    out_path = _HERE.parent / "results" / f"{bench_name}_dp_perturb.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1: DP perturb from cascade init
    log(f"  Phase 1: DP perturb (iter={perturb_iters}) from cascade init")
    perturbed, dp_wall, dp_status = run_dp_perturb(
        bench, cascade_placement, perturb_iters, log,
    )
    if perturbed is None:
        log(f"  ABORT: DP perturb failed ({dp_status})")
        out_path.write_text(json.dumps({
            "bench": bench_name, "status": f"dp_{dp_status}",
            "cached_cascade_proxy": cascade_proxy,
        }, indent=2))
        return
    ovl_dp = compute_overlap_metrics(perturbed, bench)["overlap_count"]
    proxy_dp = float(compute_proxy_cost(perturbed, bench, plc)["proxy_cost"])
    log(f"  Phase 1 done: wall={dp_wall:.0f}s ovl={ovl_dp} perturbed_proxy={proxy_dp:.5f} "
        f"(delta vs cascade init: {(proxy_dp - cascade_proxy):+.5f})")

    # Phase 2: greedy legalize
    log(f"  Phase 2: greedy_macro_legalize")
    t0 = time.time()
    legal, leg_stats = greedy_macro_legalize(perturbed, bench, step_size_frac=0.01)
    legal, _ = project_overlaps(legal, bench)
    ovl_leg = compute_overlap_metrics(legal, bench)["overlap_count"]
    proxy_leg = float(compute_proxy_cost(legal, bench, plc)["proxy_cost"])
    log(f"  Phase 2 done: wall={time.time() - t0:.1f}s ovl={ovl_leg} legal_proxy={proxy_leg:.5f}")

    if ovl_leg > 0:
        log(f"  WARNING: {ovl_leg} residual overlaps post-legalize")

    # Phase 3: cascading saddle polish
    remaining = budget_s - (time.time() - t_total0)
    cascade_budget_s = max(remaining - 30.0, 300.0)
    log(f"  Phase 3: cascading saddle (budget={cascade_budget_s:.0f}s)")
    t0 = time.time()
    final_state, stats = cascading_saddle_escape(
        legal, bench, plc,
        max_iters=5,
        eps_values=(0.3, 1.0, 3.0),
        polish_budget=180.0,
        total_budget_s=cascade_budget_s,
        log=lambda s: None,
    )
    final_state, _ = project_overlaps(
        final_state.to(torch.float32), bench,
    )
    final_proxy = float(compute_proxy_cost(final_state, bench, plc)["proxy_cost"])
    log(f"  Phase 3 done: wall={time.time() - t0:.0f}s final_proxy={final_proxy:.5f} "
        f"iters={stats.get('iters_run','?')}")

    total_wall = time.time() - t_total0
    result = {
        "bench": bench_name,
        "status": "ok",
        "cached_cascade_proxy": cascade_proxy,
        "perturbed_proxy": proxy_dp,
        "perturbed_ovl": int(ovl_dp),
        "legal_proxy": proxy_leg,
        "legal_ovl": int(ovl_leg),
        "final_proxy": final_proxy,
        "final_ovl": int(compute_overlap_metrics(final_state, bench)["overlap_count"]),
        "dp_wall": dp_wall,
        "total_wall": total_wall,
        "delta_vs_cached_cascade": final_proxy - cascade_proxy,
        "perturb_iters": perturb_iters,
        "cascade_saddle_iters": stats.get("iters_run", -1),
    }
    out_path.write_text(json.dumps(result, indent=2))
    pt_path = out_path.with_suffix(".pt")
    torch.save({"placement": final_state, "proxy": final_proxy}, pt_path)

    log(f"\n=== E91 B-R4 {bench_name} RESULT ===")
    log(f"  cached cascade : {cascade_proxy:.5f}")
    log(f"  + DP perturb   : {proxy_dp:.5f}  (Δ={proxy_dp - cascade_proxy:+.5f})")
    log(f"  + greedy legal : {proxy_leg:.5f}  (ovl={ovl_leg})")
    log(f"  + cascade poli : {final_proxy:.5f}")
    log(f"  delta vs cached: {final_proxy - cascade_proxy:+.5f} "
        f"({(final_proxy - cascade_proxy) / cascade_proxy * 100:+.2f}%)")
    log(f"  total wall: {total_wall:.0f}s")
    log(f"  → {'IMPROVED' if final_proxy < cascade_proxy else 'NOT IMPROVED'}")


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm10"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 1800.0
    main(bench, budget)
