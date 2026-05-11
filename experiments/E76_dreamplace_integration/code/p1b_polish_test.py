"""P1b: test DP → extended CD polish to measure DP-basin final proxy.

For each bench:
  1. Run DP (stock config: stop_overflow=0.07, iteration=1000)
  2. project_overlaps (50 iters default)
  3. run_cd_adaptive(min_time_s=300, hard_cap_s=900) → DP-polished basin
  4. compute final proxy + overlap_count

Compare to E25 cached plateau (~0.89 on ibm01) and E48 reference (~1.08 aggregate).

Goal: validate that the polished DP basin is competitive (within 2% of E25)
on at least ONE bench, justifying full --all run.
"""
import os, sys, time, json, tempfile, subprocess
os.environ["DREAMPLACE_ROOT"] = "/opt/DREAMPlace/install"
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

import torch
from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_proxy_cost, compute_overlap_metrics
import tilos_to_bookshelf as bk_writer
import bookshelf_to_pt as bk_reader


def run_dp(bench, plc):
    """Run DP with stock config; return (placement_torch_float32, dp_wall) or (None, msg)."""
    dp_root = "/opt/DREAMPlace/install"
    placer_py = f"{dp_root}/dreamplace/Placer.py"
    SCALE = float(bk_writer.SCALE)
    with tempfile.TemporaryDirectory(prefix=f"dp_{bench.name}_") as tmp:
        tmp = Path(tmp)
        bk_writer._write_nodes(bench, tmp); bk_writer._write_pl(bench, tmp)
        bk_writer._write_nets(bench, tmp); bk_writer._write_scl(bench, tmp)
        bk_writer._write_wts(bench, tmp); bk_writer._write_aux(bench, tmp)
        cfg = tmp / "dp.json"
        cfg.write_text(json.dumps({
            "aux_input": str(tmp / (bench.name + ".aux")),
            "target_density": 0.85, "density_weight": 8e-5, "gpu": 1,
            "global_place_stages": [{"num_bins_x": 1024, "num_bins_y": 1024,
                "iteration": 1000, "learning_rate": 0.01,
                "wirelength": "weighted_average", "optimizer": "nesterov"}],
            "legalize_flag": 0, "detailed_place_flag": 0,
            "stop_overflow": 0.07, "result_dir": str(tmp),
        }))
        t0 = time.time()
        proc = subprocess.run(["/usr/bin/python3", placer_py, str(cfg)],
            cwd=dp_root, capture_output=True, text=True, timeout=600)
        wall = time.time() - t0
        if proc.returncode != 0:
            return None, f"DP exited {proc.returncode}: {proc.stderr[-300:]}"
        gp_pl = tmp / f"{bench.name}.gp.pl"
        if not gp_pl.exists():
            for c in tmp.rglob("*.gp.pl"):
                gp_pl = c; break
        pl_map = bk_reader.parse_pl(gp_pl)
        n = bench.num_macros
        sizes = bench.macro_sizes.cpu().numpy()
        placement = bench.macro_positions.clone().detach()
        for i in range(n):
            if i in pl_map:
                llx, lly = pl_map[i]
                placement[i, 0] = llx / SCALE + float(sizes[i, 0]) / 2.0
                placement[i, 1] = lly / SCALE + float(sizes[i, 1]) / 2.0
        return placement.to(torch.float32), wall


def main(bench_name: str):
    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))
    print(f"\n=== {bench_name} (n_macros={bench.num_macros} n_hard={bench.num_hard_macros}) ===", flush=True)

    # Step 1: DP
    print(f"  [1] running DP (stock)...", flush=True)
    out = run_dp(bench, plc)
    if out is None or out[0] is None:
        print(f"  DP FAILED: {out[1] if out else 'unknown'}", flush=True)
        return
    dp_placement, dp_wall = out
    print(f"  [1] DP done in {dp_wall:.0f}s", flush=True)

    # Step 2: project_overlaps
    t0 = time.time()
    dp_proj, _ = project_overlaps(dp_placement, bench)
    proj_wall = time.time() - t0
    ovl_proj = compute_overlap_metrics(dp_proj, bench)["overlap_count"]
    proxy_proj = float(compute_proxy_cost(dp_proj, bench, plc)["proxy_cost"])
    print(f"  [2] project_overlaps: ovl={ovl_proj} proxy={proxy_proj:.5f} ({proj_wall:.0f}s)", flush=True)

    # Step 3: extended CD polish (min_time_s=300, hard_cap=900)
    print(f"  [3] CD polish (min_time_s=300, hard_cap=900s)...", flush=True)
    t0 = time.time()
    ev = IncrementalProxyEvaluator(bench, plc, dp_proj.clone())
    n_hard = bench.num_hard_macros
    fixed = bench.macro_fixed.cpu().numpy()
    movable = [i for i in range(n_hard) if not bool(fixed[i])]
    info = run_cd_adaptive(
        ev, bench, plc, movable,
        min_time_s=300.0, hard_cap_s=900.0,
        patience=3, plateau_threshold=0.001, log_fn=None,
    )
    cd_wall = time.time() - t0
    polished = ev.placement.detach().clone().to(torch.float32)
    polished, _ = project_overlaps(polished, bench)
    ovl_final = compute_overlap_metrics(polished, bench)["overlap_count"]
    proxy_final = float(compute_proxy_cost(polished, bench, plc)["proxy_cost"])
    print(f"  [3] CD polish: ovl={ovl_final} proxy={proxy_final:.5f} ({cd_wall:.0f}s)", flush=True)

    delta_from_e25 = proxy_final - 0.89  # ibm01 E25 reference
    print(f"  RESULT: {bench_name} DP→polish proxy={proxy_final:.5f} ovl={ovl_final} "
          f"total_wall={dp_wall+proj_wall+cd_wall:.0f}s", flush=True)


if __name__ == "__main__":
    benches = sys.argv[1:] or ["ibm01"]
    for b in benches:
        main(b)
