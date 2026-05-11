"""P1 integrated test: DP → custom legalize → CD polish → measure.

Combines:
  - DP (stock config, ~15s)
  - macro_legalize (greedy, deterministic, <1s, guarantees ~0 overlaps)
  - project_overlaps (cleanup any residual)
  - run_cd_adaptive (300-900s polish)

Output: final proxy + overlap count per bench. Compare to E25 cached.
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
from macro_legalizer import greedy_macro_legalize


def run_dp(bench, plc):
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
            return None, f"DP exited {proc.returncode}: {proc.stderr[-200:]}"
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


def main(bench_name: str, cd_cap: float = 900.0):
    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))
    print(f"\n=== {bench_name} (n_macros={bench.num_macros} n_hard={bench.num_hard_macros}) ===", flush=True)

    # 1. DP
    print(f"  [1] DP...", flush=True)
    out = run_dp(bench, plc)
    if out is None or out[0] is None:
        print(f"  DP FAILED: {out[1] if out else 'unknown'}", flush=True)
        return
    dp_placement, dp_wall = out
    ovl_dp = compute_overlap_metrics(dp_placement, bench)["overlap_count"]
    print(f"  [1] DP done in {dp_wall:.0f}s, ovl={ovl_dp}", flush=True)

    # 2. greedy legalize
    print(f"  [2] greedy legalize...", flush=True)
    t0 = time.time()
    legal, leg_stats = greedy_macro_legalize(dp_placement, bench, step_size_frac=0.01)
    leg_wall = time.time() - t0
    ovl_leg = compute_overlap_metrics(legal, bench)["overlap_count"]
    proxy_leg = float(compute_proxy_cost(legal, bench, plc)["proxy_cost"])
    print(f"  [2] legalize: ovl={ovl_leg} proxy={proxy_leg:.5f} moved={leg_stats['n_moved']} "
          f"failed={leg_stats['n_failed']} wall={leg_wall:.2f}s", flush=True)

    # 3. project_overlaps cleanup (if any residual)
    if ovl_leg > 0:
        legal, _ = project_overlaps(legal, bench)
        ovl_leg = compute_overlap_metrics(legal, bench)["overlap_count"]
        print(f"  [2b] post-project: ovl={ovl_leg}", flush=True)

    # 4. CD polish
    print(f"  [3] CD polish (min_time_s=300, hard_cap={cd_cap}s)...", flush=True)
    t0 = time.time()
    ev = IncrementalProxyEvaluator(bench, plc, legal.clone())
    n_hard = bench.num_hard_macros
    fixed = bench.macro_fixed.cpu().numpy()
    movable = [i for i in range(n_hard) if not bool(fixed[i])]
    run_cd_adaptive(
        ev, bench, plc, movable,
        min_time_s=300.0, hard_cap_s=cd_cap,
        patience=3, plateau_threshold=0.001, log_fn=None,
    )
    cd_wall = time.time() - t0
    polished = ev.placement.detach().clone().to(torch.float32)
    polished, _ = project_overlaps(polished, bench)
    ovl_final = compute_overlap_metrics(polished, bench)["overlap_count"]
    proxy_final = float(compute_proxy_cost(polished, bench, plc)["proxy_cost"])
    total_wall = dp_wall + leg_wall + cd_wall

    print(f"  RESULT: {bench_name}", flush=True)
    print(f"    DP-legalize-polish: ovl={ovl_final} proxy={proxy_final:.5f}", flush=True)
    print(f"    total wall: {total_wall:.0f}s = DP {dp_wall:.0f} + legal {leg_wall:.1f} + CD {cd_wall:.0f}", flush=True)


if __name__ == "__main__":
    benches = sys.argv[1:] or ["ibm01"]
    for b in benches:
        main(b)
