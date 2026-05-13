"""E96: DP config sensitivity probe.

Question: does varying DREAMPlace config (target_density, density_weight, lr,
iter, noise, seed) produce meaningfully different basin/legalize quality
on the 4 hard IBM benches?

The B-R0' result showed:
- ibm10 DP basin 1.215, legalize 1.444 (+0.229 damage), polished 1.095 (+1.6% vs cascade-capped 1.0775)
- ibm12 basin 1.442, legalize 1.477 (+0.035), polished 1.129 (-13.3% vs 1.3031)
- ibm14 basin 1.494, legalize 1.523 (+0.028), polished 1.243 (-3.8%)
- ibm17 basin 1.543, legalize 1.572 (+0.029), polished 1.307 (-10.2%)

ibm10's legalize damage is 8× the others. If varying config reduces this damage,
multi-config selection (choose best legalize_proxy per bench) is a high-EV layer
to add to the hybrid placer.

Test budget: 8 configs × 4 benches = 32 DP runs.
Per-run estimated wall: 20-40s DP + 5s legalize on CPU lambda.ai box.
Total: ~25 min wall.

Usage on cloud:
  cd ~/macro-place-challenge-2026
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
    DREAMPLACE_ROOT=/home/ubuntu/DREAMPlace_cpu/install \\
    DP_DOCKER_IMAGE=dreamplace:custom DP_USE_GPU=0 \\
    python3 experiments/E96_multi_config_dp/code/dp_config_probe.py ibm10
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))
_E76 = _ROOT / "experiments" / "E76_dreamplace_integration" / "code"
sys.path.insert(0, str(_E76))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.cd_core import project_overlaps
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

import tilos_to_bookshelf as bk_writer
import bookshelf_to_pt as bk_reader
from macro_legalizer import greedy_macro_legalize


def _auto_target_density(bench) -> float:
    sizes = bench.macro_sizes.cpu().numpy()
    hard_idx = bench.num_hard_macros
    macro_area = float((sizes[:hard_idx, 0] * sizes[:hard_idx, 1]).sum())
    canvas_area = float(bench.canvas_width * bench.canvas_height)
    macro_density = macro_area / max(canvas_area, 1e-9)
    return float(min(0.85, max(0.40, macro_density * 1.5)))


def build_config_grid(bench) -> List[Dict]:
    """8 config variants per bench, varying along salient axes."""
    auto_td = _auto_target_density(bench)
    return [
        {"label": "baseline_auto", "target_density": auto_td, "density_weight": 8e-5,
         "iteration": 2000, "learning_rate": 0.005, "gp_noise_ratio": 0.025,
         "random_seed": 1000, "stop_overflow": 0.02},
        {"label": "td_low",        "target_density": max(0.40, auto_td - 0.20), "density_weight": 8e-5,
         "iteration": 2000, "learning_rate": 0.005, "gp_noise_ratio": 0.025,
         "random_seed": 1000, "stop_overflow": 0.02},
        {"label": "td_high",       "target_density": min(0.95, auto_td + 0.10), "density_weight": 8e-5,
         "iteration": 2000, "learning_rate": 0.005, "gp_noise_ratio": 0.025,
         "random_seed": 1000, "stop_overflow": 0.02},
        {"label": "dw_low",        "target_density": auto_td, "density_weight": 2e-5,
         "iteration": 2000, "learning_rate": 0.005, "gp_noise_ratio": 0.025,
         "random_seed": 1000, "stop_overflow": 0.02},
        {"label": "dw_high",       "target_density": auto_td, "density_weight": 4e-4,
         "iteration": 2000, "learning_rate": 0.005, "gp_noise_ratio": 0.025,
         "random_seed": 1000, "stop_overflow": 0.02},
        {"label": "long_slow",     "target_density": auto_td, "density_weight": 8e-5,
         "iteration": 4000, "learning_rate": 0.002, "gp_noise_ratio": 0.025,
         "random_seed": 1000, "stop_overflow": 0.01},
        {"label": "high_noise",    "target_density": auto_td, "density_weight": 8e-5,
         "iteration": 2000, "learning_rate": 0.005, "gp_noise_ratio": 0.10,
         "random_seed": 1000, "stop_overflow": 0.02},
        {"label": "seed_2024",     "target_density": auto_td, "density_weight": 8e-5,
         "iteration": 2000, "learning_rate": 0.005, "gp_noise_ratio": 0.025,
         "random_seed": 2024, "stop_overflow": 0.02},
    ]


def run_dp_docker(bench, cfg: Dict, log=print) -> Tuple[Optional[torch.Tensor], float, str]:
    dp_image = os.environ.get("DP_DOCKER_IMAGE", "dreamplace:custom")
    dp_root_host = os.environ.get("DREAMPLACE_ROOT", "/home/ubuntu/DREAMPlace_cpu/install")
    use_gpu = int(os.environ.get("DP_USE_GPU", "0"))
    SCALE = float(bk_writer.SCALE)

    with tempfile.TemporaryDirectory(prefix=f"e96_{bench.name}_{cfg['label']}_", dir="/tmp") as tmp:
        tmp = Path(tmp)
        bk_writer._write_nodes(bench, tmp)
        bk_writer._write_pl(bench, tmp)
        bk_writer._write_nets(bench, tmp)
        bk_writer._write_scl(bench, tmp)
        bk_writer._write_wts(bench, tmp)
        bk_writer._write_aux(bench, tmp)
        cfg_path = tmp / "dp.json"
        cfg_path.write_text(json.dumps({
            "aux_input": f"/work/{bench.name}.aux",
            "target_density": cfg["target_density"],
            "density_weight": cfg["density_weight"],
            "gpu": use_gpu,
            "num_threads": int(os.environ.get("DP_NUM_THREADS", "4")),
            "random_seed": cfg["random_seed"],
            "gp_noise_ratio": cfg["gp_noise_ratio"],
            "global_place_stages": [{
                "num_bins_x": 1024, "num_bins_y": 1024,
                "iteration": cfg["iteration"], "learning_rate": cfg["learning_rate"],
                "wirelength": "weighted_average", "optimizer": "nesterov",
            }],
            "legalize_flag": 0,
            "detailed_place_flag": 0,
            "stop_overflow": cfg["stop_overflow"],
            "result_dir": "/work",
        }))

        uid = os.getuid()
        gid = os.getgid()
        docker_args = ["sudo", "docker", "run", "--rm",
                       "--user", f"{uid}:{gid}",
                       "-v", f"{tmp}:/work",
                       "-v", f"{dp_root_host}:/dp_install:ro",
                       "-w", "/work"]
        if use_gpu:
            docker_args.extend(["--gpus", "all"])
        cmd = docker_args + [
            dp_image,
            "python", "/dp_install/dreamplace/Placer.py", "/work/dp.json",
        ]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return None, time.time() - t0, "timeout"
        wall = time.time() - t0
        if proc.returncode != 0:
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


def probe_bench(bench_name: str, save_basins: bool = False) -> Dict:
    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))
    configs = build_config_grid(bench)
    print(f"\n=== E96 probe: {bench_name} (n_macros={bench.num_macros}, n_hard={bench.num_hard_macros}) ===", flush=True)
    print(f"  auto_target_density={_auto_target_density(bench):.3f}", flush=True)

    results = []
    out_dir = _HERE.parent / "results" / bench_name
    if save_basins:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, cfg in enumerate(configs, 1):
        print(f"\n  [{i}/{len(configs)}] {cfg['label']}: td={cfg['target_density']:.2f} "
              f"dw={cfg['density_weight']:.1e} iter={cfg['iteration']} "
              f"lr={cfg['learning_rate']:.3f} noise={cfg['gp_noise_ratio']:.3f} "
              f"seed={cfg['random_seed']}", flush=True)
        t_start = time.time()
        placement, dp_wall, status = run_dp_docker(bench, cfg)
        if placement is None:
            print(f"    FAILED: status={status} wall={dp_wall:.0f}s", flush=True)
            results.append({**cfg, "status": status, "dp_wall": dp_wall})
            continue
        basin_ovl = int(compute_overlap_metrics(placement, bench)["overlap_count"])
        basin_proxy = float(compute_proxy_cost(placement, bench, plc)["proxy_cost"])

        # Greedy legalize
        t_leg = time.time()
        legal, leg_stats = greedy_macro_legalize(placement, bench, step_size_frac=0.01)
        legal, _ = project_overlaps(legal, bench)
        leg_wall = time.time() - t_leg
        leg_ovl = int(compute_overlap_metrics(legal, bench)["overlap_count"])
        leg_proxy = float(compute_proxy_cost(legal, bench, plc)["proxy_cost"])

        result = {
            **cfg,
            "status": "ok",
            "basin_proxy": basin_proxy,
            "basin_ovl": basin_ovl,
            "legalize_proxy": leg_proxy,
            "legalize_ovl": leg_ovl,
            "legalize_delta": leg_proxy - basin_proxy,
            "legalize_moved": leg_stats.get("n_moved", -1),
            "legalize_failed": leg_stats.get("n_failed", -1),
            "dp_wall": dp_wall,
            "leg_wall": leg_wall,
            "total_wall": time.time() - t_start,
        }
        results.append(result)
        print(f"    basin: proxy={basin_proxy:.4f} ovl={basin_ovl} | "
              f"legalize: proxy={leg_proxy:.4f} ovl={leg_ovl} "
              f"Δ=+{leg_proxy - basin_proxy:.4f} moved={leg_stats.get('n_moved','?')} "
              f"failed={leg_stats.get('n_failed','?')} | wall={result['total_wall']:.0f}s", flush=True)

        if save_basins:
            torch.save({
                "placement_basin": placement,
                "placement_legal": legal,
                "cfg": cfg,
                "result": {k: v for k, v in result.items()},
            }, out_dir / f"{cfg['label']}.pt")

    # Summary
    ok = [r for r in results if r.get("status") == "ok" and r["legalize_ovl"] == 0]
    if ok:
        best = min(ok, key=lambda r: r["legalize_proxy"])
        worst = max(ok, key=lambda r: r["legalize_proxy"])
        spread = worst["legalize_proxy"] - best["legalize_proxy"]
        baseline = next((r for r in ok if r["label"] == "baseline_auto"), None)

        print(f"\n=== {bench_name} SUMMARY ===", flush=True)
        print(f"  Best legalize_proxy: {best['legalize_proxy']:.4f} ({best['label']})", flush=True)
        print(f"  Worst legalize_proxy: {worst['legalize_proxy']:.4f} ({worst['label']})", flush=True)
        print(f"  Spread: {spread:.4f} ({spread / best['legalize_proxy'] * 100:.1f}%)", flush=True)
        if baseline:
            improvement = baseline["legalize_proxy"] - best["legalize_proxy"]
            print(f"  Baseline auto: {baseline['legalize_proxy']:.4f}", flush=True)
            print(f"  Best vs baseline: -{improvement:.4f} ({improvement / baseline['legalize_proxy'] * 100:.1f}%)", flush=True)

    out_summary = _HERE.parent / "results" / f"{bench_name}_probe.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps({
        "bench": bench_name,
        "auto_target_density": _auto_target_density(bench),
        "n_macros": int(bench.num_macros),
        "n_hard": int(bench.num_hard_macros),
        "results": results,
    }, indent=2))
    print(f"\n  Wrote {out_summary}", flush=True)
    return {"bench": bench_name, "results": results}


def main():
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm10"
    save_basins = "--save-basins" in sys.argv
    probe_bench(bench, save_basins=save_basins)


if __name__ == "__main__":
    main()
