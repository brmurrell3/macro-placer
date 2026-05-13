"""E96: Multi-seed/multi-config DP basin generator with best-of-K selection.

Given a bench and a budget, run DP with K different configs (or seeds),
greedy-legalize each, and return the best (lowest legalize_proxy, zero ovl).

The B-R0' single-DP-config approach leaves variance on the table:
- DP has stochastic init (random_center_init_flag + gp_noise_ratio + random_seed).
- Even same-config runs give different basins (observed in E96 probe).
- DP basin quality varies meaningfully with target_density / density_weight.

This module provides:
- `multi_dp_basin(bench, K, ...)` — runs K configs, returns best legalize basin.
- A drop-in replacement for `_try_run_dreamplace` in
  `submissions/cd_lns_sa_cascade_dp_lane/placer.py`.

Selection metric: `legalize_proxy + 10 × residual_overlap_count`. Penalizes
basins that don't fully legalize. Among zero-overlap basins, picks lowest
legalize_proxy.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))
_E76 = _ROOT / "experiments" / "E76_dreamplace_integration" / "code"
sys.path.insert(0, str(_E76))

from macro_place.cd_core import project_overlaps
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

import tilos_to_bookshelf as bk_writer
import bookshelf_to_pt as bk_reader
from macro_legalizer import greedy_macro_legalize

# Stricter legalizer (E71 jitter+project loop) for small-residual rescue.
_E91 = _ROOT / "experiments" / "E91_dp_full_polish" / "code"
sys.path.insert(0, str(_E91))
from extended_legalize import extended_legalize


def _auto_target_density(bench) -> float:
    sizes = bench.macro_sizes.cpu().numpy()
    hard_idx = bench.num_hard_macros
    macro_area = float((sizes[:hard_idx, 0] * sizes[:hard_idx, 1]).sum())
    canvas_area = float(bench.canvas_width * bench.canvas_height)
    macro_density = macro_area / max(canvas_area, 1e-9)
    return float(min(0.85, max(0.40, macro_density * 1.5)))


def build_config_grid_K(bench, K: int) -> List[Dict]:
    """Generate K DP configs that vary along salient axes.

    Order is based on E96 probe findings on ibm10/12/14 (2026-05-13):
    - dw_low wins on benches with clamped auto_td (ibm10 at 0.85, ibm14 at 0.40)
    - baseline_auto wins on "natural" benches (ibm12 at 0.775)
    - dw_high wins on some benches (ibm12 second-best)
    - td_high and long_slow give better basins but worse legalize quality

    Same algorithm for every bench (rule-compliant); only `K` varies.
    """
    auto_td = _auto_target_density(bench)
    base_seed = 1000
    full_grid = [
        # 0: baseline (wins on natural-density benches like ibm12)
        {"label": "baseline_auto", "target_density": auto_td, "density_weight": 8e-5,
         "iteration": 2000, "learning_rate": 0.005, "gp_noise_ratio": 0.025,
         "random_seed": base_seed, "stop_overflow": 0.02},
        # 1: weaker density (wins on clamped-density benches like ibm10/ibm14)
        {"label": "dw_low", "target_density": auto_td, "density_weight": 2e-5,
         "iteration": 2000, "learning_rate": 0.005, "gp_noise_ratio": 0.025,
         "random_seed": base_seed, "stop_overflow": 0.02},
        # 2: stronger density (sometimes second-best, e.g. ibm12)
        {"label": "dw_high", "target_density": auto_td, "density_weight": 4e-4,
         "iteration": 2000, "learning_rate": 0.005, "gp_noise_ratio": 0.025,
         "random_seed": base_seed, "stop_overflow": 0.02},
        # 3: different seed (basin diversity, low overhead)
        {"label": "seed_2024", "target_density": auto_td, "density_weight": 8e-5,
         "iteration": 2000, "learning_rate": 0.005, "gp_noise_ratio": 0.025,
         "random_seed": 2024, "stop_overflow": 0.02},
        # 4: looser packing (low-td, basin diversity)
        {"label": "td_low", "target_density": max(0.40, auto_td - 0.20), "density_weight": 8e-5,
         "iteration": 2000, "learning_rate": 0.005, "gp_noise_ratio": 0.025,
         "random_seed": base_seed, "stop_overflow": 0.02},
        # 5: more init noise (broader sampling)
        {"label": "high_noise", "target_density": auto_td, "density_weight": 8e-5,
         "iteration": 2000, "learning_rate": 0.005, "gp_noise_ratio": 0.10,
         "random_seed": base_seed, "stop_overflow": 0.02},
        # 6: another seed (basin diversity)
        {"label": "seed_42", "target_density": auto_td, "density_weight": 8e-5,
         "iteration": 2000, "learning_rate": 0.005, "gp_noise_ratio": 0.025,
         "random_seed": 42, "stop_overflow": 0.02},
        # 7: more iters + smaller lr (better convergence, but high-ovl basin)
        {"label": "long_slow", "target_density": auto_td, "density_weight": 8e-5,
         "iteration": 4000, "learning_rate": 0.002, "gp_noise_ratio": 0.025,
         "random_seed": base_seed, "stop_overflow": 0.01},
    ]
    return full_grid[:max(1, min(K, len(full_grid)))]


def run_dp_once(
    bench, cfg: Dict, *, log=print
) -> Tuple[Optional[torch.Tensor], float, str]:
    """Run DP via Docker; return (placement, wall_s, status)."""
    dp_image = os.environ.get("DP_DOCKER_IMAGE", "dreamplace:custom")
    dp_root_host = os.environ.get("DREAMPLACE_ROOT", "/home/ubuntu/DREAMPlace_cpu/install")
    use_gpu = int(os.environ.get("DP_USE_GPU", "0"))
    SCALE = float(bk_writer.SCALE)

    with tempfile.TemporaryDirectory(prefix=f"e96mb_{bench.name}_{cfg['label']}_", dir="/tmp") as tmp:
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
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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


def multi_dp_basin(
    bench,
    plc,
    K: int = 4,
    *,
    deadline: Optional[float] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict]:
    """Run K DP configs, return best legalized basin.

    Returns:
      (raw_placement, legal_placement, stats) — raw is the DP output of the
      winning config (still overlapping); legal is the greedy-legalized
      version. Caller polishes legal further.
    Stats dict has per-config metrics and the selection log.

    If all configs fail / never legalize, returns (None, None, stats).
    """
    if log is None:
        log = print
    configs = build_config_grid_K(bench, K)
    log(f"  [multi-DP] running {len(configs)} configs (K={K})")

    candidates = []
    for i, cfg in enumerate(configs, 1):
        if deadline is not None and time.time() > deadline - 30.0:
            log(f"  [multi-DP] deadline hit; stopping after {i-1} configs")
            break
        t_start = time.time()
        placement, dp_wall, status = run_dp_once(bench, cfg, log=log)
        if placement is None:
            log(f"  [multi-DP {i}/{len(configs)}] {cfg['label']}: FAILED ({status}) wall={dp_wall:.0f}s")
            continue
        basin_ovl = int(compute_overlap_metrics(placement, bench)["overlap_count"])
        basin_proxy = float(compute_proxy_cost(placement, bench, plc)["proxy_cost"])

        # Greedy legalize
        legal, leg_stats = greedy_macro_legalize(placement, bench, step_size_frac=0.01)
        legal, _ = project_overlaps(legal, bench)
        leg_ovl = int(compute_overlap_metrics(legal, bench)["overlap_count"])
        leg_proxy = float(compute_proxy_cost(legal, bench, plc)["proxy_cost"])

        # If greedy left small-to-moderate residuals (1-50), try extended_legalize
        # (jitter+project). Observed cases:
        # - dw_low/dw_high on ibm10 sometimes give 1-residual basins with
        #   otherwise excellent proxies that greedy can't quite seat.
        # - long_slow on ibm10 has 75-residual basin with BEST basin_proxy;
        #   extended_legalize at threshold 50 doesn't reach this case but
        #   raising to 100+ risks long fallback walls.
        if 0 < leg_ovl <= 50:
            try:
                ext_legal, ext_stats = extended_legalize(
                    legal, bench, max_passes=15, jitter_scale=0.5, seed=42,
                )
                ext_ovl = ext_stats["final_overlap"]
                ext_proxy = float(compute_proxy_cost(ext_legal, bench, plc)["proxy_cost"])
                if ext_ovl < leg_ovl:
                    log(f"    [legalize fallback] greedy ovl={leg_ovl} → extended ovl={ext_ovl} "
                        f"({leg_proxy:.4f} → {ext_proxy:.4f})")
                    legal = ext_legal
                    leg_ovl = ext_ovl
                    leg_proxy = ext_proxy
            except Exception as exc:
                log(f"    [legalize fallback] extended_legalize raised: {exc}")
        total_wall = time.time() - t_start

        # Selection score: lower is better. Heavy penalty for residual overlaps.
        score = leg_proxy + 10.0 * leg_ovl
        log(f"  [multi-DP {i}/{len(configs)}] {cfg['label']}: "
            f"basin {basin_proxy:.4f}/{basin_ovl} → legal {leg_proxy:.4f}/{leg_ovl} "
            f"score={score:.4f} wall={total_wall:.0f}s")

        candidates.append({
            "label": cfg["label"],
            "cfg": cfg,
            "raw_placement": placement,
            "legal_placement": legal,
            "basin_proxy": basin_proxy,
            "basin_ovl": basin_ovl,
            "legalize_proxy": leg_proxy,
            "legalize_ovl": leg_ovl,
            "score": score,
            "dp_wall": dp_wall,
            "leg_wall": total_wall - dp_wall,
            "total_wall": total_wall,
        })

    if not candidates:
        log(f"  [multi-DP] all {len(configs)} configs failed")
        return None, None, {"K": K, "candidates": [], "winner": None}

    candidates.sort(key=lambda c: c["score"])
    winner = candidates[0]
    log(f"  [multi-DP] winner: {winner['label']} (legal_proxy={winner['legalize_proxy']:.4f}, "
        f"ovl={winner['legalize_ovl']}) — beat {len(candidates) - 1} others")
    if len(candidates) > 1:
        best_proxy = winner['legalize_proxy']
        worst_proxy = max(c['legalize_proxy'] for c in candidates if c['legalize_ovl'] == 0)
        spread = worst_proxy - best_proxy
        log(f"  [multi-DP] spread: {spread:.4f} ({spread/best_proxy*100:.1f}%)")

    stats = {
        "K": K,
        "n_configs_run": len(candidates),
        "candidates": [{k: v for k, v in c.items()
                        if k not in ("raw_placement", "legal_placement")}
                       for c in candidates],
        "winner_label": winner["label"],
        "winner_legalize_proxy": winner["legalize_proxy"],
    }
    return winner["raw_placement"], winner["legal_placement"], stats


if __name__ == "__main__":
    """Smoke test: run multi-DP basin selector on a single bench."""
    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.loader import load_benchmark_from_dir

    bench_name = sys.argv[1] if len(sys.argv) > 1 else "ibm10"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))
    print(f"=== multi_dp_basin: {bench_name}, K={K} ===", flush=True)
    raw, legal, stats = multi_dp_basin(bench, plc, K=K, log=print)
    out_path = _HERE.parent / "results" / f"multi_dp_{bench_name}_K{K}.pt"
    if legal is not None:
        torch.save({
            "raw_placement": raw, "legal_placement": legal, "stats": stats,
            "bench_name": bench_name, "K": K,
        }, out_path)
    json_path = _HERE.parent / "results" / f"multi_dp_{bench_name}_K{K}.json"
    json_path.write_text(json.dumps(stats, indent=2))
    print(f"\n  Wrote {json_path}")
