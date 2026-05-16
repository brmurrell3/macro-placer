"""B-R2 with full cascade-equivalent polish.

Same pipeline as E91 (dp_full_polish.py) but uses NATIVE DREAMPlace
with B-R2 canonical-loss patch (top-K density + RUDY) instead of
stock Docker DP.

Pipeline:
  1. Native DP with BR2_LAMBDA_TOPK + BR2_LAMBDA_RUDY → basin
  2. greedy_macro_legalize → project_overlaps cleanup
  3. CD-adaptive + LNS-gridbin + SA-v2 polish
  4. Cascading saddle escape
  5. Validate zero overlaps

Usage:
  BR2_LAMBDA_TOPK=1.0 BR2_LAMBDA_RUDY=0.5 python3 dp_br2_full_polish.py <bench> [budget_s]
"""
from __future__ import annotations

import json
import os
import sys
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
_E91 = _ROOT / "experiments" / "E91_dp_full_polish" / "code"
sys.path.insert(0, str(_E91))

from macro_place.loader import load_benchmark_from_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

import tilos_to_bookshelf as bk_writer
import bookshelf_to_pt as bk_reader
from macro_legalizer import greedy_macro_legalize
from extended_legalize import extended_legalize

import importlib.util
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
run_lns_gridbin = _E25_MOD.run_lns_gridbin
run_sa_polish_v2 = _E25_MOD.run_sa_polish_v2

from cascading_saddle import cascading_saddle_escape


def run_native_dp_br2(bench, log, target_density=0.85, stop_overflow=0.07,
                      dp_iter=1000, dp_lr=0.01):
    """Run native DREAMPlace with B-R2 canonical losses (via env vars)."""
    dp_install = Path(os.environ.get("DREAMPLACE_ROOT", "/home/ubuntu/DREAMPlace_cpu/install"))
    sys.path.insert(0, str(dp_install))
    sys.path.insert(0, str(dp_install / "dreamplace"))

    import Params, PlaceDB, NonLinearPlace

    SCALE = float(bk_writer.SCALE)
    work_dir = Path(f"/tmp/br2_polish_{bench.name}_{os.getpid()}")
    work_dir.mkdir(parents=True, exist_ok=True)
    bk_writer._write_nodes(bench, work_dir)
    bk_writer._write_pl(bench, work_dir)
    bk_writer._write_nets(bench, work_dir)
    bk_writer._write_scl(bench, work_dir)
    bk_writer._write_wts(bench, work_dir)
    bk_writer._write_aux(bench, work_dir)

    log(f"  [DP-B-R2] λ_topk={os.environ.get('BR2_LAMBDA_TOPK', 0)} "
        f"λ_rudy={os.environ.get('BR2_LAMBDA_RUDY', 0)} "
        f"target_density={target_density} stop_overflow={stop_overflow} "
        f"iter={dp_iter} lr={dp_lr}")

    cfg = work_dir / "dp.json"
    cfg.write_text(json.dumps({
        "aux_input": str(work_dir / f"{bench.name}.aux"),
        "target_density": float(target_density),
        "density_weight": 8e-5,
        "gpu": 0,
        "num_threads": 4,
        "global_place_stages": [{
            "num_bins_x": 1024, "num_bins_y": 1024,
            "iteration": int(dp_iter), "learning_rate": float(dp_lr),
            "wirelength": "weighted_average", "optimizer": "nesterov",
            "Llambda_density_weight_iteration": 1, "Lsub_iteration": 1,
        }],
        "result_dir": str(work_dir / "results"),
        "RePlAce_LOWER_PCOF": 0.95, "RePlAce_UPPER_PCOF": 1.05,
        "RePlAce_ref_hpwl": 350000.0,
        "stop_overflow": float(stop_overflow),
        "macro_place_flag": 0, "routability_opt_flag": 0,
        "legalize_flag": 0, "detailed_place_flag": 0,
        "random_seed": 42, "global_place_flag": 1,
        "scale_factor": 1.0, "shift_factor": [0.0, 0.0],
        "ignore_net_degree": 100,
    }))

    params = Params.Params()
    params.load(str(cfg))
    placedb = PlaceDB.PlaceDB()
    placedb(params)

    t0 = time.time()
    placer = NonLinearPlace.NonLinearPlace(params, placedb, None)
    placer(params, placedb, params.global_place_stages[0]["learning_rate"])
    dp_wall = time.time() - t0

    # Write and read back
    out_dir = Path(params.result_dir) / bench.name
    out_dir.mkdir(parents=True, exist_ok=True)
    gp_out = out_dir / f"{bench.name}.gp.pl"
    placedb.write(params, str(gp_out))

    d = bk_reader.parse_pl(gp_out)
    pos = torch.zeros(bench.num_macros, 2, dtype=torch.float32)
    for i in range(bench.num_macros):
        if i in d:
            llx, lly = d[i]
            w, h = float(bench.macro_sizes[i, 0]), float(bench.macro_sizes[i, 1])
            pos[i, 0] = llx / SCALE + w / 2.0
            pos[i, 1] = lly / SCALE + h / 2.0
        else:
            pos[i] = bench.macro_positions[i]
    return pos, dp_wall


def full_polish(bench, plc, init_pos, log, budget_s=2400.0):
    """Cascade-equivalent polish on init_pos.

    CD-adaptive + LNS-gridbin + SA-v2 + project_overlaps + cascading saddle.
    """
    t0 = time.time()

    # Legalize
    log(f"  [polish] greedy_macro_legalize")
    legal, _ = greedy_macro_legalize(init_pos, bench)
    legal, _ = project_overlaps(legal, bench)
    ovl_legal = compute_overlap_metrics(legal, bench)['overlap_count']
    if ovl_legal > 0:
        log(f"  [polish] greedy left {ovl_legal} overlaps, trying extended_legalize")
        legal, _ext_stats = extended_legalize(
            legal, bench, max_passes=20, jitter_scale=0.5, seed=42,
        )
        legal, _ = project_overlaps(legal, bench)
        ovl_legal = compute_overlap_metrics(legal, bench)['overlap_count']
    legal_proxy = compute_proxy_cost(legal, bench, plc)['proxy_cost']
    log(f"  [polish] legal proxy={legal_proxy:.5f} ovl={ovl_legal}")
    if ovl_legal > 0:
        log(f"  [polish] FAIL: {ovl_legal} residual overlaps after extended_legalize")
        return None

    # Setup evaluator + movable lists (match cd_lns_sa_cascade_dp_lane signatures)
    legal_f64 = legal.detach().clone().to(torch.float64)
    evaluator = IncrementalProxyEvaluator(bench, plc, legal_f64)
    n_hard = bench.num_hard_macros
    fixed = bench.macro_fixed.cpu().numpy()
    movable = [i for i in range(bench.num_macros) if not bool(fixed[i])]
    hard_movable = [i for i in range(n_hard) if not bool(fixed[i])]

    # CD-adaptive
    cd_budget = 0.45 * budget_s
    log(f"  [polish] CD-adaptive (cap {cd_budget:.0f}s)")
    run_cd_adaptive(
        evaluator=evaluator, benchmark=bench, plc=plc, movable=movable,
        min_time_s=min(60.0, cd_budget * 0.1),
        hard_cap_s=cd_budget,
        patience=3, plateau_threshold=0.001, log_fn=None,
    )
    cd_pos = evaluator.placement.detach().clone().to(torch.float32)
    cd_proxy = compute_proxy_cost(cd_pos, bench, plc)['proxy_cost']
    log(f"  [polish] CD done: proxy={cd_proxy:.5f} wall={time.time()-t0:.0f}s")

    # LNS gridbin
    lns_budget = 0.15 * budget_s
    log(f"  [polish] LNS-gridbin ({lns_budget:.0f}s)")
    run_lns_gridbin(
        evaluator=evaluator, benchmark=bench, plc=plc,
        hard_movable=hard_movable, time_budget_s=lns_budget,
        destroy_frac=0.05, destroy_cap=30, seed=42, log_fn=None,
    )
    lns_pos = evaluator.placement.detach().clone().to(torch.float32)
    lns_proxy = compute_proxy_cost(lns_pos, bench, plc)['proxy_cost']
    log(f"  [polish] LNS done: proxy={lns_proxy:.5f} wall={time.time()-t0:.0f}s")

    # SA-v2
    sa_budget = 0.15 * budget_s
    log(f"  [polish] SA-v2 ({sa_budget:.0f}s)")
    run_sa_polish_v2(
        evaluator=evaluator, benchmark=bench, plc=plc,
        hard_movable=hard_movable, time_budget_s=sa_budget,
        T0=5e-4, Tf=1e-6, seed=42, breakpoint_budget=12, log_fn=None,
    )
    sa_pos = evaluator.placement.detach().clone().to(torch.float32)
    sa_pos, _ = project_overlaps(sa_pos, bench)
    sa_proxy = compute_proxy_cost(sa_pos, bench, plc)['proxy_cost']
    log(f"  [polish] SA done: proxy={sa_proxy:.5f} wall={time.time()-t0:.0f}s")

    # Cascading saddle
    saddle_budget = max(60.0, budget_s - (time.time() - t0) - 30.0)
    log(f"  [polish] cascading saddle ({saddle_budget:.0f}s)")
    cascade_pos, stats = cascading_saddle_escape(
        sa_pos, bench, plc,
        max_iters=5,
        eps_values=(0.3, 1.0, 3.0),
        polish_budget=120.0,
        total_budget_s=saddle_budget,
        log=lambda s: None,
    )
    cascade_proxy = compute_proxy_cost(cascade_pos, bench, plc)['proxy_cost']
    log(f"  [polish] cascade done: proxy={cascade_proxy:.5f} iters={stats.get('iters_run','?')} "
        f"wall={time.time()-t0:.0f}s")

    return {
        'legal_proxy': float(legal_proxy),
        'cd_proxy': float(cd_proxy),
        'lns_proxy': float(lns_proxy),
        'sa_proxy': float(sa_proxy),
        'final_proxy': float(cascade_proxy),
        'final_pos': cascade_pos,
        'polish_wall': time.time() - t0,
    }


def main():
    bench_name = sys.argv[1] if len(sys.argv) > 1 else 'ibm01'
    budget_s = float(sys.argv[2]) if len(sys.argv) > 2 else 3300.0

    def log(s):
        print(f"[{time.time():.0f}] {s}", flush=True)

    log(f"Loading {bench_name}")
    bench_dir = _ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench_name
    if not bench_dir.exists():
        bench_dir = _ROOT / "external" / "MacroPlacement" / "Testcases" / "NanGate45_2024" / bench_name
    bench, plc = load_benchmark_from_dir(str(bench_dir))
    init_canon = compute_proxy_cost(bench.macro_positions, bench, plc)
    log(f"init canon: {init_canon['proxy_cost']:.5f}")

    # Phase 1: B-R2 DP
    t0 = time.time()
    dp_pos, dp_wall = run_native_dp_br2(bench, log)
    dp_basin = compute_proxy_cost(dp_pos, bench, plc)
    dp_basin_ovl = compute_overlap_metrics(dp_pos, bench)['overlap_count']
    log(f"DP-B-R2 basin: proxy={dp_basin['proxy_cost']:.5f} ovl={dp_basin_ovl} wall={dp_wall:.0f}s")

    # Phase 2: polish
    polish_budget = budget_s - dp_wall - 60.0
    polish = full_polish(bench, plc, dp_pos, log, polish_budget)
    if polish is None:
        log(f"FAILED polish")
        return
    total_wall = time.time() - t0
    log(f"=== FINAL ===")
    log(f"bench={bench_name} final_proxy={polish['final_proxy']:.5f} total_wall={total_wall:.0f}s")
    log(f"  init={init_canon['proxy_cost']:.5f} → dp_basin={dp_basin['proxy_cost']:.5f} "
        f"→ legal={polish['legal_proxy']:.5f} → cd={polish['cd_proxy']:.5f} "
        f"→ lns={polish['lns_proxy']:.5f} → sa={polish['sa_proxy']:.5f} → cascade={polish['final_proxy']:.5f}")

    # Save result
    out_dir = _HERE.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{bench_name}_br2_topk{os.environ.get('BR2_LAMBDA_TOPK','0')}_rudy{os.environ.get('BR2_LAMBDA_RUDY','0')}.json"
    with open(out_file, "w") as f:
        json.dump({
            'bench': bench_name,
            'init_proxy': float(init_canon['proxy_cost']),
            'dp_basin_proxy': float(dp_basin['proxy_cost']),
            'dp_basin_ovl': int(dp_basin_ovl),
            'legal_proxy': polish['legal_proxy'],
            'cd_proxy': polish['cd_proxy'],
            'lns_proxy': polish['lns_proxy'],
            'sa_proxy': polish['sa_proxy'],
            'final_proxy': polish['final_proxy'],
            'dp_wall': dp_wall,
            'polish_wall': polish['polish_wall'],
            'total_wall': total_wall,
            'lambda_topk': float(os.environ.get('BR2_LAMBDA_TOPK', 0.0)),
            'lambda_rudy': float(os.environ.get('BR2_LAMBDA_RUDY', 0.0)),
        }, f, indent=2)
    log(f"saved {out_file}")


if __name__ == '__main__':
    main()
