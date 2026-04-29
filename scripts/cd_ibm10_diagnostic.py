"""
E2 diagnostic: Full-proxy coordinate descent on ibm10 from SDF init.

Goal: How far does pure CD (no LNS, no DPO) get in 40 min wallclock from a basic
SDF init? Per E8, congestion is 74% of proxy cost so we cannot rely on closed-
form HPWL median — we must search the FULL proxy via numerical 1D line search
per coordinate.

Output:
  results/cd_ibm10_diagnostic.json    trajectory + final breakdown
  docs/cd_ibm10_results.md            comparison table

Usage:
  uv run python scripts/cd_ibm10_diagnostic.py
  uv run python scripts/cd_ibm10_diagnostic.py --time-budget 60   # quick smoke

The CD primitives this script orchestrates (sdf_init, project_overlaps,
legal_axis_range, search_axis) are imported from ``macro_place.cd_core`` —
the same module the production placers use, so this diagnostic and the
champion run on identical underlying code.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from macro_place.cd_core import (
    legal_axis_range,
    project_overlaps,
    sdf_init,
    search_axis,
)
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


ROOT = Path(__file__).resolve().parent.parent
TESTCASE_ROOT = ROOT / "external/MacroPlacement/Testcases/ICCAD04"


# ── Main loop ─────────────────────────────────────────────────────────────


def run(time_budget_s: float, log_path: Path, md_path: Path) -> dict:
    print(f"=== E2 CD diagnostic: ibm10, budget {time_budget_s:.0f}s ===")
    print("Loading benchmark ...")
    bench, plc = load_benchmark_from_dir(str(TESTCASE_ROOT / "ibm10"))
    n_hard = bench.num_hard_macros
    n_macros = bench.num_macros
    print(f"  num_macros={n_macros}, n_hard={n_hard}, n_soft={n_macros - n_hard}")
    print(f"  canvas={bench.canvas_width:.1f}x{bench.canvas_height:.1f}, "
          f"grid={bench.grid_cols}x{bench.grid_rows}, num_nets={int(plc.net_cnt)}")

    # ── Init ──
    print("Running SDF init ...")
    t_init0 = time.perf_counter()
    placement = sdf_init(bench)
    t_init = time.perf_counter() - t_init0
    print(f"  SDF init wall = {t_init:.1f} s")

    # Project overlaps if any
    placement, proj_iters = project_overlaps(placement, bench)
    print(f"  overlap projection iterations: {proj_iters}")

    init_overlaps = compute_overlap_metrics(placement, bench)
    print(f"  init overlap_count={init_overlaps['overlap_count']}, "
          f"area={init_overlaps['total_overlap_area']:.4f}")
    if init_overlaps['overlap_count'] > 0:
        print("  ! WARNING: SDF init still has overlaps after projection.")
        # Continue but flag

    # ── Build evaluator ──
    print("Building IncrementalProxyEvaluator ...")
    t_e0 = time.perf_counter()
    placement_f64 = placement.detach().clone().to(torch.float64)
    evaluator = IncrementalProxyEvaluator(bench, plc, placement_f64)
    init_cost = evaluator.current_cost()
    t_eval_init = time.perf_counter() - t_e0
    print(f"  evaluator init = {t_eval_init:.1f} s")
    print(f"  init proxy={init_cost['proxy']:.5f}, wl={init_cost['wl']:.5f}, "
          f"density={init_cost['density']:.5f}, congestion={init_cost['congestion']:.5f}")

    # ── Cross-check vs full compute_proxy_cost ──
    # (Reload plc to avoid mutation by SDFPlacer's plc usage)
    bench2, plc2 = load_benchmark_from_dir(str(TESTCASE_ROOT / "ibm10"))
    full_init = compute_proxy_cost(placement_f64, bench2, plc2)
    print(f"  full compute_proxy_cost: proxy={full_init['proxy_cost']:.5f}, "
          f"wl={full_init['wirelength_cost']:.5f}, dens={full_init['density_cost']:.5f}, "
          f"cong={full_init['congestion_cost']:.5f}")

    # Grid-line arrays (for breakpoint enumeration)
    gw = float(plc.width / plc.grid_col)
    gh = float(plc.height / plc.grid_row)
    grid_lines_x = np.arange(plc.grid_col + 1, dtype=np.float64) * gw
    grid_lines_y = np.arange(plc.grid_row + 1, dtype=np.float64) * gh

    # ── Build movable macro list (excluding fixed) ──
    movable = [i for i in range(n_macros) if not bool(bench.macro_fixed[i])]
    print(f"  movable macros: {len(movable)}")

    # ── Main CD loop ──
    trajectory = []
    sweep_idx = 0
    total_moves = 0
    total_probes = 0
    total_gs_fallbacks = 0

    # Establish start time (CD only — exclude SDF + evaluator init from budget)
    t_start_cd = time.perf_counter()

    cur_cost = init_cost["proxy"]

    # Log init point as sweep 0
    trajectory.append({
        "sweep": 0,
        "elapsed_s": 0.0,
        "proxy": init_cost["proxy"],
        "wl": init_cost["wl"],
        "density": init_cost["density"],
        "congestion": init_cost["congestion"],
        "accepted": 0,
        "probes": 0,
        "gs_fallbacks": 0,
    })

    print(f"=== Starting CD sweeps; budget = {time_budget_s:.0f}s ===")

    while True:
        elapsed = time.perf_counter() - t_start_cd
        if elapsed >= time_budget_s:
            break
        sweep_idx += 1
        sweep_t0 = time.perf_counter()
        sweep_accepted = 0
        sweep_probes = 0
        sweep_gs = 0

        # Visit movable macros in random order (different per sweep)
        rng = np.random.default_rng(seed=sweep_idx)
        order = movable.copy()
        rng.shuffle(order)

        for macro_idx in order:
            # Check time budget within sweep too
            if time.perf_counter() - t_start_cd >= time_budget_s:
                break
            # ── X-axis ──
            lo_x, hi_x = legal_axis_range(
                macro_idx, evaluator.placement, evaluator.macro_sizes,
                bench.macro_fixed, n_hard, axis=0,
                canvas_w=bench.canvas_width, canvas_h=bench.canvas_height,
            )
            cur_xy = (float(evaluator.placement[macro_idx, 0]),
                      float(evaluator.placement[macro_idx, 1]))
            best_x, best_c, mode = search_axis(
                macro_idx, 0, evaluator, grid_lines_x,
                lo_x, hi_x, cur_cost, cur_xy[1],
            )
            if mode == "gs":
                sweep_gs += 1
                total_gs_fallbacks += 1
            sweep_probes += 1  # per-axis sweep counts
            if best_c < cur_cost - 1e-9 and abs(best_x - cur_xy[0]) > 1e-7:
                # Commit
                evaluator.move(macro_idx, (float(best_x), cur_xy[1]))
                cur_cost = best_c
                sweep_accepted += 1
                total_moves += 1

            # ── Y-axis ──
            cur_xy = (float(evaluator.placement[macro_idx, 0]),
                      float(evaluator.placement[macro_idx, 1]))
            lo_y, hi_y = legal_axis_range(
                macro_idx, evaluator.placement, evaluator.macro_sizes,
                bench.macro_fixed, n_hard, axis=1,
                canvas_w=bench.canvas_width, canvas_h=bench.canvas_height,
            )
            best_y, best_c, mode = search_axis(
                macro_idx, 1, evaluator, grid_lines_y,
                lo_y, hi_y, cur_cost, cur_xy[0],
            )
            if mode == "gs":
                sweep_gs += 1
                total_gs_fallbacks += 1
            sweep_probes += 1
            if best_c < cur_cost - 1e-9 and abs(best_y - cur_xy[1]) > 1e-7:
                evaluator.move(macro_idx, (cur_xy[0], float(best_y)))
                cur_cost = best_c
                sweep_accepted += 1
                total_moves += 1

        sweep_wall = time.perf_counter() - sweep_t0
        elapsed = time.perf_counter() - t_start_cd
        # Get full breakdown via evaluator
        cost_break = evaluator.current_cost()
        cur_cost = cost_break["proxy"]
        trajectory.append({
            "sweep": sweep_idx,
            "elapsed_s": elapsed,
            "sweep_s": sweep_wall,
            "proxy": cost_break["proxy"],
            "wl": cost_break["wl"],
            "density": cost_break["density"],
            "congestion": cost_break["congestion"],
            "accepted": sweep_accepted,
            "probes": sweep_probes,
            "gs_fallbacks": sweep_gs,
        })
        total_probes += sweep_probes
        print(f"  sweep {sweep_idx:3d}  elapsed={elapsed:7.1f}s  sweep_t={sweep_wall:6.1f}s  "
              f"proxy={cost_break['proxy']:.5f}  accepted={sweep_accepted}/{sweep_probes}  "
              f"gs={sweep_gs}  "
              f"[wl={cost_break['wl']:.4f} d={cost_break['density']:.4f} c={cost_break['congestion']:.4f}]")

    # ── Validate final placement ──
    print("=== Done. Validating final placement ===")
    final_placement = evaluator.placement.detach().clone()
    final_overlaps = compute_overlap_metrics(final_placement, bench)
    print(f"  final overlap_count={final_overlaps['overlap_count']}, "
          f"area={final_overlaps['total_overlap_area']:.4f}")

    # Cross-check final via full compute_proxy_cost
    bench3, plc3 = load_benchmark_from_dir(str(TESTCASE_ROOT / "ibm10"))
    full_final = compute_proxy_cost(final_placement.to(torch.float32), bench3, plc3)
    print(f"  full check: proxy={full_final['proxy_cost']:.5f} (incr says "
          f"{cur_cost:.5f}, diff={abs(full_final['proxy_cost']-cur_cost):.6f})")

    final_cost = evaluator.current_cost()

    out = {
        "benchmark": "ibm10",
        "time_budget_s": time_budget_s,
        "wall_total_s": time.perf_counter() - t_start_cd,
        "init_proxy": init_cost["proxy"],
        "init_components": {
            "wl": init_cost["wl"],
            "density": init_cost["density"],
            "congestion": init_cost["congestion"],
        },
        "init_overlap_count": int(init_overlaps["overlap_count"]),
        "trajectory": trajectory,
        "final_proxy": final_cost["proxy"],
        "final_components": {
            "wl": final_cost["wl"],
            "density": final_cost["density"],
            "congestion": final_cost["congestion"],
        },
        "final_overlap_count": int(final_overlaps["overlap_count"]),
        "final_full_check_proxy": float(full_final["proxy_cost"]),
        "total_sweeps": sweep_idx,
        "total_moves": total_moves,
        "total_probes": total_probes,
        "total_gs_fallbacks": total_gs_fallbacks,
        "n_hard_macros": n_hard,
        "n_soft_macros": n_macros - n_hard,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(out, indent=2))
    print(f"=== JSON written: {log_path} ===")

    # ── Markdown summary ──
    write_md(md_path, out)
    print(f"=== MD written: {md_path} ===")

    return out


def write_md(path: Path, result: dict) -> None:
    init_p = result["init_proxy"]
    final_p = result["final_proxy"]
    delta_pct = 100.0 * (init_p - final_p) / init_p if init_p > 0 else 0.0
    ic = result["init_components"]
    fc = result["final_components"]
    body = f"""# E2: CD-Only Diagnostic on ibm10

Run timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
Wall budget: {result['time_budget_s']:.0f} s   (CD-only; SDF init excluded)
Total sweeps: {result['total_sweeps']}
Total accepted moves: {result['total_moves']}
Total per-axis probes: {result['total_probes']}
Golden-section fallbacks: {result['total_gs_fallbacks']}
Final overlap count: {result['final_overlap_count']}

## Comparison Table

| Method | proxy | WL | Density | Congestion |
|---|---|---|---|---|
| SDF init (E2 start) | {init_p:.4f} | {ic['wl']:.4f} | {ic['density']:.4f} | {ic['congestion']:.4f} |
| RePlAce baseline (avg, 17 IBM) | 1.4578 | – | – | – |
| DPO best_of_v2 (champion on ibm10) | 1.254 | 0.080 | 0.269 | 0.905 |
| **E2 final (CD-only, {result['time_budget_s']:.0f}s from SDF)** | **{final_p:.4f}** | **{fc['wl']:.4f}** | **{fc['density']:.4f}** | **{fc['congestion']:.4f}** |
| Leaderboard target (avg, all 17) | 1.117 | – | – | – |

Improvement vs SDF init: **{delta_pct:.1f}%**

## Decision rule triggered

"""
    if final_p <= 1.20:
        body += (
            "**E2 final ≤ 1.20:** CD-only is the answer. "
            "Recommend graduating to all 17 benchmarks and writing a CD-only placer.\n"
        )
    elif final_p <= 1.30:
        body += (
            "**E2 final in (1.20, 1.30]:** CD helps but plateaus. "
            "Recommend E3 (LNS) is the load-bearing piece.\n"
        )
    else:
        body += (
            "**E2 final > 1.30:** CD-only doesn't move the needle from SDF. "
            "Either projection is breaking things or CD's local optima trap us. "
            "Recommend DPO seed (E6) instead.\n"
        )

    body += "\n## Trajectory (first/last sweeps)\n\n"
    body += "| sweep | elapsed (s) | proxy | wl | density | congestion | accepted | probes | gs |\n"
    body += "|---|---|---|---|---|---|---|---|---|\n"
    traj = result["trajectory"]
    show = traj[: min(10, len(traj))]
    if len(traj) > 20:
        show = traj[:5] + traj[-5:]
    for r in show:
        body += (
            f"| {r['sweep']} | {r['elapsed_s']:.1f} | {r['proxy']:.4f} | "
            f"{r['wl']:.4f} | {r['density']:.4f} | {r['congestion']:.4f} | "
            f"{r.get('accepted', 0)} | {r.get('probes', 0)} | {r.get('gs_fallbacks', 0)} |\n"
        )
    path.write_text(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-budget", type=float, default=2400.0,
                        help="CD wall budget in seconds (default 2400 = 40 min)")
    parser.add_argument(
        "--out-json", default="results/cd_ibm10_diagnostic.json",
    )
    parser.add_argument(
        "--out-md", default="docs/cd_ibm10_results.md",
    )
    args = parser.parse_args()
    run(args.time_budget, ROOT / args.out_json, ROOT / args.out_md)
