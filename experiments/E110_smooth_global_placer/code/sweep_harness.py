"""Sweep harness for E110 SmoothGlobalPlacer.

Runs N configs on a chosen benchmark subset. Each config tested as
E110 descent + 60s CD polish. Writes per-config result to a JSON log.

Designed for overnight autonomous execution. Resume-safe: skips configs
already present in the log file.

Usage:
    uv run python experiments/E110_smooth_global_placer/code/sweep_harness.py \
        --bench ibm04 --output-name ibm04_sweep --max-configs 20

    uv run python experiments/E110_smooth_global_placer/code/sweep_harness.py \
        --bench-set fast --output-name fast_top5_sweep --configs top5
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost
from smooth_global_placer import SmoothGlobalPlacer


# ── Bench sets ────────────────────────────────────────────────────────────
FAST_BENCHES = ["ibm01", "ibm04", "ibm09", "ibm13"]
HARD_FAST_BENCHES = ["ibm04", "ibm17"]   # the loss cases
ALL_BENCHES = [f"ibm{i:02d}" for i in (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)]


# ── Config grids ──────────────────────────────────────────────────────────
def _label(d: Dict) -> str:
    """Compact label for a config dict."""
    parts = []
    for k, v in sorted(d.items()):
        if isinstance(v, float):
            if abs(v) < 1e-2 and v != 0:
                parts.append(f"{k}={v:.0e}")
            else:
                parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    return "_".join(parts)


def grid_a_diagnose_ibm04() -> List[Dict]:
    """Target ibm04 loss. Vary num_steps, lr, overlap, init."""
    configs = []
    for num_steps in (300, 800, 1500):
        for lr_frac in (0.003, 0.005, 0.01):
            for overlap_lambda_end in (30.0, 100.0):
                cfg = dict(
                    num_steps=num_steps,
                    lr_frac=lr_frac,
                    gamma_end_frac=5e-5,
                    overlap_lambda_end=overlap_lambda_end,
                    overlap_ramp_pct=0.7,
                    legalize_step_frac=0.005,
                    legalize_radius_steps=200,
                    init="sdf",
                )
                configs.append(cfg)
    return configs


def grid_b_gamma_anneal() -> List[Dict]:
    """Test γ-anneal end points + ramp shape."""
    configs = []
    for gamma_end_frac in (5e-3, 5e-4, 5e-5, 5e-6):
        for overlap_ramp_pct in (0.3, 0.5, 0.7, 0.9):
            cfg = dict(
                num_steps=500,
                lr_frac=0.005,
                gamma_end_frac=gamma_end_frac,
                overlap_lambda_end=50.0,
                overlap_ramp_pct=overlap_ramp_pct,
                legalize_step_frac=0.005,
                legalize_radius_steps=200,
                init="sdf",
            )
            configs.append(cfg)
    return configs


def grid_c_boundary_cong() -> List[Dict]:
    """Test boundary_lambda + cong on/off."""
    configs = []
    for boundary_lambda in (10.0, 50.0, 200.0):
        for include_congestion in (True, False):
            cfg = dict(
                num_steps=500,
                lr_frac=0.005,
                gamma_end_frac=5e-5,
                overlap_lambda_end=50.0,
                overlap_ramp_pct=0.7,
                boundary_lambda=boundary_lambda,
                include_congestion=include_congestion,
                legalize_step_frac=0.005,
                legalize_radius_steps=200,
                init="sdf",
            )
            configs.append(cfg)
    return configs


def grid_d_promising_handpicked() -> List[Dict]:
    """Targeted variants based on first-pass insight."""
    return [
        # Baseline (winner on 3/4)
        dict(num_steps=500, lr_frac=0.005, gamma_end_frac=5e-5,
             overlap_lambda_end=50.0, overlap_ramp_pct=0.7,
             legalize_step_frac=0.005, legalize_radius_steps=200, init="sdf"),
        # Longer + smaller lr
        dict(num_steps=1500, lr_frac=0.002, gamma_end_frac=5e-5,
             overlap_lambda_end=50.0, overlap_ramp_pct=0.7,
             legalize_step_frac=0.005, legalize_radius_steps=200, init="sdf"),
        # Longer + slower overlap ramp
        dict(num_steps=1500, lr_frac=0.003, gamma_end_frac=5e-5,
             overlap_lambda_end=50.0, overlap_ramp_pct=0.9,
             legalize_step_frac=0.005, legalize_radius_steps=200, init="sdf"),
        # No congestion in descent (lets WL+density dominate; CD adds cong)
        dict(num_steps=500, lr_frac=0.005, gamma_end_frac=5e-5,
             overlap_lambda_end=50.0, overlap_ramp_pct=0.7,
             include_congestion=False,
             legalize_step_frac=0.005, legalize_radius_steps=200, init="sdf"),
        # Very low overlap_lambda (let descent pack, legalize handles)
        dict(num_steps=500, lr_frac=0.005, gamma_end_frac=5e-5,
             overlap_lambda_end=10.0, overlap_ramp_pct=0.7,
             legalize_step_frac=0.005, legalize_radius_steps=200, init="sdf"),
        # Coarser legalize step
        dict(num_steps=500, lr_frac=0.005, gamma_end_frac=5e-5,
             overlap_lambda_end=50.0, overlap_ramp_pct=0.7,
             legalize_step_frac=0.02, legalize_radius_steps=80, init="sdf"),
        # Higher boundary lambda
        dict(num_steps=500, lr_frac=0.005, gamma_end_frac=5e-5,
             overlap_lambda_end=50.0, overlap_ramp_pct=0.7,
             boundary_lambda=200.0,
             legalize_step_frac=0.005, legalize_radius_steps=200, init="sdf"),
    ]


def grid_e_ibm04_winners_on_fast() -> List[Dict]:
    """Test the ibm04 winners on --fast to check generalization.

    From ibm04_diag partial: cfg (lr=3e-3, num_steps=300, ovl=30) gave
    ibm04 -1.58% win. cfg (lr=3e-3, num_steps=800, ovl=30) gave -0.66%.
    Test if these generalize across --fast.
    """
    return [
        dict(num_steps=300, lr_frac=0.003, gamma_end_frac=5e-5,
             overlap_lambda_end=30.0, overlap_ramp_pct=0.7,
             legalize_step_frac=0.005, legalize_radius_steps=200, init="sdf"),
        dict(num_steps=800, lr_frac=0.003, gamma_end_frac=5e-5,
             overlap_lambda_end=30.0, overlap_ramp_pct=0.7,
             legalize_step_frac=0.005, legalize_radius_steps=200, init="sdf"),
        dict(num_steps=300, lr_frac=0.003, gamma_end_frac=5e-5,
             overlap_lambda_end=100.0, overlap_ramp_pct=0.7,
             legalize_step_frac=0.005, legalize_radius_steps=200, init="sdf"),
        # short-step variants that may scale better to ibm17
        dict(num_steps=200, lr_frac=0.005, gamma_end_frac=5e-5,
             overlap_lambda_end=30.0, overlap_ramp_pct=0.7,
             legalize_step_frac=0.005, legalize_radius_steps=200, init="sdf"),
        dict(num_steps=150, lr_frac=0.01, gamma_end_frac=5e-5,
             overlap_lambda_end=30.0, overlap_ramp_pct=0.7,
             legalize_step_frac=0.005, legalize_radius_steps=200, init="sdf"),
        # No-cong variants (saw promising signs on ibm04)
        dict(num_steps=500, lr_frac=0.003, gamma_end_frac=5e-5,
             overlap_lambda_end=30.0, overlap_ramp_pct=0.7,
             include_congestion=False,
             legalize_step_frac=0.005, legalize_radius_steps=200, init="sdf"),
    ]


GRID_CATALOG = {
    "ibm04_diag": grid_a_diagnose_ibm04,
    "gamma": grid_b_gamma_anneal,
    "boundary_cong": grid_c_boundary_cong,
    "handpicked": grid_d_promising_handpicked,
    "ibm04_winners_on_fast": grid_e_ibm04_winners_on_fast,
}


def get_sdf_cd_reference(
    bench_name: str, cd_budget_s: float = 60.0,
) -> Tuple[float, float]:
    """SDF + project_overlaps + CD polish baseline. Returns (proxy, wall_s)."""
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    movable = [
        i for i in range(benchmark.num_macros)
        if not bool(benchmark.macro_fixed[i])
    ]
    t0 = time.time()
    sdf_pos = sdf_init(benchmark)
    sdf_pos, _ = project_overlaps(sdf_pos, benchmark)
    ev = IncrementalProxyEvaluator(benchmark, plc, sdf_pos)
    run_cd_adaptive(
        ev, benchmark, plc, movable,
        min_time_s=cd_budget_s, hard_cap_s=cd_budget_s,
        patience=3, plateau_threshold=0.001,
    )
    proxy = float(compute_proxy_cost(ev.placement, benchmark, plc)["proxy_cost"])
    return proxy, time.time() - t0


def run_e110_cd(
    bench_name: str, cfg: Dict, cd_budget_s: float = 60.0,
) -> Dict:
    """Run E110 + CD polish on one bench. Returns result dict."""
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    movable = [
        i for i in range(benchmark.num_macros)
        if not bool(benchmark.macro_fixed[i])
    ]

    t0 = time.time()
    try:
        placer = SmoothGlobalPlacer(verbose=False, **cfg)
        pos = placer.place(benchmark)
        descent_wall = time.time() - t0
        raw_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        raw_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]

        ev = IncrementalProxyEvaluator(benchmark, plc, pos)
        run_cd_adaptive(
            ev, benchmark, plc, movable,
            min_time_s=cd_budget_s, hard_cap_s=cd_budget_s,
            patience=3, plateau_threshold=0.001,
        )
        cd_pos = ev.placement.detach().clone().to(torch.float32)
        cd_proxy = float(compute_proxy_cost(cd_pos, benchmark, plc)["proxy_cost"])
        cd_ovl = compute_overlap_metrics(cd_pos, benchmark)["overlap_count"]

        return {
            "bench": bench_name,
            "config": cfg,
            "raw_proxy": raw_proxy,
            "raw_ovl": raw_ovl,
            "cd_proxy": cd_proxy,
            "cd_ovl": cd_ovl,
            "descent_wall_s": descent_wall,
            "total_wall_s": time.time() - t0,
            "error": None,
        }
    except Exception as exc:
        import traceback
        return {
            "bench": bench_name,
            "config": cfg,
            "raw_proxy": None,
            "raw_ovl": None,
            "cd_proxy": None,
            "cd_ovl": None,
            "descent_wall_s": None,
            "total_wall_s": time.time() - t0,
            "error": f"{exc}\n{traceback.format_exc()}",
        }


def load_existing(jsonl_path: Path) -> List[Dict]:
    if not jsonl_path.exists():
        return []
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="append", help="bench name (repeatable)")
    ap.add_argument("--bench-set", choices=["fast", "hard", "all"], default=None)
    ap.add_argument("--grid", action="append", choices=list(GRID_CATALOG.keys()),
                    default=None, help="config grid name (repeatable)")
    ap.add_argument("--max-configs", type=int, default=None)
    ap.add_argument("--cd-budget", type=float, default=60.0)
    ap.add_argument("--output-name", required=True)
    args = ap.parse_args()

    if args.bench_set == "fast":
        benches = FAST_BENCHES
    elif args.bench_set == "hard":
        benches = HARD_FAST_BENCHES
    elif args.bench_set == "all":
        benches = ALL_BENCHES
    else:
        benches = args.bench or []
    if not benches:
        ap.error("Must specify --bench or --bench-set")

    grids = args.grid or ["handpicked"]
    configs = []
    seen_labels = set()
    for grid_name in grids:
        for cfg in GRID_CATALOG[grid_name]():
            lbl = _label(cfg)
            if lbl in seen_labels:
                continue
            seen_labels.add(lbl)
            configs.append(cfg)
    if args.max_configs:
        configs = configs[: args.max_configs]

    out_dir = _HERE.parent / "results"
    out_dir.mkdir(exist_ok=True)
    jsonl_path = out_dir / f"{args.output_name}.jsonl"
    summary_path = out_dir / f"{args.output_name}_summary.json"

    existing = load_existing(jsonl_path)
    done_keys = {(r["bench"], _label(r["config"])) for r in existing}

    # Compute SDF references for each bench (once).
    sdf_refs = {}
    for bench in benches:
        existing_ref = next(
            (r for r in existing if r.get("type") == "sdf_ref" and r["bench"] == bench),
            None,
        )
        if existing_ref:
            sdf_refs[bench] = existing_ref["cd_proxy"]
            continue
        print(f"[ref] SDF+CD{args.cd_budget:.0f}s on {bench}...", flush=True)
        proxy, wall = get_sdf_cd_reference(bench, cd_budget_s=args.cd_budget)
        sdf_refs[bench] = proxy
        with jsonl_path.open("a") as f:
            f.write(json.dumps({
                "type": "sdf_ref",
                "bench": bench,
                "config": {},
                "cd_proxy": proxy,
                "wall_s": wall,
            }) + "\n")
        print(f"  SDF+CD ref [{bench}]: proxy={proxy:.5f} wall={wall:.1f}s",
              flush=True)

    # Sweep.
    total_configs = len(configs)
    print(f"\n=== Sweep {total_configs} configs × {len(benches)} benches "
          f"(CD budget {args.cd_budget:.0f}s) ===", flush=True)
    sweep_t0 = time.time()
    config_i = 0
    for cfg in configs:
        config_i += 1
        lbl = _label(cfg)
        for bench in benches:
            if (bench, lbl) in done_keys:
                print(f"  [{config_i}/{total_configs}] {bench} {lbl}: SKIP (done)",
                      flush=True)
                continue
            t0 = time.time()
            print(f"  [{config_i}/{total_configs}] {bench} {lbl}: running...",
                  flush=True)
            result = run_e110_cd(bench, cfg, cd_budget_s=args.cd_budget)
            result["type"] = "sweep_run"
            result["config_label"] = lbl
            if result["cd_proxy"] is not None:
                ref = sdf_refs[bench]
                result["delta_vs_sdf_cd"] = result["cd_proxy"] - ref
                result["delta_pct"] = (
                    (result["cd_proxy"] - ref) / ref * 100 if ref else None
                )
                verdict = (
                    "WIN" if result["delta_pct"] < -0.5
                    else ("TIE" if abs(result["delta_pct"]) <= 0.5 else "LOSS")
                )
                print(f"    {bench} {lbl}: cd_proxy={result['cd_proxy']:.5f} "
                      f"Δ={result['delta_pct']:+.2f}% [{verdict}] "
                      f"raw={result['raw_proxy']:.5f} "
                      f"ovl={result['cd_ovl']} wall={result['total_wall_s']:.0f}s",
                      flush=True)
            else:
                print(f"    {bench} {lbl}: ERROR — {result['error']}",
                      flush=True)
            with jsonl_path.open("a") as f:
                f.write(json.dumps(result) + "\n")

    # Summary.
    rows = load_existing(jsonl_path)
    runs = [r for r in rows if r.get("type") == "sweep_run"]
    refs = {r["bench"]: r["cd_proxy"] for r in rows if r.get("type") == "sdf_ref"}

    # Per-config aggregate
    per_config = {}
    for r in runs:
        if r["cd_proxy"] is None:
            continue
        cfg_lbl = r["config_label"]
        per_config.setdefault(cfg_lbl, []).append(r)
    config_rankings = []
    for lbl, rrows in per_config.items():
        bench_proxies = {r["bench"]: r["cd_proxy"] for r in rrows}
        if not bench_proxies:
            continue
        bench_deltas = [
            (r["cd_proxy"] - refs[r["bench"]]) / refs[r["bench"]] * 100
            for r in rrows if r["bench"] in refs
        ]
        if not bench_deltas:
            continue
        wins = sum(1 for d in bench_deltas if d < -0.5)
        losses = sum(1 for d in bench_deltas if d > 0.5)
        config_rankings.append({
            "label": lbl,
            "config": rrows[0]["config"],
            "benches": len(bench_deltas),
            "avg_delta_pct": sum(bench_deltas) / len(bench_deltas),
            "min_delta_pct": min(bench_deltas),
            "max_delta_pct": max(bench_deltas),
            "wins": wins,
            "losses": losses,
            "per_bench": bench_proxies,
        })
    config_rankings.sort(key=lambda r: r["avg_delta_pct"])
    summary_path.write_text(json.dumps({
        "benches": benches,
        "sdf_refs": refs,
        "total_configs_tested": len(per_config),
        "rankings": config_rankings,
    }, indent=2))

    print(f"\n=== SWEEP SUMMARY (sweep wall {time.time() - sweep_t0:.0f}s) ===",
          flush=True)
    print(f"  Wrote {jsonl_path}", flush=True)
    print(f"  Wrote {summary_path}", flush=True)
    print(f"  Top 5 configs by avg Δ%:", flush=True)
    for r in config_rankings[:5]:
        print(f"    Δ_avg={r['avg_delta_pct']:+.2f}% W={r['wins']} L={r['losses']} "
              f"({r['label']})", flush=True)


if __name__ == "__main__":
    main()
