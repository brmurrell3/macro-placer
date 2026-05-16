"""E104 — Worse-init saddle escape test.

Hypothesis: vmallela's 28% lift (1.4152→1.0109) comes from running Hessian
saddle escape from a WORSE init (raw CD output, no LNS, no SA polish) which
has MANY soft eigenvalues to exploit. Our E25+E41 polish drives init to a
deep basin (1.08) where few soft modes remain → only 3% saddle lift.

Test: build pipelines with progressively LESS polish before saddle, measure
lift per pipeline:
  A. SDF only (worst, no CD even)            — eigenvalues should be huge soft
  B. SDF + 100s CD only                       — somewhat polished
  C. SDF + 660s CD only (E25's CD cap)       — fully CD-polished but no LNS/SA
  D. Full E25 pipeline (CD + LNS + SA)        — our current init quality (baseline)

For each: run cascade saddle on the resulting plateau. Measure:
  - init proxy
  - post-cascade proxy
  - lift fraction

If A or B gives 10%+ lift (vs our D's ~3% lift), confirmed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost
from macro_place.sdf_init import SDFPlacer

_E84_CODE = _ROOT / "experiments" / "E84_cascading_saddle" / "code"
if str(_E84_CODE) not in sys.path:
    sys.path.insert(0, str(_E84_CODE))
from cascading_saddle import cascading_saddle_escape

# E25 for full polish baseline (D)
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer


def build_init_pipeline(level: str, bench: Benchmark, plc, log) -> tuple[torch.Tensor, dict]:
    """Build init at chosen quality level.

    Returns (placement, stats dict with 'proxy' and 'wall_s').
    """
    fixed = bench.macro_fixed.cpu().numpy()
    movable = [i for i in range(bench.num_macros) if not bool(fixed[i])]

    t0 = time.time()
    log(f"[init={level}] building...")

    if level == "A_sdf":
        # SDF only
        sdf = SDFPlacer(seed=42, n_iters=500).place(bench).to(torch.float32)
        sdf, _ = project_overlaps(sdf, bench)
        proxy = float(compute_proxy_cost(sdf, bench, plc)["proxy_cost"])
        return sdf, {"proxy": proxy, "wall_s": time.time() - t0, "phase": "SDF only"}

    if level == "B_cd100":
        sdf = SDFPlacer(seed=42, n_iters=500).place(bench).to(torch.float32)
        sdf, _ = project_overlaps(sdf, bench)
        ev = IncrementalProxyEvaluator(bench, plc, sdf.clone().to(torch.float64))
        run_cd_adaptive(ev, bench, plc, movable, min_time_s=30.0, hard_cap_s=100.0,
                        patience=3, plateau_threshold=0.001, log_fn=None)
        out = ev.placement.detach().clone().to(torch.float32)
        out, _ = project_overlaps(out, bench)
        proxy = float(compute_proxy_cost(out, bench, plc)["proxy_cost"])
        return out, {"proxy": proxy, "wall_s": time.time() - t0, "phase": "SDF + CD 100s"}

    if level == "C_cd660":
        sdf = SDFPlacer(seed=42, n_iters=500).place(bench).to(torch.float32)
        sdf, _ = project_overlaps(sdf, bench)
        ev = IncrementalProxyEvaluator(bench, plc, sdf.clone().to(torch.float64))
        run_cd_adaptive(ev, bench, plc, movable, min_time_s=60.0, hard_cap_s=660.0,
                        patience=3, plateau_threshold=0.001, log_fn=None)
        out = ev.placement.detach().clone().to(torch.float32)
        out, _ = project_overlaps(out, bench)
        proxy = float(compute_proxy_cost(out, bench, plc)["proxy_cost"])
        return out, {"proxy": proxy, "wall_s": time.time() - t0, "phase": "SDF + CD 660s only"}

    if level == "D_e25full":
        # Full E25 pipeline
        placer = CDLNSSAPlacer(cd_hard_cap_s=660.0, lns_budget_s=200.0, sa_budget_s=200.0)
        out = placer.place(bench)
        proxy = float(compute_proxy_cost(out, bench, plc)["proxy_cost"])
        return out, {"proxy": proxy, "wall_s": time.time() - t0, "phase": "Full E25"}

    raise ValueError(f"Unknown level: {level}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--level", choices=["A_sdf", "B_cd100", "C_cd660", "D_e25full"], required=True)
    ap.add_argument("--saddle-budget", type=float, default=1200.0,
                    help="cascade saddle budget seconds")
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--polish-budget", type=float, default=60.0)
    args = ap.parse_args()

    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(args.bench)))
    log = lambda s: print(s, flush=True)

    init, init_stats = build_init_pipeline(args.level, bench, plc, log)
    log(f"[init] {init_stats}")

    log(f"[saddle] starting cascade saddle escape (budget {args.saddle_budget}s, max_iters {args.max_iters})")
    t0 = time.time()
    final, sad_stats = cascading_saddle_escape(
        init, bench, plc,
        max_iters=args.max_iters,
        eps_values=(0.3, 1.0, 3.0),
        polish_budget=args.polish_budget,
        total_budget_s=args.saddle_budget,
        log=lambda s: print(s, flush=True),
    )
    saddle_wall = time.time() - t0
    final_proxy = float(compute_proxy_cost(final, bench, plc)["proxy_cost"])
    final_ovl = int(compute_overlap_metrics(final, bench)["overlap_count"])
    lift = (init_stats['proxy'] - final_proxy) / init_stats['proxy']

    log(f"\n=== {args.bench} {args.level} RESULT ===")
    log(f"  init proxy:   {init_stats['proxy']:.5f} ({init_stats['phase']}, {init_stats['wall_s']:.0f}s)")
    log(f"  final proxy:  {final_proxy:.5f} (after saddle, {saddle_wall:.0f}s)")
    log(f"  lift:         {lift*100:+.3f}%   (saddle iters={sad_stats.get('iters_run', '?')})")
    log(f"  overlap:      {final_ovl}")

    out_dir = _HERE.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.bench}_{args.level}.json"
    out_path.write_text(json.dumps({
        "bench": args.bench, "level": args.level,
        "init_proxy": init_stats['proxy'], "init_wall_s": init_stats['wall_s'],
        "init_phase": init_stats['phase'],
        "final_proxy": final_proxy, "saddle_wall_s": saddle_wall,
        "lift_frac": lift, "iters_run": sad_stats.get('iters_run', None),
        "final_overlap": final_ovl,
    }, indent=2))


if __name__ == "__main__":
    main()
