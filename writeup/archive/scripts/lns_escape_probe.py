"""Proof B — LNS escape probe.

Two-phase test of whether CD plateaus are escapable via multi-macro moves:

  Phase 1 (--mode baseline): Run CDOnly to plateau on a benchmark, save the
    final placement tensor and proxy to disk.

  Phase 2 (--mode sample --in baseline.pt --seed N): Load baseline, pick a
    random destroy set of K hard macros (seeded), run CD restricted to that
    subset for `--inner-budget` seconds, write final proxy to JSON.

A driver script runs Phase 1 once, then spawns N Phase 2 workers in parallel.
If any sample's final proxy is meaningfully (>=1%) below baseline, the
plateau is escapable -> LNS lever is real.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics
from submissions.polyhedra.init.sdf import SDFPlacer

_diag_spec = importlib.util.spec_from_file_location(
    "cd_ibm10_diagnostic", str(ROOT / "scripts" / "cd_ibm10_diagnostic.py")
)
_diag = importlib.util.module_from_spec(_diag_spec)
_diag_spec.loader.exec_module(_diag)
project_overlaps = _diag.project_overlaps

_cd_spec = importlib.util.spec_from_file_location(
    "cd_only_placer", str(ROOT / "submissions" / "cd" / "cd_only_placer.py")
)
_cd = importlib.util.module_from_spec(_cd_spec)
_cd_spec.loader.exec_module(_cd)
run_cd = _cd.run_cd

TESTCASE = ROOT / "external/MacroPlacement/Testcases/ICCAD04"


def _build_evaluator(benchmark, placement):
    bench_dir = TESTCASE / benchmark.name
    _, plc = load_benchmark_from_dir(str(bench_dir))
    placement_f64 = placement.detach().clone().to(torch.float64)
    return IncrementalProxyEvaluator(benchmark, plc, placement_f64), plc


def run_baseline(args) -> None:
    """Phase 1: CDOnly to plateau, save final placement."""
    t0 = time.perf_counter()
    bench_dir = TESTCASE / args.benchmark
    benchmark, _ = load_benchmark_from_dir(str(bench_dir))

    placement = SDFPlacer(seed=42).place(benchmark)
    placement, _ = project_overlaps(placement, benchmark)

    evaluator, plc = _build_evaluator(benchmark, placement)
    init_cost = evaluator.current_cost()
    print(f"[baseline] init proxy={init_cost['proxy']:.5f} (CD budget={args.budget:.0f}s)", flush=True)

    movable = [i for i in range(benchmark.num_macros) if not bool(benchmark.macro_fixed[i])]

    def _log(msg: str) -> None:
        print(f"[baseline] {msg}", flush=True)

    stats = run_cd(
        evaluator=evaluator, benchmark=benchmark, plc=plc,
        movable=movable, time_budget_s=args.budget, log_fn=_log,
    )
    final_cost = evaluator.current_cost()
    final_placement = evaluator.placement.detach().clone().cpu().to(torch.float32)
    overlaps = compute_overlap_metrics(final_placement, benchmark)
    if overlaps["overlap_count"] > 0:
        raise RuntimeError(f"baseline has {overlaps['overlap_count']} overlaps")

    out_pt = Path(args.out)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "benchmark": args.benchmark,
            "placement": final_placement,
            "proxy": float(final_cost["proxy"]),
            "breakdown": {k: float(v) for k, v in final_cost.items()},
            "sweeps": int(stats["sweeps"]),
            "moves": int(stats["total_moves"]),
            "wall_s": time.perf_counter() - t0,
        },
        out_pt,
    )
    print(
        f"[baseline] DONE proxy={final_cost['proxy']:.5f} sweeps={stats['sweeps']} "
        f"moves={stats['total_moves']} wall={time.perf_counter() - t0:.1f}s "
        f"-> {out_pt}",
        flush=True,
    )


def run_sample(args) -> None:
    """Phase 2: Load baseline, run one LNS sample (destroy + CD on subset)."""
    t0 = time.perf_counter()
    state = torch.load(args.in_path, weights_only=False)
    bench_name = state["benchmark"]
    placement_baseline = state["placement"]
    baseline_proxy = state["proxy"]

    bench_dir = TESTCASE / bench_name
    benchmark, _ = load_benchmark_from_dir(str(bench_dir))

    evaluator, plc = _build_evaluator(benchmark, placement_baseline)
    init_cost = evaluator.current_cost()

    # Sanity: rebuilt evaluator should match saved baseline within float noise
    if abs(init_cost["proxy"] - baseline_proxy) > 1e-3:
        print(
            f"[seed={args.seed}] WARN: rebuild proxy {init_cost['proxy']:.5f} "
            f"!= saved baseline {baseline_proxy:.5f}",
            flush=True,
        )

    # Pick K hard movable macros to destroy
    n_hard = benchmark.num_hard_macros
    fixed = benchmark.macro_fixed.cpu().numpy()
    hard_movable = [i for i in range(n_hard) if not bool(fixed[i])]
    rng = np.random.default_rng(seed=args.seed)
    if args.destroy_size > len(hard_movable):
        raise ValueError(f"destroy_size={args.destroy_size} > hard_movable={len(hard_movable)}")

    if args.destroy_strategy == "random":
        destroy = sorted(rng.choice(hard_movable, size=args.destroy_size, replace=False).tolist())
    else:  # cost_aware: pick top-K by single-macro proxy contribution
        # Approximate per-macro cost contribution: proxy delta when macro moved
        # to canvas center (a "removal" proxy proxy). Computing exactly requires
        # per-net hpwl + per-cell density isolation; we approximate by:
        #   - move macro to canvas center (with overlap legality temporarily disabled)
        #   - record proxy delta vs baseline
        #   - revert
        # This is O(num_hard) calls to the evaluator and gives the magnitude each
        # macro contributes by being at its current position.
        cw_local = float(benchmark.canvas_width) / 2.0
        ch_local = float(benchmark.canvas_height) / 2.0
        baseline_p = init_cost["proxy"]
        scores = []
        for idx in hard_movable:
            cur_xy = (
                float(evaluator.placement[idx, 0]),
                float(evaluator.placement[idx, 1]),
            )
            try:
                evaluator.move(idx, (cw_local, ch_local))
                p = evaluator.current_cost()["proxy"]
                evaluator.revert()
                scores.append((idx, p - baseline_p))
            except Exception:
                # Some moves may be illegal (overlap); skip those
                scores.append((idx, 0.0))
        # Larger positive delta when moved-to-center == this macro is
        # contributing negatively; we want to destroy macros whose CURRENT
        # position is "expensive" relative to a center move. Smaller (or
        # negative) delta means the macro is at a good position relative to
        # center; we don't want to destroy those.
        # Pick the K with the smallest delta (most "stuck", most likely to
        # benefit from a different position).
        scores.sort(key=lambda x: x[1])
        destroy = sorted([s[0] for s in scores[: args.destroy_size]])
        print(
            f"[seed={args.seed}] cost-aware destroy picked indices "
            f"{destroy[:5]}{'...' if len(destroy) > 5 else ''} "
            f"(top deltas {[f'{s[1]:.4f}' for s in scores[:3]]})",
            flush=True,
        )

    # ── True destroy: relocate destroyed macros ──
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.cpu().numpy()
    perturbed = evaluator.placement.detach().clone().cpu().numpy()
    for idx in destroy:
        half_w = sizes[idx, 0] / 2.0
        half_h = sizes[idx, 1] / 2.0
        if args.reinsert == "uniform":
            new_x = float(rng.uniform(half_w, cw - half_w))
            new_y = float(rng.uniform(half_h, ch - half_h))
        else:  # jitter
            cur_x = perturbed[idx, 0]
            cur_y = perturbed[idx, 1]
            new_x = float(np.clip(
                cur_x + rng.normal(0.0, args.jitter_sigma * cw),
                half_w, cw - half_w,
            ))
            new_y = float(np.clip(
                cur_y + rng.normal(0.0, args.jitter_sigma * ch),
                half_h, ch - half_h,
            ))
        perturbed[idx, 0] = new_x
        perturbed[idx, 1] = new_y
    perturbed_t = torch.from_numpy(perturbed).to(torch.float32)
    perturbed_t, _ = project_overlaps(perturbed_t, benchmark)

    # Rebuild evaluator on perturbed placement (cheaper than tracking many moves)
    evaluator, plc = _build_evaluator(benchmark, perturbed_t)
    perturbed_cost = evaluator.current_cost()

    print(
        f"[seed={args.seed}] baseline={baseline_proxy:.5f}, destroying {len(destroy)} "
        f"macros (random reinsert), perturbed proxy={perturbed_cost['proxy']:.5f}, "
        f"running CD on subset for {args.inner_budget:.0f}s",
        flush=True,
    )

    def _log(msg: str) -> None:
        print(f"[seed={args.seed}] {msg}", flush=True)

    stats = run_cd(
        evaluator=evaluator, benchmark=benchmark, plc=plc,
        movable=destroy, time_budget_s=args.inner_budget, log_fn=_log,
    )
    final_cost = evaluator.current_cost()
    final_placement = evaluator.placement.detach().clone().cpu().to(torch.float32)
    overlaps = compute_overlap_metrics(final_placement, benchmark)

    delta = baseline_proxy - final_cost["proxy"]
    pct = (delta / baseline_proxy) * 100.0

    result = {
        "benchmark": bench_name,
        "seed": args.seed,
        "destroy_size": args.destroy_size,
        "destroy_strategy": args.destroy_strategy,
        "reinsert": args.reinsert,
        "jitter_sigma": args.jitter_sigma if args.reinsert == "jitter" else None,
        "destroy_indices": destroy,
        "inner_budget_s": args.inner_budget,
        "baseline_proxy": float(baseline_proxy),
        "perturbed_proxy": float(perturbed_cost["proxy"]),
        "final_proxy": float(final_cost["proxy"]),
        "delta_proxy": float(delta),
        "pct_improvement": float(pct),
        "breakdown_final": {k: float(v) for k, v in final_cost.items()},
        "sweeps": int(stats["sweeps"]),
        "moves": int(stats["total_moves"]),
        "overlaps": int(overlaps["overlap_count"]),
        "wall_s": time.perf_counter() - t0,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(
        f"[seed={args.seed}] DONE final={final_cost['proxy']:.5f} "
        f"delta={delta:+.5f} ({pct:+.2f}%) sweeps={stats['sweeps']} "
        f"moves={stats['total_moves']} wall={time.perf_counter()-t0:.1f}s",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    b = sub.add_parser("baseline")
    b.add_argument("--benchmark", required=True)
    b.add_argument("--budget", type=float, required=True)
    b.add_argument("--out", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--in", dest="in_path", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--destroy-size", type=int, required=True)
    s.add_argument("--inner-budget", type=float, required=True)
    s.add_argument("--out", required=True)
    s.add_argument(
        "--destroy-strategy", choices=["random", "cost_aware"], default="random",
        help="Which macros to destroy. cost_aware picks top-K by per-macro proxy cost contribution.",
    )
    s.add_argument(
        "--reinsert", choices=["uniform", "jitter"], default="uniform",
        help="How to reinsert destroyed macros. uniform = any legal canvas position; "
             "jitter = current pos + Gaussian(sigma * canvas).",
    )
    s.add_argument(
        "--jitter-sigma", type=float, default=0.05,
        help="Jitter sigma as fraction of canvas (only used when --reinsert=jitter).",
    )

    args = ap.parse_args()
    if args.mode == "baseline":
        run_baseline(args)
    else:
        run_sample(args)


if __name__ == "__main__":
    main()
