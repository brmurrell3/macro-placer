"""Single-run worker for the multi-init CD basin-diversity probe.

Runs SDFPlacer(seed=N) -> project_overlaps -> run_cd(budget) on one benchmark
and writes the result as JSON. Spawn K instances in parallel from a shell
driver to test whether different SDF seeds land CD in different basins.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--jitter", type=float, default=0.0,
        help="Gaussian jitter sigma as a fraction of canvas (applied to SDF "
             "output before legalization+CD). 0 = no jitter (control).",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    bench_dir = TESTCASE / args.benchmark
    benchmark, _ = load_benchmark_from_dir(str(bench_dir))

    placer = SDFPlacer(seed=args.seed)
    placement = placer.place(benchmark)

    if args.jitter > 0.0:
        import numpy as np
        rng = np.random.default_rng(seed=args.seed)
        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sigma_x = args.jitter * cw
        sigma_y = args.jitter * ch
        fixed = benchmark.macro_fixed.cpu().numpy()
        noise = rng.normal(
            loc=0.0,
            scale=[sigma_x, sigma_y],
            size=(benchmark.num_macros, 2),
        ).astype("float32")
        # Don't jitter fixed macros
        noise[fixed] = 0.0
        # Don't jitter soft macros either (only hard movables — keeps semantics simple)
        if n_hard < benchmark.num_macros:
            noise[n_hard:] = 0.0
        placement = placement + torch.from_numpy(noise)
        # Clamp into canvas (center within canvas with half-margin)
        sizes = benchmark.macro_sizes
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        placement[:, 0] = placement[:, 0].clamp(min=half_w, max=cw - half_w)
        placement[:, 1] = placement[:, 1].clamp(min=half_h, max=ch - half_h)

    placement, _ = project_overlaps(placement, benchmark)

    # Reload plc fresh for the evaluator (SDFPlacer mutates its own copy)
    _, plc = load_benchmark_from_dir(str(bench_dir))
    placement_f64 = placement.detach().clone().to(torch.float64)
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
    init_cost = evaluator.current_cost()

    movable = [
        i for i in range(benchmark.num_macros)
        if not bool(benchmark.macro_fixed[i])
    ]

    def _log(msg: str) -> None:
        print(f"[seed={args.seed}] {msg}", flush=True)

    _log(
        f"init proxy={init_cost['proxy']:.5f} "
        f"[wl={init_cost['wl']:.4f} d={init_cost['density']:.4f} "
        f"c={init_cost['congestion']:.4f}] — running CD "
        f"({args.budget:.0f}s budget)"
    )

    stats = run_cd(
        evaluator=evaluator,
        benchmark=benchmark,
        plc=plc,
        movable=movable,
        time_budget_s=args.budget,
        log_fn=_log,
    )
    final_cost = evaluator.current_cost()
    final_placement = evaluator.placement.detach().clone().cpu().to(torch.float32)
    overlaps = compute_overlap_metrics(final_placement, benchmark)

    result = {
        "benchmark": args.benchmark,
        "seed": args.seed,
        "jitter": args.jitter,
        "budget_s": args.budget,
        "init_proxy": float(init_cost["proxy"]),
        "final_proxy": float(final_cost["proxy"]),
        "breakdown_init": {k: float(v) for k, v in init_cost.items()},
        "breakdown_final": {k: float(v) for k, v in final_cost.items()},
        "sweeps": int(stats["sweeps"]),
        "moves": int(stats["total_moves"]),
        "gs_fallbacks": int(stats["total_gs_fallbacks"]),
        "overlaps": int(overlaps["overlap_count"]),
        "wall_total_s": time.perf_counter() - t0,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    _log(
        f"DONE final_proxy={final_cost['proxy']:.5f} "
        f"sweeps={stats['sweeps']} moves={stats['total_moves']} "
        f"wall={result['wall_total_s']:.1f}s"
    )


if __name__ == "__main__":
    main()
