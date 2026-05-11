"""E27 — Run a single CD trajectory from a chosen init, log per-sweep proxy.

This is the diagnostic primitive for THE GATE: do the post-E25 plateaus all
sit in one big basin (H_0 = 1 long-lived feature) or several distinct deep
basins (H_0 = many features below the E25 floor)?

For each (benchmark, init, seed) combo, this script:

  1. Loads the benchmark.
  2. Builds an initial placement using one of five strategies:
       sdf / sdf_jitter / uniform / greedy / dpo
  3. Project-overlaps to legalize.
  4. Runs CD plateau-detection sweeps via run_cd_adaptive primitives,
     recording per-sweep {sweep, elapsed, proxy, wl, density, congestion}.
  5. Dumps the trajectory + final placement to JSON.

The JSON schema (one file per trajectory):

  {
    "benchmark":   str,
    "init":        str,                  # one of {sdf, sdf_jitter, uniform, greedy, dpo}
    "seed":        int,
    "budget_s":    float,
    "init_proxy":  float,
    "trajectory":  [
        {"sweep": int, "t": float, "proxy": float,
         "wl": float, "d": float, "c": float},
        ...
    ],
    "final_proxy":     float,
    "exit_reason":     str,
    "wall_total_s":    float,
    "final_placement": [[x, y], ...]     # full placement, hard + soft
  }

CLI:

  uv run python experiments/E27_basin_persistence/code/run_trajectory.py \\
      --benchmark ibm11 --init sdf --seed 42 \\
      --budget-s 600 --out trajectories/ibm11_sdf_42.json

The orchestrator script (`run_orchestrator.sh`) sequences the (bench, init,
seed) combos. Parent session may parallelize via background processes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

# Repo root → sys.path so `macro_place.*` imports resolve when run from
# anywhere via uv run python.
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import (
    _grid_lines,
    _sweep_macros,
    project_overlaps,
)
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics
from macro_place.sdf_init import SDFPlacer


# ── Init builders ──────────────────────────────────────────────────────────


def build_sdf_init(benchmark: Benchmark, seed: int) -> torch.Tensor:
    """Vanilla SDF init at chosen seed."""
    return SDFPlacer(seed=seed).place(benchmark)


def build_sdf_jitter_init(
    benchmark: Benchmark, seed: int, sigma_frac: float = 0.05
) -> torch.Tensor:
    """SDF init then add Gaussian noise σ = sigma_frac * canvas_diag.

    Noise is applied only to hard movable macros, then we re-legalize via
    project_overlaps. Soft macros and fixed macros stay where SDF placed
    them.
    """
    placement = SDFPlacer(seed=seed).place(benchmark)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    canvas_diag = math.sqrt(cw ** 2 + ch ** 2)
    sigma = sigma_frac * canvas_diag

    rng = np.random.default_rng(seed=seed + 10000)
    pos_np = placement.cpu().numpy().astype(np.float64).copy()
    sizes_np = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed_np = benchmark.macro_fixed.cpu().numpy()
    n_hard = benchmark.num_hard_macros

    for i in range(n_hard):
        if fixed_np[i]:
            continue
        half_w = sizes_np[i, 0] / 2
        half_h = sizes_np[i, 1] / 2
        nx = pos_np[i, 0] + rng.normal(0.0, sigma)
        ny = pos_np[i, 1] + rng.normal(0.0, sigma)
        pos_np[i, 0] = float(np.clip(nx, half_w, cw - half_w))
        pos_np[i, 1] = float(np.clip(ny, half_h, ch - half_h))

    jittered = torch.tensor(pos_np, dtype=placement.dtype)
    legalized, _ = project_overlaps(jittered, benchmark)
    return legalized


def build_uniform_init(benchmark: Benchmark, seed: int) -> torch.Tensor:
    """Random uniform legal init via shuffled-greedy rejection.

    For each hard movable macro (in random order), draw uniform (cx, cy)
    in the legal canvas band; reject if overlaps already-placed hard
    macros; retry up to 1000 times. Fall back to SDF position for any
    macro that fails to place (rare, only on densely packed benchmarks).
    """
    sdf_fallback = SDFPlacer(seed=seed).place(benchmark)
    placement = sdf_fallback.clone()

    rng = np.random.default_rng(seed=seed + 20000)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes_np = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed_np = benchmark.macro_fixed.cpu().numpy()
    n_hard = benchmark.num_hard_macros

    movable_hard = [i for i in range(n_hard) if not bool(fixed_np[i])]
    rng.shuffle(movable_hard)

    placed = np.zeros(n_hard, dtype=bool)
    # Fixed macros count as already placed (they cannot move).
    for i in range(n_hard):
        if fixed_np[i]:
            placed[i] = True

    pos_np = placement.cpu().numpy().astype(np.float64).copy()

    for idx in movable_hard:
        half_w = sizes_np[idx, 0] / 2
        half_h = sizes_np[idx, 1] / 2
        success = False
        for _attempt in range(1000):
            cx = rng.uniform(half_w, cw - half_w)
            cy = rng.uniform(half_h, ch - half_h)
            # Check overlap with already-placed hard macros.
            other = np.flatnonzero(placed)
            if len(other) > 0:
                dx = np.abs(pos_np[other, 0] - cx)
                dy = np.abs(pos_np[other, 1] - cy)
                min_dx = half_w + sizes_np[other, 0] / 2
                min_dy = half_h + sizes_np[other, 1] / 2
                conflict = (dx < min_dx) & (dy < min_dy)
                if conflict.any():
                    continue
            pos_np[idx, 0] = cx
            pos_np[idx, 1] = cy
            placed[idx] = True
            success = True
            break
        # If we exhausted 1000 attempts, leave SDF fallback in place;
        # project_overlaps below will clean any residual overlaps.

    rebuilt = torch.tensor(pos_np, dtype=placement.dtype)
    legalized, _ = project_overlaps(rebuilt, benchmark)
    return legalized


def build_greedy_init(benchmark: Benchmark) -> torch.Tensor:
    """GreedyRowPlacer — shelf packing, deterministic."""
    greedy_path = _ROOT / "submissions" / "examples" / "greedy_row_placer.py"
    spec_loader = _import_path("greedy_row_placer", greedy_path)
    GreedyRowPlacer = spec_loader.GreedyRowPlacer
    return GreedyRowPlacer().place(benchmark)


def build_dpo_init(benchmark: Benchmark, seed: int) -> torch.Tensor:
    """DPO BestOfV2 init from writeup/archive/submissions/dpo/."""
    dpo_path = _ROOT / "writeup" / "archive" / "submissions" / "dpo" / "best_of_v2_placer.py"
    mod = _import_path("dpo_best_of_v2", dpo_path)
    BestOfV2Placer = mod.BestOfV2Placer
    return BestOfV2Placer(seed=seed).place(benchmark)


def _import_path(name: str, path: Path):
    """Import a Python module from a file path (no editing of sys.path)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_init(benchmark: Benchmark, init: str, seed: int) -> torch.Tensor:
    """Dispatch on `init` flag, then project-overlaps once more for safety.

    Returns a [num_macros, 2] tensor.
    """
    if init == "sdf":
        placement = build_sdf_init(benchmark, seed)
    elif init == "sdf_jitter":
        placement = build_sdf_jitter_init(benchmark, seed)
    elif init == "uniform":
        placement = build_uniform_init(benchmark, seed)
    elif init == "greedy":
        placement = build_greedy_init(benchmark)
    elif init == "dpo":
        placement = build_dpo_init(benchmark, seed)
    else:
        raise ValueError(f"unknown init: {init!r}")

    placement, _ = project_overlaps(placement, benchmark)
    return placement


# ── CD trajectory loop (mirrors run_cd_adaptive but logs per-sweep dicts) ──


def run_cd_with_trajectory(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    movable: list,
    hard_cap_s: float = 600.0,
    min_time_s: float = 60.0,
    patience: int = 3,
    plateau_threshold: float = 0.001,
) -> dict:
    """Adaptive CD with structured per-sweep telemetry.

    Mirrors `macro_place.cd_core.run_cd_adaptive` exactly — same exit
    conditions and same `_sweep_macros` body — but instead of taking a
    string log_fn, returns a list of per-sweep dicts (proxy, wl, d, c, t).
    """
    n_hard = benchmark.num_hard_macros
    grid_lines_x, grid_lines_y = _grid_lines(plc)

    cost_break = evaluator.current_cost()
    cur_cost = cost_break["proxy"]
    init_proxy = cur_cost

    trajectory: list = []
    sweep_idx = 0
    total_moves = 0
    deltas: deque = deque(maxlen=patience)
    exit_reason = "cap"

    t_start = time.perf_counter()
    # Sweep 0 = the init proxy (pre-CD). Useful for the persistence plot.
    trajectory.append({
        "sweep": 0,
        "t": 0.0,
        "proxy": float(cost_break["proxy"]),
        "wl": float(cost_break["wl"]),
        "d": float(cost_break["density"]),
        "c": float(cost_break["congestion"]),
    })

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= hard_cap_s:
            exit_reason = "cap"
            break
        sweep_idx += 1
        prev_cost = cur_cost

        rng = np.random.default_rng(seed=sweep_idx)
        order = list(movable)
        rng.shuffle(order)

        cur_cost, sweep_accepted, _, _ = _sweep_macros(
            evaluator, benchmark, grid_lines_x, grid_lines_y,
            order, cur_cost, n_hard,
            deadline_check=lambda: time.perf_counter() - t_start >= hard_cap_s,
        )

        elapsed = time.perf_counter() - t_start
        cost_break = evaluator.current_cost()
        cur_cost = cost_break["proxy"]
        total_moves += sweep_accepted

        delta = prev_cost - cur_cost
        deltas.append(delta)

        trajectory.append({
            "sweep": sweep_idx,
            "t": float(elapsed),
            "proxy": float(cost_break["proxy"]),
            "wl": float(cost_break["wl"]),
            "d": float(cost_break["density"]),
            "c": float(cost_break["congestion"]),
        })

        if elapsed >= hard_cap_s:
            exit_reason = "cap"
            break
        if (
            elapsed >= min_time_s
            and len(deltas) == patience
            and all(d < plateau_threshold for d in deltas)
        ):
            exit_reason = "plateau"
            break

    return {
        "trajectory": trajectory,
        "init_proxy": float(init_proxy),
        "final_proxy": float(cur_cost),
        "exit_reason": exit_reason,
        "sweeps": sweep_idx,
        "total_moves": total_moves,
        "wall_total_s": float(time.perf_counter() - t_start),
    }


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one CD trajectory from a diverse init for E27."
    )
    parser.add_argument("--benchmark", required=True,
                        help="benchmark name, e.g. ibm11")
    parser.add_argument("--init", required=True,
                        choices=["sdf", "sdf_jitter", "uniform", "greedy", "dpo"],
                        help="init strategy")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for stochastic inits (sdf/sdf_jitter/uniform/dpo)")
    parser.add_argument("--budget-s", type=float, default=600.0,
                        help="CD wall-clock cap per trajectory (seconds)")
    parser.add_argument("--min-time-s", type=float, default=60.0,
                        help="min CD wall before plateau exit can trigger")
    parser.add_argument("--patience", type=int, default=3,
                        help="plateau patience (sweeps with Δ < threshold)")
    parser.add_argument("--plateau-threshold", type=float, default=0.001,
                        help="per-sweep proxy delta below which counts as plateau")
    parser.add_argument("--out", required=True,
                        help="output JSON path (relative to cwd or absolute)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-sweep stdout")
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"[E27] benchmark={args.benchmark} init={args.init} "
              f"seed={args.seed} budget={args.budget_s:.0f}s "
              f"out={out_path}", flush=True)

    # 1. Load benchmark.
    bench_dir = find_benchmark_dir(args.benchmark)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    # 2. Build init.
    t_init0 = time.perf_counter()
    placement = build_init(benchmark, args.init, args.seed)
    init_wall = time.perf_counter() - t_init0
    init_overlaps = compute_overlap_metrics(placement, benchmark)
    if not args.quiet:
        print(f"[E27] init build wall={init_wall:.1f}s, "
              f"residual overlaps={init_overlaps['overlap_count']}", flush=True)

    # 3. Build evaluator.
    placement_f64 = placement.detach().clone().to(torch.float64)
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
    init_cost = evaluator.current_cost()
    if not args.quiet:
        print(f"[E27] init proxy={init_cost['proxy']:.5f} "
              f"[wl={init_cost['wl']:.4f} d={init_cost['density']:.4f} "
              f"c={init_cost['congestion']:.4f}]", flush=True)

    # 4. Run CD with structured logging.
    movable = [
        i for i in range(benchmark.num_macros)
        if not bool(benchmark.macro_fixed[i])
    ]
    cd_result = run_cd_with_trajectory(
        evaluator=evaluator,
        benchmark=benchmark,
        plc=plc,
        movable=movable,
        hard_cap_s=args.budget_s,
        min_time_s=args.min_time_s,
        patience=args.patience,
        plateau_threshold=args.plateau_threshold,
    )

    # 5. Pull final placement; preserve fixed macros.
    final_placement_f64 = evaluator.placement.detach().clone().cpu()
    fixed_mask = benchmark.macro_fixed
    if fixed_mask.any():
        original_positions = benchmark.macro_positions.to(torch.float64)
        final_placement_f64[fixed_mask] = original_positions[fixed_mask]
    final_placement_f32 = final_placement_f64.to(torch.float32)

    overlaps = compute_overlap_metrics(final_placement_f32, benchmark)
    if overlaps["overlap_count"] > 0:
        # Trajectory completed but ended up illegal — flag in JSON,
        # don't crash (the analyzer can decide what to do with it).
        if not args.quiet:
            print(f"[E27] WARNING: {overlaps['overlap_count']} residual "
                  f"overlaps in final placement", flush=True)

    # 6. Dump JSON.
    payload = {
        "benchmark": args.benchmark,
        "init": args.init,
        "seed": int(args.seed),
        "budget_s": float(args.budget_s),
        "init_proxy": cd_result["init_proxy"],
        "trajectory": cd_result["trajectory"],
        "final_proxy": cd_result["final_proxy"],
        "exit_reason": cd_result["exit_reason"],
        "sweeps": cd_result["sweeps"],
        "total_moves": cd_result["total_moves"],
        "wall_total_s": cd_result["wall_total_s"],
        "final_overlap_count": int(overlaps["overlap_count"]),
        "final_overlap_area": float(overlaps["total_overlap_area"]),
        "final_placement": final_placement_f32.cpu().numpy().tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)

    if not args.quiet:
        print(f"[E27] DONE init={cd_result['init_proxy']:.5f} → "
              f"final={cd_result['final_proxy']:.5f} "
              f"({cd_result['exit_reason']}, {cd_result['sweeps']} sweeps, "
              f"{cd_result['wall_total_s']:.1f}s) → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
