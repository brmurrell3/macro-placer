"""Speedup micro-benchmark: IncrementalProxyEvaluator vs full compute_proxy_cost.

Run 10 000 random single-macro moves on ibm10 and compare wall time. Print
speedup ratio.

Usage:
    uv run python scripts/bench_incremental.py
    uv run python scripts/bench_incremental.py --benchmark ibm01 --n-moves 1000
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


TESTCASE_ROOT = Path("external/MacroPlacement/Testcases/ICCAD04")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="ibm10")
    parser.add_argument("--n-moves", type=int, default=10_000)
    parser.add_argument("--n-full-recompute", type=int, default=0,
                        help="If 0, defaults to min(n-moves, 30) for full-recompute timing "
                             "(scaled small because ibm10 full recompute is ~20 s/call).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = TESTCASE_ROOT / args.benchmark
    if not path.exists():
        raise SystemExit(f"benchmark missing: {path}")

    print(f"Loading {args.benchmark} ...")
    bench, plc = load_benchmark_from_dir(str(path))
    placement = bench.macro_positions.detach().clone().to(torch.float64)

    n_moves = args.n_moves
    n_full = args.n_full_recompute or min(n_moves, 30)
    rng = random.Random(args.seed)

    # Pre-compute the move sequence
    movable = (~bench.macro_fixed).nonzero(as_tuple=True)[0].tolist()
    moves = []
    for _ in range(n_moves):
        i = rng.choice(movable)
        w = float(bench.macro_sizes[i, 0])
        h = float(bench.macro_sizes[i, 1])
        nx = rng.uniform(w / 2 + 1e-3, bench.canvas_width - w / 2 - 1e-3)
        ny = rng.uniform(h / 2 + 1e-3, bench.canvas_height - h / 2 - 1e-3)
        moves.append((i, nx, ny))

    print(f"Benchmark: {args.benchmark} "
          f"(num_macros={bench.num_macros}, num_nets={int(plc.net_cnt)}, "
          f"grid={plc.grid_col}x{plc.grid_row})")

    # ── Build the evaluator (init cost is one-time) ──
    print("Initializing IncrementalProxyEvaluator ...")
    t0 = time.perf_counter()
    evaluator = IncrementalProxyEvaluator(bench, plc, placement)
    init_cost = evaluator.current_cost()
    init_time = time.perf_counter() - t0
    print(f"  init time = {init_time:.3f} s, cost={init_cost['proxy']:.5f}")

    # ── Time n_moves of incremental updates ──
    print(f"Timing {n_moves} incremental moves ...")
    t0 = time.perf_counter()
    for (i, nx, ny) in moves:
        evaluator.move(i, (nx, ny))
    incr_time = time.perf_counter() - t0
    incr_per_move = incr_time / n_moves
    print(f"  total incremental wall time: {incr_time:.3f} s "
          f"({incr_per_move*1000:.3f} ms/move, {1/incr_per_move:.1f} moves/s)")

    # ── Time n_full full-recompute calls (extrapolate to n_moves) ──
    print(f"Timing {n_full} full compute_proxy_cost calls ...")
    # Reset state by reloading benchmark
    bench2, plc2 = load_benchmark_from_dir(str(path))
    placement2 = bench2.macro_positions.detach().clone().to(torch.float64)
    full_moves = moves[:n_full]
    t0 = time.perf_counter()
    for (i, nx, ny) in full_moves:
        placement2[i, 0] = nx
        placement2[i, 1] = ny
        _ = compute_proxy_cost(placement2, bench2, plc2)
    full_time = time.perf_counter() - t0
    full_per_move = full_time / n_full
    full_extrap_total = full_per_move * n_moves
    print(f"  total full-recompute wall time ({n_full} calls): {full_time:.3f} s "
          f"({full_per_move*1000:.3f} ms/call)")
    print(f"  extrapolated to {n_moves} calls: {full_extrap_total:.3f} s")

    # ── Speedup ──
    speedup = full_per_move / incr_per_move
    print()
    print("=" * 60)
    print(f"SPEEDUP: {speedup:.1f}x  (incremental vs full recompute)")
    print(f"  incremental: {incr_per_move*1000:.3f} ms/move")
    print(f"  full recom:  {full_per_move*1000:.3f} ms/call")
    print("=" * 60)


if __name__ == "__main__":
    main()
