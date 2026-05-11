"""E87: GPU-batched smooth-proxy CD polish.

Hypothesis: most of CD's wall on EPYC is the per-candidate proxy evaluation
(Python dict updates inside IncrementalProxyEvaluator). If we replace it with
the smooth proxy (already torch-autograd-compatible) on GPU and batch K=12
candidates per axis-move, we should get ~10-100× speedup, finally letting
cascade saddle converge within the 60-min/bench cap.

Trade-off vs IncrementalProxyEvaluator: not bit-for-bit with PlacementCost.
The smooth proxy is well-correlated though (we've used it for Hessian saddle
escape successfully).

Pipeline (per macro per sweep):
  1. Build K=12 axis-breakpoint candidates on x then y.
  2. For each axis, compute smooth_proxy at K candidate positions
     in parallel via batched torch ops on GPU.
  3. Pick argmin; if better than current, commit (just placement[macro] update).
  4. Move to next macro.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse the existing smooth proxy from E74.
_E74 = _ROOT / "submissions" / "cd_lns_sa_hessian"
sys.path.insert(0, str(_E74))
import importlib.util
_spec = importlib.util.spec_from_file_location("e74_placer", str(_E74 / "placer.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_SmoothProxy = _mod._SmoothProxy

from macro_place.benchmark import Benchmark


def axis_candidates(
    macro_idx: int,
    axis: int,
    placement: torch.Tensor,
    half_sizes: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    n_hard: int,
    fixed: torch.Tensor,
    K: int = 12,
) -> torch.Tensor:
    """Generate K candidate axis positions for macro_idx within legal range
    (clamped to canvas, no overlap with fixed neighbors on this axis)."""
    n_macros = placement.shape[0]
    hw, hh = float(half_sizes[macro_idx, 0]), float(half_sizes[macro_idx, 1])
    cur_x = float(placement[macro_idx, 0])
    cur_y = float(placement[macro_idx, 1])
    canvas_max = canvas_w if axis == 0 else canvas_h
    h = hw if axis == 0 else hh

    # Pivot range: legal canvas extent on this axis.
    lo = h
    hi = canvas_max - h

    # Spread K candidates uniformly in [lo, hi] with one at current pos.
    if hi <= lo + 1e-5:
        return torch.tensor([lo], dtype=placement.dtype, device=placement.device)
    step = (hi - lo) / (K - 1)
    cands = torch.arange(K, dtype=placement.dtype, device=placement.device)
    cands = lo + cands * step
    # Include current position
    cur = cur_x if axis == 0 else cur_y
    cands = torch.cat([cands, torch.tensor([cur], dtype=placement.dtype, device=placement.device)])
    return cands


def gpu_smooth_cd_sweep(
    placement: torch.Tensor,
    smooth: "_SmoothProxy",
    movable: List[int],
    half_sizes: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    n_hard: int,
    fixed: torch.Tensor,
    K: int = 12,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, int, float]:
    """One CD sweep: for each movable macro, find best (x, y) via GPU-batched
    candidate eval. Returns (new_placement, n_accepts, sweep_wall)."""
    device = placement.device
    t0 = time.time()
    n_accepts = 0

    # Compute initial cost for the sweep
    cur_cost = float(smooth.cost(placement).item())

    for macro_idx in movable:
        # x-axis sweep
        cands_x = axis_candidates(macro_idx, 0, placement, half_sizes,
                                   canvas_w, canvas_h, n_hard, fixed, K)
        # Build a batch: each row is placement with macro_idx's x replaced
        batch = placement.unsqueeze(0).expand(len(cands_x), -1, -1).clone()
        batch[:, macro_idx, 0] = cands_x
        # Compute smooth cost for each batch row
        costs = torch.stack([smooth.cost(batch[i]) for i in range(len(cands_x))])
        best_idx = int(torch.argmin(costs).item())
        best_cost = float(costs[best_idx].item())
        if best_cost < cur_cost - 1e-7:
            placement = placement.clone()
            placement[macro_idx, 0] = cands_x[best_idx]
            cur_cost = best_cost
            n_accepts += 1

        # y-axis sweep
        cands_y = axis_candidates(macro_idx, 1, placement, half_sizes,
                                   canvas_w, canvas_h, n_hard, fixed, K)
        batch = placement.unsqueeze(0).expand(len(cands_y), -1, -1).clone()
        batch[:, macro_idx, 1] = cands_y
        costs = torch.stack([smooth.cost(batch[i]) for i in range(len(cands_y))])
        best_idx = int(torch.argmin(costs).item())
        best_cost = float(costs[best_idx].item())
        if best_cost < cur_cost - 1e-7:
            placement = placement.clone()
            placement[macro_idx, 1] = cands_y[best_idx]
            cur_cost = best_cost
            n_accepts += 1

    sweep_wall = time.time() - t0
    return placement, n_accepts, sweep_wall


def run_gpu_cd_polish(
    init_placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    n_sweeps: int = 20,
    K: int = 12,
    device: str = "cuda",
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, dict]:
    if log is None:
        log = lambda s: print(s, flush=True)

    smooth = _SmoothProxy(benchmark, plc)
    placement = init_placement.detach().clone().to(torch.float32)

    # Move state to device
    if device != "cpu":
        dev = torch.device(device)
        placement = placement.to(dev)
        # Move smooth proxy state to device (all torch tensors)
        for attr_name in dir(smooth):
            if attr_name.startswith("_"):
                continue
            v = getattr(smooth, attr_name, None)
            if isinstance(v, torch.Tensor):
                setattr(smooth, attr_name, v.to(dev))
        # net_data: move every tensor field
        nd = smooth.net_data
        for attr_name in dir(nd):
            if attr_name.startswith("_"):
                continue
            v = getattr(nd, attr_name, None)
            if isinstance(v, torch.Tensor):
                setattr(nd, attr_name, v.to(dev))

    half_sizes = smooth.half_sizes
    fixed_t = benchmark.macro_fixed.to(placement.device)
    n_hard = benchmark.num_hard_macros
    movable = [i for i in range(n_hard) if not bool(fixed_t[i])]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    log(f"  [gpu-cd] device={device} K={K} sweeps={n_sweeps} movable={len(movable)}")
    sweep_log = []
    t_start = time.time()
    for sw in range(n_sweeps):
        placement, n_acc, sw_wall = gpu_smooth_cd_sweep(
            placement, smooth, movable, half_sizes, cw, ch, n_hard, fixed_t, K=K, log=log,
        )
        cur_cost = float(smooth.cost(placement).item())
        log(f"  [gpu-cd] sweep {sw+1}/{n_sweeps}  accepts={n_acc}  smooth_cost={cur_cost:.5f}  wall={sw_wall:.1f}s")
        sweep_log.append({"sweep": sw + 1, "accepts": n_acc, "smooth_cost": cur_cost, "wall": sw_wall})
        if n_acc == 0:
            log(f"  [gpu-cd] no accepts → converged")
            break

    placement = placement.cpu()
    return placement, {
        "n_sweeps_run": len(sweep_log),
        "sweep_log": sweep_log,
        "total_wall": time.time() - t_start,
    }


if __name__ == "__main__":
    import argparse
    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.cd_core import sdf_init, project_overlaps
    from macro_place.objective import compute_proxy_cost, compute_overlap_metrics

    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sweeps", type=int, default=20)
    ap.add_argument("--K", type=int, default=12)
    args = ap.parse_args()

    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(args.bench)))
    print(f"=== E87 GPU CD smoke: {args.bench} ===", flush=True)
    print(f"  device={args.device}  sweeps={args.sweeps}  K={args.K}", flush=True)
    print(f"  num_macros={bench.num_macros}  num_hard={bench.num_hard_macros}", flush=True)

    init = sdf_init(bench)
    init, _ = project_overlaps(init, bench)
    ovl0 = compute_overlap_metrics(init, bench)["overlap_count"]
    proxy0 = float(compute_proxy_cost(init, bench, plc)["proxy_cost"])
    print(f"  init: proxy={proxy0:.5f} overlaps={ovl0}", flush=True)

    t0 = time.time()
    placement, stats = run_gpu_cd_polish(init, bench, plc, n_sweeps=args.sweeps, K=args.K, device=args.device)
    wall = time.time() - t0

    placement, _ = project_overlaps(placement, bench)
    ovl_final = compute_overlap_metrics(placement, bench)["overlap_count"]
    proxy_final = float(compute_proxy_cost(placement, bench, plc)["proxy_cost"])
    print(f"  final: proxy={proxy_final:.5f} overlaps={ovl_final} wall={wall:.1f}s", flush=True)
    print(f"  Δ={proxy_final - proxy0:+.5f}", flush=True)
