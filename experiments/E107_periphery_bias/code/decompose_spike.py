"""E107 decompose: is the -0.95% on ariane133 from periphery direction,
or just any small perturbation?

Test 4 cases at same RMS-displacement magnitude:
  A. periphery push (move toward nearest edge)
  B. random Gaussian kick
  C. center push (move toward canvas center) — opposite direction
  D. no push, just re-polish baseline (control)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def edge_dist_mean(positions, canvas_w, canvas_h):
    pos = positions.cpu().numpy()
    dx = np.minimum(pos[:, 0], canvas_w - pos[:, 0]) / canvas_w
    dy = np.minimum(pos[:, 1], canvas_h - pos[:, 1]) / canvas_h
    return float(np.minimum(dx, dy).mean())


def push_periphery(positions, benchmark, alpha):
    cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
    pos = positions.cpu().numpy().copy()
    half_w = (benchmark.macro_sizes[:, 0] / 2).cpu().numpy()
    half_h = (benchmark.macro_sizes[:, 1] / 2).cpu().numpy()
    fixed = benchmark.macro_fixed.cpu().numpy()
    for i in range(pos.shape[0]):
        if fixed[i]:
            continue
        x, y = pos[i, 0], pos[i, 1]
        dx_l = x - half_w[i]; dx_r = (cw - half_w[i]) - x
        dy_b = y - half_h[i]; dy_t = (ch - half_h[i]) - y
        dists = [dx_l, dx_r, dy_b, dy_t]
        nearest = int(np.argmin([abs(d) for d in dists]))
        if nearest == 0:   pos[i, 0] = x - alpha * dx_l
        elif nearest == 1: pos[i, 0] = x + alpha * dx_r
        elif nearest == 2: pos[i, 1] = y - alpha * dy_b
        else:              pos[i, 1] = y + alpha * dy_t
        pos[i, 0] = np.clip(pos[i, 0], half_w[i], cw - half_w[i])
        pos[i, 1] = np.clip(pos[i, 1], half_h[i], ch - half_h[i])
    return torch.tensor(pos, dtype=torch.float32)


def push_center(positions, benchmark, alpha):
    cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
    cx, cy = cw / 2, ch / 2
    pos = positions.cpu().numpy().copy()
    half_w = (benchmark.macro_sizes[:, 0] / 2).cpu().numpy()
    half_h = (benchmark.macro_sizes[:, 1] / 2).cpu().numpy()
    fixed = benchmark.macro_fixed.cpu().numpy()
    for i in range(pos.shape[0]):
        if fixed[i]:
            continue
        pos[i, 0] = pos[i, 0] + alpha * (cx - pos[i, 0])
        pos[i, 1] = pos[i, 1] + alpha * (cy - pos[i, 1])
        pos[i, 0] = np.clip(pos[i, 0], half_w[i], cw - half_w[i])
        pos[i, 1] = np.clip(pos[i, 1], half_h[i], ch - half_h[i])
    return torch.tensor(pos, dtype=torch.float32)


def push_random(positions, benchmark, rms_target, seed=42):
    """Random Gaussian kick with controlled RMS magnitude."""
    cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
    pos = positions.cpu().numpy().copy()
    half_w = (benchmark.macro_sizes[:, 0] / 2).cpu().numpy()
    half_h = (benchmark.macro_sizes[:, 1] / 2).cpu().numpy()
    fixed = benchmark.macro_fixed.cpu().numpy()
    rng = np.random.RandomState(seed)
    delta = rng.randn(*pos.shape).astype(np.float64) * rms_target
    for i in range(pos.shape[0]):
        if fixed[i]:
            continue
        pos[i] += delta[i]
        pos[i, 0] = np.clip(pos[i, 0], half_w[i], cw - half_w[i])
        pos[i, 1] = np.clip(pos[i, 1], half_h[i], ch - half_h[i])
    return torch.tensor(pos, dtype=torch.float32)


def polish_and_score(perturbed, bench, plc, cd_budget_s=180.0):
    legal, _ = project_overlaps(perturbed, bench)
    fixed = bench.macro_fixed.cpu().numpy()
    movable_idx = [i for i in range(bench.num_macros) if not bool(fixed[i])]
    evaluator = IncrementalProxyEvaluator(bench, plc, legal.clone())
    run_cd_adaptive(
        evaluator, bench, plc, movable_idx,
        min_time_s=30.0, hard_cap_s=cd_budget_s,
        patience=3, plateau_threshold=0.001, log_fn=None,
    )
    final = evaluator.placement.detach().clone().to(torch.float32)
    return final


def measure_rms(pos_new, pos_old):
    delta = (pos_new - pos_old).cpu().numpy()
    return float(np.sqrt(np.mean(delta**2)))


def main(bench_name="ariane133", alpha=0.01):
    bench_dir = find_benchmark_dir(bench_name)
    bench, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"[bench] {bench_name}: {bench.num_macros} macros, canvas={bench.canvas_width:.0f}×{bench.canvas_height:.0f}")

    placement_path = _ROOT / f"experiments/E91_dp_full_polish/results/{bench_name}_dp_full_polish.pt"
    data = torch.load(placement_path, map_location="cpu", weights_only=False)
    baseline = data["placement"].to(torch.float32)
    cw, ch = float(bench.canvas_width), float(bench.canvas_height)

    base_proxy = float(compute_proxy_cost(baseline, bench, plc)["proxy_cost"])
    base_edge = edge_dist_mean(baseline, cw, ch)
    print(f"[baseline] proxy={base_proxy:.5f} edge_dist={base_edge:.4f}")

    # Measure RMS of periphery push to match other tests.
    periph = push_periphery(baseline, bench, alpha)
    periph_rms = measure_rms(periph, baseline)
    print(f"[RMS calibration] periphery push α={alpha} → RMS={periph_rms:.4f}")

    cases = [
        ("D. no push (control)", baseline.clone()),
        ("A. periphery push", periph),
        ("B. random kick", push_random(baseline, bench, periph_rms)),
        ("C. center push", push_center(baseline, bench, alpha)),
    ]

    print(f"\n{'case':30s} {'proxy':>9} {'Δproxy':>9} {'edge':>7} {'Δedge':>9} {'ovl':>4} {'wall':>5}")
    print("-" * 80)
    for name, perturbed in cases:
        t0 = time.time()
        rms = measure_rms(perturbed, baseline)
        polished = polish_and_score(perturbed, bench, plc, cd_budget_s=180.0)
        p = float(compute_proxy_cost(polished, bench, plc)["proxy_cost"])
        e = edge_dist_mean(polished, cw, ch)
        ovl = compute_overlap_metrics(polished, bench)["overlap_count"]
        dp = (p - base_proxy) / base_proxy * 100
        de = (e - base_edge) / base_edge * 100
        wall = time.time() - t0
        print(f"{name:30s} {p:9.5f} {dp:+8.2f}% {e:7.4f} {de:+8.2f}% {ovl:4d} {wall:5.0f}s")


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "ariane133"
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    main(bench, alpha)
