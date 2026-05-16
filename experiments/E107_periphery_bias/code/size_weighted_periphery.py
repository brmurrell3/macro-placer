"""E107 variant: size-weighted periphery push.

Hypothesis: ReMaP-style benefit comes from placing LARGE macros at
periphery (creating "ring of memories" around std-cell-dense center).
Standard periphery push moves all macros equally toward nearest edge,
which is information-free.

Size-weighted: push factor = α * (macro_area / max_macro_area). Big
macros get full push; small macros stay nearly in place.
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
sys.path.insert(0, str(Path(__file__).parent))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost
from decompose_spike import edge_dist_mean, polish_and_score, measure_rms


def push_periphery_size_weighted(positions, benchmark, alpha):
    """Push each macro α fraction toward nearest edge, weighted by
    relative macro area (big macros pushed harder)."""
    cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
    pos = positions.cpu().numpy().copy()
    sizes = benchmark.macro_sizes.cpu().numpy()
    half_w = sizes[:, 0] / 2; half_h = sizes[:, 1] / 2
    areas = sizes[:, 0] * sizes[:, 1]
    max_area = float(areas.max())
    fixed = benchmark.macro_fixed.cpu().numpy()

    for i in range(pos.shape[0]):
        if fixed[i]:
            continue
        weight = float(areas[i] / max_area)  # ∈ [0, 1]
        eff_alpha = alpha * weight
        x, y = pos[i, 0], pos[i, 1]
        dx_l = x - half_w[i]; dx_r = (cw - half_w[i]) - x
        dy_b = y - half_h[i]; dy_t = (ch - half_h[i]) - y
        dists = [dx_l, dx_r, dy_b, dy_t]
        nearest = int(np.argmin([abs(d) for d in dists]))
        if nearest == 0:   pos[i, 0] = x - eff_alpha * dx_l
        elif nearest == 1: pos[i, 0] = x + eff_alpha * dx_r
        elif nearest == 2: pos[i, 1] = y - eff_alpha * dy_b
        else:              pos[i, 1] = y + eff_alpha * dy_t
        pos[i, 0] = np.clip(pos[i, 0], half_w[i], cw - half_w[i])
        pos[i, 1] = np.clip(pos[i, 1], half_h[i], ch - half_h[i])
    return torch.tensor(pos, dtype=torch.float32)


def main(bench_name="ariane133", alphas=(0.01, 0.05, 0.1, 0.2)):
    bench_dir = find_benchmark_dir(bench_name)
    bench, plc = load_benchmark_from_dir(str(bench_dir))
    cw, ch = float(bench.canvas_width), float(bench.canvas_height)

    placement_path = _ROOT / f"experiments/E91_dp_full_polish/results/{bench_name}_dp_full_polish.pt"
    data = torch.load(placement_path, map_location="cpu", weights_only=False)
    baseline = data["placement"].to(torch.float32) if isinstance(data, dict) else data.to(torch.float32)
    base_proxy = float(compute_proxy_cost(baseline, bench, plc)["proxy_cost"])
    base_edge = edge_dist_mean(baseline, cw, ch)

    sizes = bench.macro_sizes.cpu().numpy()
    areas = sizes[:, 0] * sizes[:, 1]
    print(f"=== {bench_name} ===")
    print(f"baseline: proxy={base_proxy:.5f} edge_dist={base_edge:.4f}")
    print(f"macro areas: min={areas.min():.2f} max={areas.max():.2f} mean={areas.mean():.2f}")
    print(f"  area ratios: top-1 weight=1.00, mean weight={float(areas.mean() / areas.max()):.3f}")

    print(f"\n{'α (size-weighted)':>18} {'proxy':>9} {'Δprx':>7} {'edge':>7} {'Δedg':>7} {'ovl':>4} {'wall':>5}")
    for alpha in alphas:
        pushed = push_periphery_size_weighted(baseline, bench, alpha)
        rms = measure_rms(pushed, baseline)
        t = time.time()
        polished = polish_and_score(pushed, bench, plc, cd_budget_s=180.0)
        p = float(compute_proxy_cost(polished, bench, plc)["proxy_cost"])
        e = edge_dist_mean(polished, cw, ch)
        ovl = compute_overlap_metrics(polished, bench)["overlap_count"]
        dp = (p - base_proxy) / base_proxy * 100
        de = (e - base_edge) / base_edge * 100
        wall = time.time() - t
        tag = "*" if ovl == 0 and dp < 0 else " "
        print(f"  {tag} α={alpha:.3f}  RMS={rms:.2f}  {p:9.5f} {dp:+6.2f}% {e:7.4f} {de:+6.2f}% {ovl:4d} {wall:4.0f}s")


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "ariane133"
    main(bench)
