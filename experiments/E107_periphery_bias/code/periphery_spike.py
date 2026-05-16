"""E107 spike: can we push macros toward periphery without destroying proxy?

Hypothesis: DAC25-ReMaP gets +34% WNS / +65% TNS from periphery-guided
relocation. Test whether we can move macros toward the nearest edge
(measured by `edge_dist`), re-legalize + CD-polish, and keep proxy
within budget (< +20% regression).

If yes → Path F is alive: ship `cd_lns_sa_cascade_periphery/placer.py`
that biases cascade saddle's polish toward more peripheral solutions.

Local test on ibm01 (M3 Max) — no cloud needed.
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


def edge_dist_mean(positions: torch.Tensor, canvas_w: float, canvas_h: float) -> float:
    """Mean normalized distance from each macro to the nearest canvas edge.
    0 = at edge, 0.5 = at center. Lower = more peripheral."""
    pos = positions.cpu().numpy()
    dx = np.minimum(pos[:, 0], canvas_w - pos[:, 0]) / canvas_w
    dy = np.minimum(pos[:, 1], canvas_h - pos[:, 1]) / canvas_h
    return float(np.minimum(dx, dy).mean())


def push_to_periphery(
    positions: torch.Tensor,
    benchmark,
    alpha: float,
) -> torch.Tensor:
    """Move each (movable) macro alpha-fraction of its distance toward
    the nearest of the four canvas edges. alpha=0 → no move; alpha=1 →
    slammed against edge."""
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    pos = positions.clone()
    pos_np = pos.cpu().numpy()
    half_w = (benchmark.macro_sizes[:, 0] / 2).cpu().numpy()
    half_h = (benchmark.macro_sizes[:, 1] / 2).cpu().numpy()
    fixed = benchmark.macro_fixed.cpu().numpy()

    new_pos = pos_np.copy()
    for i in range(pos_np.shape[0]):
        if fixed[i]:
            continue
        x, y = pos_np[i, 0], pos_np[i, 1]
        # Find nearest edge of the four.
        dx_left = x - half_w[i]
        dx_right = (cw - half_w[i]) - x
        dy_bot = y - half_h[i]
        dy_top = (ch - half_h[i]) - y
        dists = [dx_left, dx_right, dy_bot, dy_top]
        nearest = int(np.argmin([abs(d) for d in dists]))
        # Move alpha fraction toward that edge.
        if nearest == 0:  # left
            new_pos[i, 0] = x - alpha * dx_left
        elif nearest == 1:  # right
            new_pos[i, 0] = x + alpha * dx_right
        elif nearest == 2:  # bottom
            new_pos[i, 1] = y - alpha * dy_bot
        else:  # top
            new_pos[i, 1] = y + alpha * dy_top
        # Clamp.
        new_pos[i, 0] = np.clip(new_pos[i, 0], half_w[i], cw - half_w[i])
        new_pos[i, 1] = np.clip(new_pos[i, 1], half_h[i], ch - half_h[i])

    return torch.tensor(new_pos, dtype=torch.float32)


def spike(bench_name: str = "ibm01", placement_path: str = None, cd_budget_s: float = 600.0):
    """Run periphery push at alphas {0.1, 0.3, 0.5}, polish, compare to baseline."""
    bench_dir = find_benchmark_dir(bench_name)
    bench, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"[bench] {bench_name}: {bench.num_macros} macros, canvas={bench.canvas_width:.0f}×{bench.canvas_height:.0f}")

    # Load cached cascade-converged placement.
    if placement_path is None:
        placement_path = str(_ROOT / f"experiments/E91_dp_full_polish/results/{bench_name}_dp_full_polish.pt")
    data = torch.load(placement_path, map_location="cpu", weights_only=False)
    baseline = data["placement"].to(torch.float32) if isinstance(data, dict) else data.to(torch.float32)
    baseline_proxy = float(compute_proxy_cost(baseline, bench, plc)["proxy_cost"])
    baseline_edge = edge_dist_mean(baseline, bench.canvas_width, bench.canvas_height)
    baseline_ovl = compute_overlap_metrics(baseline, bench)["overlap_count"]
    print(f"[baseline] proxy={baseline_proxy:.5f} edge_dist={baseline_edge:.4f} ovl={baseline_ovl}")

    fixed = bench.macro_fixed.cpu().numpy()
    movable_idx = [i for i in range(bench.num_macros) if not bool(fixed[i])]

    results = [{"alpha": 0.0, "proxy": baseline_proxy, "edge_dist": baseline_edge, "ovl": baseline_ovl, "wall": 0.0}]

    for alpha in [0.005, 0.01]:
        t0 = time.time()
        pushed = push_to_periphery(baseline, bench, alpha)
        pushed_edge_raw = edge_dist_mean(pushed, bench.canvas_width, bench.canvas_height)
        # Re-legalize.
        legal, _ = project_overlaps(pushed, bench)
        legal_ovl = compute_overlap_metrics(legal, bench)["overlap_count"]
        legal_proxy = float(compute_proxy_cost(legal, bench, plc)["proxy_cost"])
        legal_edge = edge_dist_mean(legal, bench.canvas_width, bench.canvas_height)
        print(f"[α={alpha:.1f}] pre-polish: edge_raw={pushed_edge_raw:.4f} → legal edge={legal_edge:.4f} proxy={legal_proxy:.5f} ovl={legal_ovl}")

        if legal_ovl > 0:
            print(f"[α={alpha:.2f}] residual {legal_ovl} after project — letting CD polish resolve")
            # CD has LNS + SA moves that can fix overlaps; try it.

        # CD polish.
        evaluator = IncrementalProxyEvaluator(bench, plc, legal.clone())
        run_cd_adaptive(
            evaluator, bench, plc, movable_idx,
            min_time_s=30.0, hard_cap_s=cd_budget_s,
            patience=3, plateau_threshold=0.001, log_fn=None,
        )
        polished = evaluator.placement.detach().clone().to(torch.float32)
        polished_proxy = float(compute_proxy_cost(polished, bench, plc)["proxy_cost"])
        polished_edge = edge_dist_mean(polished, bench.canvas_width, bench.canvas_height)
        polished_ovl = compute_overlap_metrics(polished, bench)["overlap_count"]
        wall = time.time() - t0
        delta_proxy = (polished_proxy - baseline_proxy) / baseline_proxy * 100
        delta_edge = (polished_edge - baseline_edge) / baseline_edge * 100
        print(f"[α={alpha:.1f}] POLISHED: proxy={polished_proxy:.5f} ({delta_proxy:+.2f}%) edge_dist={polished_edge:.4f} ({delta_edge:+.2f}%) ovl={polished_ovl} wall={wall:.0f}s")
        results.append({
            "alpha": alpha, "proxy": polished_proxy, "edge_dist": polished_edge,
            "ovl": polished_ovl, "wall": wall, "delta_proxy_pct": delta_proxy,
            "delta_edge_pct": delta_edge,
        })

    print("\n=== SUMMARY ===")
    print(f"{'alpha':>6} {'proxy':>9} {'Δproxy':>10} {'edge_dist':>10} {'Δedge':>10} {'ovl':>5}")
    for r in results:
        dp = r.get("delta_proxy_pct", 0)
        de = r.get("delta_edge_pct", 0)
        print(f"{r['alpha']:6.2f} {r['proxy']:9.5f} {dp:+9.2f}% {r['edge_dist']:10.4f} {de:+9.2f}% {r['ovl']:5d}")

    # Save.
    out = _ROOT / "experiments" / "E107_periphery_bias" / "results" / f"{bench_name}_spike.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"bench": bench_name, "baseline_proxy": baseline_proxy, "results": results}, out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
    placement = sys.argv[2] if len(sys.argv) > 2 else None
    spike(bench, placement)
