"""Tier 2 macro-clearance diagnostic.

ORFS pushes macros apart to maintain >= 12 um edge-to-edge clearance for
PDN channel routing. If our submitted placement has any pairs with
clearance < 12 um, ORFS moves them at Tier 2 evaluation, which can change
WL / density / congestion at the routed level (and thus WNS / TNS / Area).

This script computes per-design clearance distributions, reports
violation counts, and estimates how much the Tier 2 push would move
macros.

Usage:
    uv run python analysis/macro_clearance_diagnostic/diagnostic.py \\
        --bench ariane133 --placement <path.pt>

    uv run python analysis/macro_clearance_diagnostic/diagnostic.py \\
        --bench ariane133  # runs cascade_adaptive on it first, then checks
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def compute_pairwise_clearance(
    placement: torch.Tensor, sizes: torch.Tensor, n_hard: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return edge-to-edge clearance matrix for all hard-macro pairs.

    Output ``clearance[i, j]`` is the L-infinity edge-to-edge gap between
    hard macros ``i`` and ``j``. Negative => overlap. Zero => touching.
    Positive => gap between edges.

    Convention: for ORFS PDN spacing, the relevant metric is the *minimum*
    of (x-gap, y-gap) — if either axis has positive separation, the macros
    are non-overlapping but the PDN channel still needs >= 12 um in BOTH
    axes for the push to be a no-op. We report both the L-inf gap and
    a "max-violation" view (the minimum of x-gap, y-gap clipped above 0).
    """
    pos = placement[:n_hard].detach().cpu().numpy().astype(np.float64)  # [n, 2]
    sz = sizes[:n_hard].detach().cpu().numpy().astype(np.float64)  # [n, 2]

    n = pos.shape[0]
    # Pairwise edge-to-edge gap per axis.
    # gap_axis[i, j] = |pos[i, k] - pos[j, k]| - (sz[i, k] + sz[j, k]) / 2
    diff = np.abs(pos[:, None, :] - pos[None, :, :])  # [n, n, 2]
    half_sum = (sz[:, None, :] + sz[None, :, :]) / 2.0  # [n, n, 2]
    gap_per_axis = diff - half_sum  # [n, n, 2]

    # Clearance: edge-to-edge "true" gap. Negative => overlap on this axis.
    # For non-overlapping macros, both axis gaps will sum to a positive
    # value but the PDN concern is the smaller of the two (channel width).
    # An "ORFS-relevant" clearance: max of the two axis gaps (since macros
    # are non-overlapping iff at least one axis gap is > 0). Channel width
    # is the magnitude of separation along the axis with the larger gap.
    orfs_clearance = np.max(gap_per_axis, axis=2)  # [n, n]
    # Hard overlaps: both axes negative => boxes overlap
    is_overlap = (gap_per_axis[:, :, 0] < 0) & (gap_per_axis[:, :, 1] < 0)
    return orfs_clearance, is_overlap


def diagnose(bench_name: str, placement_path: str | None, threshold: float = 12.0) -> dict:
    """Compute clearance stats for a placement on a benchmark.

    If ``placement_path`` is None, runs cascade_adaptive with a short
    budget to generate one (M3 Max: ~5-10 min on NG45-sized benches).
    """
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    n_hard = benchmark.num_hard_macros

    if placement_path:
        print(f"loading placement from {placement_path}")
        data = torch.load(placement_path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            for key in ("placement", "polished_placement", "transplanted_placement"):
                if key in data:
                    placement = data[key]
                    break
            else:
                raise RuntimeError(f"No placement key in {placement_path}: {list(data.keys())}")
        else:
            placement = data
        placement = placement.to(torch.float32)
    else:
        print(f"running cascade_adaptive on {bench_name} (short budget for diagnostic)")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "placer_adaptive",
            str(_ROOT / "submissions" / "cd_lns_sa_cascade" / "placer_adaptive.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        p = mod.CDLNSSACascadeAdaptivePlacer(budget_seconds=600.0, verbose=False)
        t0 = time.time()
        placement = p.place(benchmark)
        print(f"  placer wall: {time.time() - t0:.0f}s")

    proxy = float(compute_proxy_cost(placement, benchmark, plc)["proxy_cost"])
    ovl = compute_overlap_metrics(placement, benchmark)["overlap_count"]
    print(f"  proxy={proxy:.5f}  overlaps={ovl}")

    # Clearance.
    clearance, is_overlap_mat = compute_pairwise_clearance(
        placement, benchmark.macro_sizes, n_hard
    )

    # Strip diagonal + upper triangle.
    triu = np.triu_indices(n_hard, k=1)
    pair_clr = clearance[triu]
    pair_overlap = is_overlap_mat[triu]
    n_pairs = len(pair_clr)

    n_overlap = int(pair_overlap.sum())
    n_violation = int(((pair_clr < threshold) & ~pair_overlap).sum())  # close but not overlapping
    n_safe = int((pair_clr >= threshold).sum())

    # Estimate Tier 2 push displacement.
    # If push enforces >= 12 um, each violating pair gets nudged by
    # (12 - clearance) / 2 per macro in the most-violating axis. Sum of
    # displacements is a rough proxy for how much routing churn ORFS
    # will introduce.
    violating_clr = pair_clr[(pair_clr < threshold) & ~pair_overlap]
    if len(violating_clr) > 0:
        push_amounts = (threshold - violating_clr) / 2.0
        push_avg = float(push_amounts.mean())
        push_max = float(push_amounts.max())
        push_total = float(push_amounts.sum())
    else:
        push_avg = push_max = push_total = 0.0

    # Distribution of clearances for the non-overlapping pairs.
    non_ovl_clr = pair_clr[~pair_overlap]
    if len(non_ovl_clr) > 0:
        q01 = float(np.percentile(non_ovl_clr, 1))
        q05 = float(np.percentile(non_ovl_clr, 5))
        q25 = float(np.percentile(non_ovl_clr, 25))
        q50 = float(np.percentile(non_ovl_clr, 50))
        clr_min = float(non_ovl_clr.min())
    else:
        q01 = q05 = q25 = q50 = clr_min = 0.0

    canvas_diag = float(
        np.hypot(benchmark.canvas_width, benchmark.canvas_height)
    )

    out = {
        "bench": bench_name,
        "n_hard_macros": n_hard,
        "n_pairs": n_pairs,
        "proxy": proxy,
        "overlap_count": int(ovl),
        "canvas_w": float(benchmark.canvas_width),
        "canvas_h": float(benchmark.canvas_height),
        "canvas_diag_um": canvas_diag,
        "threshold_um": float(threshold),
        "n_overlap_pairs": n_overlap,
        "n_clearance_violations": n_violation,
        "n_safe_pairs": n_safe,
        "violation_pct": 100.0 * n_violation / max(1, n_pairs),
        "min_clearance_um": clr_min,
        "p01_clearance_um": q01,
        "p05_clearance_um": q05,
        "p25_clearance_um": q25,
        "p50_clearance_um": q50,
        "push_avg_um": push_avg,
        "push_max_um": push_max,
        "push_total_um": push_total,
        "push_avg_as_canvas_frac": push_avg / canvas_diag if canvas_diag > 0 else 0.0,
    }
    return out


def _fmt(o: dict) -> str:
    return (
        f"\n=== {o['bench']} clearance diagnostic ===\n"
        f"  n_hard_macros={o['n_hard_macros']}, n_pairs={o['n_pairs']}\n"
        f"  proxy={o['proxy']:.5f}, overlaps={o['overlap_count']}\n"
        f"  canvas {o['canvas_w']:.1f} x {o['canvas_h']:.1f} um "
        f"(diag {o['canvas_diag_um']:.1f})\n"
        f"  ORFS threshold = {o['threshold_um']:.0f} um (PDN channel)\n"
        f"\n  clearance distribution (non-overlapping pairs):\n"
        f"    min:  {o['min_clearance_um']:8.2f} um\n"
        f"    p01:  {o['p01_clearance_um']:8.2f} um\n"
        f"    p05:  {o['p05_clearance_um']:8.2f} um\n"
        f"    p25:  {o['p25_clearance_um']:8.2f} um\n"
        f"    p50:  {o['p50_clearance_um']:8.2f} um\n"
        f"\n  Tier 2 PDN-push impact:\n"
        f"    pairs needing push: {o['n_clearance_violations']:>7} "
        f"({o['violation_pct']:.2f}% of all pairs)\n"
        f"    avg push:           {o['push_avg_um']:8.2f} um per macro\n"
        f"    max push:           {o['push_max_um']:8.2f} um per macro\n"
        f"    push as fraction of canvas diagonal: "
        f"{100*o['push_avg_as_canvas_frac']:.4f}%\n"
        f"  hard overlap pairs (should be 0): {o['n_overlap_pairs']}\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True,
                    help="Benchmark name (e.g. ariane133, ariane136, mempool_tile, nvdla, ibm10)")
    ap.add_argument("--placement", default=None,
                    help="Path to .pt with placement tensor; if omitted, run cascade_adaptive locally")
    ap.add_argument("--threshold", type=float, default=12.0,
                    help="Clearance threshold in um (ORFS PDN: 12 um)")
    args = ap.parse_args()

    out = diagnose(args.bench, args.placement, threshold=args.threshold)
    print(_fmt(out))


if __name__ == "__main__":
    main()
