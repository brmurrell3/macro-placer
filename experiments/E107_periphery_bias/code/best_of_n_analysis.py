"""Best-of-N ceiling analysis.

Walk all cached .pt files. For each that contains a placement tensor:
  - Identify the bench by filename / dict key
  - Compute canonical proxy (re-evaluate, in case cached proxy is stale)
  - Track per-bench best
Then merge with AWS-cpu fresh IBM scores (where placements aren't local).
Report 21-bench aggregate ceiling and per-bench winners.
"""
from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


BENCHES_17_IBM = [f"ibm{i:02d}" for i in [1, 2, 3, 4] + list(range(6, 19))]
BENCHES_4_NG45 = ["ariane133", "ariane136", "mempool_tile", "nvdla"]
ALL_BENCHES = BENCHES_17_IBM + BENCHES_4_NG45

# AWS-cpu fresh IBM scores (no placement files saved)
AWS_CPU_FRESH = {
    "ibm01": 0.8797, "ibm02": 1.0623, "ibm03": 0.9467, "ibm04": 0.9859,
    "ibm06": 1.1489, "ibm07": 1.0773, "ibm08": 1.0897, "ibm09": 0.8267,
    "ibm10": 1.0180, "ibm11": 0.8729, "ibm12": 1.2092, "ibm13": 0.9549,
    "ibm14": 1.2186, "ibm15": 1.1727, "ibm16": 1.1460, "ibm17": 1.3419,
    "ibm18": 1.3344,
}
# E107 fresh NG45 wrapper scores (placements saved as JSON logs but compute fresh)
M3_FRESH_NG45 = {
    "ariane133": 0.65212,
    "ariane136": 0.64980,
    "mempool_tile": 0.73750,
    "nvdla": 0.67646,
}


def infer_bench(path: Path, data) -> str | None:
    """Try to infer which bench a .pt file is for."""
    # Try dict key
    if isinstance(data, dict):
        if "bench" in data:
            return str(data["bench"])
        if "bench_name" in data:
            return str(data["bench_name"])
    # Try filename pattern
    name = path.stem.lower()
    for b in ALL_BENCHES:
        if b in name:
            return b
    return None


def get_placement_tensor(data):
    """Extract the placement [N, 2] tensor from various .pt formats."""
    if isinstance(data, torch.Tensor) and data.ndim == 2 and data.shape[1] == 2:
        return data
    if isinstance(data, dict):
        for key in ("placement", "positions", "best_placement", "final", "pos"):
            if key in data:
                v = data[key]
                if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[1] == 2:
                    return v
        # Try first 2D tensor
        for v in data.values():
            if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[1] == 2:
                return v
    return None


def main():
    # Per-bench best record: {bench: (proxy, path, ovl)}
    best = {}

    # Pre-seed with the known fresh scores (no path needed; treat as cached).
    for b, p in AWS_CPU_FRESH.items():
        best[b] = (p, "AWS-cpu fresh (cd_lns_sa_cascade/placer_adaptive.py)", 0)
    for b, p in M3_FRESH_NG45.items():
        best[b] = (p, "M3 fresh (cd_lns_sa_cascade_levy_periphery)", 0)

    # Walk all .pt files in the repo.
    pt_files = []
    for d in ("results", "experiments", "submissions"):
        if (_ROOT / d).exists():
            pt_files.extend((_ROOT / d).rglob("*.pt"))

    print(f"Scanning {len(pt_files)} .pt files...\n")

    # Cache benchmark + plc per bench (loading is slow).
    bench_cache = {}
    def get_bench(name):
        if name not in bench_cache:
            try:
                bench_dir = find_benchmark_dir(name)
                bench, plc = load_benchmark_from_dir(str(bench_dir))
                bench_cache[name] = (bench, plc)
            except Exception:
                bench_cache[name] = None
        return bench_cache[name]

    scanned = 0
    new_wins = 0
    for pt in pt_files:
        try:
            data = torch.load(pt, map_location="cpu", weights_only=False)
        except Exception:
            continue
        bench_name = infer_bench(pt, data)
        if bench_name is None or bench_name not in ALL_BENCHES:
            continue
        placement = get_placement_tensor(data)
        if placement is None:
            continue
        bp = get_bench(bench_name)
        if bp is None:
            continue
        bench, plc = bp
        if placement.shape[0] != bench.num_macros:
            continue
        scanned += 1
        try:
            proxy = float(compute_proxy_cost(placement.to(torch.float32), bench, plc)["proxy_cost"])
            ovl = compute_overlap_metrics(placement.to(torch.float32), bench)["overlap_count"]
        except Exception:
            continue
        if ovl > 0:
            continue
        current_best = best.get(bench_name, (float("inf"), "", 0))
        if proxy < current_best[0] - 1e-6:
            best[bench_name] = (proxy, str(pt.relative_to(_ROOT)), ovl)
            new_wins += 1
            print(f"  WIN {bench_name}: {proxy:.5f} < {current_best[0]:.5f}  ({pt.relative_to(_ROOT)})")

    print(f"\nScanned {scanned} valid (bench, placement) pairs from {len(pt_files)} files; {new_wins} new wins.\n")

    # Report.
    print(f"\n{'='*80}\nBest per bench:\n{'='*80}")
    print(f"{'bench':>14} {'proxy':>9} {'source':>50}")
    print("-" * 80)
    ibm_sum = 0.0
    ng45_sum = 0.0
    for b in ALL_BENCHES:
        if b not in best:
            print(f"  {b:>14}  NO RESULT")
            continue
        proxy, source, ovl = best[b]
        marker = " *" if "AWS-cpu" in source or "M3 fresh" in source else "  "
        print(f"{marker}{b:>12} {proxy:9.5f}   {source[:50]}")
        if b in BENCHES_17_IBM:
            ibm_sum += proxy
        else:
            ng45_sum += proxy

    print("-" * 80)
    print(f"IBM 17 aggregate:   {ibm_sum / 17:.5f}")
    print(f"NG45 4 aggregate:    {ng45_sum / 4:.5f}")
    print(f"21-bench aggregate: {(ibm_sum + ng45_sum) / 21:.5f}")


if __name__ == "__main__":
    main()
