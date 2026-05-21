"""E119 — Sweep PerNetTraceCongestion hyperparameters on hard benches.

Tunable hparams (from `experiments/E111_per_net_trace_congestion/code/per_net_trace_proxy.py`):
  - `smooth_range`: 1, 2, 3 — kernel width for box smoothing (canonical: 2)
  - `sigma_cell_frac`: 0.3, 0.5, 0.7 — Gaussian σ for soft cell-assignment
  - `beta_minmax`: 4, 6, 10 — logsumexp steepness for soft row/col endpoints
  - `beta_range_per_cell`: 4, 6, 10 — sigmoid steepness for in-range indicator

Metric: relative scalar mismatch vs canonical on cached cascade placements
for the three "hard" benches (ibm10, ibm12, ibm17 — see memory:
`diff_proxy_rudy_mismatch`). We also include ibm04 / ibm17 as additional
references.

Output: `results/sweep_table.json` with one row per config × bench, and
`results/sweep_summary.md` with a ranked best-of table.

Run:
  uv run python experiments/E119_per_net_trace_tuning/code/sweep_hparams.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

from per_net_trace_proxy import PerNetTraceCongestion


HARD_BENCHES = ["ibm10", "ibm12", "ibm17"]


def load_start(bench_name: str):
    """Load cached cascade placement (or fall back to macro_positions)."""
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    cached = _ROOT / "experiments" / "E84_cascading_saddle" / "results" / f"cascade_{bench_name}.pt"
    if cached.exists():
        data = torch.load(cached, weights_only=False, map_location="cpu")
        start = (data.get("placement") if "placement" in data else data["polished_placement"]).to(torch.float32)
        source = "cached_cascade"
    else:
        start = benchmark.macro_positions.clone().float()
        source = "macro_positions"
    return benchmark, plc, start, source


def evaluate_config_inplace(proxy, start, can: float, cfg: Dict) -> Dict:
    """Mutate `proxy` hparams in place, return scalar smooth + relative gap.

    The PerNetTraceCongestion class reads `self.smooth_range`,
    `self.sigma_cell_frac`, `self.beta_minmax`, `self.beta_range_per_cell`
    at evaluation time (`compute_congestion`). All four are simple float
    attributes, so we can swap them per config without re-running the
    expensive `_extract_net_data` + `_build_pair_data` setup.
    """
    proxy.smooth_range = int(cfg["smooth_range"])
    proxy.sigma_cell_frac = float(cfg["sigma_frac"])
    proxy.beta_minmax = float(cfg["beta_minmax"])
    proxy.beta_range_per_cell = float(cfg["beta_range"])
    with torch.no_grad():
        sm = float(proxy.compute_congestion(start))
    rel = (sm - can) / max(abs(can), 1e-9) * 100.0
    return {
        "smooth": sm,
        "canonical": can,
        "rel_pct": rel,
        "abs_rel_pct": abs(rel),
    }


def main():
    # Build sweep grid. Started from the requested {sr,σ,β_mm,β_rg} grid
    # and added a wider σ + β_mm exploration after the ibm10 smoke showed
    # σ=0.3 and β_mm=10 are the most-improving directions.
    smooth_ranges = [1, 2, 3]
    sigma_fracs = [0.2, 0.3, 0.5, 0.7]
    beta_minmaxes = [4, 6, 10, 16]
    beta_ranges = [4, 6, 10]

    grid = list(itertools.product(smooth_ranges, sigma_fracs, beta_minmaxes, beta_ranges))
    print(f"Sweep size: {len(grid)} configs × {len(HARD_BENCHES)} benches "
          f"= {len(grid)*len(HARD_BENCHES)} evals", flush=True)

    # Pre-load each benchmark once (expensive — cascade .pt + plc setup),
    # build one PerNetTraceCongestion per bench, cache canonical cong.
    bench_data = {}
    for b in HARD_BENCHES:
        t0 = time.time()
        benchmark, plc, start, source = load_start(b)
        can_full = compute_proxy_cost(start, benchmark, plc)
        can_cong = float(can_full["congestion_cost"])
        # Build proxy once. hparams are mutated in-place per config below.
        proxy = PerNetTraceCongestion(benchmark, plc, device="cpu")
        wall = time.time() - t0
        bench_data[b] = (benchmark, plc, start, proxy, can_cong)
        print(f"  Loaded {b}: source={source}, n_pairs={proxy.n_pairs}, "
              f"canonical_cong={can_cong:.5f}, wall={wall:.1f}s", flush=True)

    rows: List[Dict] = []
    t_grid_start = time.time()
    for i, (sr, sigma, b_mm, b_rg) in enumerate(grid):
        cfg = {
            "smooth_range": sr,
            "sigma_frac": sigma,
            "beta_minmax": b_mm,
            "beta_range": b_rg,
        }
        cfg_label = f"sr={sr} σ={sigma} β_mm={b_mm} β_rg={b_rg}"
        for b in HARD_BENCHES:
            _, _, start, proxy, can_cong = bench_data[b]
            t0 = time.time()
            try:
                m = evaluate_config_inplace(proxy, start, can_cong, cfg)
                row = {
                    **cfg,
                    "bench": b,
                    "smooth": m["smooth"],
                    "canonical": m["canonical"],
                    "rel_pct": m["rel_pct"],
                    "abs_rel_pct": m["abs_rel_pct"],
                    "wall_s": time.time() - t0,
                    "error": None,
                }
            except Exception as e:
                row = {
                    **cfg,
                    "bench": b,
                    "smooth": None, "canonical": None,
                    "rel_pct": None, "abs_rel_pct": None,
                    "wall_s": time.time() - t0,
                    "error": repr(e),
                }
            rows.append(row)
        if (i + 1) % 10 == 0 or i == len(grid) - 1:
            elapsed = time.time() - t_grid_start
            print(f"  [{i+1:3d}/{len(grid)}] {cfg_label} | elapsed {elapsed:.0f}s "
                  f"({elapsed/(i+1):.1f}s/cfg)", flush=True)

    # Aggregate per-config max + mean abs gap
    cfg_rows: Dict[tuple, Dict] = {}
    for r in rows:
        key = (r["smooth_range"], r["sigma_frac"], r["beta_minmax"], r["beta_range"])
        if r["abs_rel_pct"] is None:
            continue
        cfg_rows.setdefault(key, {
            "smooth_range": r["smooth_range"],
            "sigma_frac": r["sigma_frac"],
            "beta_minmax": r["beta_minmax"],
            "beta_range": r["beta_range"],
            "per_bench_abs": {},
            "per_bench_signed": {},
        })
        cfg_rows[key]["per_bench_abs"][r["bench"]] = r["abs_rel_pct"]
        cfg_rows[key]["per_bench_signed"][r["bench"]] = r["rel_pct"]

    summary = []
    for cfg, info in cfg_rows.items():
        per_b = info["per_bench_abs"]
        if len(per_b) < len(HARD_BENCHES):
            continue
        max_gap = max(per_b.values())
        mean_gap = sum(per_b.values()) / len(per_b)
        info["max_abs"] = max_gap
        info["mean_abs"] = mean_gap
        summary.append(info)

    # Sort: lowest max → lowest mean.
    summary.sort(key=lambda x: (x["max_abs"], x["mean_abs"]))

    out_dir = _HERE.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "sweep_table.json"
    json_path.write_text(json.dumps({
        "rows": rows,
        "summary": summary,
        "grid_size": len(grid),
        "wall_total_s": time.time() - t_grid_start,
    }, indent=2))
    print(f"\nWrote {json_path}", flush=True)

    md_path = out_dir / "sweep_summary.md"
    md_lines = []
    md_lines.append(f"# E119 hparam sweep — {len(grid)} configs × {len(HARD_BENCHES)} benches\n")
    md_lines.append(f"Wall: {time.time()-t_grid_start:.0f}s\n")
    md_lines.append("\n## Top 15 by max-bench absolute gap (lower is better)\n")
    md_lines.append("| Rank | sr | σ | β_mm | β_rg | ibm10 | ibm12 | ibm17 | max | mean |\n")
    md_lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for i, info in enumerate(summary[:15]):
        md_lines.append(
            f"| {i+1} | {info['smooth_range']} | {info['sigma_frac']} | "
            f"{info['beta_minmax']} | {info['beta_range']} | "
            f"{info['per_bench_signed']['ibm10']:+.1f}% | "
            f"{info['per_bench_signed']['ibm12']:+.1f}% | "
            f"{info['per_bench_signed']['ibm17']:+.1f}% | "
            f"{info['max_abs']:.1f}% | "
            f"{info['mean_abs']:.1f}% |\n"
        )
    md_lines.append("\n## Baseline (defaults: sr=2 σ=0.5 β_mm=6 β_rg=4)\n")
    for info in summary:
        if (info["smooth_range"] == 2 and info["sigma_frac"] == 0.5
                and info["beta_minmax"] == 6 and info["beta_range"] == 4):
            md_lines.append(
                f"- ibm10 {info['per_bench_signed']['ibm10']:+.1f}%, "
                f"ibm12 {info['per_bench_signed']['ibm12']:+.1f}%, "
                f"ibm17 {info['per_bench_signed']['ibm17']:+.1f}%, "
                f"max {info['max_abs']:.1f}%, mean {info['mean_abs']:.1f}%\n"
            )
            break
    md_path.write_text("".join(md_lines))
    print(f"Wrote {md_path}", flush=True)

    print("\n=== TOP 5 ===", flush=True)
    for i, info in enumerate(summary[:5]):
        print(
            f"  {i+1}. sr={info['smooth_range']} σ={info['sigma_frac']} "
            f"β_mm={info['beta_minmax']} β_rg={info['beta_range']}: "
            f"max={info['max_abs']:.1f}% mean={info['mean_abs']:.1f}%   "
            f"[ibm10 {info['per_bench_signed']['ibm10']:+.1f}%, "
            f"ibm12 {info['per_bench_signed']['ibm12']:+.1f}%, "
            f"ibm17 {info['per_bench_signed']['ibm17']:+.1f}%]",
            flush=True,
        )
    print(f"  Default sr=2/σ=0.5/β_mm=6/β_rg=4 → ", end="", flush=True)
    for info in summary:
        if (info["smooth_range"] == 2 and info["sigma_frac"] == 0.5
                and info["beta_minmax"] == 6 and info["beta_range"] == 4):
            print(
                f"max={info['max_abs']:.1f}% mean={info['mean_abs']:.1f}%",
                flush=True
            )
            break
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
