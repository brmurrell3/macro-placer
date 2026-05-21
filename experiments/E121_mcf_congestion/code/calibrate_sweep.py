"""E121 — Calibration sweep across (tau_frac, rudy_mix, n_bend) configs.

Compares smooth MCF vs canonical congestion on cached cascade placements
for the three hardest benches: ibm10, ibm12, ibm17.

Reports:
  - rel mismatch %
  - Pearson ρ on Δcong (16 random perturbations)
  - Spearman ρ (rank correlation)
  - sign-agreement %

Picks the config with best (min max |rel_pct|, max Pearson) Pareto front.

Run:
  uv run python experiments/E121_mcf_congestion/code/calibrate_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

from mcf_congestion import MCFCongestion


BENCHES = ["ibm10", "ibm12", "ibm17"]

CONFIGS = [
    # tau_frac, rudy_mix, n_bend
    (0.05, 1.00, 3),   # near-RUDY (sanity vs E111)
    (0.10, 0.50, 5),
    (0.15, 0.30, 5),
    (0.20, 0.20, 7),
    (0.30, 0.00, 7),   # pure MCF, moderate diffusion
    (0.40, 0.00, 9),
    (0.50, 0.00, 9),   # broad MCF
    (0.30, 0.10, 7),   # MCF + 10% RUDY anchor
    (0.25, 0.15, 7),
]


def eval_config(bench_name, tau_frac, rudy_mix, n_bend, n_perturb=12):
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    cached = _ROOT / "experiments" / "E84_cascading_saddle" / "results" / f"cascade_{bench_name}.pt"
    if cached.exists():
        data = torch.load(cached, weights_only=False, map_location="cpu")
        start = (data.get("placement") if "placement" in data else data["polished_placement"]).to(torch.float32)
    else:
        start = benchmark.macro_positions.clone().float()

    proxy = MCFCongestion(
        benchmark, plc, device="cpu",
        tau_frac=tau_frac, n_bend_samples=n_bend, rudy_mix=rudy_mix,
    )

    can = compute_proxy_cost(start, benchmark, plc)
    can_cong = float(can["congestion_cost"])
    with torch.no_grad():
        sm_cong = float(proxy.compute_congestion(start))
    rel_pct = (sm_cong - can_cong) / max(abs(can_cong), 1e-6) * 100.0

    rng = np.random.default_rng(42)
    fixed = benchmark.macro_fixed.cpu().numpy()
    cw = float(benchmark.canvas_width)
    movable_hard = [i for i in range(benchmark.num_hard_macros) if not bool(fixed[i])]
    scale = 0.02 * cw

    can_deltas = []
    sm_deltas = []
    for _ in range(n_perturb):
        p = start.clone()
        targets = rng.choice(movable_hard, size=min(5, len(movable_hard)), replace=False)
        for t in targets:
            dx = float(rng.normal(0.0, scale))
            dy = float(rng.normal(0.0, scale))
            x0 = float(benchmark.macro_sizes[t, 0]) / 2.0
            y0 = float(benchmark.macro_sizes[t, 1]) / 2.0
            p[t, 0] = torch.clamp(p[t, 0] + dx, x0, cw - x0)
            p[t, 1] = torch.clamp(p[t, 1] + dy, y0, float(benchmark.canvas_height) - y0)
        c_can = float(compute_proxy_cost(p, benchmark, plc)["congestion_cost"])
        with torch.no_grad():
            c_sm = float(proxy.compute_congestion(p))
        can_deltas.append(c_can - can_cong)
        sm_deltas.append(c_sm - sm_cong)

    can_arr = np.asarray(can_deltas)
    sm_arr = np.asarray(sm_deltas)
    pearson = float(np.corrcoef(can_arr, sm_arr)[0, 1]) if len(can_arr) > 2 else float("nan")
    rk_c = np.argsort(np.argsort(can_arr))
    rk_s = np.argsort(np.argsort(sm_arr))
    spearman = float(np.corrcoef(rk_c, rk_s)[0, 1]) if len(can_arr) > 2 else float("nan")
    sign_agree = float(np.mean(np.sign(can_arr) == np.sign(sm_arr)))

    return {
        "bench": bench_name,
        "tau_frac": tau_frac,
        "rudy_mix": rudy_mix,
        "n_bend": n_bend,
        "canonical": can_cong,
        "smooth": sm_cong,
        "rel_pct": rel_pct,
        "pearson": pearson,
        "spearman": spearman,
        "sign_agree": sign_agree,
    }


def main():
    out_dir = _HERE.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibration_sweep.json"

    all_results = []
    for tau_frac, rudy_mix, n_bend in CONFIGS:
        row = {"tau_frac": tau_frac, "rudy_mix": rudy_mix, "n_bend": n_bend, "by_bench": {}}
        rels = []
        pearsons = []
        for b in BENCHES:
            r = eval_config(b, tau_frac, rudy_mix, n_bend)
            row["by_bench"][b] = r
            rels.append(abs(r["rel_pct"]))
            pearsons.append(r["pearson"])
            print(
                f"  τ={tau_frac:.2f} rm={rudy_mix:.2f} K={n_bend}  "
                f"{b}: smooth={r['smooth']:.5f}  can={r['canonical']:.5f}  "
                f"Δ={r['rel_pct']:+.1f}%  ρ={r['pearson']:+.2f}",
                flush=True,
            )
        row["max_abs_rel_pct"] = float(max(rels))
        row["mean_pearson"] = float(np.mean(pearsons))
        all_results.append(row)
        # Save incrementally
        out_path.write_text(json.dumps(all_results, indent=2))

    print("\n=== Summary (sorted by max_abs_rel_pct) ===", flush=True)
    sorted_rows = sorted(all_results, key=lambda r: r["max_abs_rel_pct"])
    print(f"  {'τ':>6} {'rudy_mix':>9} {'K':>3}  {'max|Δ|':>8} {'mean ρ':>8}", flush=True)
    for r in sorted_rows:
        print(
            f"  {r['tau_frac']:>6.2f} {r['rudy_mix']:>9.2f} {r['n_bend']:>3d}  "
            f"{r['max_abs_rel_pct']:>7.1f}% {r['mean_pearson']:>+8.3f}",
            flush=True,
        )

    best = sorted_rows[0]
    print(f"\nBest config (min max|Δ|): τ={best['tau_frac']}, rudy_mix={best['rudy_mix']}, K={best['n_bend']}", flush=True)
    print(f"  max|Δ| = {best['max_abs_rel_pct']:.1f}%   mean ρ = {best['mean_pearson']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
