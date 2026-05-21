"""Quick calibration: just base comparison (no perturbations). ~30s per bench."""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

from mcf_congestion import MCFCongestion


def quick_eval(bench_name, configs):
    """For each config, return (smooth, canonical, rel_pct)."""
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    cached = _ROOT / "experiments" / "E84_cascading_saddle" / "results" / f"cascade_{bench_name}.pt"
    if cached.exists():
        data = torch.load(cached, weights_only=False, map_location="cpu")
        start = (data.get("placement") if "placement" in data else data["polished_placement"]).to(torch.float32)
    else:
        start = benchmark.macro_positions.clone().float()

    # Canonical: compute once
    t0 = time.time()
    can = compute_proxy_cost(start, benchmark, plc)
    can_cong = float(can["congestion_cost"])
    can_wall = time.time() - t0
    print(f"  [{bench_name}] canonical={can_cong:.5f} (wall {can_wall:.1f}s)", flush=True)

    results = []
    for cfg_name, kwargs in configs:
        t0 = time.time()
        proxy = MCFCongestion(benchmark, plc, device="cpu", **kwargs)
        with torch.no_grad():
            sm_cong = float(proxy.compute_congestion(start))
        rel_pct = (sm_cong - can_cong) / max(abs(can_cong), 1e-6) * 100.0
        wall = time.time() - t0
        print(
            f"  [{bench_name}] {cfg_name:30s} smooth={sm_cong:.5f}  Δ={rel_pct:+6.1f}%  ({wall:.1f}s)",
            flush=True,
        )
        results.append({
            "bench": bench_name,
            "cfg_name": cfg_name,
            "config": kwargs,
            "canonical": can_cong,
            "smooth": sm_cong,
            "rel_pct": rel_pct,
            "wall": wall,
        })
    return results


CONFIGS = [
    # (label, kwargs)
    ("rudy_mix=1.00 K=3 τ=0.05",   dict(tau_frac=0.05, rudy_mix=1.0, n_bend_samples=3)),
    ("rudy_mix=0.50 K=5 τ=0.15",   dict(tau_frac=0.15, rudy_mix=0.50, n_bend_samples=5)),
    ("rudy_mix=0.30 K=5 τ=0.20",   dict(tau_frac=0.20, rudy_mix=0.30, n_bend_samples=5)),
    ("rudy_mix=0.20 K=7 τ=0.25",   dict(tau_frac=0.25, rudy_mix=0.20, n_bend_samples=7)),
    ("rudy_mix=0.10 K=7 τ=0.30",   dict(tau_frac=0.30, rudy_mix=0.10, n_bend_samples=7)),
    ("rudy_mix=0.00 K=7 τ=0.30",   dict(tau_frac=0.30, rudy_mix=0.0, n_bend_samples=7)),
    ("rudy_mix=0.00 K=9 τ=0.40",   dict(tau_frac=0.40, rudy_mix=0.0, n_bend_samples=9)),
    ("rudy_mix=0.00 K=9 τ=0.50",   dict(tau_frac=0.50, rudy_mix=0.0, n_bend_samples=9)),
]


def main():
    out_dir = _HERE.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for bench in ["ibm10", "ibm12", "ibm17"]:
        print(f"\n=== {bench} ===", flush=True)
        rows = quick_eval(bench, CONFIGS)
        all_results.extend(rows)
        (out_dir / "quick_calibration.json").write_text(json.dumps(all_results, indent=2))

    print("\n=== Summary by config (across 3 benches) ===", flush=True)
    by_cfg = {}
    for r in all_results:
        by_cfg.setdefault(r["cfg_name"], []).append(r["rel_pct"])
    for cfg_name, deltas in by_cfg.items():
        max_abs = max(abs(d) for d in deltas)
        mean = sum(deltas) / len(deltas)
        print(f"  {cfg_name:35s} max|Δ|={max_abs:6.1f}%  mean Δ={mean:+6.1f}%  ({deltas})", flush=True)


if __name__ == "__main__":
    main()
