"""E107 multi-seed test: is periphery push genuinely better than random kicks?

The single-seed test on ariane133 gave:
  Control      -0.28%
  Periphery    -0.95%
  Random       -0.62%
  Center       -0.50%

The Periphery advantage of -0.33% over Random could be just lucky seed.
Run 5 random kicks (different seeds) + periphery + center; report
mean ± std vs the deterministic periphery.
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
from decompose_spike import (
    edge_dist_mean, push_periphery, push_center, push_random,
    polish_and_score, measure_rms,
)


def main(bench_name="ariane133", alpha=0.01, n_random_seeds=5, cd_budget_s=180.0):
    bench_dir = find_benchmark_dir(bench_name)
    bench, plc = load_benchmark_from_dir(str(bench_dir))
    cw, ch = float(bench.canvas_width), float(bench.canvas_height)

    placement_path = _ROOT / f"experiments/E91_dp_full_polish/results/{bench_name}_dp_full_polish.pt"
    data = torch.load(placement_path, map_location="cpu", weights_only=False)
    baseline = data["placement"].to(torch.float32) if isinstance(data, dict) else data.to(torch.float32)
    base_proxy = float(compute_proxy_cost(baseline, bench, plc)["proxy_cost"])
    base_edge = edge_dist_mean(baseline, cw, ch)
    print(f"[{bench_name}] baseline: proxy={base_proxy:.5f} edge_dist={base_edge:.4f}")

    periph = push_periphery(baseline, bench, alpha)
    periph_rms = measure_rms(periph, baseline)
    print(f"RMS calibration (α={alpha}): {periph_rms:.4f}\n")

    # 1. Periphery (deterministic).
    t = time.time()
    pol = polish_and_score(periph, bench, plc, cd_budget_s)
    p = float(compute_proxy_cost(pol, bench, plc)["proxy_cost"])
    e = edge_dist_mean(pol, cw, ch)
    ovl = compute_overlap_metrics(pol, bench)["overlap_count"]
    dp_periph = (p - base_proxy) / base_proxy * 100
    de_periph = (e - base_edge) / base_edge * 100
    print(f"  Periphery        Δproxy={dp_periph:+.3f}% Δedge={de_periph:+.3f}% ovl={ovl} wall={time.time()-t:.0f}s")

    # 2. Center (deterministic).
    cen = push_center(baseline, bench, alpha)
    t = time.time()
    pol = polish_and_score(cen, bench, plc, cd_budget_s)
    p = float(compute_proxy_cost(pol, bench, plc)["proxy_cost"])
    e = edge_dist_mean(pol, cw, ch)
    ovl = compute_overlap_metrics(pol, bench)["overlap_count"]
    dp_center = (p - base_proxy) / base_proxy * 100
    de_center = (e - base_edge) / base_edge * 100
    print(f"  Center           Δproxy={dp_center:+.3f}% Δedge={de_center:+.3f}% ovl={ovl} wall={time.time()-t:.0f}s")

    # 3. Random kicks (5 seeds).
    random_dps = []
    random_des = []
    random_ovls = []
    for seed in range(42, 42 + n_random_seeds):
        rnd = push_random(baseline, bench, periph_rms, seed=seed)
        t = time.time()
        pol = polish_and_score(rnd, bench, plc, cd_budget_s)
        p = float(compute_proxy_cost(pol, bench, plc)["proxy_cost"])
        e = edge_dist_mean(pol, cw, ch)
        ovl = compute_overlap_metrics(pol, bench)["overlap_count"]
        dp = (p - base_proxy) / base_proxy * 100
        de = (e - base_edge) / base_edge * 100
        random_dps.append(dp); random_des.append(de); random_ovls.append(ovl)
        print(f"  Random seed={seed}  Δproxy={dp:+.3f}% Δedge={de:+.3f}% ovl={ovl} wall={time.time()-t:.0f}s")

    # 4. Control (just re-polish).
    t = time.time()
    pol = polish_and_score(baseline.clone(), bench, plc, cd_budget_s)
    p = float(compute_proxy_cost(pol, bench, plc)["proxy_cost"])
    e = edge_dist_mean(pol, cw, ch)
    ovl = compute_overlap_metrics(pol, bench)["overlap_count"]
    dp_control = (p - base_proxy) / base_proxy * 100
    de_control = (e - base_edge) / base_edge * 100
    print(f"  Control          Δproxy={dp_control:+.3f}% Δedge={de_control:+.3f}% ovl={ovl} wall={time.time()-t:.0f}s")

    # Summary.
    rdp = np.array(random_dps)
    rde = np.array(random_des)
    print(f"\n=== {bench_name} α={alpha} (n_seeds={n_random_seeds}) ===")
    print(f"  Control       : Δproxy={dp_control:+.3f}%")
    print(f"  Periphery     : Δproxy={dp_periph:+.3f}% Δedge={de_periph:+.3f}%")
    print(f"  Random (mean) : Δproxy={rdp.mean():+.3f}% ± {rdp.std():.3f}%  Δedge={rde.mean():+.3f}% ± {rde.std():.3f}%")
    print(f"  Center        : Δproxy={dp_center:+.3f}% Δedge={de_center:+.3f}%")
    # Statistical signal: periphery beats random by how many random-std?
    z = (rdp.mean() - dp_periph) / max(rdp.std(), 1e-6)
    print(f"  Periphery vs Random: {z:+.2f}σ ({'periphery wins' if dp_periph < rdp.mean() else 'random wins'} by {abs(dp_periph - rdp.mean()):.3f}%)")
    out = _ROOT / "experiments" / "E107_periphery_bias" / "results" / f"{bench_name}_multiseed_alpha{int(alpha*1000):03d}.pt"
    torch.save({
        "bench": bench_name, "alpha": alpha,
        "baseline_proxy": base_proxy, "baseline_edge": base_edge,
        "control": (dp_control, de_control),
        "periphery": (dp_periph, de_periph),
        "center": (dp_center, de_center),
        "random_dps": rdp.tolist(),
        "random_des": rde.tolist(),
        "random_ovls": random_ovls,
    }, out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "ariane133"
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    main(bench, alpha)
