"""E108 smoke: A/B test SA-v2 vs SA-critnet on a cached plateau.

Loads a cascade-final placement (already at the plateau), then runs:
  A) plain run_sa_polish_v2 for budget_s seconds
  B) run_sa_polish_v2_critnet for same budget_s

Both use seed=42. Compares best_proxy and improvement_vs_init.

Usage:
  uv run python experiments/E108_critical_net_sa/code/smoke_sa_ab.py \\
      [bench=ibm01] [budget_s=120] [weight_power=2.0]
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# Import original SA from E25
_E25_SPEC = importlib.util.spec_from_file_location(
    "_e25", str(_ROOT / "submissions" / "cd_lns_sa" / "placer.py"))
_E25 = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25)
run_sa_polish_v2 = _E25.run_sa_polish_v2

from critical_net_sa import run_sa_polish_v2_critnet, compute_critical_net_weights


def find_cached_placement(bench_name: str) -> Path:
    """Look for any cached cascade .pt for this bench."""
    candidates = [
        _ROOT / "results" / "cloud_snapshot_20260516_002023" / "e84_results" / f"cascade_{bench_name}.pt",
        _ROOT / "experiments" / "E84_cascading_saddle" / "results" / f"cascade_{bench_name}.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Last resort: scan
    matches = list(_ROOT.glob(f"**/cascade_{bench_name}.pt")) + \
              list(_ROOT.glob(f"**/{bench_name}_dp_full_polish.pt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No cached placement for {bench_name}")


def run_one(label, sa_fn, bench, plc, placement, budget_s, **sa_kwargs):
    placement_f64 = placement.detach().clone().to(torch.float64)
    ev = IncrementalProxyEvaluator(bench, plc, placement_f64)
    n_hard = bench.num_hard_macros
    fixed = bench.macro_fixed.cpu().numpy()
    hard_movable = [i for i in range(n_hard) if not bool(fixed[i])]

    t0 = time.time()
    stats = sa_fn(
        evaluator=ev, benchmark=bench, plc=plc,
        hard_movable=hard_movable, time_budget_s=budget_s,
        T0=5e-4, Tf=1e-6, seed=42, breakpoint_budget=12,
        log_fn=None, **sa_kwargs,
    )
    wall = time.time() - t0
    final_proxy = float(compute_proxy_cost(
        ev.placement.detach().clone().to(torch.float32), bench, plc)["proxy_cost"])
    ovl = compute_overlap_metrics(
        ev.placement.detach().clone().to(torch.float32), bench)["overlap_count"]
    print(f"  [{label}] wall={wall:.1f}s  init={stats['init_proxy']:.5f}  "
          f"best={stats['best_proxy']:.5f}  final={final_proxy:.5f}  ovl={ovl}  "
          f"prop={stats['proposed']}  acc_better={stats['accepted_better']}  "
          f"acc_worse={stats['accepted_worse']}", flush=True)
    return stats, final_proxy, ovl


def main():
    bench_name = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
    budget_s = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    weight_power = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    mix_uniform = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

    print(f"\n=== E108 SA A/B {bench_name} budget={budget_s}s "
          f"weight_power={weight_power} mix_uniform={mix_uniform} ===\n", flush=True)

    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))
    cached_path = find_cached_placement(bench_name)
    print(f"  cached placement: {cached_path}", flush=True)
    data = torch.load(cached_path, weights_only=False)
    placement = data.get("placement", data)
    if not isinstance(placement, torch.Tensor):
        placement = data["placement"] if "placement" in data else data["polished_placement"]
    placement = placement.to(torch.float32)
    placement, _ = project_overlaps(placement, bench)

    init_proxy = float(compute_proxy_cost(placement, bench, plc)["proxy_cost"])
    n_hard = bench.num_hard_macros
    fixed = bench.macro_fixed.cpu().numpy()
    hard_movable = [i for i in range(n_hard) if not bool(fixed[i])]

    # Compute and print weight distribution stats
    weights = compute_critical_net_weights(bench, hard_movable, weight_power=weight_power)
    nonzero = weights[weights > 0]
    print(f"  init_proxy={init_proxy:.5f}  n_hard_movable={len(hard_movable)}", flush=True)
    print(f"  weight distribution: min={float(nonzero.min()) if len(nonzero) else 0:.2e}  "
          f"median={float(np.median(nonzero)) if len(nonzero) else 0:.2e}  "
          f"max={float(weights.max()):.2e}  "
          f"top-10/rest-90 ratio={(weights[weights.argsort()[-len(weights)//10:]].sum() / max(weights.sum(), 1e-12)):.3f}", flush=True)
    print(f"\n  A: vanilla SA-v2 (uniform sampling)", flush=True)
    stats_a, final_a, ovl_a = run_one(
        "A vanilla", run_sa_polish_v2, bench, plc, placement, budget_s)

    print(f"\n  B: SA-critnet (weight_power={weight_power}, mix_uniform={mix_uniform})", flush=True)
    stats_b, final_b, ovl_b = run_one(
        "B critnet", run_sa_polish_v2_critnet, bench, plc, placement, budget_s,
        weight_power=weight_power, mix_uniform=mix_uniform)

    print(f"\n=== SUMMARY {bench_name} ===", flush=True)
    print(f"  A vanilla:  best={stats_a['best_proxy']:.5f}  Δ_vs_init={stats_a['improvement_vs_init']:+.5f}  ovl={ovl_a}", flush=True)
    print(f"  B critnet:  best={stats_b['best_proxy']:.5f}  Δ_vs_init={stats_b['improvement_vs_init']:+.5f}  ovl={ovl_b}", flush=True)
    delta = stats_b['best_proxy'] - stats_a['best_proxy']
    pct = delta / max(stats_a['best_proxy'], 1e-9) * 100
    print(f"  Δ(B-A) = {delta:+.5f}  ({pct:+.3f}%)  "
          f"{'WIN' if delta < -1e-6 else ('TIE' if abs(delta) <= 1e-6 else 'LOSE')}", flush=True)


if __name__ == "__main__":
    import numpy as np
    main()
