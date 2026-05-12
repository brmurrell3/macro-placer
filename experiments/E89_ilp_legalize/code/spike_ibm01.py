"""E89 spike — single pass of ILP detailed legalize on cached cascade ibm01.

Loads cascade-cached placement (canon 0.84528, 0 overlaps). Partitions movable
hard macros into k-NN clusters. For each cluster, builds a candidate grid +
solves an ILP that picks a joint assignment minimizing proxy delta. Applies
all clusters' moves sequentially. Reports canonical proxy + overlap count.

Kill gate (per manifest): ibm01 lift < 0.1 % vs cascade input -> kill.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from ilp_legalize import (
    build_kmeans_clusters_by_knn,
    build_cluster_ilp_data,
    solve_cluster_ilp,
    apply_picks_sequentially,
)


def main():
    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    cached = torch.load(
        "experiments/E84_cascading_saddle/results/cascade_ibm01.pt",
        weights_only=False,
        map_location="cpu",
    )
    placement = cached["placement"].clone().float()
    n_hard = int(benchmark.num_hard_macros)
    print(f"[setup] benchmark={benchmark.name}, n_hard={n_hard}, n_macros={int(benchmark.num_macros)}")

    canon_in = compute_proxy_cost(placement, benchmark, plc)
    print(f"[baseline] cascade canon={canon_in['proxy_cost']:.5f}  ovl={int(canon_in['overlap_count'])}")
    assert int(canon_in["overlap_count"]) == 0, "Expected zero-overlap input"

    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement)
    inc_proxy_start = float(evaluator.current_cost()["proxy"])
    print(f"[evaluator] init proxy (incremental) = {inc_proxy_start:.5f} "
          f"(should ≈ canonical {canon_in['proxy_cost']:.5f})")

    CLUSTER_SIZE = 8
    GRID = 5              # 5x5 candidate grid per macro
    SPAN_FRAC = 2.0       # candidate window = ±2 * macro_dim around current
    N_PASSES = 5          # multi-pass, re-randomize cluster seeds each pass

    sizes_np = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed

    log = []
    t_total = time.time()
    total_clusters_changed = 0
    total_moves = 0
    pass_canon_history = []

    for p in range(N_PASSES):
        clusters = build_kmeans_clusters_by_knn(
            placement, fixed, n_hard, cluster_size=CLUSTER_SIZE, rng_seed=p,
        )
        print(f"\n[pass {p+1}/{N_PASSES}] {len(clusters)} clusters of size up to "
              f"{CLUSTER_SIZE} covering {sum(len(c) for c in clusters)} movable hard macros")
        pass_start_proxy = float(evaluator.current_cost()["proxy"])
        pass_t0 = time.time()
        pass_moves = 0
        pass_changed = 0
        for ci, members in enumerate(clusters):
            c_data = build_cluster_ilp_data(
                members, evaluator, benchmark, n_hard,
                grid=GRID, span_frac=SPAN_FRAC,
            )
            if c_data is None:
                continue
            picks = solve_cluster_ilp(c_data, sizes_np, time_limit_s=2.0)
            if picks is None:
                continue
            apply_stats = apply_picks_sequentially(c_data, picks, evaluator)
            if apply_stats["n_moved"] > 0:
                pass_changed += 1
                pass_moves += apply_stats["n_moved"]
        wall_pass = time.time() - pass_t0
        pass_end_proxy = float(evaluator.current_cost()["proxy"])
        placement_pass = evaluator.placement.to(torch.float32).clone()
        canon_pass = compute_proxy_cost(placement_pass, benchmark, plc)
        pass_canon_history.append(float(canon_pass["proxy_cost"]))
        total_clusters_changed += pass_changed
        total_moves += pass_moves
        print(f"[pass {p+1}] {wall_pass:.1f}s  moves={pass_moves}  changed_clusters={pass_changed}  "
              f"inc_proxy={pass_start_proxy:.5f} -> {pass_end_proxy:.5f}  "
              f"canon={canon_pass['proxy_cost']:.5f}  ovl={int(canon_pass['overlap_count'])}")

    wall_clusters = time.time() - t_total
    print(f"\n[ilp] {N_PASSES} passes done in {wall_clusters:.1f}s  total_moves={total_moves}  "
          f"changed_clusters={total_clusters_changed}")

    inc_proxy_end = float(evaluator.current_cost()["proxy"])
    placement_out = evaluator.placement.to(torch.float32).clone()
    canon_out = compute_proxy_cost(placement_out, benchmark, plc)
    print(f"[result] canon: {canon_in['proxy_cost']:.5f} -> {canon_out['proxy_cost']:.5f}  "
          f"({(canon_out['proxy_cost'] - canon_in['proxy_cost']) / canon_in['proxy_cost'] * 100:+.3f}%)")
    print(f"[result] ovl: {int(canon_in['overlap_count'])} -> {int(canon_out['overlap_count'])}")
    print(f"[result] incremental proxy: {inc_proxy_start:.5f} -> {inc_proxy_end:.5f} "
          f"(should match canonical within 1e-3)")

    lift_pct = (canon_in["proxy_cost"] - canon_out["proxy_cost"]) / canon_in["proxy_cost"] * 100
    GATE = 0.1
    verdict = "pass" if (lift_pct >= GATE and int(canon_out["overlap_count"]) == 0) else "fail"
    print(f"\n=== SPIKE GATE: {GATE:.2f}% lift, zero overlaps preserved ===")
    print(f"Lift: {lift_pct:+.3f}%  verdict: {verdict}")

    def _to_jsonable(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if hasattr(o, "tolist"):
            return o.tolist()
        return str(o)

    out = {
        "config": {
            "cluster_size": CLUSTER_SIZE, "grid": GRID, "span_frac": SPAN_FRAC,
            "n_passes": N_PASSES,
        },
        "pass_canon_history": pass_canon_history,
        "baseline_canon": float(canon_in["proxy_cost"]),
        "result_canon": float(canon_out["proxy_cost"]),
        "result_ovl": int(canon_out["overlap_count"]),
        "lift_pct": float(lift_pct),
        "incremental_proxy_start": float(inc_proxy_start),
        "incremental_proxy_end": float(inc_proxy_end),
        "total_moves": int(total_moves),
        "clusters_changed": int(total_clusters_changed),
        "wall_seconds": float(wall_clusters),
        "log": log,
        "verdict": verdict,
        "gate_lift_pct": GATE,
    }
    out_path = Path("experiments/E89_ilp_legalize/results/spike_ibm01.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=_to_jsonable))
    print(f"Wrote {out_path}")
    torch.save(
        {"placement": placement_out, "bench_name": "ibm01", "label": "E89_ilp"},
        "experiments/E89_ilp_legalize/results/spike_ibm01.pt",
    )


if __name__ == "__main__":
    main()
