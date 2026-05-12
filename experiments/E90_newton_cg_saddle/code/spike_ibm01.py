"""E90 spike — multi-direction saddle escape on cached cascade ibm01.

Two probes:
  (1) only_rank_at_least=2 with K=3 — does ANY rank-≥2 combination beat the
      cached cascade input? If yes → multi-direction has value.
  (2) (only if (1) is null) K=3, all ranks — sanity check that single-direction
      also can't improve (cascade already saturated single-direction).

Reduced polish_budget_s=45 to fit in ~10 min on M3. Cloud runs the full
budget.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from multi_saddle import multi_saddle_escape

from macro_place.loader import load_benchmark_from_dir


def main():
    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    cached = torch.load(
        "experiments/E84_cascading_saddle/results/cascade_ibm01.pt",
        weights_only=False,
        map_location="cpu",
    )
    plateau = cached["placement"].clone().float()
    print(f"[spike] loaded plateau {tuple(plateau.shape)} from cascade-cached ibm01")

    out_dir = Path("experiments/E90_newton_cg_saddle/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Probe 1: rank ≥ 2 only (multi-direction contribution alone).
    print("\n=== PROBE 1: only_rank_at_least=2, K=3, polish=45s ===")
    t0 = time.time()
    best_state_p1, stats_p1 = multi_saddle_escape(
        plateau, benchmark, plc,
        K=3, eps_values=(0.5, 2.0),
        polish_budget_s=45.0,
        only_rank_at_least=2,
        total_budget_s=900.0,
    )
    wall_p1 = time.time() - t0
    print(f"[probe 1] wall={wall_p1:.0f}s  init={stats_p1['init_proxy']:.5f} "
          f"-> best={stats_p1['best_proxy']:.5f}  Δ={stats_p1['improvement']:+.5f}")

    decision_p1 = {
        "improvement_frac": (stats_p1["improvement"] / stats_p1["init_proxy"]) if stats_p1["init_proxy"] > 0 else 0.0,
        "found_lift": stats_p1["best_proxy"] < stats_p1["init_proxy"] - 1e-7,
    }
    print(f"[probe 1 decision] found_lift={decision_p1['found_lift']}  "
          f"improvement_frac={decision_p1['improvement_frac']:+.4%}")

    out_p1 = out_dir / "spike_ibm01_probe1.json"
    out_p1.write_text(json.dumps({
        "config": {"K": 3, "eps_values": [0.5, 2.0], "polish": 45.0, "only_rank_at_least": 2},
        "wall": wall_p1,
        "stats": stats_p1,
        "decision": decision_p1,
    }, indent=2))
    torch.save(
        {"placement": best_state_p1, "bench_name": "ibm01", "label": "E90_probe1"},
        out_dir / "spike_ibm01_probe1.pt",
    )

    # Probe 2: rank=1 baseline (E74-like) — sanity-check that single-direction
    # has saturated. Only run if probe 1 was null, else we already know
    # multi-direction wins.
    if decision_p1["found_lift"]:
        print("\n=== PROBE 1 ALREADY PASSED — skip baseline probe 2 ===")
        print(f"\nSPIKE VERDICT: pass (multi-direction lift = {decision_p1['improvement_frac']:+.4%})")
        return

    print("\n=== PROBE 2: rank=1 baseline (E74-like, K=3, polish=45s) ===")
    t0 = time.time()
    _, stats_p2 = multi_saddle_escape(
        plateau, benchmark, plc,
        K=3, eps_values=(0.5, 2.0),
        polish_budget_s=45.0,
        only_rank_at_least=1, max_rank=1,
        total_budget_s=900.0,
    )
    wall_p2 = time.time() - t0
    print(f"[probe 2] wall={wall_p2:.0f}s  init={stats_p2['init_proxy']:.5f} "
          f"-> best={stats_p2['best_proxy']:.5f}  Δ={stats_p2['improvement']:+.5f}")
    decision_p2 = {
        "improvement_frac": (stats_p2["improvement"] / stats_p2["init_proxy"]) if stats_p2["init_proxy"] > 0 else 0.0,
        "found_lift": stats_p2["best_proxy"] < stats_p2["init_proxy"] - 1e-7,
    }
    out_p2 = out_dir / "spike_ibm01_probe2.json"
    out_p2.write_text(json.dumps({
        "config": {"K": 3, "eps_values": [0.5, 2.0], "polish": 45.0,
                   "only_rank_at_least": 1, "max_rank": 1},
        "wall": wall_p2,
        "stats": stats_p2,
        "decision": decision_p2,
    }, indent=2))

    print("\n=== SPIKE VERDICT ===")
    if decision_p1["found_lift"]:
        print(f"PASS: multi-direction (rank≥2) found lift {decision_p1['improvement_frac']:+.4%}")
    elif decision_p2["found_lift"]:
        print(f"PARTIAL: rank-1 found lift {decision_p2['improvement_frac']:+.4%}  "
              f"(but cascade should already have done this) — debug cascade convergence?")
    else:
        print(f"FAIL: neither rank-1 nor rank-≥2 found a lift at this plateau")


if __name__ == "__main__":
    main()
