"""E113 — Sweep V5 (with margin) on the 4 --fast benches.

Quick validation that V5 + margin generalizes beyond ibm01.
The --fast benches are: ibm01, ibm04, ibm09, ibm13.

Compares:
  - V3 baseline (Adam)
  - V5 multi-stage + margin=0.003, λ_C=200 (the ibm01 sweet spot)

All with 60s CD polish. Reports per-bench raw and +CD60s.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE, _ROOT,
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v3 import SmoothGlobalPlacerV3
from smooth_global_placer_v5 import SmoothGlobalPlacerV5


class quiet:
    def __enter__(self): self.old = sys.stdout; sys.stdout = io.StringIO(); return self
    def __exit__(self, *a): sys.stdout = self.old


FAST_BENCHES = ["ibm01", "ibm04", "ibm09", "ibm13"]
CD_BUDGET_S = 60.0


def run_placer(placer_class, bench, **kwargs):
    bd = find_benchmark_dir(bench)
    b, plc = load_benchmark_from_dir(str(bd))
    placer = placer_class(**kwargs)
    t0 = time.time()
    with quiet():
        pos = placer.place(b)
    wall_raw = time.time() - t0
    proxy_raw = float(compute_proxy_cost(pos, b, plc)["proxy_cost"])
    ovl_raw = compute_overlap_metrics(pos, b)["overlap_count"]
    if ovl_raw > 0:
        return {"bench": bench, "raw_proxy": proxy_raw, "raw_ovl": ovl_raw,
                "raw_wall_s": wall_raw, "polish_proxy": float("inf"),
                "polish_wall_s": 0.0, "polish_ovl": ovl_raw}
    ev = IncrementalProxyEvaluator(b, plc, pos.to(torch.float64))
    movable = [i for i in range(b.num_macros) if not bool(b.macro_fixed[i])]
    t1 = time.time()
    run_cd_adaptive(ev, b, plc, movable, min_time_s=0.0, hard_cap_s=CD_BUDGET_S,
                    patience=3, plateau_threshold=1e-4)
    polished = ev.placement.clone().to(torch.float32)
    polish_proxy = float(compute_proxy_cost(polished, b, plc)["proxy_cost"])
    polish_ovl = compute_overlap_metrics(polished, b)["overlap_count"]
    return {
        "bench": bench, "raw_proxy": proxy_raw, "raw_ovl": ovl_raw,
        "raw_wall_s": wall_raw, "polish_proxy": polish_proxy,
        "polish_ovl": polish_ovl, "polish_wall_s": time.time() - t1,
    }


def main():
    print(f"=== E113 sweep on --fast benches ({FAST_BENCHES}) ===", flush=True)
    results = {}
    for placer_label, placer_class, kwargs in [
        ("V3 (Adam baseline)", SmoothGlobalPlacerV3,
         dict(num_steps=500, lr_frac=0.005, init="sdf", verbose=False)),
        ("V5 m=0.003 λ_C=200", SmoothGlobalPlacerV5,
         dict(stage_steps=(250, 200, 150), base_lr_frac=0.005,
              overlap_lambda_stage_C=(50.0, 200.0),
              overlap_margin_frac=0.003, verbose=False)),
    ]:
        print(f"\n--- {placer_label} ---", flush=True)
        results[placer_label] = []
        for bench in FAST_BENCHES:
            print(f"  {bench}... ", end="", flush=True)
            r = run_placer(placer_class, bench, **kwargs)
            print(
                f"raw={r['raw_proxy']:.5f}/{r['raw_ovl']:2d}  "
                f"+CD60={r['polish_proxy']:.5f}  "
                f"wall={r['raw_wall_s']:.0f}+{r['polish_wall_s']:.0f}s",
                flush=True,
            )
            results[placer_label].append(r)

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'bench':6s} | {'V3 raw':10s} | {'V3 +CD60':10s} || "
          f"{'V5 raw':10s} | {'V5 +CD60':10s} | {'Δ+CD %':10s}", flush=True)
    print("-" * 90, flush=True)
    avg_v3, avg_v5, n = 0.0, 0.0, 0
    for i, bench in enumerate(FAST_BENCHES):
        v3 = results["V3 (Adam baseline)"][i]
        v5 = results["V5 m=0.003 λ_C=200"][i]
        if v3["polish_proxy"] != float("inf") and v5["polish_proxy"] != float("inf"):
            delta_pct = (v5["polish_proxy"] - v3["polish_proxy"]) / v3["polish_proxy"] * 100.0
            print(
                f"{bench:6s} | {v3['raw_proxy']:10.5f} | {v3['polish_proxy']:10.5f} || "
                f"{v5['raw_proxy']:10.5f} | {v5['polish_proxy']:10.5f} | {delta_pct:+.2f}%",
                flush=True,
            )
            avg_v3 += v3["polish_proxy"]
            avg_v5 += v5["polish_proxy"]
            n += 1
    if n > 0:
        print(f"\nMean +CD60s: V3 = {avg_v3/n:.5f}, V5 = {avg_v5/n:.5f}, "
              f"delta = {(avg_v5/n - avg_v3/n)/(avg_v3/n)*100:+.2f}%", flush=True)

    out_path = _HERE.parent / "results" / "sweep_fast.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
