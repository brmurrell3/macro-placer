"""E113 — Test V5 on ariane133 (NG45) for generalization.

Per E54 lesson: don't promote based on IBM-only data. NG45 commercial
designs have a different macro distribution (much smaller hard macros,
many more soft cells, denser routing). If V5 catastrophically regresses
on ariane133, the lift on IBM doesn't matter.

Compares:
  - V3 baseline (Adam)
  - V5 m=0.003 λ_C=200 (the ibm01 sweet spot)
Then 60s CD polish on each.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

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


BENCH = "ariane133"
CD_BUDGET_S = 120.0


def main():
    bd = find_benchmark_dir(BENCH)
    benchmark, plc = load_benchmark_from_dir(str(bd))
    print(
        f"Loaded {BENCH}: {benchmark.num_macros} macros "
        f"({benchmark.num_hard_macros} hard), nets={benchmark.num_nets}",
        flush=True,
    )

    results = []
    for label, placer_class, kwargs in [
        ("V3 baseline (Adam)", SmoothGlobalPlacerV3,
         dict(num_steps=500, lr_frac=0.005, init="sdf", verbose=False)),
        ("V5 m=0.003 λ_C=200", SmoothGlobalPlacerV5,
         dict(stage_steps=(250, 200, 150), base_lr_frac=0.005,
              overlap_lambda_stage_C=(50.0, 200.0),
              overlap_margin_frac=0.003, verbose=False)),
    ]:
        print(f"\n--- {label} ---", flush=True)
        benchmark, plc = load_benchmark_from_dir(str(bd))
        placer = placer_class(**kwargs)
        t0 = time.time()
        with quiet():
            pos = placer.place(benchmark)
        wall_raw = time.time() - t0
        proxy_raw = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        ovl_raw = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        print(f"  RAW: proxy={proxy_raw:.5f} ovl={ovl_raw} wall={wall_raw:.0f}s",
              flush=True)

        if ovl_raw > 0:
            print(f"  SKIP CD (overlap)", flush=True)
            results.append({
                "label": label, "raw_proxy": proxy_raw, "raw_ovl": ovl_raw,
                "raw_wall_s": wall_raw,
                "polish_proxy": float("inf"), "polish_wall_s": 0.0,
            })
            continue

        ev = IncrementalProxyEvaluator(benchmark, plc, pos.to(torch.float64))
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        t1 = time.time()
        run_cd_adaptive(ev, benchmark, plc, movable, min_time_s=0.0,
                        hard_cap_s=CD_BUDGET_S, patience=3, plateau_threshold=1e-4)
        polished = ev.placement.clone().to(torch.float32)
        polish_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
        print(f"  +CD{CD_BUDGET_S:.0f}s: proxy={polish_proxy:.5f} wall={time.time()-t1:.0f}s",
              flush=True)
        results.append({
            "label": label, "raw_proxy": proxy_raw, "raw_ovl": ovl_raw,
            "raw_wall_s": wall_raw, "polish_proxy": polish_proxy,
            "polish_wall_s": time.time() - t1,
        })

    print("\n=== SUMMARY (ariane133) ===", flush=True)
    for r in results:
        print(
            f"  {r['label']:35s}  raw={r['raw_proxy']:.5f}  +CD{CD_BUDGET_S:.0f}s={r['polish_proxy']:.5f}",
            flush=True,
        )

    out_path = _HERE.parent / "results" / "test_ariane133.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
