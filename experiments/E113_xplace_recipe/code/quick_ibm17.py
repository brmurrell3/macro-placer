"""E113 — Quick V5 m=0.003 test on ibm17 (single config, no V3 baseline).

Saves 30 min vs the full 5-variant test_ibm17.py.
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

from smooth_global_placer_v5 import SmoothGlobalPlacerV5


class quiet:
    def __enter__(self): self.old = sys.stdout; sys.stdout = io.StringIO(); return self
    def __exit__(self, *a): sys.stdout = self.old


def main():
    bd = find_benchmark_dir("ibm17")
    b, plc = load_benchmark_from_dir(str(bd))
    print(f"ibm17: {b.num_macros} macros ({b.num_hard_macros} hard), nets={b.num_nets}", flush=True)

    placer = SmoothGlobalPlacerV5(
        stage_steps=(250, 200, 150), base_lr_frac=0.005,
        overlap_lambda_stage_C=(50.0, 200.0),
        overlap_margin_frac=0.003, verbose=False,
    )
    t0 = time.time()
    with quiet():
        pos = placer.place(b)
    wall_raw = time.time() - t0
    proxy_raw = float(compute_proxy_cost(pos, b, plc)['proxy_cost'])
    ovl_raw = compute_overlap_metrics(pos, b)['overlap_count']
    print(f"V5 m=0.003 RAW: proxy={proxy_raw:.5f} ovl={ovl_raw} wall={wall_raw:.0f}s",
          flush=True)

    if ovl_raw > 0:
        print(f"SKIP CD (overlap)", flush=True)
        return

    # CD 60s polish
    ev = IncrementalProxyEvaluator(b, plc, pos.to(torch.float64))
    movable = [i for i in range(b.num_macros) if not bool(b.macro_fixed[i])]
    t1 = time.time()
    run_cd_adaptive(ev, b, plc, movable, min_time_s=0.0, hard_cap_s=60.0,
                    patience=3, plateau_threshold=1e-4)
    polished = ev.placement.clone().to(torch.float32)
    proxy_cd = float(compute_proxy_cost(polished, b, plc)['proxy_cost'])
    print(f"V5 m=0.003 +CD60s: proxy={proxy_cd:.5f} wall={time.time()-t1:.0f}s",
          flush=True)
    print(f"\nTarget: ibm17 +CD60s < 1.24  Result: {proxy_cd:.4f}  "
          f"{'PASS' if proxy_cd < 1.24 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
