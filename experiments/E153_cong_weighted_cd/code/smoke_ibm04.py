"""E153 smoke — congestion-weighted CD second pass on ibm04.

Pipeline:
  1. Load v2-extCD shipped placement (from a quick run, or use sdf_init+greedy)
  2. CD polish under canonical weights (congestion=0.5)
  3. Record cost A
  4. Restart CD with weights["congestion"] *= 2.0
  5. Only commit final placement if canonical cost B <= A; else fallback to A.

The goal isn't to ship — it's to see whether the canonical objective can be
pushed below the CD-saturated plateau by *biasing* the search direction.

If yes (B < A by > 0.3%), it's an actionable lift mechanism for the v3.
If no (B >= A or worsens), the plateau is rigid w.r.t. congestion weight
and we've ruled out idea C+ as an EV lever.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def main():
    bench_name = "ibm04"
    print(f"=== E153 cong-weighted CD smoke ({bench_name}) ===", flush=True)

    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    # ── Stage 1: SDF init + project + canonical CD polish ──
    t0 = time.time()
    pos = sdf_init(benchmark)
    pos, _ = project_overlaps(pos, benchmark)
    ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    init_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
    print(f"  init: proxy={init_proxy:.5f} ovl={ovl} wall={time.time()-t0:.0f}s", flush=True)

    movable = [i for i in range(benchmark.num_macros)
               if not bool(benchmark.macro_fixed[i])]

    # First CD: canonical weights {wl:1.0, d:0.5, c:0.5}
    evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
    print(f"  CD pass A weights={evaluator.weights}", flush=True)
    run_cd_adaptive(
        evaluator, benchmark, plc, movable,
        min_time_s=120.0,
        hard_cap_s=200.0,
        patience=5,
        plateau_threshold=0.001,
        log_fn=None,
    )
    posA = evaluator.placement.detach().clone().to(torch.float32)
    proxyA = float(compute_proxy_cost(posA, benchmark, plc)["proxy_cost"])
    detailsA = compute_proxy_cost(posA, benchmark, plc)
    print(
        f"  CD A: proxy={proxyA:.5f} wl={detailsA['wirelength_cost']:.4f} "
        f"d={detailsA['density_cost']:.4f} c={detailsA['congestion_cost']:.4f} "
        f"wall={time.time()-t0:.0f}s",
        flush=True,
    )

    # ── Stage 2: same starting point as A, but with cong-weighted search ──
    for cong_w_scale in [1.5, 2.0, 3.0]:
        evaluator2 = IncrementalProxyEvaluator(benchmark, plc, posA.clone())
        evaluator2.weights = {
            "wirelength": 1.0,
            "density": 0.5,
            "congestion": 0.5 * cong_w_scale,
        }
        t1 = time.time()
        run_cd_adaptive(
            evaluator2, benchmark, plc, movable,
            min_time_s=60.0,
            hard_cap_s=120.0,
            patience=5,
            plateau_threshold=0.001,
            log_fn=None,
        )
        posB = evaluator2.placement.detach().clone().to(torch.float32)
        detailsB = compute_proxy_cost(posB, benchmark, plc)
        proxyB = float(detailsB["proxy_cost"])
        # Final canonical check — does the cong-weighted basin recover lower CANONICAL cost?
        delta = proxyB - proxyA
        print(
            f"  CD B w_cong x{cong_w_scale}: canonical_proxy={proxyB:.5f} "
            f"wl={detailsB['wirelength_cost']:.4f} d={detailsB['density_cost']:.4f} "
            f"c={detailsB['congestion_cost']:.4f} "
            f"Δ_vs_A={delta:+.5f} ({delta/proxyA*100:+.2f}%) wall={time.time()-t1:.0f}s",
            flush=True,
        )

        # Re-polish back under canonical to validate the new basin
        evaluator3 = IncrementalProxyEvaluator(benchmark, plc, posB.clone())
        # weights default = canonical
        t2 = time.time()
        run_cd_adaptive(
            evaluator3, benchmark, plc, movable,
            min_time_s=60.0,
            hard_cap_s=120.0,
            patience=5,
            plateau_threshold=0.001,
            log_fn=None,
        )
        posC = evaluator3.placement.detach().clone().to(torch.float32)
        proxyC = float(compute_proxy_cost(posC, benchmark, plc)["proxy_cost"])
        delta_c = proxyC - proxyA
        print(
            f"  CD C (re-polish canonical): proxy={proxyC:.5f} "
            f"Δ_vs_A={delta_c:+.5f} ({delta_c/proxyA*100:+.2f}%) wall={time.time()-t2:.0f}s",
            flush=True,
        )

    print(f"  TOTAL wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
