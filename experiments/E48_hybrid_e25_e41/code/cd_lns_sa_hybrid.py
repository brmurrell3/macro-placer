"""E48 hybrid placer: run E25 and E41 pipelines, return per-bench best.

Per-bench analysis of the verified --all results (committed 2026-04-30):
  E25: 1.09540 — wins 5/17 (ibm01, ibm06, ibm07, ibm17, ibm18)
  E41: 1.08480 — wins 12/17 (hard plateau benches + most others)
  E18: 1.08979 — dominated per-bench by max(E25, E41)
  best-of avg: 1.08121 = -0.33 % vs E41, -1.62 % vs E12

The per-bench winner is determined by which init class lands in the
right basin for that benchmark — DPO basin (E41) on most, but SDF
basin (E25) on the 5 where DPO is globally worse. Local moves can't
recover the basin choice once made.

This placer runs BOTH pipelines, evaluates compute_proxy_cost on both
outputs, and returns the lower-cost placement. No per-benchmark
hardcoded logic — the pipeline runs deterministically on every bench;
the per-bench dispatch is by PROXY VALUE.

Pipeline:
  1. Run E25 pipeline (CDLNSSAPlacer): SDF + CD + LNS + SA-v2.
  2. Run E41 pipeline (CDLNSSADPOKJointPlacer): DPO + CD + LNS + SA-v2 + K-joint.
  3. Compare proxies; return the lower one.

Wall: ~30 min (E25) + ~50 min (E41) per bench = ~80 min/bench
sequential. Total --all wall ~22-26 hr serial; needs 2-core parallelism
to fit the 17-hr envelope.

Reference:
- E25 — `submissions/cd_lns_sa/placer.py` (SDF-init pipeline).
- E41 — `experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`
        (DPO-init + K-joint pipeline).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# E25 pipeline at submissions/cd_lns_sa/placer.py.
import importlib.util
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer

from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import (
    CDLNSSADPOKJointPlacer,
)


class CDLNSSAHybridPlacer:
    """E48 hybrid: run E25 and E41 pipelines, return per-bench best."""

    def __init__(self, **kwargs):
        # Each inner placer takes its own kwargs; we pass through any
        # E41-style kwargs (kjoint_*) to E41 only and the rest to both.
        e41_only_keys = {
            "kjoint_K", "kjoint_top_N", "kjoint_budget_s", "kjoint_seed"
        }
        e41_kwargs = {k: v for k, v in kwargs.items() if k in e41_only_keys}
        common_kwargs = {k: v for k, v in kwargs.items() if k not in e41_only_keys}
        self._e25 = CDLNSSAPlacer(**common_kwargs)
        self._e41 = CDLNSSADPOKJointPlacer(**common_kwargs, **e41_kwargs)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True)
        log(f"=== E48 HYBRID placer: running E25 then E41, returning lower-proxy output ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Run E25 pipeline.
        t0 = time.perf_counter()
        e25_placement = self._e25.place(benchmark)
        e25_wall = time.perf_counter() - t0
        e25_proxy = compute_proxy_cost(e25_placement, benchmark, plc)
        e25_ov = compute_overlap_metrics(e25_placement, benchmark)
        log(
            f"  E25 done: proxy={e25_proxy['proxy_cost']:.5f} "
            f"overlaps={e25_ov['overlap_count']} wall={e25_wall:.1f}s"
        )

        # Run E41 pipeline. NOTE: this re-runs DPO + CD + LNS + SA + K-joint
        # from the original benchmark state (ignores the E25 output).
        t1 = time.perf_counter()
        e41_placement = self._e41.place(benchmark)
        e41_wall = time.perf_counter() - t1
        e41_proxy = compute_proxy_cost(e41_placement, benchmark, plc)
        e41_ov = compute_overlap_metrics(e41_placement, benchmark)
        log(
            f"  E41 done: proxy={e41_proxy['proxy_cost']:.5f} "
            f"overlaps={e41_ov['overlap_count']} wall={e41_wall:.1f}s"
        )

        # Pick the lower-proxy-with-zero-overlaps output.
        e25_valid = e25_ov["overlap_count"] == 0
        e41_valid = e41_ov["overlap_count"] == 0

        if e25_valid and e41_valid:
            if e25_proxy["proxy_cost"] <= e41_proxy["proxy_cost"]:
                winner = "E25"
                final = e25_placement
                final_proxy = e25_proxy["proxy_cost"]
            else:
                winner = "E41"
                final = e41_placement
                final_proxy = e41_proxy["proxy_cost"]
        elif e25_valid:
            winner = "E25 (E41 had overlaps)"
            final = e25_placement
            final_proxy = e25_proxy["proxy_cost"]
        elif e41_valid:
            winner = "E41 (E25 had overlaps)"
            final = e41_placement
            final_proxy = e41_proxy["proxy_cost"]
        else:
            raise RuntimeError(
                f"E48 hybrid: BOTH pipelines produced overlapping placements "
                f"(E25 {e25_ov['overlap_count']} overlaps, "
                f"E41 {e41_ov['overlap_count']} overlaps)"
            )
        total_wall = time.perf_counter() - t0
        log(
            f"  E48 winner: {winner} proxy={final_proxy:.5f} "
            f"(E25={e25_proxy['proxy_cost']:.5f}, E41={e41_proxy['proxy_cost']:.5f}) "
            f"total_wall={total_wall:.1f}s"
        )
        return final
