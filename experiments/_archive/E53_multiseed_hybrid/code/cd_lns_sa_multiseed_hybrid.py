"""E53 — 3-way hybrid: E25 + E41(seed=42) + E41(seed=1), per-bench best.

E48 (E25 + E41_seed42) realized -0.17% on --fast vs theoretical -0.57%
because DPO seed-noise on ibm04 happened to be unfavorable for seed=42.
E52 (E41 seed=1) showed bench-by-bench DPO seed-noise:
  ibm01: tied (-0.12%)
  ibm04: seed=42 better by +0.79%
  ibm09: tied (-0.11%)
  ibm13: seed=1 better by -0.54%

Different seeds win on different benches. A 3-way hybrid that runs
E25 + E41 seed=42 + E41 seed=1 captures BOTH basin choice AND seed
choice per bench. Projected --fast result: best-of-{0.8910, 0.9845,
0.8403, 0.9441} avg ~0.91498 = -0.74% over E41 fast.

Pipeline: per benchmark, run all three placers, return lowest-cost
(zero-overlap) output.
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

import importlib.util
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer

from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import (
    CDLNSSADPOKJointPlacer,
)


class CDLNSSAMultiseedHybridPlacer:
    """E53: best-of-{E25, E41 seed=42, E41 seed=1} per bench."""

    def __init__(self, **kwargs):
        e41_only_keys = {"kjoint_K", "kjoint_top_N", "kjoint_budget_s"}
        e41_kwargs = {k: v for k, v in kwargs.items() if k in e41_only_keys}
        common_kwargs = {k: v for k, v in kwargs.items() if k not in e41_only_keys}

        self._e25 = CDLNSSAPlacer(**common_kwargs)
        self._e41_seed42 = CDLNSSADPOKJointPlacer(
            **{**common_kwargs, **e41_kwargs}
        )
        # E41 with all seeds = 1.
        kw1 = {**common_kwargs, **e41_kwargs}
        kw1.setdefault("seed", 1)
        kw1.setdefault("sa_seed", 1)
        kw1.setdefault("kjoint_seed", 1)
        kw1.setdefault("lns_seed", 1)
        self._e41_seed1 = CDLNSSADPOKJointPlacer(**kw1)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True)
        log(f"=== E53 MULTISEED-HYBRID: E25 + E41(seed=42) + E41(seed=1) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        candidates = []
        for name, placer in [
            ("E25", self._e25),
            ("E41_s42", self._e41_seed42),
            ("E41_s1", self._e41_seed1),
        ]:
            t0 = time.perf_counter()
            placement = placer.place(benchmark)
            wall = time.perf_counter() - t0
            ov = compute_overlap_metrics(placement, benchmark)
            px = compute_proxy_cost(placement, benchmark, plc)
            log(
                f"  {name} done: proxy={px['proxy_cost']:.5f} "
                f"overlaps={ov['overlap_count']} wall={wall:.1f}s"
            )
            if ov["overlap_count"] == 0:
                candidates.append((name, placement, px["proxy_cost"]))

        if not candidates:
            raise RuntimeError("E53: ALL placers produced overlapping placements")
        candidates.sort(key=lambda x: x[2])
        winner_name, winner_placement, winner_proxy = candidates[0]
        all_proxies = ", ".join(f"{n}={p:.5f}" for n, _, p in candidates)
        log(f"  E53 winner: {winner_name} proxy={winner_proxy:.5f} ({all_proxies})")
        return winner_placement
