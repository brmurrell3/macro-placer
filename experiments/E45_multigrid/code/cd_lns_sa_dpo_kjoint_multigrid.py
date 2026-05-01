"""E45 placer: CDLNSSA + DPO + K-joint + super-macro block LNS.

Pipeline (per benchmark):
  1. Run E41's pipeline (DPO best_of_v2 init -> CD plateau -> grid-bin
     LNS -> SA-v2 polish -> K=3 K-joint -> validate).
  2. Cluster hard macros into K = round(sqrt(n_hard)) super-macros via
     pymetis.
  3. Build a fresh IncrementalProxyEvaluator on the post-K-joint
     placement.
  4. Run super-macro block LNS (new mechanism): for each super-macro,
     try translating all constituents as a block to canvas-grid target
     centroids; commit the best-improving rigid translation.
  5. Validate (zero overlaps), preserve fixed macros, return.

The block-LNS phase exploits the multigrid CLUSTER STRUCTURE as a
new MOVE TYPE — moving 30-100 macros simultaneously vs E41's K=3.
This is the post-smoother that breaks past E41's saturation floor by
escaping the basin choice that single- through 3-macro moves can't
recover.

Reference:
- E41 — DPO + K=3 K-joint parent. Pipeline phases 1-7 are the same.
- E39 — K-joint primitive (eps=1e-9 strict separation).
- E45 multigrid_clustering — pymetis-based partitioning.
- E45 multigrid_block_lns — block-LNS implementation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics

from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import (
    CDLNSSADPOKJointPlacer,
)

from multigrid_clustering import cluster_macros, super_macro_geometry  # noqa: E402
from multigrid_block_lns import run_super_block_lns  # noqa: E402


class CDLNSSADPOKJointMultigridPlacer:
    """E45 placer: E41 pipeline + super-macro block-LNS post-smoother."""

    def __init__(
        self,
        block_lns_budget_s: float = 600.0,
        block_lns_top_M: int = 8,
        block_lns_seed: int = 42,
        K_super: Optional[int] = None,
        **kwargs,
    ):
        self._inner = CDLNSSADPOKJointPlacer(**kwargs)
        self.block_lns_budget_s = float(block_lns_budget_s)
        self.block_lns_top_M = int(block_lns_top_M)
        self.block_lns_seed = int(block_lns_seed)
        self.K_super = K_super  # None → default sqrt(n_hard) in cluster_macros

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True)
        log(
            f"=== E45 CDLNSSADPOKJointMultigridPlacer: phases 1-7 from E41, "
            f"then super-block-LNS budget={self.block_lns_budget_s:.0f}s "
            f"top_M={self.block_lns_top_M} ==="
        )

        # Phase 1-7: E41 pipeline.
        t0 = time.perf_counter()
        e41_placement = self._inner.place(benchmark)
        log(f"  E41 pipeline wall = {time.perf_counter() - t0:.1f}s")

        # Phase 8: cluster.
        t1 = time.perf_counter()
        n_hard = benchmark.num_hard_macros
        K = self.K_super if self.K_super is not None else max(
            2, int(round(n_hard ** 0.5))
        )
        cluster_ids = cluster_macros(benchmark, K)
        _super_sizes_t, _super_areas, super_members = super_macro_geometry(
            benchmark, cluster_ids, K
        )
        n_nonempty = sum(1 for m in super_members if m)
        log(
            f"  multigrid clustering: K={K}, non-empty super-macros={n_nonempty}, "
            f"wall={time.perf_counter() - t1:.1f}s"
        )

        # Phase 9: build evaluator on E41 output.
        t2 = time.perf_counter()
        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        evaluator = IncrementalProxyEvaluator(benchmark, plc, e41_placement)
        log(
            f"  evaluator setup: proxy={evaluator.current_cost()['proxy']:.5f}, "
            f"wall={time.perf_counter() - t2:.1f}s"
        )

        # Phase 10: super-block-LNS.
        t3 = time.perf_counter()
        stats = run_super_block_lns(
            evaluator,
            benchmark,
            plc,
            super_members,
            time_budget_s=self.block_lns_budget_s,
            top_M=self.block_lns_top_M,
            seed=self.block_lns_seed,
            log_fn=log,
        )
        log(
            f"  super-block-LNS done: "
            f"blocks_tried={stats['blocks_tried']}, "
            f"committed={stats['blocks_committed']}, "
            f"Δ={stats['total_improvement']:+.5f}, "
            f"passes={stats['passes']}, "
            f"wall={stats['wall_total_s']:.1f}s"
        )

        # Phase 11: validate.
        final_placement = evaluator.placement.clone()
        ov = compute_overlap_metrics(final_placement, benchmark)
        if ov["overlap_count"] > 0:
            raise RuntimeError(
                f"E45: post-block-LNS placement has {ov['overlap_count']} "
                f"overlap(s), area={ov['total_overlap_area']:.6e}"
            )
        log(
            f"  total wall: {time.perf_counter() - t0:.1f}s "
            f"(E41 + clustering + block-LNS + validate)"
        )
        return final_placement
