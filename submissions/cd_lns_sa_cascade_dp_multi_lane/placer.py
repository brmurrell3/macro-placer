"""CDLNSSACascadeDPMultiLanePlacer — cascade + MULTI-CONFIG DP basin lane.

Identical to `submissions/cd_lns_sa_cascade_dp_lane/placer.py` BUT replaces
the single-DP basin generator with `multi_dp_basin(K=4)` from E96. That
function runs DP K=4 times (diverse configs: baseline_auto, dw_low, dw_high,
seed_2024), greedy-legalizes each, applies extended_legalize as fallback
for 1-10 residual overlaps, and selects the lowest legalize_proxy.

Rule-compliant: same algorithm for every bench (K=4 fixed configs derived
from bench geometry via auto_target_density).

Status (2026-05-13 ~02:30 UTC): UNVALIDATED on `--all`. E96 probe data:
- ibm10: K=4 picks dw_low or dw_high (probe winner). +9% basin improvement
  vs baseline. Polish from 1.39 → ~1.04 expected.
- ibm12: K=4 picks baseline_auto (already optimal). No regression.
- ibm14: K=4 picks dw_low. +3% basin improvement vs baseline.
- ibm17: K=4 probe pending; expected td_high or dw_low based on partial data.

Trade-off vs single-DP variant:
- Cost: +4 × ~150s = +600s DP wall (~10 min). Within 3300s budget.
- Win: 1-9% basin improvement on benches where alternative config wins.

To use:
  export DREAMPLACE_ROOT=/home/ubuntu/DREAMPlace_cpu/install
  export DP_DOCKER_IMAGE=dreamplace:custom
  export DP_USE_GPU=0
  uv run evaluate submissions/cd_lns_sa_cascade_dp_multi_lane/placer.py \\
    --all --json --hypothesis E96multilane

Same env requirements as cd_lns_sa_cascade_dp_lane.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# E25 placer (SDF init + polish primitives).
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer
run_lns_gridbin = _E25_MOD.run_lns_gridbin
run_sa_polish_v2 = _E25_MOD.run_sa_polish_v2

# E41 placer (DPO init + K-joint pipeline).
from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import CDLNSSADPOKJointPlacer

# Cascading saddle escape (E84).
_E84_CODE = _ROOT / "experiments" / "E84_cascading_saddle" / "code"
if str(_E84_CODE) not in sys.path:
    sys.path.insert(0, str(_E84_CODE))
from cascading_saddle import cascading_saddle_escape

# E96 multi-config DP basin selector — the NEW primitive.
_E96 = _ROOT / "experiments" / "E96_multi_config_dp" / "code"
if str(_E96) not in sys.path:
    sys.path.insert(0, str(_E96))
from multi_dp_basin import multi_dp_basin

# Stricter legalizer fallback.
_E91 = _ROOT / "experiments" / "E91_dp_full_polish" / "code"
if str(_E91) not in sys.path:
    sys.path.insert(0, str(_E91))
from extended_legalize import extended_legalize


def _polish_dp_basin(
    legal_placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    cd_hard_cap_s: float,
    lns_budget_s: float,
    sa_budget_s: float,
    log,
) -> Tuple[Optional[torch.Tensor], float]:
    """Apply E25 polish pipeline (CD + LNS + SA) to a (legalized) DP basin.

    Caller is responsible for ensuring `legal_placement` has 0 hard-macro
    overlaps (i.e., already greedy_legalized + extended_legalize'd by
    multi_dp_basin).
    """
    placement_f64 = legal_placement.detach().clone().to(torch.float64)
    ev = IncrementalProxyEvaluator(benchmark, plc, placement_f64)

    n_hard = benchmark.num_hard_macros
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable = [i for i in range(benchmark.num_macros) if not bool(fixed[i])]
    hard_movable = [i for i in range(n_hard) if not bool(fixed[i])]

    log(f"  [DP polish] CD (cap {cd_hard_cap_s:.0f}s)")
    run_cd_adaptive(
        evaluator=ev, benchmark=benchmark, plc=plc, movable=movable,
        min_time_s=min(60.0, cd_hard_cap_s * 0.1),
        hard_cap_s=cd_hard_cap_s,
        patience=3, plateau_threshold=0.001, log_fn=None,
    )

    log(f"  [DP polish] LNS ({lns_budget_s:.0f}s)")
    run_lns_gridbin(
        evaluator=ev, benchmark=benchmark, plc=plc,
        hard_movable=hard_movable, time_budget_s=lns_budget_s,
        destroy_frac=0.05, destroy_cap=30, seed=42, log_fn=None,
    )

    log(f"  [DP polish] SA-v2 ({sa_budget_s:.0f}s)")
    run_sa_polish_v2(
        evaluator=ev, benchmark=benchmark, plc=plc,
        hard_movable=hard_movable, time_budget_s=sa_budget_s,
        T0=5e-4, Tf=1e-6, seed=42, breakpoint_budget=12, log_fn=None,
    )

    polished = ev.placement.detach().clone().to(torch.float32)
    polished, _ = project_overlaps(polished, benchmark)
    final_ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
    proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
    if final_ovl > 0:
        log(f"  [DP polish] FAILED VALIDITY: {final_ovl} residual overlaps "
            f"after polish — DP lane will be excluded from plateau pick")
        return None, float('inf')
    return polished, proxy


class CDLNSSACascadeDPMultiLanePlacer:
    """Cascade with multi-config DP as a third basin lane.

    Uses `multi_dp_basin(K=4)` to select the best DP basin from 4 diverse
    configs, instead of a single auto-config DP run.
    """

    def __init__(
        self,
        max_iters: int = 5,
        eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
        polish_budget: float = 180.0,
        budget_seconds: Optional[float] = 3300.0,
        K_dp_configs: int = 4,
        verbose: bool = True,
    ):
        self.max_iters = max_iters
        self.eps_values = eps_values
        self.polish_budget = polish_budget
        self.budget_seconds = budget_seconds
        self.K_dp_configs = K_dp_configs
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSACascadeDPMultiLanePlacer ({benchmark.name}, K_dp={self.K_dp_configs}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Budget: rebalanced to give DP-lane enough room for both K=4 multi-DP
        # AND a proper polish (CD+LNS+SA). Compared to cd_lns_sa_cascade_dp_lane
        # (single-DP), DP lane share goes from 22% → 30%. E25/E41 shares
        # shrink slightly (22→18%, 24→20%). Cascade saddle stays at ~28%.
        #
        # With K=4 and B=3300s:
        #   - multi-DP basin selection: 4 × 150s = 600s
        #   - DP polish (CD/LNS/SA): 30% × 3300 - 600 = 390s total
        #     → CD ~250s, LNS ~70s, SA ~70s
        #   - Matches the *minimum* polish the basin needs to converge.
        #
        # If K is reduced (e.g., 2), DP polish gets correspondingly more time.
        if self.budget_seconds is not None:
            B = self.budget_seconds
            e25_kwargs = dict(
                cd_hard_cap_s=B * 0.12, lns_budget_s=B * 0.03, sa_budget_s=B * 0.03,
            )
            e41_kwargs = dict(
                cd_hard_cap_s=B * 0.13, lns_budget_s=B * 0.03, sa_budget_s=B * 0.03,
                kjoint_budget_s=B * 0.01,
            )
            multi_dp_basin_estimate_s = self.K_dp_configs * 150.0
            dp_polish_budget = max(120.0, B * 0.30 - multi_dp_basin_estimate_s)
            dp_kwargs = dict(
                cd_hard_cap_s=max(120.0, dp_polish_budget * 0.65),
                lns_budget_s=max(30.0, dp_polish_budget * 0.18),
                sa_budget_s=max(30.0, dp_polish_budget * 0.18),
            )
            log(f"  budget {B:.0f}s; E25={sum(e25_kwargs.values()):.0f}s "
                f"E41={sum(e41_kwargs.values()):.0f}s "
                f"DP=basin {multi_dp_basin_estimate_s:.0f}s + polish {sum(dp_kwargs.values()):.0f}s "
                f"cascade saddle target ~{B * 0.28:.0f}s")
        else:
            e25_kwargs = e41_kwargs = {}
            dp_kwargs = dict(cd_hard_cap_s=400.0, lns_budget_s=120.0, sa_budget_s=120.0)

        # Phase 1: E25 (SDF init + polish)
        log("  Phase 1: E25 (SDF + CD + LNS + SA)")
        e25 = CDLNSSAPlacer(**e25_kwargs).place(benchmark)
        e25_proxy = float(compute_proxy_cost(e25, benchmark, plc)["proxy_cost"])
        log(f"  E25 done: proxy={e25_proxy:.5f} (wall={time.time() - t0:.0f}s)")

        # Phase 2: E41
        e41 = None
        e41_proxy = float("inf")
        if deadline is not None and (deadline - time.time()) < 800.0:
            log(f"  Phase 2: SKIPPED ({deadline - time.time():.0f}s left)")
        else:
            log("  Phase 2: E41 (DPO + CD + LNS + SA + K-joint)")
            e41 = CDLNSSADPOKJointPlacer(**e41_kwargs).place(benchmark)
            e41_proxy = float(compute_proxy_cost(e41, benchmark, plc)["proxy_cost"])
            log(f"  E41 done: proxy={e41_proxy:.5f} (wall={time.time() - t0:.0f}s)")

        # Phase 3: Multi-config DP (with fair polish)
        dp = None
        dp_proxy = float("inf")
        # Need ~basin_estimate + polish + some margin for cascade
        dp_min_time = self.K_dp_configs * 200.0 + sum(dp_kwargs.values()) + 60.0
        if deadline is not None and (deadline - time.time()) < dp_min_time:
            log(f"  Phase 3: DP SKIPPED ({deadline - time.time():.0f}s left, need {dp_min_time:.0f}s)")
        else:
            log(f"  Phase 3: Multi-config DREAMPlace (K={self.K_dp_configs}) + polish")
            raw, legal, basin_stats = multi_dp_basin(
                benchmark, plc, K=self.K_dp_configs,
                deadline=(deadline - sum(dp_kwargs.values()) - 60.0) if deadline else None,
                log=log,
            )
            if legal is not None:
                legal_ovl = int(compute_overlap_metrics(legal, benchmark)["overlap_count"])
                if legal_ovl > 0:
                    log(f"  [multi-DP] winner has {legal_ovl} residual overlaps; DP lane skipped")
                else:
                    dp, dp_proxy = _polish_dp_basin(
                        legal, benchmark, plc,
                        cd_hard_cap_s=dp_kwargs["cd_hard_cap_s"],
                        lns_budget_s=dp_kwargs["lns_budget_s"],
                        sa_budget_s=dp_kwargs["sa_budget_s"],
                        log=log,
                    )
                    log(f"  DP polished done: proxy={dp_proxy:.5f} "
                        f"(wall={time.time() - t0:.0f}s)")

        # Phase 4: plateau pick = best of {E25, E41, DP}
        candidates = [(e25_proxy, e25, "E25")]
        if e41 is not None:
            candidates.append((e41_proxy, e41, "E41"))
        if dp is not None:
            candidates.append((dp_proxy, dp, "DP"))
        candidates.sort(key=lambda c: c[0])
        plateau_proxy, plateau, plateau_label = candidates[0]
        log(f"  hybrid plateau pick: {plateau_label} ({plateau_proxy:.5f}) "
            f"[{', '.join(f'{n}={p:.5f}' for p, _, n in candidates)}]")

        # Phase 5: cascading saddle escape
        cascade_state = plateau
        cascade_proxy = plateau_proxy
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining < 60.0:
                log(f"  Phase 5: SKIPPED ({remaining:.0f}s left)")
            else:
                cascade_budget = remaining - 10.0
                log(f"  Phase 5: cascading saddle (budget {cascade_budget:.0f}s, "
                    f"max_iters {self.max_iters})")
                try:
                    cascade_state, stats = cascading_saddle_escape(
                        plateau, benchmark, plc,
                        max_iters=self.max_iters,
                        eps_values=self.eps_values,
                        polish_budget=self.polish_budget,
                        total_budget_s=cascade_budget,
                        log=lambda s: None,
                    )
                    cascade_proxy = float(compute_proxy_cost(
                        cascade_state, benchmark, plc)["proxy_cost"])
                    log(f"  cascade done: {cascade_proxy:.5f} "
                        f"iters={stats.get('iters_run', '?')}")
                except Exception as exc:
                    log(f"  cascade failed: {exc}; falling back to plateau")
                    cascade_state = plateau
                    cascade_proxy = plateau_proxy

        # Phase 6: best of all
        all_candidates = [(e25_proxy, e25, "E25"),
                          (cascade_proxy, cascade_state, "cascade")]
        if e41 is not None:
            all_candidates.append((e41_proxy, e41, "E41"))
        if dp is not None:
            all_candidates.append((dp_proxy, dp, "DP"))
        all_candidates.sort(key=lambda c: c[0])
        best_proxy, best_placement, best_name = all_candidates[0]
        log(f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"({', '.join(f'{n}={p:.5f}' for p, _, n in all_candidates)}) "
            f"total wall={time.time() - t0:.0f}s")

        ovl = compute_overlap_metrics(best_placement, benchmark)["overlap_count"]
        if ovl > 0:
            raise RuntimeError(
                f"CascadeDPMultiLane winner has {ovl} hard-macro overlaps"
            )
        return best_placement
