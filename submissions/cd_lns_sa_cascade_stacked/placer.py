"""CDLNSSACascadeStackedPlacer — cascade_saddle FOLLOWED BY portfolio_saddle.

Architecture (2026-05-16 H4-stacked, after observing portfolio_levy ibm10
single-bench 0.99468 vs cached cascade 0.98940 = +0.5% LOSS, diagnosed as
"portfolio_saddle exhausts wall budget in 1 iter, doesn't saturate canonical"):

  1. E25 lane (SDF + CD + LNS + SA)
  2. E41 lane (DPO + CD + LNS + SA + K-joint)
  3. Plateau pick: best of {E25, E41}
  4. Phase 3a: cascade_saddle (canonical only, fast iters until saturation
     OR budget exhausted)
  5. Phase 3b: portfolio_saddle on TOP of cascade plateau, with 3 non-canonical
     weights (cong-focus, density-focus, non-WL). Canonical was already done
     by cascade in 3a, so we skip it here.
  6. Pick best of {E25, E41, cascade_saddle, portfolio_saddle}

Expected behavior on ibm10 (single bench, E100+E84 spike data):
  - E25 plateau ≈ 1.056
  - E41 plateau ≈ 1.019
  - cascade_saddle plateau (best canonical iter) ≈ 0.989
  - portfolio_saddle on cascade plateau ≈ 0.978 (-1.1% E100 spike)

Wall split (B=3300s):
  - E25 lane: 0.20 B = 660s
  - E41 lane: 0.22 B = 726s
  - cascade_saddle: 0.20 B = 660s
  - portfolio_saddle (3 weights, K_eps=2, polish=60s): 0.13 B = 429s
  - Slack for raw init, eval, save: 0.25 B = 825s
  Total: 1.00 B = 3300s
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer

from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import CDLNSSADPOKJointPlacer

# Cascading saddle escape from E84 (canonical eigvec, iterates until plateau).
_E84_CODE = _ROOT / "experiments" / "E84_cascading_saddle" / "code"
if str(_E84_CODE) not in sys.path:
    sys.path.insert(0, str(_E84_CODE))
from cascading_saddle import cascading_saddle_escape

# Portfolio (multi-weight) saddle from E100.
_E100_CODE = _ROOT / "experiments" / "E100_weight_portfolio_saddle" / "code"
if str(_E100_CODE) not in sys.path:
    sys.path.insert(0, str(_E100_CODE))
from portfolio_saddle import portfolio_saddle_escape


class CDLNSSACascadeStackedPlacer:
    """Cascade (canonical) saddle + portfolio (3 non-canonical weights) saddle.

    Properly composes E84 cascade with E100 portfolio: cascade saturates
    the canonical eigvec direction (multiple iters), then portfolio explores
    3 non-canonical weights (cong, density, non-WL) for additional lift.
    """

    def __init__(
        self,
        cascade_max_iters: int = 5,
        cascade_eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
        cascade_polish_budget: float = 180.0,
        portfolio_max_iters: int = 2,
        portfolio_K_eps: int = 2,
        portfolio_eps_scale: float = 1.0,
        portfolio_polish_budget: float = 60.0,
        budget_seconds: Optional[float] = 3300.0,
        rng_seed: int = 42,
        verbose: bool = True,
    ):
        self.cascade_max_iters = cascade_max_iters
        self.cascade_eps_values = cascade_eps_values
        self.cascade_polish_budget = cascade_polish_budget
        self.portfolio_max_iters = portfolio_max_iters
        self.portfolio_K_eps = portfolio_K_eps
        self.portfolio_eps_scale = portfolio_eps_scale
        self.portfolio_polish_budget = portfolio_polish_budget
        self.budget_seconds = budget_seconds
        self.rng_seed = rng_seed
        self.verbose = verbose
        # Three non-canonical weights: canonical is handled by cascade_saddle.
        self.portfolio = [
            (1.0, 0.0, 1.0),  # cong-focus
            (1.0, 1.0, 0.0),  # density-focus
            (0.0, 1.0, 1.0),  # non-WL
        ]

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSACascadeStackedPlacer ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        if self.budget_seconds is not None:
            B = self.budget_seconds
            e25_kwargs = dict(
                cd_hard_cap_s=B * 0.20, lns_budget_s=B * 0.06, sa_budget_s=B * 0.06,
            )
            e41_kwargs = dict(
                cd_hard_cap_s=B * 0.20, lns_budget_s=B * 0.05, sa_budget_s=B * 0.05,
                kjoint_budget_s=B * 0.05,
            )
            cascade_budget_target = B * 0.20  # ~660s for canonical saturation
            portfolio_budget_target = B * 0.13  # ~429s for 3 non-canonical weights
            log(f"  budget {self.budget_seconds:.0f}s; "
                f"E25={sum(e25_kwargs.values()):.0f}s "
                f"E41={sum(e41_kwargs.values()):.0f}s "
                f"cascade≈{cascade_budget_target:.0f}s "
                f"portfolio≈{portfolio_budget_target:.0f}s")
        else:
            e25_kwargs = e41_kwargs = {}
            cascade_budget_target = portfolio_budget_target = None

        log("  Phase 1: E25 (CDLNSSAPlacer)")
        e25 = None
        e25_proxy = float("inf")
        try:
            e25 = CDLNSSAPlacer(**e25_kwargs).place(benchmark)
            e25_proxy = float(compute_proxy_cost(e25, benchmark, plc)["proxy_cost"])
            log(f"  E25 done: proxy={e25_proxy:.5f} (wall={time.time() - t0:.0f}s)")
        except Exception as exc:
            log(f"  E25 FAILED ({exc}); continuing with E41 only")

        e41 = None
        e41_proxy = float("inf")
        if deadline is not None and (deadline - time.time()) < 300.0:
            log(f"  Phase 2: SKIPPED ({deadline - time.time():.0f}s left)")
        else:
            log("  Phase 2: E41 (CDLNSSADPOKJointPlacer)")
            try:
                e41 = CDLNSSADPOKJointPlacer(**e41_kwargs).place(benchmark)
                e41_proxy = float(compute_proxy_cost(e41, benchmark, plc)["proxy_cost"])
                log(f"  E41 done: proxy={e41_proxy:.5f} (wall={time.time() - t0:.0f}s)")
            except Exception as exc:
                log(f"  E41 FAILED ({exc}); will skip E41 lane")

        if e25 is None and e41 is None:
            raise RuntimeError("Both E25 and E41 placers failed; cannot proceed")

        if e25 is None:
            plateau, plateau_label, plateau_proxy = e41, "E41", e41_proxy
        elif e41 is None or e25_proxy <= e41_proxy:
            plateau, plateau_label, plateau_proxy = e25, "E25", e25_proxy
        else:
            plateau, plateau_label, plateau_proxy = e41, "E41", e41_proxy
        log(f"  plateau = {plateau_label} ({plateau_proxy:.5f})")

        # Phase 3a: cascade_saddle (canonical only, multiple iters until plateau).
        cascade_state = plateau
        cascade_proxy = plateau_proxy
        if deadline is not None:
            remaining = deadline - time.time()
            cascade_budget = min(
                cascade_budget_target if cascade_budget_target else remaining,
                max(remaining - portfolio_budget_target - 60.0, 60.0)
                if portfolio_budget_target else remaining,
            )
        else:
            cascade_budget = 600.0

        if cascade_budget < 60.0:
            log(f"  Phase 3a: cascade SKIPPED (budget {cascade_budget:.0f}s)")
        else:
            log(f"  Phase 3a: cascade_saddle canonical (budget {cascade_budget:.0f}s, "
                f"max_iters={self.cascade_max_iters})")
            try:
                cascade_state, stats = cascading_saddle_escape(
                    plateau, benchmark, plc,
                    max_iters=self.cascade_max_iters,
                    eps_values=self.cascade_eps_values,
                    polish_budget=self.cascade_polish_budget,
                    total_budget_s=cascade_budget,
                    log=lambda s: None,
                )
                cascade_proxy = float(compute_proxy_cost(
                    cascade_state, benchmark, plc)["proxy_cost"])
                log(f"  cascade done: {cascade_proxy:.5f} iters={stats.get('iters_run', '?')}")
            except Exception as exc:
                log(f"  cascade failed: {exc}; falling back to plateau")
                cascade_state = plateau
                cascade_proxy = plateau_proxy

        # Phase 3b: portfolio_saddle (3 non-canonical weights on top of cascade plateau).
        portfolio_state = cascade_state
        portfolio_proxy = cascade_proxy
        if deadline is not None:
            remaining = deadline - time.time()
            portfolio_budget = remaining - 30.0
        else:
            portfolio_budget = 600.0

        if portfolio_budget < 90.0:
            log(f"  Phase 3b: portfolio SKIPPED (budget {portfolio_budget:.0f}s)")
        else:
            log(f"  Phase 3b: portfolio_saddle 3 non-canonical weights "
                f"(budget {portfolio_budget:.0f}s, K_eps={self.portfolio_K_eps})")
            try:
                portfolio_state, stats = portfolio_saddle_escape(
                    cascade_state, benchmark, plc,
                    max_iters=self.portfolio_max_iters,
                    portfolio=self.portfolio,
                    K_eps=self.portfolio_K_eps,
                    eps_scale=self.portfolio_eps_scale,
                    polish_budget=self.portfolio_polish_budget,
                    total_budget_s=portfolio_budget,
                    rng_seed=self.rng_seed,
                    log=lambda s: None,
                )
                portfolio_proxy = float(compute_proxy_cost(
                    portfolio_state, benchmark, plc)["proxy_cost"])
                log(f"  portfolio done: {portfolio_proxy:.5f} iters={stats.get('iters_run', '?')}")
            except Exception as exc:
                log(f"  portfolio failed: {exc}; falling back to cascade plateau")
                portfolio_state = cascade_state
                portfolio_proxy = cascade_proxy

        candidates = [
            (cascade_proxy, cascade_state, "cascade"),
            (portfolio_proxy, portfolio_state, "portfolio"),
        ]
        if e25 is not None:
            candidates.append((e25_proxy, e25, "E25"))
        if e41 is not None:
            candidates.append((e41_proxy, e41, "E41"))
        candidates.sort(key=lambda c: c[0])

        chosen = None
        for proxy_i, placement_i, name_i in candidates:
            ovl_i = compute_overlap_metrics(placement_i, benchmark)["overlap_count"]
            if ovl_i == 0:
                chosen = (proxy_i, placement_i, name_i)
                break
            log(f"  WARNING: {name_i} has {ovl_i} overlaps; falling through")
        if chosen is None:
            raise RuntimeError(
                "Stacked pipeline returned no overlap-free candidate"
            )
        best_proxy, best_placement, best_name = chosen

        log(f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"({', '.join(f'{n}={p:.5f}' for p, _, n in candidates)})  "
            f"total wall={time.time() - t0:.0f}s")
        return best_placement
