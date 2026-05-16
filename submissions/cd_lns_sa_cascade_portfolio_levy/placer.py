"""CDLNSSACascadePortfolioLevyPlacer — Lévy saddle with FULL 4-WEIGHT portfolio.

Maximal variant of E100 portfolio: cycles through 4 weight vectors per iter,
finding each's softest Hessian eigvec and running Lévy ε-perturbation on it.

Portfolio (4 weights):
  - (1.0, 0.5, 0.5) — canonical
  - (1.0, 0.0, 1.0) — congestion-focus (cong-only Hessian)
  - (1.0, 1.0, 0.0) — density-focus
  - (0.0, 1.0, 1.0) — non-WL

E100 spike on cached cascade_ibm10.pt: all 4 weights contributed lift,
aggregate -1.089% vs pure Lévy single-vec -0.260% (**4.15× on hard bench**).

Wall budget per iter: 4× eigsh + 4× K_eps × 2 signs polishes. At K_eps=3,
that's 24 polishes/iter × ~60s = 24 min/iter — fits in 0.33×3300=1100s
saddle budget for ~1 iter. For wall-tighter budgets prefer the 2-weight
dual_levy variant.

If hard benches dominate aggregate (which they do for IBM --all), full
portfolio may beat dual_levy. Worth validating both.
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

# Portfolio (multi-weight) saddle from E100.
_E100_CODE = _ROOT / "experiments" / "E100_weight_portfolio_saddle" / "code"
if str(_E100_CODE) not in sys.path:
    sys.path.insert(0, str(_E100_CODE))
from portfolio_saddle import portfolio_saddle_escape


class CDLNSSACascadePortfolioLevyPlacer:
    """E48 plateau + dual-eigvec (canonical + cong-focus) Lévy saddle."""

    def __init__(
        self,
        max_iters: int = 5,
        K_eps: int = 3,
        eps_scale: float = 1.0,
        polish_budget: float = 180.0,
        budget_seconds: Optional[float] = 3300.0,
        rng_seed: int = 42,
        verbose: bool = True,
    ):
        self.max_iters = max_iters
        self.K_eps = K_eps
        self.eps_scale = eps_scale
        self.polish_budget = polish_budget
        self.budget_seconds = budget_seconds
        self.rng_seed = rng_seed
        self.verbose = verbose
        # Four-weight full portfolio. E100 spike on cached cascade_ibm10.pt
        # showed all 4 weights contribute lift (cumulative -1.089% vs init);
        # pure Lévy single-vec on same plateau only got -0.260%.
        # Heavier wall (4× eigsh per iter, 4× polishes) but bigger payoff on
        # hard benches. For tight budget use _dual_levy variant (2 weights).
        self.portfolio = [
            (1.0, 0.5, 0.5),  # canonical
            (1.0, 0.0, 1.0),  # cong-focus  (λ_w typically 1.5-2× softer)
            (1.0, 1.0, 0.0),  # density-focus  (often near-zero λ_w, but still hits)
            (0.0, 1.0, 1.0),  # non-WL  (cross-component soft direction)
        ]

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSACascadePortfolioLevyPlacer ({benchmark.name}) ===")
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
            log(f"  budget {self.budget_seconds:.0f}s; "
                f"E25={sum(e25_kwargs.values()):.0f}s "
                f"E41={sum(e41_kwargs.values()):.0f}s "
                f"dual saddle≈{B * 0.33:.0f}s")
        else:
            e25_kwargs = e41_kwargs = {}

        log("  Phase 1: E25 (CDLNSSAPlacer)")
        e25 = CDLNSSAPlacer(**e25_kwargs).place(benchmark)
        e25_proxy = float(compute_proxy_cost(e25, benchmark, plc)["proxy_cost"])
        log(f"  E25 done: proxy={e25_proxy:.5f} (wall={time.time() - t0:.0f}s)")

        e41 = None
        e41_proxy = float("inf")
        if deadline is not None and (deadline - time.time()) < 300.0:
            log(f"  Phase 2: SKIPPED ({deadline - time.time():.0f}s left)")
        else:
            log("  Phase 2: E41 (CDLNSSADPOKJointPlacer)")
            e41 = CDLNSSADPOKJointPlacer(**e41_kwargs).place(benchmark)
            e41_proxy = float(compute_proxy_cost(e41, benchmark, plc)["proxy_cost"])
            log(f"  E41 done: proxy={e41_proxy:.5f} (wall={time.time() - t0:.0f}s)")

        if e41 is None or e25_proxy <= e41_proxy:
            plateau, plateau_label, plateau_proxy = e25, "E25", e25_proxy
        else:
            plateau, plateau_label, plateau_proxy = e41, "E41", e41_proxy
        log(f"  plateau = {plateau_label} ({plateau_proxy:.5f})")

        cascade_state = plateau
        cascade_proxy = plateau_proxy
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining < 60.0:
                log(f"  Phase 3: SKIPPED ({remaining:.0f}s left)")
            else:
                cascade_budget = remaining - 10.0
                log(f"  Phase 3: dual-eigvec Lévy saddle "
                    f"(budget {cascade_budget:.0f}s, |portfolio|={len(self.portfolio)}, "
                    f"K={self.K_eps})")
                try:
                    cascade_state, stats = portfolio_saddle_escape(
                        plateau, benchmark, plc,
                        max_iters=self.max_iters,
                        portfolio=self.portfolio,
                        K_eps=self.K_eps,
                        eps_scale=self.eps_scale,
                        polish_budget=self.polish_budget,
                        total_budget_s=cascade_budget,
                        rng_seed=self.rng_seed,
                        log=lambda s: None,
                    )
                    cascade_proxy = float(compute_proxy_cost(
                        cascade_state, benchmark, plc)["proxy_cost"])
                    log(f"  dual saddle done: {cascade_proxy:.5f} "
                        f"iters={stats.get('iters_run', '?')}")
                except Exception as exc:
                    log(f"  dual saddle failed: {exc}; falling back to plateau")
                    cascade_state = plateau
                    cascade_proxy = plateau_proxy
        else:
            try:
                cascade_state, stats = portfolio_saddle_escape(
                    plateau, benchmark, plc,
                    max_iters=self.max_iters, portfolio=self.portfolio,
                    K_eps=self.K_eps, eps_scale=self.eps_scale,
                    polish_budget=self.polish_budget, total_budget_s=3600.0,
                    rng_seed=self.rng_seed, log=lambda s: None,
                )
                cascade_proxy = float(compute_proxy_cost(
                    cascade_state, benchmark, plc)["proxy_cost"])
                log(f"  dual saddle done: {cascade_proxy:.5f} iters={stats.get('iters_run', '?')}")
            except Exception as exc:
                log(f"  dual saddle failed: {exc}; falling back to plateau")

        candidates = [(e25_proxy, e25, "E25"), (cascade_proxy, cascade_state, "dual_levy")]
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
                "Dual-Lévy pipeline returned no overlap-free candidate"
            )
        best_proxy, best_placement, best_name = chosen

        log(f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"({', '.join(f'{n}={p:.5f}' for p, _, n in candidates)})  "
            f"total wall={time.time() - t0:.0f}s")
        return best_placement
