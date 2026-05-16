"""CDLNSSACascadeLevyPlacer — E48 hybrid + Lévy-flight cascading saddle escape.

Drop-in variant of `submissions/cd_lns_sa_cascade/placer.py` (E84). Same E25
plateau → E41 plateau → best-of pick → cascading saddle escape. The only
change: the saddle escape draws ε magnitudes from a heavy-tailed (half-Cauchy)
distribution instead of the fixed grid `(0.3, 1.0, 3.0)`.

Heavy-tail rationale: the fixed Gaussian grid biases toward "typical" jumps.
Cauchy magnitudes give most draws near the median (~σ) with occasional 5-10×
larger jumps (escape distant basins) and occasional ~0.1× smaller probes
(find adjacent valleys). Same eigvec direction; only magnitude distribution
differs.

Spike H2H (2026-05-13, same wall budget, same eigvec):
  ibm01  Lévy K=6 best=0.91045 (-0.35%)  vs  Gaussian K=3 best=0.91148 (-0.22%)
  ibm03  Lévy K=3 best=0.99323 (-0.317%) vs  Gaussian K=6 best=0.99555 (-0.091%)

Both seeds 42, polish_budget=60s. Lévy's K=3 beats Gaussian's K=6 on ibm03
because draws [0.82, 2.70, 4.43] hit a sweet spot the fixed [0.3, 1.0, 3.0]
grid misses.

Wall budget enforcement model: identical to the parent E84 placer
(`budget_seconds` default 3300s = 55 min, 3-min margin to 60-min cap).
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

# Lévy saddle escape (E97).
_E97_CODE = _ROOT / "experiments" / "E97_levy_saddle" / "code"
if str(_E97_CODE) not in sys.path:
    sys.path.insert(0, str(_E97_CODE))
from levy_saddle import levy_saddle_escape


class CDLNSSACascadeLevyPlacer:
    """E48 plateau + Lévy-flight cascading saddle escape, wall-budgeted."""

    def __init__(
        self,
        max_iters: int = 5,
        K_eps: int = 3,            # match Gaussian's count for fair wall
        eps_scale: float = 1.0,    # half-Cauchy median
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

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSACascadeLevyPlacer ({benchmark.name}) ===")
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
            log(f"  budget enforced: {self.budget_seconds:.0f}s; "
                f"E25 cap≈{sum(e25_kwargs.values()):.0f}s, "
                f"E41 cap≈{sum(e41_kwargs.values()):.0f}s, "
                f"cascade≈{B * 0.33:.0f}s, K_eps={self.K_eps}")
        else:
            e25_kwargs, e41_kwargs = {}, {}
            log(f"  no wall budget (offline mode), K_eps={self.K_eps}")

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
            log(f"  Phase 2: SKIPPED (only {deadline - time.time():.0f}s left)")
        else:
            log("  Phase 2: E41 (CDLNSSADPOKJointPlacer)")
            e41 = CDLNSSADPOKJointPlacer(**e41_kwargs).place(benchmark)
            e41_proxy = float(compute_proxy_cost(e41, benchmark, plc)["proxy_cost"])
            log(f"  E41 done: proxy={e41_proxy:.5f} (wall={time.time() - t0:.0f}s)")

        if e25 is None and e41 is None:
            raise RuntimeError("Both E25 and E41 placers failed; cannot proceed")
        if e25 is None:
            plateau, plateau_label, plateau_proxy = e41, "E41", e41_proxy
        elif e41 is None or e25_proxy <= e41_proxy:
            plateau, plateau_label, plateau_proxy = e25, "E25", e25_proxy
        else:
            plateau, plateau_label, plateau_proxy = e41, "E41", e41_proxy
        log(f"  E48 plateau = {plateau_label} ({plateau_proxy:.5f})")

        cascade_state = plateau
        cascade_proxy = plateau_proxy
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining < 60.0:
                log(f"  Phase 3: SKIPPED (only {remaining:.0f}s left)")
            else:
                cascade_budget = remaining - 10.0
                log(f"  Phase 3: Lévy cascading saddle (budget {cascade_budget:.0f}s, "
                    f"max_iters {self.max_iters}, K={self.K_eps}, σ={self.eps_scale})")
                try:
                    cascade_state, stats = levy_saddle_escape(
                        plateau, benchmark, plc,
                        max_iters=self.max_iters,
                        K_eps=self.K_eps,
                        eps_scale=self.eps_scale,
                        polish_budget=self.polish_budget,
                        total_budget_s=cascade_budget,
                        rng_seed=self.rng_seed,
                        log=log if self.verbose else None,
                    )
                    cascade_proxy = float(compute_proxy_cost(
                        cascade_state, benchmark, plc)["proxy_cost"])
                    log(f"  Lévy cascade done: {cascade_proxy:.5f} "
                        f"iters={stats['iters_run']}")
                except Exception as exc:
                    log(f"  Lévy cascade failed: {exc}; falling back to plateau")
                    cascade_state = plateau
                    cascade_proxy = plateau_proxy
        else:
            log(f"  Phase 3: Lévy cascading saddle (offline, max_iters {self.max_iters})")
            try:
                cascade_state, stats = levy_saddle_escape(
                    plateau, benchmark, plc,
                    max_iters=self.max_iters,
                    K_eps=self.K_eps,
                    eps_scale=self.eps_scale,
                    polish_budget=self.polish_budget,
                    total_budget_s=3600.0,
                    rng_seed=self.rng_seed,
                    log=log if self.verbose else None,
                )
                cascade_proxy = float(compute_proxy_cost(
                    cascade_state, benchmark, plc)["proxy_cost"])
                log(f"  Lévy cascade done: {cascade_proxy:.5f} iters={stats['iters_run']}")
            except Exception as exc:
                log(f"  Lévy cascade failed: {exc}; falling back to plateau")

        # Best-of fallback chain with overlap validation.
        candidates = [(e25_proxy, e25, "E25"), (cascade_proxy, cascade_state, "levy_cascade")]
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
                "Lévy cascade pipeline returned no overlap-free candidate "
                "(all of E25 / E41 / levy_cascade have hard-macro overlaps)"
            )
        best_proxy, best_placement, best_name = chosen

        log(f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"({', '.join(f'{n}={p:.5f}' for p, _, n in candidates)})  "
            f"total wall={time.time() - t0:.0f}s")
        return best_placement
