"""Cascade with MORE cascade budget at expense of E25/E41 phase budgets.

Current cascade b=3000 allocation: E25 ~32%, E41 ~35%, cascade ~33%.
Hypothesis: E25/E41 over-polish plateaus that cascade then has to lift; trim
their budgets and give cascade more room (more iters, deeper polish per ε).

New allocation:
  E25: cd=600 + lns=150 + sa=150 = 900s (was 1056)  → 30%
  E41: cd=600 + lns=120 + sa=120 + kjoint=60 = 900s (was 1155)  → 30%
  Cascade gets the remaining ~1200s (was ~990s)  → 40%
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Need experiments path for cascading_saddle
_E84 = _ROOT / "experiments" / "E84_cascading_saddle" / "code"
sys.path.insert(0, str(_E84))

import time
import torch
from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# Reuse the existing components with custom budgets
from submissions.cd_lns_sa_cascade.placer import CDLNSSACascadePlacer
import importlib.util
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer_mc", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer
from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import CDLNSSADPOKJointPlacer
from cascading_saddle import cascading_saddle_escape
from macro_place.cd_core import project_overlaps


class CDLNSSACascadeMoreCascade(CDLNSSACascadePlacer):
    """Override phase budget allocation to favor cascade."""

    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 3000.0)
        super().__init__(**kwargs)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSACascadeMoreCascade ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # NEW allocation: E25=900s, E41=900s, cascade=1200s
        e25_kwargs = dict(cd_hard_cap_s=600, lns_budget_s=150, sa_budget_s=150)
        e41_kwargs = dict(cd_hard_cap_s=600, lns_budget_s=120, sa_budget_s=120,
                          kjoint_budget_s=60)
        log(f"  budget {self.budget_seconds}s: E25=900s + E41=900s + cascade≈1200s")

        log("  Phase 1: E25")
        e25 = CDLNSSAPlacer(**e25_kwargs).place(benchmark)
        e25_proxy = float(compute_proxy_cost(e25, benchmark, plc)["proxy_cost"])
        log(f"  E25 done: proxy={e25_proxy:.5f} wall={time.time() - t0:.0f}s")

        e41 = None
        e41_proxy = float("inf")
        if deadline is not None and (deadline - time.time()) < 300.0:
            log("  Phase 2: SKIPPED")
        else:
            log("  Phase 2: E41")
            e41 = CDLNSSADPOKJointPlacer(**e41_kwargs).place(benchmark)
            e41_proxy = float(compute_proxy_cost(e41, benchmark, plc)["proxy_cost"])
            log(f"  E41 done: proxy={e41_proxy:.5f} wall={time.time() - t0:.0f}s")

        if e41 is None or e25_proxy <= e41_proxy:
            plateau, label, p_proxy = e25, "E25", e25_proxy
        else:
            plateau, label, p_proxy = e41, "E41", e41_proxy
        log(f"  plateau = {label} ({p_proxy:.5f})")

        cascade_state, cascade_proxy = plateau, p_proxy
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining < 60.0:
                log(f"  cascade SKIPPED ({remaining:.0f}s left)")
            else:
                cascade_budget = remaining - 10.0
                log(f"  Phase 3: cascade (budget {cascade_budget:.0f}s, max_iters {self.max_iters})")
                try:
                    cascade_state, stats = cascading_saddle_escape(
                        plateau, benchmark, plc,
                        max_iters=self.max_iters,
                        eps_values=self.eps_values,
                        polish_budget=self.polish_budget,
                        total_budget_s=cascade_budget,
                        log=log if self.verbose else None,
                    )
                    cascade_proxy = float(compute_proxy_cost(cascade_state, benchmark, plc)["proxy_cost"])
                    log(f"  cascade done: {cascade_proxy:.5f} iters={stats['iters_run']}")
                except Exception as exc:
                    log(f"  cascade failed: {exc}")

        candidates = [(e25_proxy, e25, "E25"), (cascade_proxy, cascade_state, "cascade")]
        if e41 is not None:
            candidates.append((e41_proxy, e41, "E41"))
        candidates.sort(key=lambda c: c[0])
        best_proxy, best_pl, best_name = candidates[0]
        log(f"  WINNER: {best_name} proxy={best_proxy:.5f} total wall={time.time() - t0:.0f}s")
        ovl = compute_overlap_metrics(best_pl, benchmark)["overlap_count"]
        if ovl > 0:
            raise RuntimeError(f"{ovl} overlaps")
        return best_pl
