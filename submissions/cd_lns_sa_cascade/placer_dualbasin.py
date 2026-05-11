"""Dual-basin cascade: run cascade saddle from BOTH E25 and E41 plateaus, take best.

Hypothesis: E48 hybrid data shows 8/17 benches favor E25, 9/17 favor E41
— they land in genuinely different basins. Current cascade picks min(E25, E41)
then cascades from that one. Maybe cascading from BOTH separately and taking
the best of two outputs lifts the per-bench score on benches where E25/E41
are close but lead to different saddle-escape landings.

Trade-off: cascade budget split 50/50 between two passes → each gets fewer
iters but explores a different basin.
"""
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_E84 = _ROOT / "experiments" / "E84_cascading_saddle" / "code"
sys.path.insert(0, str(_E84))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

import importlib.util
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer_db", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer
from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import CDLNSSADPOKJointPlacer
from cascading_saddle import cascading_saddle_escape

from submissions.cd_lns_sa_cascade.placer import CDLNSSACascadePlacer


class CDLNSSACascadeDualBasin(CDLNSSACascadePlacer):
    """Cascade from BOTH E25 and E41 plateaus, return best."""

    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 3000.0)
        super().__init__(**kwargs)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSACascadeDualBasin ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        B = self.budget_seconds
        e25_kwargs = dict(cd_hard_cap_s=B*0.18, lns_budget_s=B*0.05, sa_budget_s=B*0.05)
        e41_kwargs = dict(cd_hard_cap_s=B*0.18, lns_budget_s=B*0.04, sa_budget_s=B*0.04,
                          kjoint_budget_s=B*0.04)

        # E25
        log("  Phase 1: E25")
        e25 = CDLNSSAPlacer(**e25_kwargs).place(benchmark)
        e25_proxy = float(compute_proxy_cost(e25, benchmark, plc)["proxy_cost"])
        log(f"  E25 done: {e25_proxy:.5f} wall={time.time()-t0:.0f}s")

        # E41
        e41 = None
        e41_proxy = float("inf")
        if (deadline - time.time()) >= 400:
            log("  Phase 2: E41")
            e41 = CDLNSSADPOKJointPlacer(**e41_kwargs).place(benchmark)
            e41_proxy = float(compute_proxy_cost(e41, benchmark, plc)["proxy_cost"])
            log(f"  E41 done: {e41_proxy:.5f} wall={time.time()-t0:.0f}s")

        # Cascade from E25 (half remaining budget)
        cascade_e25_state, cascade_e25_proxy = e25, e25_proxy
        cascade_e41_state, cascade_e41_proxy = e41, e41_proxy

        if (deadline - time.time()) >= 120:
            half_remaining = (deadline - time.time() - 10) / 2
            log(f"  Cascade from E25 (budget {half_remaining:.0f}s)")
            try:
                cascade_e25_state, _ = cascading_saddle_escape(
                    e25, benchmark, plc,
                    max_iters=self.max_iters, eps_values=self.eps_values,
                    polish_budget=self.polish_budget,
                    total_budget_s=half_remaining,
                    log=log if self.verbose else None,
                )
                cascade_e25_proxy = float(compute_proxy_cost(cascade_e25_state, benchmark, plc)["proxy_cost"])
                log(f"  E25-cascade: {cascade_e25_proxy:.5f}")
            except Exception as exc:
                log(f"  E25-cascade FAIL: {exc}")

        if e41 is not None and (deadline - time.time()) >= 60:
            remaining = deadline - time.time() - 10
            log(f"  Cascade from E41 (budget {remaining:.0f}s)")
            try:
                cascade_e41_state, _ = cascading_saddle_escape(
                    e41, benchmark, plc,
                    max_iters=self.max_iters, eps_values=self.eps_values,
                    polish_budget=self.polish_budget,
                    total_budget_s=remaining,
                    log=log if self.verbose else None,
                )
                cascade_e41_proxy = float(compute_proxy_cost(cascade_e41_state, benchmark, plc)["proxy_cost"])
                log(f"  E41-cascade: {cascade_e41_proxy:.5f}")
            except Exception as exc:
                log(f"  E41-cascade FAIL: {exc}")

        candidates = [
            (e25_proxy, e25, "E25"),
            (cascade_e25_proxy, cascade_e25_state, "E25-cascade"),
        ]
        if e41 is not None:
            candidates.append((e41_proxy, e41, "E41"))
            if cascade_e41_state is not None:
                candidates.append((cascade_e41_proxy, cascade_e41_state, "E41-cascade"))
        candidates.sort(key=lambda c: c[0])
        best_proxy, best_pl, best_name = candidates[0]
        log(f"  WINNER: {best_name} proxy={best_proxy:.5f} total wall={time.time()-t0:.0f}s")
        ovl = compute_overlap_metrics(best_pl, benchmark)["overlap_count"]
        if ovl > 0:
            raise RuntimeError(f"{ovl} overlaps")
        return best_pl
