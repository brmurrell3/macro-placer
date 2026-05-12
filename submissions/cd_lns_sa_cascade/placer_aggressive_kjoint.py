"""Cascade with AGGRESSIVE K-joint (K=8 vs default K=3, top_N=15 vs 5).

Hypothesis: larger group moves (K=8 macros at once) can escape conflict
clusters that pairwise moves can't. K-joint Hungarian solves the K-tuple
optimally; larger K explores bigger combinatorial neighborhoods.

Costs more time per K-joint pass; trade off by shortening cascade phase.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import time
import torch
from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

import importlib.util
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer_ak", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer
from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import CDLNSSADPOKJointPlacer

_E84 = _ROOT / "experiments" / "E84_cascading_saddle" / "code"
sys.path.insert(0, str(_E84))
from cascading_saddle import cascading_saddle_escape
from submissions.cd_lns_sa_cascade.placer import CDLNSSACascadePlacer


class CDLNSSACascadeAggressiveKJoint(CDLNSSACascadePlacer):
    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 3000.0)
        super().__init__(**kwargs)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== AggressiveKJoint ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds
        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        B = self.budget_seconds
        # Same E25 budget. E41 gets aggressive K-joint (K=8, top_N=15, budget bumped).
        e25_kwargs = dict(cd_hard_cap_s=B*0.20, lns_budget_s=B*0.06, sa_budget_s=B*0.06)
        e41_kwargs = dict(
            cd_hard_cap_s=B*0.18, lns_budget_s=B*0.05, sa_budget_s=B*0.05,
            kjoint_budget_s=B*0.10,  # 300s vs 150s default
            kjoint_K=8, kjoint_top_N=15,
        )

        log("  Phase 1: E25")
        e25 = CDLNSSAPlacer(**e25_kwargs).place(benchmark)
        e25_proxy = float(compute_proxy_cost(e25, benchmark, plc)["proxy_cost"])
        log(f"  E25: {e25_proxy:.5f}")

        e41 = None
        e41_proxy = float("inf")
        if (deadline - time.time()) >= 400:
            log("  Phase 2: E41 (aggressive K-joint K=8)")
            e41 = CDLNSSADPOKJointPlacer(**e41_kwargs).place(benchmark)
            e41_proxy = float(compute_proxy_cost(e41, benchmark, plc)["proxy_cost"])
            log(f"  E41: {e41_proxy:.5f}")

        if e41 is None or e25_proxy <= e41_proxy:
            plateau, label, p_proxy = e25, "E25", e25_proxy
        else:
            plateau, label, p_proxy = e41, "E41", e41_proxy

        cascade_state, cascade_proxy = plateau, p_proxy
        if (deadline - time.time()) >= 60:
            cb = deadline - time.time() - 10
            log(f"  Phase 3: cascade ({cb:.0f}s)")
            try:
                cascade_state, _ = cascading_saddle_escape(
                    plateau, benchmark, plc,
                    max_iters=self.max_iters, eps_values=self.eps_values,
                    polish_budget=self.polish_budget, total_budget_s=cb,
                    log=log if self.verbose else None)
                cascade_proxy = float(compute_proxy_cost(cascade_state, benchmark, plc)["proxy_cost"])
            except Exception as exc:
                log(f"  cascade fail: {exc}")

        cands = [(e25_proxy, e25, "E25"), (cascade_proxy, cascade_state, "cascade")]
        if e41 is not None:
            cands.append((e41_proxy, e41, "E41"))
        cands.sort(key=lambda c: c[0])
        bp, bpl, bn = cands[0]
        log(f"  WINNER: {bn} {bp:.5f} wall={time.time()-t0:.0f}s")
        if compute_overlap_metrics(bpl, benchmark)["overlap_count"] > 0:
            raise RuntimeError("ovl")
        return bpl
