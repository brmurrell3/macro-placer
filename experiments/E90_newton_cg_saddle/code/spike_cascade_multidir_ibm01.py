"""Quick local probe: does cascade_multidir get MORE lift than a single-pass
multi_saddle_escape on cached ibm01?
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from cascade_multidir import cascade_multidir_escape
from macro_place.loader import load_benchmark_from_dir


def main():
    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    cached = torch.load(
        "experiments/E84_cascading_saddle/results/cascade_ibm01.pt",
        weights_only=False, map_location="cpu",
    )
    plateau = cached["placement"].clone().float()
    t0 = time.time()
    best_state, stats = cascade_multidir_escape(
        plateau, benchmark, plc,
        K=3, eps_values=(0.5, 2.0),
        polish_budget_s=45.0,
        max_iters=3,
        total_budget_s=2400.0,
        only_rank_at_least=2,
    )
    wall = time.time() - t0
    print(f"\n[FINAL] init={stats['init_proxy']:.5f} -> best={stats['best_proxy']:.5f}  "
          f"(Δ={stats['improvement']:+.5f} = {100 * stats['improvement_frac']:+.3f}%)  "
          f"iters={stats['iters_run']}  wall={wall:.0f}s")
    out_dir = Path("experiments/E90_newton_cg_saddle/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cascade_multidir_ibm01.json").write_text(json.dumps({
        **stats,
        "config": {"K": 3, "eps": [0.5, 2.0], "polish": 45.0, "max_iters": 3,
                    "only_rank_at_least": 2},
    }, indent=2))
    torch.save({"placement": best_state, "bench_name": "ibm01",
                "label": "E90_cascade_multidir"}, out_dir / "cascade_multidir_ibm01.pt")
    print(f"Wrote {out_dir}/cascade_multidir_ibm01.{{json,pt}}")


if __name__ == "__main__":
    main()
