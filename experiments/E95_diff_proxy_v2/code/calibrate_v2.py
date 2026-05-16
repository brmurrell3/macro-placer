"""E95 — calibration v2: does annealed γ shrink the smooth-vs-canonical gap at cascade?

E88 calibration with γ=0.0005·canvas: smooth(cascade) = 0.863 vs
canonical(cascade) = 0.845 → +2.1 % gap. This probe rebuilds DiffProxy with
γ ∈ {0.005, 1e-3, 1e-4, 1e-5}·canvas and checks the gap shrinks.

If the gap shrinks toward 0 with γ → 0, the LSE-HPWL-bias diagnosis holds
and γ-annealing during descent should let smooth gradient agree with
canonical at the optimum.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
_E88_DIR = _ROOT / "experiments" / "E88_diff_proxy" / "code"
for p in (_ROOT, _E88_DIR, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from diff_proxy_v2 import DiffProxyV2


def main():
    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    proxy = DiffProxyV2(benchmark, plc, device="cpu", gamma_frac=0.005)

    cached = torch.load(
        "experiments/E84_cascading_saddle/results/cascade_ibm01.pt",
        weights_only=False,
        map_location="cpu",
    )
    base = cached["placement"].clone().float()

    canonical = compute_proxy_cost(base, benchmark, plc)
    canon_v = float(canonical["proxy_cost"])
    print(f"canonical(cascade) = {canon_v:.5f} ovl={int(canonical['overlap_count'])}")

    gamma_fracs = [0.005, 1e-3, 5e-4, 1e-4, 1e-5]
    rows = []
    for gf in gamma_fracs:
        proxy.set_gamma_frac(gf)
        with torch.no_grad():
            smooth_cost, parts = proxy.cost(base)
        smooth_v = float(smooth_cost)
        gap_pct = (smooth_v - canon_v) / canon_v * 100.0
        rows.append({
            "gamma_frac": gf,
            "smooth": smooth_v,
            "canonical": canon_v,
            "gap_pct": gap_pct,
            "smooth_wl": float(parts["wl"]),
            "smooth_density": float(parts["density"]),
            "smooth_cong": float(parts["cong"]),
        })
        print(f"  γfrac={gf:.5f}  smooth={smooth_v:.5f}  gap={gap_pct:+.2f}%  "
              f"(wl={float(parts['wl']):.4f} dens={float(parts['density']):.4f} "
              f"cong={float(parts['cong']):.4f})")

    out = {
        "canonical": canon_v,
        "rows": rows,
        "gate": {
            "best_gap_pct": min(abs(r["gap_pct"]) for r in rows),
            "best_gamma": min(rows, key=lambda r: abs(r["gap_pct"]))["gamma_frac"],
            "pass": min(abs(r["gap_pct"]) for r in rows) < 1.0,
        },
    }
    out_path = Path("experiments/E95_diff_proxy_v2/results/calibrate_v2_ibm01.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    print(f"\nBest gap: {out['gate']['best_gap_pct']:.3f}% at γfrac={out['gate']['best_gamma']:.5f}  "
          f"→ {'pass' if out['gate']['pass'] else 'fail'}")


if __name__ == "__main__":
    main()
