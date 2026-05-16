"""Summarize all E107 periphery test results."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
RESULTS = _ROOT / "experiments" / "E107_periphery_bias" / "results"


def summarize_pt(pt_file):
    """Print a single result summary."""
    try:
        d = torch.load(pt_file, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  ERROR loading {pt_file.name}: {e}")
        return None
    return d


def main():
    print(f"{'='*80}")
    print(f"E107 Periphery Bias Results")
    print(f"{'='*80}")

    # Single-bench spikes.
    print(f"\n## Single-bench spikes\n")
    for pt in sorted(RESULTS.glob("*_spike.pt")):
        d = summarize_pt(pt)
        if d is None:
            continue
        print(f"### {d['bench']}\nbaseline proxy = {d['baseline_proxy']:.5f}")
        for r in d.get("results", []):
            alpha = r["alpha"]
            dp = r.get("delta_proxy_pct", 0)
            de = r.get("delta_edge_pct", 0)
            ovl = r["ovl"]
            print(f"  α={alpha:.3f}  proxy={r['proxy']:.5f} ({dp:+.2f}%)  edge={r['edge_dist']:.4f} ({de:+.2f}%)  ovl={ovl}")
        print()

    # Multi-seed tests.
    print(f"\n## Multi-seed tests (5 random seeds vs periphery)\n")
    print(f"{'bench':>16} {'α':>6} {'Ctrl%':>8} {'Peri%':>8} {'Rand μ%':>8} {'Rand σ%':>8} {'z':>6} {'Δedge_p%':>9}")
    print("-" * 90)
    for pt in sorted(RESULTS.glob("*_multiseed_*.pt")):
        d = summarize_pt(pt)
        if d is None:
            continue
        rdp = np.array(d["random_dps"])
        rde = np.array(d["random_des"])
        z = (rdp.mean() - d["periphery"][0]) / max(rdp.std(), 1e-6)
        print(f"{d['bench']:>16} {d['alpha']:6.3f} {d['control'][0]:+7.2f} {d['periphery'][0]:+7.2f} {rdp.mean():+7.2f} {rdp.std():7.2f} {z:+5.1f} {d['periphery'][1]:+8.2f}")

    # Sweep results.
    print(f"\n## NG45 sweep ({RESULTS / 'ng45_sweep_alpha001.pt'})\n")
    sweep_pt = RESULTS / "ng45_sweep_alpha001.pt"
    if sweep_pt.exists():
        d = torch.load(sweep_pt, map_location="cpu", weights_only=False)
        for r in d:
            ctrl = r["rows"][0]
            per = r["rows"][1]
            rnd = r["rows"][2]
            cen = r["rows"][3]
            print(f"  {r['bench']:>16}: Ctrl {ctrl[2]:+.2f}% | Peri {per[2]:+.2f}% (ovl={per[5]}) | "
                  f"Rand {rnd[2]:+.2f}% (ovl={rnd[5]}) | Cen {cen[2]:+.2f}% (ovl={cen[5]})")

    # Cascade tests.
    print(f"\n## Periphery cascade tests\n")
    for pt in sorted(RESULTS.glob("*_periphery_cascade.pt")):
        d = summarize_pt(pt)
        if d is None:
            continue
        s = d.get("stats", {})
        print(f"  {d['bench']}: init={s.get('init_proxy', '?'):.5f} final={s.get('best_proxy', '?'):.5f}  "
              f"Δ={s.get('best_proxy', 0) - s.get('init_proxy', 0):+.5f}  wall={s.get('wall_seconds', 0):.0f}s")


if __name__ == "__main__":
    main()
