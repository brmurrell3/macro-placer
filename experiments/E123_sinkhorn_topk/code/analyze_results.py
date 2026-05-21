"""E123 — analyze results and produce summary table.

Reads results/quick_smoke_*.json + results/eps_sweep_*.json +
results/ibm17_smoke.json (when available) and prints a unified summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE.parent / "results"


def main():
    # eps_sweep (calibration)
    sweeps = sorted(_RESULTS.glob("eps_sweep_*.json"))
    if sweeps:
        print("=" * 80)
        print("EPS SWEEP (canonical vs Sinkhorn smooth-proxy scalar)")
        print("=" * 80)
        for fp in sweeps:
            data = json.loads(fp.read_text())
            bench = data["bench"]
            can = data["canonical"]
            print(f"\n  bench: {bench}  canonical: {can:.5f}")
            print(f"  {'variant':>25s} {'cong':>9s} {'rel%':>8s} {'movers':>10s} {'top10g':>10s}")
            for r in data["rows"]:
                v = r["variant"]
                cong = r["cong"]
                rel = r["rel_pct"]
                if "n_movers" in r:
                    mv = f"{r['n_movers']}/{r['total_macros']}"
                else:
                    mv = "—"
                tg = r.get("top10_grad", float("nan"))
                print(f"  {v:>25s} {cong:>9.5f} {rel:>+7.2f}% {mv:>10s} {tg:>10.4e}")

    # quick smoke (Adam descent + legalize only)
    smokes = sorted(_RESULTS.glob("quick_smoke_*.json"))
    if smokes:
        print("\n" + "=" * 80)
        print("QUICK SMOKE (V3 Adam descent + legalize, NO CD polish)")
        print("=" * 80)
        for fp in smokes:
            data = json.loads(fp.read_text())
            bench = fp.stem.replace("quick_smoke_", "")
            print(f"\n  bench: {bench}")
            baseline = None
            for r in data:
                if "topk" in r["label"] or "use_sinkhorn" in r.get("config", {}) and not r["config"].get("use_sinkhorn", True):
                    baseline = r["raw_proxy"]
                    break
            print(f"  {'label':>45s} {'raw_proxy':>10s} {'Δ vs topk':>12s} {'ovl':>4s} {'wall':>6s}")
            for r in data:
                if baseline is not None:
                    delta = r["raw_proxy"] - baseline
                    pct = delta / baseline * 100
                    delta_str = f"{delta:+.5f} ({pct:+.2f}%)"
                else:
                    delta_str = "—"
                print(f"  {r['label']:>45s} {r['raw_proxy']:>10.5f} {delta_str:>12s} "
                      f"{r['ovl']:>4d} {r['wall_s']:>5.0f}s")

    # full smoke (Adam + legalize + CD600s)
    fulls = sorted(_RESULTS.glob("*_smoke.json"))
    fulls = [f for f in fulls if "quick" not in f.name and "eps_sweep" not in f.name]
    if fulls:
        print("\n" + "=" * 80)
        print("FULL SMOKE (V3 Adam + legalize + CD polish)")
        print("=" * 80)
        for fp in fulls:
            data = json.loads(fp.read_text())
            bench = fp.stem.replace("_smoke", "")
            print(f"\n  bench: {bench}")
            print(f"  {'label':>50s} {'raw':>9s} {'polish':>9s} {'wall':>7s}")
            for r in data:
                w = r.get("raw_wall_s", 0) + r.get("polish_wall_s", 0)
                print(f"  {r['label']:>50s} {r['raw_proxy']:>9.5f} {r['polish_proxy']:>9.5f} {w:>6.0f}s")


if __name__ == "__main__":
    main()
