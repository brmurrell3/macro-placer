"""Compare all variant --all results to current champion.

When experimental variants ship JSON results, run this to see which
ones beat the champion (V3Min ovl10 720s, IBM 1.00279 M3).
"""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
RESULTS = _ROOT / "results"

CHAMPION_IBM = 1.00279
CHAMPION_NG45 = 0.67861
OPTION_C_IBM = 1.0575
CARROTATO = 0.967

# Variant catalog with display names
VARIANT_PATTERNS = {
    "v1 cascade": "CDLNSSACascadeStackedPeripheryPlacer_2026*.json",
    "v2 champion (V3 ovl10 720s)": "E111MinimalOvl10_720sPlacer_2026*.json",
    "v2 EPYC verification": "EPYC_thinkorplace_v2_validation_20260520.json",
    # New variants from parallel agents
    "E114 eDensity": "E111Minimal*EDensity*.json",
    "E115 Triton": "E111Minimal*Triton*.json",
    "E116 patched DREAMPlace": "*PatchedDreamplace*.json",
    "E117 Gaussian density": "*Gaussian*.json",
    "E118 multi-stage Adam": "*Multistage*.json",
    "E119 E111 tuned": "*TunedTrace*.json",
    "E120 continuous Hessian": "*HessianSaddle*.json",
    "E121 MCF congestion": "*MCF*.json",
    "E122 3-pin Steiner": "*Steiner*.json",
    "E123 Sinkhorn top-K": "*Sinkhorn*.json",
    "Hybrid V3+cascade": "*Hybrid*.json",
}


def main():
    print("=" * 80)
    print("ALL VARIANTS — comparison vs current champion (V3 ovl10 720s)")
    print("=" * 80)
    print(f"Reference:")
    print(f"  Champion (v2):     IBM {CHAMPION_IBM:.5f}   NG45 {CHAMPION_NG45:.5f}")
    print(f"  v1 (Option C):     IBM {OPTION_C_IBM:.5f}")
    print(f"  Carrotato (#1):    IBM {CARROTATO:.5f}  ← gap to close: +{(CHAMPION_IBM-CARROTATO)/CARROTATO*100:.2f}%")
    print()

    print(f"  {'variant':<32} {'mode':>6} {'avg':>9} {'Δ vs champ':>12} {'Δ vs Carrotato':>14}")
    print(f"  {'-'*32} {'-'*6} {'-'*9} {'-'*12} {'-'*14}")
    rows = []
    for label, pattern in VARIANT_PATTERNS.items():
        for path in sorted(RESULTS.glob(pattern), key=lambda p: p.stat().st_mtime):
            try:
                d = json.loads(path.read_text())
                avg = d.get("avg_proxy_cost")
                mode = d.get("mode", "?")
                ovl = d.get("total_overlaps", "?")
                qual = d.get("qualified", "?")
                if avg is None:
                    continue
                if mode == "all":
                    delta_c = (avg - CHAMPION_IBM) / CHAMPION_IBM * 100
                    delta_carr = (avg - CARROTATO) / CARROTATO * 100
                else:
                    delta_c = 0
                    delta_carr = 0
                rows.append((avg, label, mode, ovl, qual, delta_c, delta_carr, path.name))
            except Exception as exc:
                print(f"  ERROR reading {path.name}: {exc}")
    rows.sort()
    for row in rows:
        avg, label, mode, ovl, qual, dc, dcarr, name = row
        marker = "✓" if isinstance(qual, bool) and qual else " "
        if mode == "all" and avg < CHAMPION_IBM:
            marker = "★ NEW CHAMP"
        elif mode == "all" and avg < CHAMPION_IBM * 1.005:
            marker = "≈ tie"
        print(f"  {label:<32} {mode:>6} {avg:>9.5f} {dc:>+10.2f}% {dcarr:>+12.2f}% {marker}")
    print()


if __name__ == "__main__":
    main()
