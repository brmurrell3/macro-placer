"""Ship preparation script. Reads all results and picks the best placer.

Decision logic (descending priority):
  1. If best --all avg < 1.05 (Option C −0.7%) AND NG45 within 1% of 0.6893:
     Ship that placer.
  2. If best --all avg < 1.05525 (Option C −0.25%) AND zero overlaps:
     Ship if NG45 doesn't regress >2%.
  3. Otherwise: Ship Option C (safe baseline).

Usage:
  uv run python experiments/E110_smooth_global_placer/code/ship_prep.py
  # Prints recommendation + the placer path to use
"""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
RESULTS = _ROOT / "results"

OPTION_C_PATH = "submissions/cd_lns_sa_cascade_stacked_periphery/placer.py"
OPTION_C_IBM = 1.0575
OPTION_C_NG45 = 0.6893

# Catalog of placers tested, with class name → path.
PLACER_CATALOG = {
    "CDLNSSACascadeStackedPeripheryE110Placer":
        "submissions/cd_lns_sa_cascade_stacked_periphery_e110/placer.py",
    "CDLNSSACascadeStackedPeripheryE110PlateauPlacer":
        "submissions/cd_lns_sa_cascade_stacked_periphery_e110_plateau/placer.py",
    "CDLNSSACascadeStackedPeripheryE110Ovl10Placer":
        "submissions/cd_lns_sa_cascade_stacked_periphery_e110_ovl10/placer.py",
    "CDLNSSACascadeStackedPeripheryE110NoCongPlacer":
        "submissions/cd_lns_sa_cascade_stacked_periphery_e110_nocong/placer.py",
    "CDLNSSACascadeStackedPeripheryE110AdaptivePlacer":
        "submissions/cd_lns_sa_cascade_stacked_periphery_e110_adaptive/placer.py",
    "CDLNSSACascadeStackedPeripheryE110PlateauOvl10Placer":
        "submissions/cd_lns_sa_cascade_stacked_periphery_e110_plateau_ovl10/placer.py",
    "E110MinimalPlacer":
        "submissions/e110_minimal/placer.py",
    "E110Minimal600sPlacer":
        "submissions/e110_minimal_600s/placer.py",
    "E110MinimalMultiseedPlacer":
        "submissions/e110_minimal_multiseed/placer.py",
    "CDLNSSACascadeStackedPeripheryPlacer":
        OPTION_C_PATH,
}


def latest_result(class_name, mode_filter=None):
    """Find latest --all (or NG45) result JSON for a class."""
    files = sorted(
        RESULTS.glob(f"{class_name}_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if mode_filter:
        files = [f for f in files
                 if mode_filter in json.loads(f.read_text()).get("mode", "")]
    return files[-1] if files else None


def all_results():
    """Catalog all available results: {placer_class: {'all': avg, 'ng45': avg}}."""
    out = {}
    for class_name in PLACER_CATALOG:
        all_file = latest_result(class_name, mode_filter="all")
        ng45_file = latest_result(class_name, mode_filter="ng45")
        if all_file is None and ng45_file is None:
            continue
        info = {"placer_path": PLACER_CATALOG[class_name]}
        if all_file:
            d = json.loads(all_file.read_text())
            info["all_avg"] = d.get("avg_proxy_cost")
            info["all_overlaps"] = d.get("total_overlaps")
            info["all_qualified"] = d.get("qualified")
            info["all_path"] = str(all_file.name)
        if ng45_file:
            d = json.loads(ng45_file.read_text())
            info["ng45_avg"] = d.get("avg_proxy_cost")
            info["ng45_overlaps"] = d.get("total_overlaps")
            info["ng45_qualified"] = d.get("qualified")
            info["ng45_path"] = str(ng45_file.name)
        out[class_name] = info
    return out


def recommend(results):
    """Return (recommended_placer_class, reason)."""
    # Filter to qualified --all results
    candidates = [
        (cls, info) for cls, info in results.items()
        if info.get("all_avg") is not None
        and info.get("all_qualified", False)
    ]
    if not candidates:
        return "CDLNSSACascadeStackedPeripheryPlacer", (
            "No qualified --all result; fall back to Option C"
        )
    # Sort by --all avg
    candidates.sort(key=lambda c: c[1]["all_avg"])
    best_cls, best_info = candidates[0]
    best_ibm = best_info["all_avg"]

    if best_ibm < 1.05:
        # Strong lift
        ng45_ok = (
            best_info.get("ng45_avg") is None  # untested = pessimistic
            or best_info.get("ng45_avg", 999) < OPTION_C_NG45 * 1.01
        )
        if ng45_ok:
            return best_cls, (
                f"Strong IBM lift ({best_ibm:.5f} < 1.05); NG45 within 1%."
            )
        return "CDLNSSACascadeStackedPeripheryPlacer", (
            f"Best IBM {best_ibm:.5f} BUT NG45 regresses; ship Option C for safety"
        )
    elif best_ibm < 1.05525:
        # Modest lift
        ng45_ok = (
            best_info.get("ng45_avg") is None
            or best_info.get("ng45_avg", 999) < OPTION_C_NG45 * 1.02
        )
        if ng45_ok:
            return best_cls, (
                f"Modest IBM lift ({best_ibm:.5f} < 1.05525); NG45 within 2%."
            )
    # Default to Option C
    return "CDLNSSACascadeStackedPeripheryPlacer", (
        f"No clear winner (best {best_ibm:.5f}); ship Option C"
    )


def main():
    print("=" * 70)
    print("SHIP PREP")
    print("=" * 70)
    print(f"Option C reference: IBM={OPTION_C_IBM}, NG45={OPTION_C_NG45}")
    print()
    results = all_results()
    print(f"Have results for {len(results)} placers:")
    rows = []
    for cls, info in results.items():
        ibm = info.get("all_avg")
        ng45 = info.get("ng45_avg")
        qual = info.get("all_qualified")
        ovl = info.get("all_overlaps")
        rows.append((ibm or 999, cls, ibm, ng45, qual, ovl))
    rows.sort()
    print(f"{'IBM':>9}  {'NG45':>9}  {'qual':>4}  {'ovl':>3}  Class")
    for _, cls, ibm, ng45, qual, ovl in rows:
        ibm_s = f"{ibm:.5f}" if ibm is not None else "—"
        ng45_s = f"{ng45:.5f}" if ng45 is not None else "—"
        marker = "*" if ibm is not None and ibm < OPTION_C_IBM else " "
        print(f"  {ibm_s:>8}{marker} {ng45_s:>9}  {str(qual)[:4]:>4}  {str(ovl)[:3]:>3}  {cls}")

    print()
    rec_cls, reason = recommend(results)
    rec_path = PLACER_CATALOG.get(rec_cls, OPTION_C_PATH)
    print("=" * 70)
    print(f"RECOMMENDATION: {rec_cls}")
    print(f"PATH: {rec_path}")
    print(f"REASON: {reason}")
    print("=" * 70)


if __name__ == "__main__":
    main()
