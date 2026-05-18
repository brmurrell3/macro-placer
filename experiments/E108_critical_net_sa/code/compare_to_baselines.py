"""Compare a candidate placer's --all output to existing baselines.

Reads results/experiment_log.jsonl for the latest entries matching
--hypothesis prefix, plus the canonical baselines (E84 cascade uncapped,
E74 Hessian loader).

Prints a per-bench delta table and aggregate stats.

Usage:
  uv run python experiments/E108_critical_net_sa/code/compare_to_baselines.py \\
      [hypothesis_prefix=E100_portfolio_all]
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


def load_latest(prefix):
    """Find the most-recent log entry whose hypothesis starts with prefix
    AND has at least 5 benchmarks in per_benchmark (filters out single-bench)."""
    log = _ROOT / "results" / "experiment_log.jsonl"
    found = []
    for line in log.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if str(r.get("hypothesis", "")).startswith(prefix) and len(r.get("per_benchmark", {})) >= 5:
            found.append(r)
    if not found:
        return None
    return found[-1]


# Canonical baselines (from results/experiment_log.jsonl).
BASELINE_E84 = {
    "hypothesis": "E84_cascade_uncapped",
    "avg_proxy_cost": 1.061191,
    "per_benchmark": {
        "ibm01": 0.8452833890914917, "ibm02": 1.0210378170013428,
        "ibm03": 0.9463424682617188, "ibm04": 0.9906584620475769,
        "ibm06": 1.1179163455963135, "ibm07": 1.0644912719726562,
        "ibm08": 1.1006910800933838, "ibm09": 0.8226866722106934,
        "ibm10": 0.9894043207168579, "ibm11": 0.8680754899978638,
        "ibm12": 1.1976633071899414, "ibm13": 0.9345810413360596,
        "ibm14": 1.2062981128692627, "ibm15": 1.1550953388214111,
        "ibm16": 1.1084659099578857, "ibm17": 1.3316220045089722,
        "ibm18": 1.3399391174316406,
    },
}
BASELINE_E74 = {
    "hypothesis": "E74_hessian",
    "avg_proxy_cost": 1.066594,
    "per_benchmark": {
        "ibm01": 0.8552724123001099, "ibm02": 1.0390732288360596,
        "ibm03": 0.9521355032920837, "ibm04": 0.9922513961791992,
        "ibm06": 1.1312695741653442, "ibm07": 1.075741171836853,
        "ibm08": 1.106062412261963, "ibm09": 0.8273468017578125,
        "ibm10": 0.991546630859375, "ibm11": 0.8729335069656372,
        "ibm12": 1.1979918479919434, "ibm13": 0.9488269686698914,
        "ibm14": 1.2094016075134277, "ibm15": 1.14163339138031,
        "ibm16": 1.113197922706604, "ibm17": 1.3332306146621704,
        "ibm18": 1.3441908359527588,
    },
}


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "E100_portfolio_all"
    candidate = load_latest(prefix)
    if candidate is None:
        print(f"No log entry found for hypothesis prefix '{prefix}'")
        return

    name = candidate.get("hypothesis", "?")
    cand_avg = candidate.get("avg_proxy_cost", float("nan"))
    cand = candidate.get("per_benchmark", {})

    print(f"\n=== {name} ===")
    print(f"  avg_proxy = {cand_avg:.5f}")
    print(f"  qualified = {candidate.get('qualified', '?')}")
    print(f"  overlaps  = {candidate.get('total_overlaps', '?')}")
    print(f"  runtime   = {candidate.get('runtime_seconds', 0):.0f}s")
    print(f"  benches   = {len(cand)}/17")

    print(f"\n  Per-bench comparison (vs E84 cascade uncapped, E74 Hessian)")
    print(f"  {'bench':<8} {'cand':>8} {'cascade':>8} {'Δcasc%':>8} {'Hessian':>8} {'ΔHess%':>8}")
    all_benches = sorted(set(cand) | set(BASELINE_E84["per_benchmark"]))
    wins_cascade = 0
    losses_cascade = 0
    total_lift_cascade = 0.0
    n_compared = 0
    for b in all_benches:
        if b not in cand:
            print(f"  {b:<8} {'MISSING':>8}")
            continue
        c = cand[b]
        e84 = BASELINE_E84["per_benchmark"].get(b, float("nan"))
        e74 = BASELINE_E74["per_benchmark"].get(b, float("nan"))
        delta_pct_84 = (c - e84) / e84 * 100 if e84 == e84 else float("nan")
        delta_pct_74 = (c - e74) / e74 * 100 if e74 == e74 else float("nan")
        if delta_pct_84 < -0.1:
            wins_cascade += 1
        elif delta_pct_84 > 0.1:
            losses_cascade += 1
        total_lift_cascade += delta_pct_84
        n_compared += 1
        print(f"  {b:<8} {c:>8.4f} {e84:>8.4f} {delta_pct_84:+8.2f} {e74:>8.4f} {delta_pct_74:+8.2f}")

    avg_lift_cascade = total_lift_cascade / n_compared if n_compared else 0.0
    print(f"\n  Summary vs E84 cascade ({BASELINE_E84['avg_proxy_cost']:.5f}):")
    print(f"    avg lift = {(cand_avg - BASELINE_E84['avg_proxy_cost']) / BASELINE_E84['avg_proxy_cost'] * 100:+.3f}%")
    print(f"    wins (>0.1%)  = {wins_cascade}/{n_compared}")
    print(f"    losses (>0.1%) = {losses_cascade}/{n_compared}")
    print(f"    avg per-bench delta = {avg_lift_cascade:+.3f}%")
    print(f"\n  vs E74 Hessian ({BASELINE_E74['avg_proxy_cost']:.5f}):")
    print(f"    avg lift = {(cand_avg - BASELINE_E74['avg_proxy_cost']) / BASELINE_E74['avg_proxy_cost'] * 100:+.3f}%")
    print(f"\n  vs current Option A (1.07820): {(cand_avg - 1.07820) / 1.07820 * 100:+.3f}%")
    print(f"  vs current Option B (1.06650): {(cand_avg - 1.06650) / 1.06650 * 100:+.3f}%")
    print(f"  vs leaderboard top (1.011):    {(cand_avg - 1.011) / 1.011 * 100:+.3f}%")
    print(f"  vs Carrotato (0.967):          {(cand_avg - 0.967) / 0.967 * 100:+.3f}%")


if __name__ == "__main__":
    main()
