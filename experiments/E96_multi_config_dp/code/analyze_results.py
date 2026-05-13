"""E96: analysis script — compare polish trajectories across runs.

Reads all available result JSONs and prints a comparison table.
References:
- B-R0' single-DP (experiments/E91_dp_full_polish/results/<bench>_dp_full_polish.json)
- E96 K=4 multi-DP polish (experiments/E96_multi_config_dp/results/multi_dp_polish_<bench>_K4.json)
- Phase 1 overnight (also B-R0', but later runs with different basin samples)

Cascade-capped reference: from autopsy table in MORNING_REPORT_E96.md.
"""
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]

CASCADE_CAPPED = {
    "ibm10": 1.0775,
    "ibm12": 1.3031,
    "ibm14": 1.2919,
    "ibm17": 1.4546,
}

B_R0_AUTOPSY = {
    "ibm01": 0.862, "ibm09": 0.789,
    "ibm10": 1.095, "ibm12": 1.129, "ibm14": 1.243, "ibm17": 1.307,
}


def load_json(path):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return None


def main():
    print(f"\n{'='*90}")
    print(f"{'BENCH':<10} {'SOURCE':<25} {'init':>9} {'CD':>9} {'LNS':>9} {'SA':>9} {'FINAL':>9} {'vs CASC':>9} {'vs B-R0':>9}")
    print(f"{'='*90}")

    for bench in ["ibm10", "ibm12", "ibm14", "ibm17"]:
        cap = CASCADE_CAPPED.get(bench, None)
        br0 = B_R0_AUTOPSY.get(bench, None)
        print(f"--- {bench} ---")

        # B-R0' from autopsy
        if br0:
            print(f"{bench:<10} {'B-R0 (autopsy table)':<25} {'-':>9} {'-':>9} {'-':>9} {'-':>9} {br0:>9.4f} "
                  f"{f'+{(br0/cap-1)*100:.1f}%' if cap else '-':>9} {'(ref)':>9}")

        # E91 B-R0' overnight (the new run)
        p = _ROOT / "experiments" / "E91_dp_full_polish" / "results" / f"{bench}_dp_full_polish.json"
        d = load_json(p)
        if d and d.get("status") == "ok":
            f = d["final_proxy"]
            vs_cap = f"{(f/cap-1)*100:+.1f}%" if cap else "-"
            vs_br0 = f"{(f/br0-1)*100:+.1f}%" if br0 else "-"
            print(f"{bench:<10} {'B-R0 overnight':<25} {d['legal_proxy']:>9.4f} "
                  f"{d['cd_proxy']:>9.4f} {d['lns_proxy']:>9.4f} {d['sa_proxy']:>9.4f} "
                  f"{f:>9.4f} {vs_cap:>9} {vs_br0:>9}")

        # E96 K=4 (latest run; may be v1 or v2 depending on order)
        p = _ROOT / "experiments" / "E96_multi_config_dp" / "results" / f"multi_dp_polish_{bench}_K4.json"
        d = load_json(p)
        if d and d.get("status") == "ok":
            f = d["final_proxy"]
            vs_cap = f"{(f/cap-1)*100:+.1f}%" if cap else "-"
            vs_br0 = f"{(f/br0-1)*100:+.1f}%" if br0 else "-"
            winner = d.get("basin_stats", {}).get("winner_label", "?")
            print(f"{bench:<10} {f'E96 K=4 ({winner})':<25} {d['legal_proxy']:>9.4f} "
                  f"{d['cd_proxy']:>9.4f} {d['lns_proxy']:>9.4f} {d['sa_proxy']:>9.4f} "
                  f"{f:>9.4f} {vs_cap:>9} {vs_br0:>9}")

        # E96 K=8 if available
        p = _ROOT / "experiments" / "E96_multi_config_dp" / "results" / f"multi_dp_polish_{bench}_K8.json"
        d = load_json(p)
        if d and d.get("status") == "ok":
            f = d["final_proxy"]
            vs_cap = f"{(f/cap-1)*100:+.1f}%" if cap else "-"
            vs_br0 = f"{(f/br0-1)*100:+.1f}%" if br0 else "-"
            winner = d.get("basin_stats", {}).get("winner_label", "?")
            print(f"{bench:<10} {f'E96 K=8 ({winner})':<25} {d['legal_proxy']:>9.4f} "
                  f"{d['cd_proxy']:>9.4f} {d['lns_proxy']:>9.4f} {d['sa_proxy']:>9.4f} "
                  f"{f:>9.4f} {vs_cap:>9} {vs_br0:>9}")

    print(f"\nCascade-capped reference: {CASCADE_CAPPED}")
    print(f"B-R0' autopsy reference: {B_R0_AUTOPSY}\n")


if __name__ == "__main__":
    main()
