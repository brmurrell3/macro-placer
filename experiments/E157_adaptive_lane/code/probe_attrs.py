"""Probe structural attributes of all 17 IBM benchmarks for E157 adaptive lane."""

import os
import sys
import json

# Allow running from repo root
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from macro_place.loader import load_benchmark_from_dir

# v2-extCD vs E138 per-bench scores
V2_EXTCD = {
    "ibm01": 0.8256, "ibm02": 1.0155, "ibm03": 0.8989, "ibm04": 0.9225,
    "ibm06": 1.0564, "ibm07": 1.0077, "ibm08": 1.0109, "ibm09": 0.7708,
    "ibm10": 0.9846, "ibm11": 0.7893, "ibm12": 1.0937, "ibm13": 0.8330,
    "ibm14": 1.0747, "ibm15": 1.0592, "ibm16": 1.0116, "ibm17": 1.1772,
    "ibm18": 1.1944,
}
E138 = {
    "ibm01": 0.8351, "ibm02": 0.9937, "ibm03": 0.8938, "ibm04": 0.9250,
    "ibm06": 1.0586, "ibm07": 1.0010, "ibm08": 1.0264, "ibm09": 0.7733,
    "ibm10": 0.9744, "ibm11": 0.7924, "ibm12": 1.0891, "ibm13": 0.8389,
    "ibm14": 1.0768, "ibm15": 1.0577, "ibm16": 1.0190, "ibm17": 1.1800,
    "ibm18": 1.1893,
}

BENCH_DIR = os.path.join(REPO, "external", "MacroPlacement", "Testcases", "ICCAD04")

records = []
benches = sorted(V2_EXTCD.keys())
for b in benches:
    path = os.path.join(BENCH_DIR, b)
    try:
        bm, plc = load_benchmark_from_dir(path)
    except Exception as e:
        print(f"FAILED {b}: {e}", file=sys.stderr)
        continue
    # Compute macro density (movable hard macro area / canvas area)
    sizes = bm.macro_sizes  # [N, 2]
    fixed_mask = bm.macro_fixed  # [N]
    # All hard macros area (first num_hard_macros entries)
    hard_sizes = sizes[: bm.num_hard_macros]
    hard_fixed = fixed_mask[: bm.num_hard_macros]
    movable_hard_mask = ~hard_fixed
    n_hard = int(bm.num_hard_macros)
    n_hard_movable = int(movable_hard_mask.sum().item())
    n_hard_fixed = n_hard - n_hard_movable
    hard_area = float((hard_sizes[:, 0] * hard_sizes[:, 1]).sum().item())
    movable_hard_area = float(
        (hard_sizes[movable_hard_mask, 0] * hard_sizes[movable_hard_mask, 1]).sum().item()
    )

    n_soft = int(bm.num_soft_macros)
    soft_sizes = sizes[bm.num_hard_macros :]
    soft_area = float((soft_sizes[:, 0] * soft_sizes[:, 1]).sum().item())

    canvas_area = float(bm.canvas_width * bm.canvas_height)
    avg_macro_pins = 0.0
    # Estimate avg pins per net
    if bm.num_nets > 0:
        total_pin_count = sum(int(nn.shape[0]) for nn in bm.net_pin_nodes)
        avg_net_size = total_pin_count / bm.num_nets
    else:
        avg_net_size = 0.0

    v2 = V2_EXTCD[b]
    e138 = E138[b]
    winner = "v2" if v2 < e138 else ("e138" if e138 < v2 else "tie")
    diff = v2 - e138  # negative => v2 better (lower), positive => e138 better
    rec = {
        "bench": b,
        "num_macros": int(bm.num_macros),
        "num_hard": n_hard,
        "num_hard_movable": n_hard_movable,
        "num_hard_fixed": n_hard_fixed,
        "num_soft": n_soft,
        "num_nets": int(bm.num_nets),
        "num_ports": int(bm.port_positions.shape[0]),
        "grid_rows": int(bm.grid_rows),
        "grid_cols": int(bm.grid_cols),
        "grid_cells": int(bm.grid_rows * bm.grid_cols),
        "canvas_w": float(bm.canvas_width),
        "canvas_h": float(bm.canvas_height),
        "canvas_area": canvas_area,
        "hard_area": hard_area,
        "movable_hard_area": movable_hard_area,
        "soft_area": soft_area,
        "hard_density": hard_area / canvas_area if canvas_area else 0.0,
        "movable_hard_density": movable_hard_area / canvas_area if canvas_area else 0.0,
        "total_density": (hard_area + soft_area) / canvas_area if canvas_area else 0.0,
        "avg_net_size": avg_net_size,
        "v2": v2,
        "e138": e138,
        "winner": winner,
        "v2_minus_e138": diff,
    }
    records.append(rec)
    print(json.dumps(rec), flush=True)

# Write all records to JSON
out = os.path.join(os.path.dirname(__file__), "..", "attrs.json")
out = os.path.abspath(out)
with open(out, "w") as f:
    json.dump(records, f, indent=2)
print(f"\nWrote {out}", file=sys.stderr)
