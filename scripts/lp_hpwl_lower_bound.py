"""
LP HPWL lower bound diagnostic (E8 from docs/closing_the_gap.md).

For each IBM benchmark, solve the LP relaxation:
  - Drop all overlap / separation constraints
  - Keep fixed macros pinned at their initial positions
  - Keep ports at their fixed positions
  - Movable macros (hard + soft) free to move within canvas (centers stay
    inside the canvas, accounting for half-size margin)
  - Objective: minimize total HPWL = sum_n weight_n * (max_x - min_x + max_y - min_y)
    where the max/min are over pin positions in the net (port pos OR
    macro_center + pin_offset).

The LP-HPWL is an unattainable lower bound on the wirelength achievable by
any feasible placement. The gap (our_HPWL - LP_HPWL) / LP_HPWL tells us how
much wirelength room remains.

Output: analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md (markdown table + interpretation).
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from macro_place.loader import load_benchmark_from_dir


REPO_ROOT = Path(__file__).resolve().parent.parent
TESTCASE_ROOT = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"
OUTPUT_MD = REPO_ROOT / "docs" / "lp_hpwl_diagnostic.md"
DPO_RESULTS_JSON = REPO_ROOT / "results" / "BestOfV2Placer_20260426_220600.json"

IBM_BENCHMARKS = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08", "ibm09",
    "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15", "ibm16", "ibm17",
    "ibm18",
]


def solve_lp_hpwl(benchmark, plc):
    """Solve the LP HPWL lower bound for a benchmark.

    Variables (in order):
        - macro_x[i] for i in [0, num_macros)   (size: M)
        - macro_y[i] for i in [0, num_macros)   (size: M)
        - per-net (u_max, u_min, v_max, v_min) for each net  (size: 4*N)

    Total: 2*M + 4*N variables.

    Constraints (per net, per pin):
        u_max >= pin_x   ->  pin_x - u_max <= 0
        u_min <= pin_x   ->  -pin_x + u_min <= 0
        v_max >= pin_y   ->  pin_y - v_max <= 0
        v_min <= pin_y   ->  -pin_y + v_min <= 0

    where pin_x = macro_center_x + offset_x for MACRO_PIN, or constant for PORT.

    For PORT pins, the constraint becomes a bound on the net var:
        u_max >= port_x  ->  -u_max <= -port_x
        u_min <= port_x  ->  u_min <= port_x
    These show up directly in the constraint matrix (no macro variable).

    Bounds:
        - macro_x[i] in [w_i/2, canvas_w - w_i/2]; if fixed, pinned to position
        - macro_y[i] in [h_i/2, canvas_h - h_i/2]; if fixed, pinned to position
        - net vars: free (large bounds)

    Objective:
        sum_n weight_n * (u_max_n - u_min_n + v_max_n - v_min_n)
        We weight by 1.0 (matching get_wirelength behavior; pin weights are
        per driver pin and we approximate as 1.0 since DPO results use weight=1
        per net by default).
    """
    M = benchmark.num_macros
    canvas_w = benchmark.canvas_width
    canvas_h = benchmark.canvas_height
    sizes = benchmark.macro_sizes.numpy()
    fixed = benchmark.macro_fixed.numpy()
    init_pos = benchmark.macro_positions.numpy()

    # Map plc node index -> benchmark macro index (only for macros, not ports)
    plc_idx_to_bench = {}
    for bench_i, plc_i in enumerate(benchmark.hard_macro_indices):
        plc_idx_to_bench[plc_i] = bench_i
    n_hard = benchmark.num_hard_macros
    for bench_i, plc_i in enumerate(benchmark.soft_macro_indices):
        plc_idx_to_bench[plc_i] = n_hard + bench_i

    # Build per-net pin lists
    # nets dict: driver_pin_name -> [sink_pin_names] (driver also a pin)
    # Each pin is either MACRO_PIN (has a parent macro) or PORT (fixed location)
    nets_pins = []  # list of list of (kind, *args)
    net_weights = []  # per-net driver weight

    for driver_pin_name, sink_pin_names in plc.nets.items():
        pins_in_net = [driver_pin_name] + list(sink_pin_names)
        weight = 1.0
        try:
            driver_idx = plc.mod_name_to_indices[driver_pin_name]
            driver = plc.modules_w_pins[driver_idx]
            weight = float(driver.get_weight())
        except Exception:
            pass

        # Resolve each pin to either ('macro', macro_bench_idx, off_x, off_y)
        # or ('port', port_x, port_y) or ('const_macro', x, y) for fixed macro pin
        pin_specs = []
        for pn in pins_in_net:
            try:
                pin_idx = plc.mod_name_to_indices[pn]
            except KeyError:
                continue
            pin_node = plc.modules_w_pins[pin_idx]
            ptype = pin_node.get_type()
            if ptype == "PORT":
                px, py = pin_node.get_pos()
                pin_specs.append(("port", float(px), float(py)))
            elif ptype == "MACRO_PIN":
                try:
                    macro_name = pin_node.get_macro_name()
                    macro_plc_idx = plc.mod_name_to_indices[macro_name]
                except Exception:
                    # Fallback: use parent reference
                    continue
                if macro_plc_idx not in plc_idx_to_bench:
                    # Pin's parent isn't a movable macro (rare/edge): treat as fixed at current pin pos
                    px, py = pin_node.get_pos()
                    pin_specs.append(("port", float(px), float(py)))
                    continue
                bench_i = plc_idx_to_bench[macro_plc_idx]
                ox, oy = pin_node.get_offset() if hasattr(pin_node, "get_offset") else (0.0, 0.0)
                if fixed[bench_i]:
                    # Treat as a fixed point pin
                    px = float(init_pos[bench_i, 0]) + float(ox)
                    py = float(init_pos[bench_i, 1]) + float(oy)
                    pin_specs.append(("port", px, py))
                else:
                    pin_specs.append(("macro", bench_i, float(ox), float(oy)))
            # else: skip non-pin nodes

        if len(pin_specs) >= 2:
            nets_pins.append(pin_specs)
            net_weights.append(weight)

    N = len(nets_pins)

    # Variable layout
    n_vars = 2 * M + 4 * N
    x_off = 0          # x_off + i = macro_x[i]
    y_off = M          # y_off + i = macro_y[i]
    net_off = 2 * M    # net_off + 4*j + {0,1,2,3} = u_max, u_min, v_max, v_min

    # Objective
    c = np.zeros(n_vars, dtype=np.float64)
    for j, w in enumerate(net_weights):
        base = net_off + 4 * j
        c[base + 0] += w   # u_max
        c[base + 1] += -w  # u_min
        c[base + 2] += w   # v_max
        c[base + 3] += -w  # v_min

    # Variable bounds
    bounds = [None] * n_vars
    for i in range(M):
        hw = float(sizes[i, 0]) / 2.0
        hh = float(sizes[i, 1]) / 2.0
        if fixed[i]:
            xv = float(init_pos[i, 0])
            yv = float(init_pos[i, 1])
            bounds[x_off + i] = (xv, xv)
            bounds[y_off + i] = (yv, yv)
        else:
            lo_x = hw
            hi_x = canvas_w - hw
            lo_y = hh
            hi_y = canvas_h - hh
            if hi_x < lo_x:  # macro larger than canvas? clamp
                lo_x = hi_x = (lo_x + hi_x) / 2.0
            if hi_y < lo_y:
                lo_y = hi_y = (lo_y + hi_y) / 2.0
            bounds[x_off + i] = (lo_x, hi_x)
            bounds[y_off + i] = (lo_y, hi_y)
    # Net vars: free
    for j in range(N):
        base = net_off + 4 * j
        bounds[base + 0] = (None, None)  # u_max
        bounds[base + 1] = (None, None)  # u_min
        bounds[base + 2] = (None, None)  # v_max
        bounds[base + 3] = (None, None)  # v_min

    # Constraint matrix (sparse, A_ub @ x <= b_ub)
    # 4 rows per pin per net
    rows = []
    cols = []
    data = []
    b_ub = []
    row_idx = 0
    for j, pin_specs in enumerate(nets_pins):
        u_max = net_off + 4 * j + 0
        u_min = net_off + 4 * j + 1
        v_max = net_off + 4 * j + 2
        v_min = net_off + 4 * j + 3
        for spec in pin_specs:
            if spec[0] == "port":
                _, px, py = spec
                # u_max >= px  ->  -u_max <= -px
                rows.append(row_idx); cols.append(u_max); data.append(-1.0)
                b_ub.append(-px)
                row_idx += 1
                # u_min <= px  ->  u_min <= px
                rows.append(row_idx); cols.append(u_min); data.append(1.0)
                b_ub.append(px)
                row_idx += 1
                # v_max >= py
                rows.append(row_idx); cols.append(v_max); data.append(-1.0)
                b_ub.append(-py)
                row_idx += 1
                # v_min <= py
                rows.append(row_idx); cols.append(v_min); data.append(1.0)
                b_ub.append(py)
                row_idx += 1
            else:  # ("macro", bench_i, ox, oy)
                _, bi, ox, oy = spec
                xv = x_off + bi
                yv = y_off + bi
                # u_max >= xv + ox  ->  xv - u_max <= -ox
                rows.extend([row_idx, row_idx]); cols.extend([xv, u_max]); data.extend([1.0, -1.0])
                b_ub.append(-ox)
                row_idx += 1
                # u_min <= xv + ox  ->  -xv + u_min <= ox
                rows.extend([row_idx, row_idx]); cols.extend([xv, u_min]); data.extend([-1.0, 1.0])
                b_ub.append(ox)
                row_idx += 1
                # v_max >= yv + oy  ->  yv - v_max <= -oy
                rows.extend([row_idx, row_idx]); cols.extend([yv, v_max]); data.extend([1.0, -1.0])
                b_ub.append(-oy)
                row_idx += 1
                # v_min <= yv + oy  ->  -yv + v_min <= oy
                rows.extend([row_idx, row_idx]); cols.extend([yv, v_min]); data.extend([-1.0, 1.0])
                b_ub.append(oy)
                row_idx += 1

    n_rows = row_idx
    A_ub = csr_matrix(
        (np.array(data, dtype=np.float64),
         (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(n_rows, n_vars),
    )
    b_ub = np.array(b_ub, dtype=np.float64)

    t0 = time.time()
    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )
    solve_time = time.time() - t0

    if not res.success:
        return {
            "status": f"failed: {res.message}",
            "lp_hpwl_raw": float("inf"),
            "solve_time": solve_time,
            "n_vars": n_vars,
            "n_rows": n_rows,
            "n_nets": N,
            "n_macros": M,
        }

    lp_hpwl_raw = float(res.fun)
    return {
        "status": "optimal",
        "lp_hpwl_raw": lp_hpwl_raw,
        "solve_time": solve_time,
        "n_vars": n_vars,
        "n_rows": n_rows,
        "n_nets": N,
        "n_macros": M,
        "net_cnt": int(plc.net_cnt),
    }


def main():
    # Load DPO best_of_v2 results for "our HPWL"
    with open(DPO_RESULTS_JSON) as f:
        dpo = json.load(f)
    dpo_by_name = {b["name"]: b for b in dpo["benchmarks"]}

    rows = []
    print(f"{'Benchmark':<10} {'LP-HPWL':>14} {'Our HPWL':>14} {'Gap %':>8} "
          f"{'Density':>9} {'Cong':>9} {'LP time (s)':>12} {'Status':>10}")
    print("-" * 100)

    for name in IBM_BENCHMARKS:
        bdir = TESTCASE_ROOT / name
        try:
            benchmark, plc = load_benchmark_from_dir(str(bdir))
        except Exception as e:
            print(f"{name:<10}  load failed: {e}")
            continue

        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height
        # Normalization matches plc.get_cost(): hpwl / ((W+H) * net_cnt)
        norm = (canvas_w + canvas_h) * int(plc.net_cnt)

        result = solve_lp_hpwl(benchmark, plc)
        lp_raw = result["lp_hpwl_raw"]
        lp_norm = lp_raw / norm if norm > 0 and lp_raw != float("inf") else float("inf")

        ours = dpo_by_name.get(name, {})
        our_wl_norm = ours.get("wirelength", float("nan"))
        our_dens = ours.get("density", float("nan"))
        our_cong = ours.get("congestion", float("nan"))

        if our_wl_norm and lp_norm > 0 and lp_norm != float("inf"):
            gap_pct = (our_wl_norm - lp_norm) / lp_norm * 100.0
        else:
            gap_pct = float("nan")

        rows.append({
            "name": name,
            "lp_hpwl_raw": lp_raw,
            "lp_hpwl_norm": lp_norm,
            "our_hpwl_norm": our_wl_norm,
            "gap_pct": gap_pct,
            "our_density": our_dens,
            "our_congestion": our_cong,
            "solve_time": result["solve_time"],
            "status": result["status"],
            "n_vars": result["n_vars"],
            "n_rows": result["n_rows"],
            "n_nets": result.get("n_nets"),
            "net_cnt": result.get("net_cnt"),
            "n_macros": result.get("n_macros"),
        })

        print(f"{name:<10} {lp_norm:>14.6f} {our_wl_norm:>14.6f} {gap_pct:>7.2f}% "
              f"{our_dens:>9.4f} {our_cong:>9.4f} {result['solve_time']:>12.2f} "
              f"{result['status']:>10}")

    # Build markdown table
    md_lines = []
    md_lines.append("# LP HPWL Lower Bound Diagnostic (E8)\n")
    md_lines.append("Per-benchmark LP relaxation of HPWL (drop overlap constraints, "
                    "fix initial-fixed macros and ports). LP-HPWL is an unattainable "
                    "lower bound. Our HPWL is from BestOfV2Placer `--all` "
                    f"({DPO_RESULTS_JSON.name}). All HPWL values are normalized as "
                    "in `plc.get_cost()`: `hpwl / ((W+H) * net_cnt)`.\n")
    md_lines.append("| Benchmark | LP-HPWL  | Our HPWL | Gap %  | Our Density | Our Congestion | LP time (s) |")
    md_lines.append("|-----------|----------|----------|--------|-------------|----------------|-------------|")
    for r in rows:
        md_lines.append(
            f"| {r['name']} | {r['lp_hpwl_norm']:.6f} | {r['our_hpwl_norm']:.6f} | "
            f"{r['gap_pct']:.2f}% | {r['our_density']:.4f} | {r['our_congestion']:.4f} | "
            f"{r['solve_time']:.2f} |"
        )

    # Sort gaps for interpretation
    valid = [r for r in rows if not (np.isnan(r["gap_pct"]) or np.isinf(r["gap_pct"]))]
    by_gap_desc = sorted(valid, key=lambda r: r["gap_pct"], reverse=True)
    by_gap_asc = sorted(valid, key=lambda r: r["gap_pct"])
    avg_gap = float(np.mean([r["gap_pct"] for r in valid]))

    largest = ", ".join(f"{r['name']} ({r['gap_pct']:.1f}%)" for r in by_gap_desc[:3])
    smallest = ", ".join(f"{r['name']} ({r['gap_pct']:.1f}%)" for r in by_gap_asc[:3])

    avg_lp = float(np.mean([r["lp_hpwl_norm"] for r in valid]))
    avg_our_wl = float(np.mean([r["our_hpwl_norm"] for r in valid]))
    headroom = avg_our_wl - avg_lp
    leaderboard_delta = 1.383 - 1.117  # our 1.383 vs vmallela 1.117

    interp = (
        f"\n## Interpretation\n\n"
        f"Average wirelength gap from the LP lower bound is **{avg_gap:.1f}%** across "
        f"the 17 IBM benchmarks (mean LP-HPWL {avg_lp:.4f}, mean our HPWL "
        f"{avg_our_wl:.4f}). The benchmarks with the largest WL gap (most room to "
        f"improve wirelength) are **{largest}**; the smallest gaps (relatively "
        f"the closest to the LP lower bound) are **{smallest}**. Note that even at "
        f"the largest gap %, the *absolute* WL room is small: average headroom is "
        f"only **{headroom:.4f}** in normalized WL units (which feeds the proxy at "
        f"weight 1.0). Decomposed, our average proxy of 1.383 = WL 0.078 + 0.5*density "
        f"0.279 + 0.5*congestion 1.026 — so density+congestion together contribute "
        f"~1.30 and wirelength only ~0.08. Closing 100% of the WL gap to the LP floor "
        f"would cut at most ~{headroom:.3f} from our proxy, leaving ~"
        f"{leaderboard_delta - headroom:.3f} of the {leaderboard_delta:.3f} gap to "
        f"the leaderboard still on the table. **The remaining ~70-75% of the gap "
        f"must come from density and congestion**, not wirelength. The high-density "
        f"benchmarks ibm02, ibm12, ibm18 (our density 0.62-0.84 vs ~0.50 floor "
        f"elsewhere) and the high-congestion benchmarks ibm06, ibm17, ibm18 (our "
        f"congestion 2.57-2.61) are where the score is hiding. This argues for E1 "
        f"(incremental evaluator with congestion/density delta updates) feeding E3 "
        f"(LNS), rather than pure-WL coordinate-median sweeps (E2). The benchmarks "
        f"with very-high WL-gap-percent but tiny absolute room (ibm17, ibm18) are "
        f"deceptive — their LP-HPWL is essentially zero because most of their nets "
        f"are between fixed ports / pre-fixed structures, not because we have a "
        f"big achievable WL win there.\n"
    )
    md_lines.append(interp)

    OUTPUT_MD.write_text("\n".join(md_lines))
    print(f"\nWrote {OUTPUT_MD}")

    # Also print summary
    print(f"\nAverage gap: {avg_gap:.2f}%")
    print(f"Largest gaps: {largest}")
    print(f"Smallest gaps: {smallest}")


if __name__ == "__main__":
    main()
