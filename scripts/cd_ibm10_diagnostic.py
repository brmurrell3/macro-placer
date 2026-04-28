"""
E2 diagnostic: Full-proxy coordinate descent on ibm10 from SDF init.

Goal: How far does pure CD (no LNS, no DPO) get in 40 min wallclock from a basic
SDF init? Per E8, congestion is 74% of proxy cost so we cannot rely on closed-
form HPWL median — we must search the FULL proxy via numerical 1D line search
per coordinate.

Output:
  results/cd_ibm10_diagnostic.json    trajectory + final breakdown
  docs/cd_ibm10_results.md            comparison table

Usage:
  uv run python scripts/cd_ibm10_diagnostic.py
  uv run python scripts/cd_ibm10_diagnostic.py --time-budget 60   # quick smoke
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from submissions.cd.sdf_init import SDFPlacer


TESTCASE_ROOT = ROOT / "external/MacroPlacement/Testcases/ICCAD04"


# ── Initialization helpers ─────────────────────────────────────────────────


def sdf_init(benchmark) -> torch.Tensor:
    """Run SDFPlacer.place(); returns full [num_macros, 2] tensor with hard macros
    legalized (zero overlaps among hard macros) and soft macros at default pos.
    """
    placer = SDFPlacer(seed=42)
    return placer.place(benchmark)


def project_overlaps(placement: torch.Tensor, benchmark) -> Tuple[torch.Tensor, int]:
    """If any hard-macro overlaps remain, run iterative push-apart legalization.

    Returns (legalized_placement, n_iters_used).
    """
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.cpu().numpy()
    pos = placement.cpu().numpy().astype(np.float64).copy()

    half_w = sizes[:n_hard, 0] / 2
    half_h = sizes[:n_hard, 1] / 2

    for it in range(50):
        dx = np.abs(pos[:n_hard, 0:1] - pos[:n_hard, 0:1].T)
        dy = np.abs(pos[:n_hard, 1:2] - pos[:n_hard, 1:2].T)
        min_dx = half_w[:, None] + half_w[None, :]
        min_dy = half_h[:, None] + half_h[None, :]
        ovl = (dx < min_dx) & (dy < min_dy)
        np.fill_diagonal(ovl, False)
        pairs = np.argwhere(np.triu(ovl))
        if len(pairs) == 0:
            return torch.tensor(pos, dtype=placement.dtype), it
        for a, b in pairs:
            mov_a = not bool(fixed[a])
            mov_b = not bool(fixed[b])
            if not (mov_a or mov_b):
                continue
            vx = float(min_dx[a, b] - dx[a, b])
            vy = float(min_dy[a, b] - dy[a, b])
            if vx < vy:
                sgn = 1.0 if pos[a, 0] < pos[b, 0] else -1.0
                if mov_a and mov_b:
                    pos[a, 0] -= sgn * vx / 2
                    pos[b, 0] += sgn * vx / 2
                elif mov_a:
                    pos[a, 0] -= sgn * vx
                else:
                    pos[b, 0] += sgn * vx
            else:
                sgn = 1.0 if pos[a, 1] < pos[b, 1] else -1.0
                if mov_a and mov_b:
                    pos[a, 1] -= sgn * vy / 2
                    pos[b, 1] += sgn * vy / 2
                elif mov_a:
                    pos[a, 1] -= sgn * vy
                else:
                    pos[b, 1] += sgn * vy
        # Clamp to canvas
        pos[:n_hard, 0] = np.clip(pos[:n_hard, 0], half_w, cw - half_w)
        pos[:n_hard, 1] = np.clip(pos[:n_hard, 1], half_h, ch - half_h)
    return torch.tensor(pos, dtype=placement.dtype), 50


# ── CD helpers ─────────────────────────────────────────────────────────────


def legal_axis_range(
    macro_idx: int,
    placement: torch.Tensor,
    macro_sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    n_hard: int,
    axis: int,
    canvas_w: float,
    canvas_h: float,
    eps: float = 1e-4,
) -> Tuple[float, float]:
    """Return (lo, hi) interval for `axis` (0=x, 1=y) such that the macro at this
    axis-coordinate has zero overlap with any *other hard macro* given the
    perpendicular axis y-coordinate is fixed at its current value.

    Soft macros do not contribute (no overlap constraints among soft).

    For soft macros (idx >= n_hard) we only enforce canvas bounds.
    """
    is_hard = macro_idx < n_hard
    half_a = float(macro_sizes[macro_idx, axis]) / 2
    other_axis = 1 - axis
    half_a_perp = float(macro_sizes[macro_idx, other_axis]) / 2
    canvas_size = canvas_w if axis == 0 else canvas_h
    lo = half_a
    hi = canvas_size - half_a

    if not is_hard:
        return lo, hi  # soft: only canvas bound

    cur_perp = float(placement[macro_idx, other_axis])
    cur_axis = float(placement[macro_idx, axis])

    # Other hard macros: find those that overlap in the perpendicular axis.
    # Among these, find the closest blocker on each side along `axis`.
    pos_axis = placement[:n_hard, axis].cpu().numpy().astype(np.float64)
    pos_perp = placement[:n_hard, other_axis].cpu().numpy().astype(np.float64)
    sz_axis = macro_sizes[:n_hard, axis].cpu().numpy().astype(np.float64)
    sz_perp = macro_sizes[:n_hard, other_axis].cpu().numpy().astype(np.float64)

    # Perpendicular overlap test: does the OTHER macro's perpendicular extent
    # overlap with this macro's perpendicular extent at cur_perp? With eps
    # margin, only count as blocker if perpendicular separation is strictly
    # less than min_sep + eps (i.e. they are touching or overlapping when we
    # add the eps safety).
    perp_dist = np.abs(pos_perp - cur_perp)
    min_perp_sep = half_a_perp + sz_perp / 2
    blockers_perp = perp_dist < (min_perp_sep + eps)
    blockers_perp[macro_idx] = False  # don't consider self

    if not blockers_perp.any():
        return lo, hi

    # For each blocker i, the forbidden axis interval (i.e. positions where the
    # bbox would overlap blocker i) is centered at pos_axis[i]:
    #   forbidden iff |center_axis - pos_axis[i]| < half_a + sz_axis[i]/2
    # We want the LEGAL closed interval; with eps safety we shrink by eps.
    #   safe right boundary (max axis value approaching from left)
    #     = pos_axis[i] - half_sep - eps
    #   safe left boundary (min axis value approaching from right)
    #     = pos_axis[i] + half_sep + eps
    half_sep = sz_axis / 2 + half_a
    safe_right_bd = pos_axis - half_sep - eps  # we may go up to this from below
    safe_left_bd = pos_axis + half_sep + eps   # we may go down to this from above

    # Right blockers: those whose forbidden region starts at or past current
    # position (pos_axis[i] - half_sep >= cur_axis - eps).
    right_mask = blockers_perp & (pos_axis - half_sep >= cur_axis - eps)
    if right_mask.any():
        # The closest right blocker imposes the smallest safe_right_bd
        new_hi = float(safe_right_bd[right_mask].min())
        hi = min(hi, new_hi)

    left_mask = blockers_perp & (pos_axis + half_sep <= cur_axis + eps)
    if left_mask.any():
        new_lo = float(safe_left_bd[left_mask].max())
        lo = max(lo, new_lo)

    if lo > hi:
        # Over-constrained (e.g. blocker on both sides too close). Pin in place.
        return cur_axis, cur_axis
    return lo, hi


def axis_breakpoints(
    macro_idx: int,
    axis: int,
    evaluator: IncrementalProxyEvaluator,
    grid_lines: np.ndarray,
    lo: float,
    hi: float,
    max_breakpoints: int = 12,
    cur_axis: float = 0.0,
) -> np.ndarray:
    """Generate candidate axis-coordinate values for macro_idx along `axis`.

    Probe cost dominates wallclock (each move+revert ≈ 6 ms on ibm10), so we
    keep this list SMALL (~10-15 candidates per axis). Strategic mix:
      - Weighted-median-of-net-pin-targets (global HPWL-ish optimum candidate).
      - Lo, Hi, midpoint of legal interval.
      - Closest 3-5 grid-line crossings (cell boundary jumps for D/C).
      - A few quantile splits in [lo, hi] for local exploration.

    All values are clamped to [lo, hi] and deduplicated. Returns sorted array
    of at most ``max_breakpoints`` floats.
    """
    nets_t = evaluator.macro_to_nets[macro_idx]
    macro_pins_t = evaluator.macro_to_pins[macro_idx]
    macro_pins_set = set(macro_pins_t.tolist())
    pin_offset = evaluator.pin_offset_x if axis == 0 else evaluator.pin_offset_y
    pin_axisp = evaluator.pin_x if axis == 0 else evaluator.pin_y

    cands: list = [lo, hi, 0.5 * (lo + hi), cur_axis]

    # Weighted-median-of-other-pin-targets per axis (HPWL-favoring point).
    # Each net gives a target = mean(other_pin_axis) for "centered" placement,
    # weighted by net weight. We just take the median of all targets across
    # nets — close in spirit to the closed-form HPWL median for one macro.
    targets: list = []
    for nidx in nets_t.tolist():
        pins_n = evaluator.net_pins[nidx].tolist()
        our_pins = [p for p in pins_n if p in macro_pins_set]
        other_pins = [p for p in pins_n if p not in macro_pins_set]
        if not our_pins or not other_pins:
            continue
        # Use first pin's offset (all pins on same macro for the same axis
        # might have different offsets but typically <few). Use the bbox of
        # other pins as the WL-favoring zone.
        other_axisp = [float(pin_axisp[p]) for p in other_pins]
        bbox_lo = min(other_axisp)
        bbox_hi = max(other_axisp)
        for p in our_pins:
            off = float(pin_offset[p])
            targets.append(bbox_lo - off)
            targets.append(bbox_hi - off)
            targets.append(0.5 * (bbox_lo + bbox_hi) - off)

    if targets:
        # Compress: median + a few quantiles (small per-net probe cost matters)
        ta = np.asarray(targets, dtype=np.float64)
        for q in (0.25, 0.5, 0.75):
            cands.append(float(np.quantile(ta, q)))

    # A handful of grid-line crossings nearest cur_axis (D/C kinks)
    half = float(evaluator.macro_sizes[macro_idx, axis]) / 2
    if len(grid_lines) > 0:
        # The relevant macro-edge crossings: macro_center = grid_line ± half
        gl_cands = np.concatenate([grid_lines - half, grid_lines + half])
        # Sort by distance to cur_axis, take nearest 3
        order = np.argsort(np.abs(gl_cands - cur_axis))
        for v in gl_cands[order[:3]]:
            cands.append(float(v))

    arr = np.asarray(cands, dtype=np.float64)
    arr = np.clip(arr, lo, hi)
    arr.sort()
    if len(arr) == 0:
        return np.array([0.5 * (lo + hi)])
    unique = [arr[0]]
    for v in arr[1:]:
        if v - unique[-1] > 1e-6:
            unique.append(v)
    out = np.asarray(unique)
    if len(out) > max_breakpoints:
        idx = np.linspace(0, len(out) - 1, max_breakpoints).astype(int)
        out = out[idx]
    return out


def golden_section(
    f, lo: float, hi: float, n_iters: int = 30, tol: float = 1e-5
) -> Tuple[float, float]:
    """Golden-section search for minimum of f on [lo, hi].

    f is called n_iters+2 times (rough). Returns (best_x, best_val).
    """
    phi = (math.sqrt(5) - 1) / 2  # ~0.618
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(n_iters):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = f(d)
        if abs(b - a) < tol:
            break
    if fc < fd:
        return c, fc
    return d, fd


def search_axis(
    macro_idx: int,
    axis: int,
    evaluator: IncrementalProxyEvaluator,
    grid_lines: np.ndarray,
    lo: float,
    hi: float,
    cur_cost: float,
    perp_val: float,
    breakpoint_budget: int = 12,
) -> Tuple[float, float, str]:
    """Search for the optimum axis-coordinate of macro_idx.

    Returns (best_axis, best_cost, mode).  mode in {"bp", "gs", "skip"}.
    The caller is responsible for committing or reverting; this function
    leaves the evaluator in the SAME STATE it was given (i.e. does revert
    after every probe).
    """
    if hi - lo < 1e-5:
        return float(evaluator.placement[macro_idx, axis]), cur_cost, "skip"

    # Build current xy
    cur_xy = [float(evaluator.placement[macro_idx, 0]),
              float(evaluator.placement[macro_idx, 1])]

    cur_axis = float(evaluator.placement[macro_idx, axis])
    candidates = axis_breakpoints(
        macro_idx, axis, evaluator, grid_lines, lo, hi,
        max_breakpoints=breakpoint_budget, cur_axis=cur_axis,
    )

    def evalc(v: float) -> float:
        new_xy = list(cur_xy)
        new_xy[axis] = float(v)
        cost = evaluator.move(macro_idx, tuple(new_xy))["proxy"]
        evaluator.revert()
        return cost

    if len(candidates) > breakpoint_budget:
        # Fall back to golden section
        best_v, best_c = golden_section(evalc, lo, hi, n_iters=25)
        return best_v, best_c, "gs"

    best_v = float(evaluator.placement[macro_idx, axis])
    best_c = cur_cost
    for v in candidates:
        c = evalc(float(v))
        if c < best_c:
            best_c = c
            best_v = float(v)
    return best_v, best_c, "bp"


# ── Main loop ─────────────────────────────────────────────────────────────


def run(time_budget_s: float, log_path: Path, md_path: Path) -> dict:
    print(f"=== E2 CD diagnostic: ibm10, budget {time_budget_s:.0f}s ===")
    print("Loading benchmark ...")
    bench, plc = load_benchmark_from_dir(str(TESTCASE_ROOT / "ibm10"))
    n_hard = bench.num_hard_macros
    n_macros = bench.num_macros
    print(f"  num_macros={n_macros}, n_hard={n_hard}, n_soft={n_macros - n_hard}")
    print(f"  canvas={bench.canvas_width:.1f}x{bench.canvas_height:.1f}, "
          f"grid={bench.grid_cols}x{bench.grid_rows}, num_nets={int(plc.net_cnt)}")

    # ── Init ──
    print("Running SDF init ...")
    t_init0 = time.perf_counter()
    placement = sdf_init(bench)
    t_init = time.perf_counter() - t_init0
    print(f"  SDF init wall = {t_init:.1f} s")

    # Project overlaps if any
    placement, proj_iters = project_overlaps(placement, bench)
    print(f"  overlap projection iterations: {proj_iters}")

    init_overlaps = compute_overlap_metrics(placement, bench)
    print(f"  init overlap_count={init_overlaps['overlap_count']}, "
          f"area={init_overlaps['total_overlap_area']:.4f}")
    if init_overlaps['overlap_count'] > 0:
        print("  ! WARNING: SDF init still has overlaps after projection.")
        # Continue but flag

    # ── Build evaluator ──
    print("Building IncrementalProxyEvaluator ...")
    t_e0 = time.perf_counter()
    placement_f64 = placement.detach().clone().to(torch.float64)
    evaluator = IncrementalProxyEvaluator(bench, plc, placement_f64)
    init_cost = evaluator.current_cost()
    t_eval_init = time.perf_counter() - t_e0
    print(f"  evaluator init = {t_eval_init:.1f} s")
    print(f"  init proxy={init_cost['proxy']:.5f}, wl={init_cost['wl']:.5f}, "
          f"density={init_cost['density']:.5f}, congestion={init_cost['congestion']:.5f}")

    # ── Cross-check vs full compute_proxy_cost ──
    # (Reload plc to avoid mutation by SDFPlacer's plc usage)
    bench2, plc2 = load_benchmark_from_dir(str(TESTCASE_ROOT / "ibm10"))
    full_init = compute_proxy_cost(placement_f64, bench2, plc2)
    print(f"  full compute_proxy_cost: proxy={full_init['proxy_cost']:.5f}, "
          f"wl={full_init['wirelength_cost']:.5f}, dens={full_init['density_cost']:.5f}, "
          f"cong={full_init['congestion_cost']:.5f}")

    # Grid-line arrays (for breakpoint enumeration)
    gw = float(plc.width / plc.grid_col)
    gh = float(plc.height / plc.grid_row)
    grid_lines_x = np.arange(plc.grid_col + 1, dtype=np.float64) * gw
    grid_lines_y = np.arange(plc.grid_row + 1, dtype=np.float64) * gh

    # ── Build movable macro list (excluding fixed) ──
    movable = [i for i in range(n_macros) if not bool(bench.macro_fixed[i])]
    print(f"  movable macros: {len(movable)}")

    # ── Main CD loop ──
    trajectory = []
    sweep_idx = 0
    total_moves = 0
    total_probes = 0
    total_gs_fallbacks = 0

    # Establish start time (CD only — exclude SDF + evaluator init from budget)
    t_start_cd = time.perf_counter()

    cur_cost = init_cost["proxy"]

    # Log init point as sweep 0
    trajectory.append({
        "sweep": 0,
        "elapsed_s": 0.0,
        "proxy": init_cost["proxy"],
        "wl": init_cost["wl"],
        "density": init_cost["density"],
        "congestion": init_cost["congestion"],
        "accepted": 0,
        "probes": 0,
        "gs_fallbacks": 0,
    })

    print(f"=== Starting CD sweeps; budget = {time_budget_s:.0f}s ===")

    while True:
        elapsed = time.perf_counter() - t_start_cd
        if elapsed >= time_budget_s:
            break
        sweep_idx += 1
        sweep_t0 = time.perf_counter()
        sweep_accepted = 0
        sweep_probes = 0
        sweep_gs = 0

        # Visit movable macros in random order (different per sweep)
        rng = np.random.default_rng(seed=sweep_idx)
        order = movable.copy()
        rng.shuffle(order)

        for macro_idx in order:
            # Check time budget within sweep too
            if time.perf_counter() - t_start_cd >= time_budget_s:
                break
            # ── X-axis ──
            lo_x, hi_x = legal_axis_range(
                macro_idx, evaluator.placement, evaluator.macro_sizes,
                bench.macro_fixed, n_hard, axis=0,
                canvas_w=bench.canvas_width, canvas_h=bench.canvas_height,
            )
            cur_xy = (float(evaluator.placement[macro_idx, 0]),
                      float(evaluator.placement[macro_idx, 1]))
            best_x, best_c, mode = search_axis(
                macro_idx, 0, evaluator, grid_lines_x,
                lo_x, hi_x, cur_cost, cur_xy[1],
            )
            if mode == "gs":
                sweep_gs += 1
                total_gs_fallbacks += 1
            sweep_probes += 1  # per-axis sweep counts
            if best_c < cur_cost - 1e-9 and abs(best_x - cur_xy[0]) > 1e-7:
                # Commit
                evaluator.move(macro_idx, (float(best_x), cur_xy[1]))
                cur_cost = best_c
                sweep_accepted += 1
                total_moves += 1

            # ── Y-axis ──
            cur_xy = (float(evaluator.placement[macro_idx, 0]),
                      float(evaluator.placement[macro_idx, 1]))
            lo_y, hi_y = legal_axis_range(
                macro_idx, evaluator.placement, evaluator.macro_sizes,
                bench.macro_fixed, n_hard, axis=1,
                canvas_w=bench.canvas_width, canvas_h=bench.canvas_height,
            )
            best_y, best_c, mode = search_axis(
                macro_idx, 1, evaluator, grid_lines_y,
                lo_y, hi_y, cur_cost, cur_xy[0],
            )
            if mode == "gs":
                sweep_gs += 1
                total_gs_fallbacks += 1
            sweep_probes += 1
            if best_c < cur_cost - 1e-9 and abs(best_y - cur_xy[1]) > 1e-7:
                evaluator.move(macro_idx, (cur_xy[0], float(best_y)))
                cur_cost = best_c
                sweep_accepted += 1
                total_moves += 1

        sweep_wall = time.perf_counter() - sweep_t0
        elapsed = time.perf_counter() - t_start_cd
        # Get full breakdown via evaluator
        cost_break = evaluator.current_cost()
        cur_cost = cost_break["proxy"]
        trajectory.append({
            "sweep": sweep_idx,
            "elapsed_s": elapsed,
            "sweep_s": sweep_wall,
            "proxy": cost_break["proxy"],
            "wl": cost_break["wl"],
            "density": cost_break["density"],
            "congestion": cost_break["congestion"],
            "accepted": sweep_accepted,
            "probes": sweep_probes,
            "gs_fallbacks": sweep_gs,
        })
        total_probes += sweep_probes
        print(f"  sweep {sweep_idx:3d}  elapsed={elapsed:7.1f}s  sweep_t={sweep_wall:6.1f}s  "
              f"proxy={cost_break['proxy']:.5f}  accepted={sweep_accepted}/{sweep_probes}  "
              f"gs={sweep_gs}  "
              f"[wl={cost_break['wl']:.4f} d={cost_break['density']:.4f} c={cost_break['congestion']:.4f}]")

    # ── Validate final placement ──
    print("=== Done. Validating final placement ===")
    final_placement = evaluator.placement.detach().clone()
    final_overlaps = compute_overlap_metrics(final_placement, bench)
    print(f"  final overlap_count={final_overlaps['overlap_count']}, "
          f"area={final_overlaps['total_overlap_area']:.4f}")

    # Cross-check final via full compute_proxy_cost
    bench3, plc3 = load_benchmark_from_dir(str(TESTCASE_ROOT / "ibm10"))
    full_final = compute_proxy_cost(final_placement.to(torch.float32), bench3, plc3)
    print(f"  full check: proxy={full_final['proxy_cost']:.5f} (incr says "
          f"{cur_cost:.5f}, diff={abs(full_final['proxy_cost']-cur_cost):.6f})")

    final_cost = evaluator.current_cost()

    out = {
        "benchmark": "ibm10",
        "time_budget_s": time_budget_s,
        "wall_total_s": time.perf_counter() - t_start_cd,
        "init_proxy": init_cost["proxy"],
        "init_components": {
            "wl": init_cost["wl"],
            "density": init_cost["density"],
            "congestion": init_cost["congestion"],
        },
        "init_overlap_count": int(init_overlaps["overlap_count"]),
        "trajectory": trajectory,
        "final_proxy": final_cost["proxy"],
        "final_components": {
            "wl": final_cost["wl"],
            "density": final_cost["density"],
            "congestion": final_cost["congestion"],
        },
        "final_overlap_count": int(final_overlaps["overlap_count"]),
        "final_full_check_proxy": float(full_final["proxy_cost"]),
        "total_sweeps": sweep_idx,
        "total_moves": total_moves,
        "total_probes": total_probes,
        "total_gs_fallbacks": total_gs_fallbacks,
        "n_hard_macros": n_hard,
        "n_soft_macros": n_macros - n_hard,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(out, indent=2))
    print(f"=== JSON written: {log_path} ===")

    # ── Markdown summary ──
    write_md(md_path, out)
    print(f"=== MD written: {md_path} ===")

    return out


def write_md(path: Path, result: dict) -> None:
    init_p = result["init_proxy"]
    final_p = result["final_proxy"]
    delta_pct = 100.0 * (init_p - final_p) / init_p if init_p > 0 else 0.0
    ic = result["init_components"]
    fc = result["final_components"]
    body = f"""# E2: CD-Only Diagnostic on ibm10

Run timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
Wall budget: {result['time_budget_s']:.0f} s   (CD-only; SDF init excluded)
Total sweeps: {result['total_sweeps']}
Total accepted moves: {result['total_moves']}
Total per-axis probes: {result['total_probes']}
Golden-section fallbacks: {result['total_gs_fallbacks']}
Final overlap count: {result['final_overlap_count']}

## Comparison Table

| Method | proxy | WL | Density | Congestion |
|---|---|---|---|---|
| SDF init (E2 start) | {init_p:.4f} | {ic['wl']:.4f} | {ic['density']:.4f} | {ic['congestion']:.4f} |
| RePlAce baseline (avg, 17 IBM) | 1.4578 | – | – | – |
| DPO best_of_v2 (champion on ibm10) | 1.254 | 0.080 | 0.269 | 0.905 |
| **E2 final (CD-only, {result['time_budget_s']:.0f}s from SDF)** | **{final_p:.4f}** | **{fc['wl']:.4f}** | **{fc['density']:.4f}** | **{fc['congestion']:.4f}** |
| Leaderboard target (avg, all 17) | 1.117 | – | – | – |

Improvement vs SDF init: **{delta_pct:.1f}%**

## Decision rule triggered

"""
    if final_p <= 1.20:
        body += (
            "**E2 final ≤ 1.20:** CD-only is the answer. "
            "Recommend graduating to all 17 benchmarks and writing a CD-only placer.\n"
        )
    elif final_p <= 1.30:
        body += (
            "**E2 final in (1.20, 1.30]:** CD helps but plateaus. "
            "Recommend E3 (LNS) is the load-bearing piece.\n"
        )
    else:
        body += (
            "**E2 final > 1.30:** CD-only doesn't move the needle from SDF. "
            "Either projection is breaking things or CD's local optima trap us. "
            "Recommend DPO seed (E6) instead.\n"
        )

    body += "\n## Trajectory (first/last sweeps)\n\n"
    body += "| sweep | elapsed (s) | proxy | wl | density | congestion | accepted | probes | gs |\n"
    body += "|---|---|---|---|---|---|---|---|---|\n"
    traj = result["trajectory"]
    show = traj[: min(10, len(traj))]
    if len(traj) > 20:
        show = traj[:5] + traj[-5:]
    for r in show:
        body += (
            f"| {r['sweep']} | {r['elapsed_s']:.1f} | {r['proxy']:.4f} | "
            f"{r['wl']:.4f} | {r['density']:.4f} | {r['congestion']:.4f} | "
            f"{r.get('accepted', 0)} | {r.get('probes', 0)} | {r.get('gs_fallbacks', 0)} |\n"
        )
    path.write_text(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-budget", type=float, default=2400.0,
                        help="CD wall budget in seconds (default 2400 = 40 min)")
    parser.add_argument(
        "--out-json", default="results/cd_ibm10_diagnostic.json",
    )
    parser.add_argument(
        "--out-md", default="docs/cd_ibm10_results.md",
    )
    args = parser.parse_args()
    run(args.time_budget, ROOT / args.out_json, ROOT / args.out_md)
