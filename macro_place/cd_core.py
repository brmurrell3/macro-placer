"""Coordinate-descent primitives shared across CD-based placers.

Owns the symbols that the cd_only / cd_adaptive / cd_lns_gridbin placers all
need:

  * ``sdf_init``           — wrapper around ``SDFPlacer`` for the initial pass
  * ``project_overlaps``   — iterative push-apart legalization
  * ``legal_axis_range``   — closed-form per-axis legal interval
  * ``axis_breakpoints``   — candidate-coordinate generator for line search
  * ``golden_section``     — golden-section search on the proxy
  * ``search_axis``        — combine breakpoints + GS into a per-axis search
  * ``run_cd``             — fixed-budget CD sweep loop (CDOnly)
  * ``run_cd_adaptive``    — plateau-detection CD sweep loop (CDAdaptive,
                             reused by CDLNSGridBin's CD phase)

The implementations here are the source of truth. Earlier versions lived in
``scripts/cd_ibm10_diagnostic.py`` (an experiment that became load-bearing)
and ``submissions/cd_adaptive/placer.py`` (with the LNS placer importing from
its sibling submission). The diagnostic and the submissions now both import
from this module.
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.sdf_init import SDFPlacer


# ── Initialization helpers ─────────────────────────────────────────────────


def sdf_init(benchmark: Benchmark) -> torch.Tensor:
    """Run ``SDFPlacer.place()``; returns ``[num_macros, 2]`` with hard macros
    legalized (zero overlaps among hard macros) and soft macros at default pos.
    """
    placer = SDFPlacer(seed=42)
    return placer.place(benchmark)


def project_overlaps(
    placement: torch.Tensor, benchmark: Benchmark
) -> Tuple[torch.Tensor, int]:
    """If any hard-macro overlaps remain, run iterative push-apart legalization.

    Returns ``(legalized_placement, n_iters_used)``. Bounded at 50 iterations.
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


# ── CD per-axis search helpers ─────────────────────────────────────────────


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
    """Return ``(lo, hi)`` interval for ``axis`` (0=x, 1=y) such that placing
    the macro at this axis-coordinate has zero overlap with any *other hard
    macro*, given the perpendicular axis y-coordinate is fixed at its current
    value.

    Soft macros do not contribute (no overlap constraints among soft).
    For soft macros (``idx >= n_hard``) we only enforce canvas bounds.
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

    pos_axis = placement[:n_hard, axis].cpu().numpy().astype(np.float64)
    pos_perp = placement[:n_hard, other_axis].cpu().numpy().astype(np.float64)
    sz_axis = macro_sizes[:n_hard, axis].cpu().numpy().astype(np.float64)
    sz_perp = macro_sizes[:n_hard, other_axis].cpu().numpy().astype(np.float64)

    perp_dist = np.abs(pos_perp - cur_perp)
    min_perp_sep = half_a_perp + sz_perp / 2
    blockers_perp = perp_dist < (min_perp_sep + eps)
    blockers_perp[macro_idx] = False  # don't consider self

    if not blockers_perp.any():
        return lo, hi

    half_sep = sz_axis / 2 + half_a
    safe_right_bd = pos_axis - half_sep - eps
    safe_left_bd = pos_axis + half_sep + eps

    right_mask = blockers_perp & (pos_axis - half_sep >= cur_axis - eps)
    if right_mask.any():
        new_hi = float(safe_right_bd[right_mask].min())
        hi = min(hi, new_hi)

    left_mask = blockers_perp & (pos_axis + half_sep <= cur_axis + eps)
    if left_mask.any():
        new_lo = float(safe_left_bd[left_mask].max())
        lo = max(lo, new_lo)

    if lo > hi:
        return cur_axis, cur_axis  # over-constrained; pin in place
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
    """Generate candidate axis-coordinate values for ``macro_idx`` along ``axis``.

    Probe cost dominates wallclock (each move+revert ≈ 6 ms on ibm10), so we
    keep this list SMALL (~10-15 candidates per axis). Strategic mix:
      - Weighted-median-of-net-pin-targets (global HPWL-ish optimum candidate).
      - Lo, Hi, midpoint of legal interval.
      - Closest 3-5 grid-line crossings (cell boundary jumps for D/C).
      - A few quantile splits in [lo, hi] for local exploration.

    All values are clamped to ``[lo, hi]`` and deduplicated. Returns sorted
    array of at most ``max_breakpoints`` floats.
    """
    nets_t = evaluator.macro_to_nets[macro_idx]
    macro_pins_t = evaluator.macro_to_pins[macro_idx]
    macro_pins_set = set(macro_pins_t.tolist())
    pin_offset = evaluator.pin_offset_x if axis == 0 else evaluator.pin_offset_y
    pin_axisp = evaluator.pin_x if axis == 0 else evaluator.pin_y

    cands: list = [lo, hi, 0.5 * (lo + hi), cur_axis]

    targets: list = []
    for nidx in nets_t.tolist():
        pins_n = evaluator.net_pins[nidx].tolist()
        our_pins = [p for p in pins_n if p in macro_pins_set]
        other_pins = [p for p in pins_n if p not in macro_pins_set]
        if not our_pins or not other_pins:
            continue
        other_axisp = [float(pin_axisp[p]) for p in other_pins]
        bbox_lo = min(other_axisp)
        bbox_hi = max(other_axisp)
        for p in our_pins:
            off = float(pin_offset[p])
            targets.append(bbox_lo - off)
            targets.append(bbox_hi - off)
            targets.append(0.5 * (bbox_lo + bbox_hi) - off)

    if targets:
        ta = np.asarray(targets, dtype=np.float64)
        for q in (0.25, 0.5, 0.75):
            cands.append(float(np.quantile(ta, q)))

    half = float(evaluator.macro_sizes[macro_idx, axis]) / 2
    if len(grid_lines) > 0:
        gl_cands = np.concatenate([grid_lines - half, grid_lines + half])
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
    """Golden-section search for minimum of ``f`` on ``[lo, hi]``.

    ``f`` is called ``n_iters + 2`` times (rough). Returns ``(best_x, best_val)``.
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
    """Search for the optimum axis-coordinate of ``macro_idx``.

    Returns ``(best_axis, best_cost, mode)``.  ``mode in {"bp", "gs", "skip"}``.
    The caller is responsible for committing or reverting; this function leaves
    the evaluator in the SAME STATE it was given (i.e. does revert after every
    probe).
    """
    if hi - lo < 1e-5:
        return float(evaluator.placement[macro_idx, axis]), cur_cost, "skip"

    cur_xy = [float(evaluator.placement[macro_idx, 0]),
              float(evaluator.placement[macro_idx, 1])]

    cur_axis = float(evaluator.placement[macro_idx, axis])
    candidates = axis_breakpoints(
        macro_idx, axis, evaluator, grid_lines, lo, hi,
        max_breakpoints=breakpoint_budget, cur_axis=cur_axis,
    )

    def evalc(v: float) -> float:
        # Single-candidate delta_cost (golden-section fallback path).
        new_xy = list(cur_xy)
        new_xy[axis] = float(v)
        return evaluator.delta_cost(macro_idx, tuple(new_xy))["proxy"]

    if len(candidates) > breakpoint_budget:
        best_v, best_c = golden_section(evalc, lo, hi, n_iters=25)
        return best_v, best_c, "gs"

    # Breakpoint mode: evaluate all K candidates in one batched call.
    # delta_cost_axis_batch amortizes "subtract old contributions" once
    # and runs density + congestion cost as batched torch ops across the
    # K hypothetical cell-tensors — much faster than K serial probes.
    best_v = float(evaluator.placement[macro_idx, axis])
    best_c = cur_cost
    if len(candidates) > 0:
        proxies = evaluator.delta_cost_axis_batch(
            macro_idx, axis, cur_xy, candidates,
        )
        for v, c in zip(candidates, proxies):
            if c < best_c:
                best_c = c
                best_v = float(v)
    return best_v, best_c, "bp"


# ── CD sweep loops ─────────────────────────────────────────────────────────


def _grid_lines(plc) -> Tuple[np.ndarray, np.ndarray]:
    gw = float(plc.width / plc.grid_col)
    gh = float(plc.height / plc.grid_row)
    grid_lines_x = np.arange(plc.grid_col + 1, dtype=np.float64) * gw
    grid_lines_y = np.arange(plc.grid_row + 1, dtype=np.float64) * gh
    return grid_lines_x, grid_lines_y


def _sweep_macros(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    grid_lines_x: np.ndarray,
    grid_lines_y: np.ndarray,
    order: List[int],
    cur_cost: float,
    n_hard: int,
    deadline_check: Callable[[], bool],
) -> Tuple[float, int, int, int]:
    """Run one CD sweep over ``order``. Returns
    ``(cur_cost, accepted, probes, gs_fallbacks)``.

    Stops mid-sweep if ``deadline_check()`` returns True between macros.
    """
    accepted = 0
    probes = 0
    gs_fallbacks = 0

    for macro_idx in order:
        if deadline_check():
            break

        # ── X-axis ──
        lo_x, hi_x = legal_axis_range(
            macro_idx, evaluator.placement, evaluator.macro_sizes,
            benchmark.macro_fixed, n_hard, axis=0,
            canvas_w=benchmark.canvas_width, canvas_h=benchmark.canvas_height,
        )
        cur_xy = (float(evaluator.placement[macro_idx, 0]),
                  float(evaluator.placement[macro_idx, 1]))
        best_x, best_c, mode = search_axis(
            macro_idx, 0, evaluator, grid_lines_x,
            lo_x, hi_x, cur_cost, cur_xy[1],
        )
        if mode == "gs":
            gs_fallbacks += 1
        probes += 1
        if best_c < cur_cost - 1e-9 and abs(best_x - cur_xy[0]) > 1e-7:
            evaluator.move(macro_idx, (float(best_x), cur_xy[1]))
            cur_cost = best_c
            accepted += 1

        # ── Y-axis ──
        cur_xy = (float(evaluator.placement[macro_idx, 0]),
                  float(evaluator.placement[macro_idx, 1]))
        lo_y, hi_y = legal_axis_range(
            macro_idx, evaluator.placement, evaluator.macro_sizes,
            benchmark.macro_fixed, n_hard, axis=1,
            canvas_w=benchmark.canvas_width, canvas_h=benchmark.canvas_height,
        )
        best_y, best_c, mode = search_axis(
            macro_idx, 1, evaluator, grid_lines_y,
            lo_y, hi_y, cur_cost, cur_xy[0],
        )
        if mode == "gs":
            gs_fallbacks += 1
        probes += 1
        if best_c < cur_cost - 1e-9 and abs(best_y - cur_xy[1]) > 1e-7:
            evaluator.move(macro_idx, (cur_xy[0], float(best_y)))
            cur_cost = best_c
            accepted += 1

    return cur_cost, accepted, probes, gs_fallbacks


def run_cd(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    movable: List[int],
    time_budget_s: float,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Coordinate-descent sweeps with a fixed wall-clock budget (CDOnly).

    Visits each ``movable`` macro in randomized order per sweep. For each
    macro, searches the legal range on each axis (x then y) using
    ``search_axis`` (closed-form breakpoint enumeration; falls back to golden
    section when the candidate set is too dense). Commits any improving move;
    reverts otherwise. Stops when wall clock crosses ``time_budget_s``.

    Returns dict with ``sweeps``, ``total_moves``, ``total_probes``,
    ``total_gs_fallbacks``, ``wall_total_s``.
    """
    n_hard = benchmark.num_hard_macros
    grid_lines_x, grid_lines_y = _grid_lines(plc)

    cur_cost = evaluator.current_cost()["proxy"]

    sweep_idx = 0
    total_moves = 0
    total_probes = 0
    total_gs_fallbacks = 0

    t_start = time.perf_counter()

    while True:
        if time.perf_counter() - t_start >= time_budget_s:
            break
        sweep_idx += 1
        sweep_t0 = time.perf_counter()

        rng = np.random.default_rng(seed=sweep_idx)
        order = list(movable)
        rng.shuffle(order)

        cur_cost, sweep_accepted, sweep_probes, sweep_gs = _sweep_macros(
            evaluator, benchmark, grid_lines_x, grid_lines_y,
            order, cur_cost, n_hard,
            deadline_check=lambda: time.perf_counter() - t_start >= time_budget_s,
        )

        sweep_wall = time.perf_counter() - sweep_t0
        elapsed = time.perf_counter() - t_start
        cost_break = evaluator.current_cost()
        cur_cost = cost_break["proxy"]
        total_moves += sweep_accepted
        total_probes += sweep_probes
        total_gs_fallbacks += sweep_gs
        if log_fn is not None:
            log_fn(
                f"  sweep {sweep_idx:3d}  elapsed={elapsed:7.1f}s  "
                f"sweep_t={sweep_wall:6.1f}s  proxy={cost_break['proxy']:.5f}  "
                f"accepted={sweep_accepted}/{sweep_probes}  gs={sweep_gs}  "
                f"[wl={cost_break['wl']:.4f} d={cost_break['density']:.4f} "
                f"c={cost_break['congestion']:.4f}]"
            )

    return {
        "sweeps": sweep_idx,
        "total_moves": total_moves,
        "total_probes": total_probes,
        "total_gs_fallbacks": total_gs_fallbacks,
        "wall_total_s": time.perf_counter() - t_start,
    }


def run_cd_adaptive(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    movable: List[int],
    min_time_s: float = 300.0,
    hard_cap_s: float = 3600.0,
    patience: int = 3,
    plateau_threshold: float = 0.005,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Adaptive coordinate-descent: per-benchmark plateau detection + hard cap.

    Same per-sweep body as ``run_cd``. After each sweep, records
    ``delta = previous_proxy - current_proxy`` (always non-negative since CD
    only accepts improving moves) into a deque of length ``patience``.

    Exit conditions, checked at the bottom of each sweep:
      1. ``elapsed >= hard_cap_s``                  → exit_reason = "cap"
      2. ``elapsed >= min_time_s`` AND
         ``len(deltas) == patience`` AND
         all ``d < plateau_threshold``              → exit_reason = "plateau"

    Returns dict with sweep stats plus ``exit_reason`` and ``final_deltas``.
    """
    n_hard = benchmark.num_hard_macros
    grid_lines_x, grid_lines_y = _grid_lines(plc)

    cur_cost = evaluator.current_cost()["proxy"]

    sweep_idx = 0
    total_moves = 0
    total_probes = 0
    total_gs_fallbacks = 0
    deltas: deque = deque(maxlen=patience)
    exit_reason = "cap"

    t_start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= hard_cap_s:
            exit_reason = "cap"
            break
        sweep_idx += 1
        sweep_t0 = time.perf_counter()
        prev_cost = cur_cost

        rng = np.random.default_rng(seed=sweep_idx)
        order = list(movable)
        rng.shuffle(order)

        cur_cost, sweep_accepted, sweep_probes, sweep_gs = _sweep_macros(
            evaluator, benchmark, grid_lines_x, grid_lines_y,
            order, cur_cost, n_hard,
            deadline_check=lambda: time.perf_counter() - t_start >= hard_cap_s,
        )

        sweep_wall = time.perf_counter() - sweep_t0
        elapsed = time.perf_counter() - t_start
        cost_break = evaluator.current_cost()
        cur_cost = cost_break["proxy"]
        total_moves += sweep_accepted
        total_probes += sweep_probes
        total_gs_fallbacks += sweep_gs

        delta = prev_cost - cur_cost
        deltas.append(delta)

        if log_fn is not None:
            log_fn(
                f"  sweep {sweep_idx:3d}  elapsed={elapsed:7.1f}s  "
                f"sweep_t={sweep_wall:6.1f}s  proxy={cost_break['proxy']:.5f}  "
                f"Δ={delta:+.5f}  "
                f"accepted={sweep_accepted}/{sweep_probes}  gs={sweep_gs}  "
                f"[wl={cost_break['wl']:.4f} d={cost_break['density']:.4f} "
                f"c={cost_break['congestion']:.4f}]"
            )

        if elapsed >= hard_cap_s:
            exit_reason = "cap"
            break
        if (
            elapsed >= min_time_s
            and len(deltas) == patience
            and all(d < plateau_threshold for d in deltas)
        ):
            exit_reason = "plateau"
            break

    return {
        "sweeps": sweep_idx,
        "total_moves": total_moves,
        "total_probes": total_probes,
        "total_gs_fallbacks": total_gs_fallbacks,
        "wall_total_s": time.perf_counter() - t_start,
        "exit_reason": exit_reason,
        "final_deltas": list(deltas),
    }
