"""E32 — SAM-CD: CD with worst-case-perturbation breakpoint scoring + LNS + SA-v2.

Hypothesis: CD's fixed point may be a sharp local minimum — an artifact
of the proxy's exact form. SAM-style (Foret et al., ICLR 2021,
"Sharpness-Aware Minimization") worst-case perturbation scoring
evaluates each candidate axis position under the worst of K random
nearby placements within radius ρ. The accepted move is the *flattest*
minimum, which should generalize better to NG45's slightly different
proxy regime.

Pipeline (per benchmark, total ≤ 3600 s legal cap):

  1. SDF init.
  2. Project overlaps.
  3. Build IncrementalProxyEvaluator.
  4. SAM-CD phase (≤ 2400 s, plateau detection at threshold 0.001).
     `sam_search_axis` replaces `search_axis`: each candidate v is
     scored as max(base, perturbations) where K=4 perturbations are
     sampled uniformly on the disk of radius ρ = 0.01 × canvas_diag.
     Accept the v with lowest max-cost (flattest minimum).
  5. LNS phase (≤ 600 s, grid-bin destroy/reinsert — unchanged from E25).
  6. SA-v2 phase (≤ 600 s, T₀=5e-4 — unchanged from E25).
  7. Validate (zero overlaps), preserve fixed macros, return.

Hyperparams `sam_K` and `sam_rho_frac` are constructor kwargs for
sweepability. Default sam_K=4, sam_rho_frac=0.01.

References:
- E25 — CD + LNS + SA-v2 reference pipeline (champion candidate 1.0954).
- Foret et al. ICLR 2021, "Sharpness-Aware Minimization for Efficiently
  Improving Generalization" — the spirit but not the gradient form.
"""
from __future__ import annotations

import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

# The eval harness loads placers via importlib.spec_from_file_location, which
# does NOT add the repo root to sys.path. We need it on sys.path so the
# `macro_place.*` imports resolve.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import (
    _grid_lines,
    axis_breakpoints,
    legal_axis_range,
    project_overlaps,
    sdf_init,
)
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics

# Reuse E25's LNS and SA-v2 primitives by importing them from the promoted
# (candidate) submission module — they're inlined and self-contained there.
_E25_DIR = _ROOT / "submissions" / "cd_lns_sa"
if str(_E25_DIR) not in sys.path:
    sys.path.insert(0, str(_E25_DIR))
from placer import run_lns_gridbin, run_sa_polish_v2  # type: ignore  # noqa: E402


# ── SAM-style per-axis search ──────────────────────────────────────────────


def sam_search_axis(
    macro_idx: int,
    axis: int,
    evaluator: IncrementalProxyEvaluator,
    grid_lines: np.ndarray,
    lo: float,
    hi: float,
    cur_cost: float,
    perp_val: float,
    benchmark: Benchmark,
    n_hard: int,
    rng: np.random.Generator,
    breakpoint_budget: int = 12,
    sam_K: int = 4,
    sam_rho: float = 0.0,
) -> Tuple[float, float, str]:
    """SAM-style axis search: score each candidate as worst over K perturbations.

    For each candidate axis-coordinate v from ``axis_breakpoints(...)``:
      1. Compute base proxy at (v, perp_val) via move + revert.
      2. Sample K perturbations δ ~ uniform on the 2-disk of radius
         ``sam_rho`` around (v, perp_val).
      3. For each δ, clamp the perturbed (x, y) to the macro's legal
         x- and y-range (computed by ``legal_axis_range``), then compute
         proxy via move + revert.
      4. Effective cost = max over (base, K perturbations).

    Returns (best_axis_value, best_max_cost, mode) where mode = "sam".
    The evaluator is left in the SAME STATE it was given (every probe is
    reverted).

    If sam_K <= 0 or sam_rho <= 0, this degenerates to the same logic
    as ``cd_core.search_axis`` (breakpoint enumeration only).

    Note: legal-range clamp uses the macro's *current* perp position to
    compute the perp-axis legal range. For perturbations along the
    accepted axis, the perp range is recomputed using the perturbed perp
    value would be expensive (O(n_hard) per probe); we use the unperturbed
    perp range as a stable proxy. The disk radius is small (1 % of canvas
    diag by default), so over/underestimation of the legal range is
    negligible compared to canvas dimensions.
    """
    if hi - lo < 1e-5:
        return float(evaluator.placement[macro_idx, axis]), cur_cost, "skip"

    cur_xy = [
        float(evaluator.placement[macro_idx, 0]),
        float(evaluator.placement[macro_idx, 1]),
    ]
    cur_axis = float(evaluator.placement[macro_idx, axis])

    candidates = axis_breakpoints(
        macro_idx, axis, evaluator, grid_lines, lo, hi,
        max_breakpoints=breakpoint_budget, cur_axis=cur_axis,
    )
    if len(candidates) == 0:
        return cur_axis, cur_cost, "skip"

    # Pre-compute the perp-axis legal range once, using the macro's
    # current perp position. We use this to clamp perturbed perp values.
    perp_axis = 1 - axis
    perp_lo, perp_hi = legal_axis_range(
        macro_idx, evaluator.placement, evaluator.macro_sizes,
        benchmark.macro_fixed, n_hard, axis=perp_axis,
        canvas_w=benchmark.canvas_width,
        canvas_h=benchmark.canvas_height,
    )

    def evalc(v: float) -> float:
        new_xy = list(cur_xy)
        new_xy[axis] = float(v)
        cost = evaluator.move(macro_idx, tuple(new_xy))["proxy"]
        evaluator.revert()
        return cost

    def eval_perturbed(v: float, delta_axis: float, delta_perp: float) -> float:
        """Eval at (v + δ_axis, perp_val + δ_perp), clamped to legal ranges."""
        v_pert = float(np.clip(v + delta_axis, lo, hi))
        perp_pert = float(np.clip(perp_val + delta_perp, perp_lo, perp_hi))
        new_xy = list(cur_xy)
        new_xy[axis] = v_pert
        new_xy[perp_axis] = perp_pert
        cost = evaluator.move(macro_idx, tuple(new_xy))["proxy"]
        evaluator.revert()
        return cost

    best_v = cur_axis
    best_max = cur_cost
    use_sam = sam_K > 0 and sam_rho > 0.0

    for v in candidates:
        v_f = float(v)
        base = evalc(v_f)
        max_cost = base
        if use_sam:
            # Sample K perturbations uniform on disk of radius sam_rho
            # using rejection-free polar sampling: r = ρ √u, θ = 2π v.
            us = rng.random(sam_K)
            vs = rng.random(sam_K)
            rs = sam_rho * np.sqrt(us)
            thetas = 2.0 * math.pi * vs
            for k in range(sam_K):
                d_axis = float(rs[k] * math.cos(thetas[k]))
                d_perp = float(rs[k] * math.sin(thetas[k]))
                pc = eval_perturbed(v_f, d_axis, d_perp)
                if pc > max_cost:
                    max_cost = pc
        if max_cost < best_max - 1e-9:
            best_max = max_cost
            best_v = v_f

    return best_v, best_max, "sam"


# ── SAM-CD sweep loop (mirror of cd_core._sweep_macros + run_cd_adaptive) ──


def _sweep_macros_sam(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    grid_lines_x: np.ndarray,
    grid_lines_y: np.ndarray,
    order: List[int],
    cur_cost: float,
    n_hard: int,
    rng: np.random.Generator,
    sam_K: int,
    sam_rho: float,
    breakpoint_budget: int,
    deadline_check: Callable[[], bool],
) -> Tuple[float, int, int]:
    """Run one SAM-CD sweep over ``order``.

    Returns ``(cur_cost, accepted, probes)``.

    Note: ``cur_cost`` here is the *plain* proxy at the current state
    (not a SAM max-cost). When we accept a move based on SAM scoring
    (best_max < some_baseline), we still need to update cur_cost to
    the *plain* proxy at the new state — that's what subsequent
    sweeps and the plateau detector compare against.

    Acceptance: we accept the SAM-best v when its plain proxy at
    (v, perp_val) is < cur_cost (i.e., a real improvement, not just
    a flatter neighborhood). This keeps plateau detection meaningful
    and avoids accepting moves that worsen cur_cost just because the
    perturbed neighborhood happens to be flatter.
    """
    accepted = 0
    probes = 0

    for macro_idx in order:
        if deadline_check():
            break

        # ── X-axis ──
        lo_x, hi_x = legal_axis_range(
            macro_idx, evaluator.placement, evaluator.macro_sizes,
            benchmark.macro_fixed, n_hard, axis=0,
            canvas_w=benchmark.canvas_width, canvas_h=benchmark.canvas_height,
        )
        cur_xy = (
            float(evaluator.placement[macro_idx, 0]),
            float(evaluator.placement[macro_idx, 1]),
        )
        best_x, _best_max_x, _mode = sam_search_axis(
            macro_idx, 0, evaluator, grid_lines_x,
            lo_x, hi_x, cur_cost, cur_xy[1],
            benchmark=benchmark, n_hard=n_hard, rng=rng,
            breakpoint_budget=breakpoint_budget,
            sam_K=sam_K, sam_rho=sam_rho,
        )
        probes += 1
        if abs(best_x - cur_xy[0]) > 1e-7:
            # Compute plain proxy at the SAM-best v to decide acceptance.
            new_proxy = evaluator.move(macro_idx, (float(best_x), cur_xy[1]))["proxy"]
            if new_proxy < cur_cost - 1e-9:
                cur_cost = new_proxy
                accepted += 1
            else:
                evaluator.revert()

        # ── Y-axis ──
        cur_xy = (
            float(evaluator.placement[macro_idx, 0]),
            float(evaluator.placement[macro_idx, 1]),
        )
        lo_y, hi_y = legal_axis_range(
            macro_idx, evaluator.placement, evaluator.macro_sizes,
            benchmark.macro_fixed, n_hard, axis=1,
            canvas_w=benchmark.canvas_width, canvas_h=benchmark.canvas_height,
        )
        best_y, _best_max_y, _mode = sam_search_axis(
            macro_idx, 1, evaluator, grid_lines_y,
            lo_y, hi_y, cur_cost, cur_xy[0],
            benchmark=benchmark, n_hard=n_hard, rng=rng,
            breakpoint_budget=breakpoint_budget,
            sam_K=sam_K, sam_rho=sam_rho,
        )
        probes += 1
        if abs(best_y - cur_xy[1]) > 1e-7:
            new_proxy = evaluator.move(macro_idx, (cur_xy[0], float(best_y)))["proxy"]
            if new_proxy < cur_cost - 1e-9:
                cur_cost = new_proxy
                accepted += 1
            else:
                evaluator.revert()

    return cur_cost, accepted, probes


def run_sam_cd_adaptive(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    movable: List[int],
    sam_K: int,
    sam_rho: float,
    breakpoint_budget: int = 12,
    seed: int = 42,
    min_time_s: float = 300.0,
    hard_cap_s: float = 2400.0,
    patience: int = 3,
    plateau_threshold: float = 0.001,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Adaptive coordinate-descent with SAM-style worst-case scoring.

    Mirrors ``cd_core.run_cd_adaptive`` but uses ``_sweep_macros_sam``
    (which calls ``sam_search_axis``) instead of ``cd_core._sweep_macros``.
    Plateau detection and exit conditions identical to run_cd_adaptive.
    """
    n_hard = benchmark.num_hard_macros
    grid_lines_x, grid_lines_y = _grid_lines(plc)
    rng = np.random.default_rng(seed=seed)

    cur_cost = evaluator.current_cost()["proxy"]

    sweep_idx = 0
    total_moves = 0
    total_probes = 0
    deltas: deque = deque(maxlen=patience)
    exit_reason = "cap"

    t_start = time.perf_counter()

    if log_fn is not None:
        log_fn(
            f"  SAM-CD adaptive: K={sam_K}, ρ={sam_rho:.4f}, "
            f"min_time={min_time_s:.0f}s, cap={hard_cap_s:.0f}s, "
            f"patience={patience}, plateau_thr={plateau_threshold}"
        )

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= hard_cap_s:
            exit_reason = "cap"
            break
        sweep_idx += 1
        sweep_t0 = time.perf_counter()
        prev_cost = cur_cost

        sweep_rng = np.random.default_rng(seed=seed + sweep_idx)
        order = list(movable)
        sweep_rng.shuffle(order)

        cur_cost, sweep_accepted, sweep_probes = _sweep_macros_sam(
            evaluator, benchmark, grid_lines_x, grid_lines_y,
            order, cur_cost, n_hard,
            rng=rng,
            sam_K=sam_K, sam_rho=sam_rho,
            breakpoint_budget=breakpoint_budget,
            deadline_check=lambda: time.perf_counter() - t_start >= hard_cap_s,
        )

        sweep_wall = time.perf_counter() - sweep_t0
        elapsed = time.perf_counter() - t_start
        cost_break = evaluator.current_cost()
        cur_cost = cost_break["proxy"]
        total_moves += sweep_accepted
        total_probes += sweep_probes

        delta = prev_cost - cur_cost
        deltas.append(delta)

        if log_fn is not None:
            log_fn(
                f"  SAM-sweep {sweep_idx:3d}  elapsed={elapsed:7.1f}s  "
                f"sweep_t={sweep_wall:6.1f}s  proxy={cost_break['proxy']:.5f}  "
                f"Δ={delta:+.5f}  "
                f"accepted={sweep_accepted}/{sweep_probes}  "
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
        "wall_total_s": time.perf_counter() - t_start,
        "exit_reason": exit_reason,
        "final_deltas": list(deltas),
    }


# ── Placer ─────────────────────────────────────────────────────────────────


class CDLNSSASAMPlacer:
    """E32 placer: SAM-CD + grid-bin LNS + SA-v2 polish.

    SAM-CD scores each candidate breakpoint as the max over K=4
    perturbations within radius ρ = 0.01 × canvas_diag. Accept moves
    whose plain proxy improves vs current state — SAM scoring is used
    only to *select* among candidates, not to gate acceptance.

    Time budget per benchmark (≤ 3600 s legal cap):
      * SAM-CD phase: 2400 s.
      * LNS phase: 600 s.
      * SA-v2 phase: 600 s.

    All hyperparameters global. No per-benchmark tuning.
    """

    def __init__(
        self,
        # SAM hyperparams
        sam_K: int = 4,
        sam_rho_frac: float = 0.01,
        # CD hyperparams (mirror E25)
        cd_hard_cap_s: float = 2400.0,
        cd_min_time_s: float = 300.0,
        cd_patience: int = 3,
        cd_plateau_threshold: float = 0.001,
        cd_breakpoint_budget: int = 12,
        cd_seed: int = 42,
        # LNS hyperparams (mirror E25)
        lns_budget_s: float = 600.0,
        lns_destroy_frac: float = 0.05,
        lns_destroy_cap: int = 30,
        lns_seed: int = 42,
        # SA-v2 hyperparams (mirror E25)
        sa_budget_s: float = 600.0,
        sa_T0: float = 5e-4,
        sa_Tf: float = 1e-6,
        sa_seed: int = 42,
        sa_breakpoint_budget: int = 12,
        verbose: bool = True,
    ) -> None:
        self.sam_K = int(sam_K)
        self.sam_rho_frac = float(sam_rho_frac)
        self.cd_hard_cap_s = float(cd_hard_cap_s)
        self.cd_min_time_s = float(cd_min_time_s)
        self.cd_patience = int(cd_patience)
        self.cd_plateau_threshold = float(cd_plateau_threshold)
        self.cd_breakpoint_budget = int(cd_breakpoint_budget)
        self.cd_seed = int(cd_seed)
        self.lns_budget_s = float(lns_budget_s)
        self.lns_destroy_frac = float(lns_destroy_frac)
        self.lns_destroy_cap = int(lns_destroy_cap)
        self.lns_seed = int(lns_seed)
        self.sa_budget_s = float(sa_budget_s)
        self.sa_T0 = float(sa_T0)
        self.sa_Tf = float(sa_Tf)
        self.sa_seed = int(sa_seed)
        self.sa_breakpoint_budget = int(sa_breakpoint_budget)
        self.verbose = bool(verbose)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _load_plc_for(self, benchmark: Benchmark):
        bench_dir = find_benchmark_dir(benchmark.name)
        return load_benchmark_from_dir(str(bench_dir))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_total0 = time.perf_counter()
        canvas_diag = math.hypot(
            float(benchmark.canvas_width), float(benchmark.canvas_height)
        )
        sam_rho = self.sam_rho_frac * canvas_diag

        self._log(
            f"=== CDLNSSASAMPlacer ({benchmark.name}): "
            f"SAM-CD cap={self.cd_hard_cap_s:.0f}s "
            f"(K={self.sam_K}, ρ={sam_rho:.3f} = "
            f"{self.sam_rho_frac*100:.2f}% of canvas diag {canvas_diag:.1f}), "
            f"LNS={self.lns_budget_s:.0f}s, "
            f"SA={self.sa_budget_s:.0f}s ==="
        )
        self._log(
            f"  num_macros={benchmark.num_macros}, "
            f"n_hard={benchmark.num_hard_macros}, "
            f"canvas={benchmark.canvas_width:.1f}x{benchmark.canvas_height:.1f}"
        )

        # 1. SDF init
        t_init0 = time.perf_counter()
        placement = sdf_init(benchmark)
        self._log(f"  SDF init wall = {time.perf_counter() - t_init0:.1f} s")

        # 2. Project overlaps
        placement, proj_iters = project_overlaps(placement, benchmark)
        init_overlaps = compute_overlap_metrics(placement, benchmark)
        self._log(
            f"  overlap projection: {proj_iters} iters, "
            f"residual count={init_overlaps['overlap_count']}"
        )

        # 3. Build evaluator
        _, plc = self._load_plc_for(benchmark)
        placement_f64 = placement.detach().clone().to(torch.float64)
        evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
        init_cost = evaluator.current_cost()
        self._log(
            f"  init proxy={init_cost['proxy']:.5f} "
            f"[wl={init_cost['wl']:.4f} d={init_cost['density']:.4f} "
            f"c={init_cost['congestion']:.4f}]"
        )

        # 4. SAM-CD phase
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        hard_movable = [
            i for i in range(benchmark.num_hard_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        self._log(
            f"  starting SAM-CD phase (cap={self.cd_hard_cap_s:.0f}s, "
            f"K={self.sam_K}, ρ={sam_rho:.3f})"
        )
        cd_stats = run_sam_cd_adaptive(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            movable=movable,
            sam_K=self.sam_K,
            sam_rho=sam_rho,
            breakpoint_budget=self.cd_breakpoint_budget,
            seed=self.cd_seed,
            min_time_s=self.cd_min_time_s,
            hard_cap_s=self.cd_hard_cap_s,
            patience=self.cd_patience,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=self._log if self.verbose else None,
        )
        cd_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  SAM-CD done: exit={cd_stats['exit_reason']}, "
            f"sweeps={cd_stats['sweeps']}, accepts={cd_stats['total_moves']}, "
            f"probes={cd_stats['total_probes']}, "
            f"wall={cd_stats['wall_total_s']:.1f}s, proxy={cd_proxy:.5f}"
        )

        # 5. LNS phase
        self._log(f"  starting LNS phase (budget={self.lns_budget_s:.0f}s)")
        lns_stats = run_lns_gridbin(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.lns_budget_s,
            destroy_frac=self.lns_destroy_frac,
            destroy_cap=self.lns_destroy_cap,
            seed=self.lns_seed,
            log_fn=self._log if self.verbose else None,
        )
        lns_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  LNS done: samples={lns_stats['samples']}, "
            f"Δ={lns_stats['total_improvement']:+.5f}, "
            f"wall={lns_stats['wall_total_s']:.1f}s, proxy={lns_proxy:.5f}"
        )

        # 6. SA-v2 phase
        self._log(f"  starting SA-v2 phase (budget={self.sa_budget_s:.0f}s)")
        sa_stats = run_sa_polish_v2(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.sa_budget_s,
            T0=self.sa_T0,
            Tf=self.sa_Tf,
            seed=self.sa_seed,
            breakpoint_budget=self.sa_breakpoint_budget,
            log_fn=self._log if self.verbose else None,
        )
        final_cost = evaluator.current_cost()
        self._log(
            f"  SA-v2 done: lift_vs_init={sa_stats['improvement_vs_init']:+.5f}, "
            f"best={sa_stats['best_proxy']:.5f}, "
            f"final proxy={final_cost['proxy']:.5f}"
        )
        self._log(
            f"  TOTAL lifts: SAM-CD={init_cost['proxy'] - cd_proxy:+.5f}, "
            f"LNS={cd_proxy - lns_proxy:+.5f}, "
            f"SA={lns_proxy - final_cost['proxy']:+.5f}"
        )

        # 7. Pull placement back; preserve fixed macros
        final_placement_f64 = evaluator.placement.detach().clone().cpu()
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            final_placement_f64[fixed_mask] = original_positions[fixed_mask]
        final_placement = final_placement_f64.to(torch.float32)

        overlaps = compute_overlap_metrics(final_placement, benchmark)
        if overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"CDLNSSASAMPlacer produced {overlaps['overlap_count']} overlaps "
                f"(area {overlaps['total_overlap_area']:.4f}) on '{benchmark.name}'"
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(SDF + project + SAM-CD + LNS + SA + validate)"
        )
        return final_placement
