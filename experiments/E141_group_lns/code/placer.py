"""E141 — v2 pipeline + group-LNS escape phase.

Hypothesis: CD plateau is single-axis line-search saturated. K=6 spatially
clustered macros (focal + 5 nearest hard neighbors) re-placed by best-first
greedy on a candidate grid around each macro's current location creates a
2D coordinated move that CD cannot reach. Spirit of E12's grid-bin LNS, but
clustered (not top-K-cost) and local-grid (not full-canvas grid).

Pipeline:
  1. V4 + Gaussian descent → legalize → project_overlaps (v2 basin).
  2. CD polish 1 (300 s).
  3. Group LNS (200 s):
       - pick random focal hard-movable macro
       - find K-1 nearest hard movables (Euclidean on current xy)
       - destroy: collect all K positions; mark "to re-place"
       - reinsert one at a time, greedy best-first by canonical Δ.
         Each macro: enumerate N=20 candidates on a grid around current
         location (radius = 30 % of canvas in each axis), evaluate proxy
         via IncrementalProxyEvaluator.move/revert, commit best legal
         improving position. If none improves, leave at original xy.
  4. CD polish 2 (300 s).

DO NOT mutate v2 source or shared evaluator code. Total budget 1500 s/bench.
"""
from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _ROOT,
    _HERE,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E127_v4_gaussian" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v4_gaussian import SmoothGlobalPlacerV4Gaussian  # noqa: E402


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── LNS primitives ──────────────────────────────────────────────────────────


def _is_legal_2d(
    idx: int,
    x: float,
    y: float,
    placement: torch.Tensor,
    macro_sizes_np: np.ndarray,
    n_hard: int,
    eps: float = 1e-4,
) -> bool:
    """True iff placing hard macro `idx` at (x, y) does not overlap any other
    hard macro at its current position. Soft macros never block."""
    if idx >= n_hard:
        return True
    half_w = float(macro_sizes_np[idx, 0]) / 2.0
    half_h = float(macro_sizes_np[idx, 1]) / 2.0
    pos = placement[:n_hard].cpu().numpy().astype(np.float64)
    sz = macro_sizes_np[:n_hard]
    dx = np.abs(pos[:, 0] - x)
    dy = np.abs(pos[:, 1] - y)
    min_dx = half_w + sz[:, 0] / 2.0
    min_dy = half_h + sz[:, 1] / 2.0
    blockers = (dx < min_dx - eps) & (dy < min_dy - eps)
    blockers[idx] = False  # exclude self
    return not bool(np.any(blockers))


def _k_nearest(
    focal_xy: Tuple[float, float],
    pool: List[int],
    positions: np.ndarray,
    K: int,
) -> List[int]:
    """K-nearest macros in `pool` by Euclidean distance from focal_xy.

    Returns `pool` indices sorted by distance, length min(K, len(pool)).
    """
    if K >= len(pool):
        return list(pool)
    pool_arr = np.asarray(pool, dtype=np.int64)
    pos = positions[pool_arr]  # (P, 2)
    d2 = (pos[:, 0] - focal_xy[0]) ** 2 + (pos[:, 1] - focal_xy[1]) ** 2
    order = np.argsort(d2)[:K]
    return [int(pool_arr[i]) for i in order]


def _local_grid_candidates(
    cur_xy: Tuple[float, float],
    canvas_w: float,
    canvas_h: float,
    half_w: float,
    half_h: float,
    n_candidates: int,
    rng: random.Random,
    radius_frac: float = 0.3,
) -> List[Tuple[float, float]]:
    """Generate up to `n_candidates` candidate (x, y) positions on a grid
    centered at cur_xy, with radius radius_frac * canvas in each axis.

    Candidates clamped into legal canvas bounds (half_w/h margin).
    """
    rx = radius_frac * canvas_w
    ry = radius_frac * canvas_h
    # ~sqrt(N) per axis grid, plus a couple of jittered candidates.
    side = max(2, int(math.sqrt(n_candidates)))
    cands: List[Tuple[float, float]] = []
    for ix in range(side):
        for iy in range(side):
            fx = (ix + 0.5) / side
            fy = (iy + 0.5) / side
            cx = cur_xy[0] - rx + 2 * rx * fx
            cy = cur_xy[1] - ry + 2 * ry * fy
            cx = max(half_w, min(canvas_w - half_w, cx))
            cy = max(half_h, min(canvas_h - half_h, cy))
            cands.append((cx, cy))
            if len(cands) >= n_candidates:
                break
        if len(cands) >= n_candidates:
            break
    # A couple of pure-random jitter shots within radius (helps when grid
    # is too regular to escape local pockets).
    n_jitter = max(2, n_candidates // 5)
    for _ in range(n_jitter):
        jx = cur_xy[0] + rng.uniform(-rx, rx)
        jy = cur_xy[1] + rng.uniform(-ry, ry)
        jx = max(half_w, min(canvas_w - half_w, jx))
        jy = max(half_h, min(canvas_h - half_h, jy))
        cands.append((jx, jy))
    return cands


def _greedy_reinsert(
    evaluator: IncrementalProxyEvaluator,
    macro_idx: int,
    canvas_w: float,
    canvas_h: float,
    macro_sizes_np: np.ndarray,
    n_hard: int,
    n_candidates: int,
    rng: random.Random,
    radius_frac: float,
) -> float:
    """Try `n_candidates` legal grid candidates around macro_idx's current
    position; commit the best canonical-improving position. Return Δ
    (negative = improvement, 0.0 if none committed).
    """
    half_w = float(macro_sizes_np[macro_idx, 0]) / 2.0
    half_h = float(macro_sizes_np[macro_idx, 1]) / 2.0
    cur_xy = (
        float(evaluator.placement[macro_idx, 0]),
        float(evaluator.placement[macro_idx, 1]),
    )
    cur_cost = evaluator.current_cost()["proxy"]
    best_cost = cur_cost
    best_xy = cur_xy
    cands = _local_grid_candidates(
        cur_xy, canvas_w, canvas_h, half_w, half_h, n_candidates, rng, radius_frac
    )
    for (cx, cy) in cands:
        if not _is_legal_2d(
            macro_idx, cx, cy, evaluator.placement, macro_sizes_np, n_hard
        ):
            continue
        evaluator.move(macro_idx, (cx, cy))
        c = evaluator.current_cost()["proxy"]
        evaluator.revert()
        if c < best_cost - 1e-9:
            best_cost = c
            best_xy = (cx, cy)
    if best_cost < cur_cost - 1e-9 and (
        abs(best_xy[0] - cur_xy[0]) > 1e-7 or abs(best_xy[1] - cur_xy[1]) > 1e-7
    ):
        evaluator.move(macro_idx, best_xy)
        return best_cost - cur_cost
    return 0.0


def run_group_lns(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    canvas_w: float,
    canvas_h: float,
    hard_movable: List[int],
    time_budget_s: float,
    K: int = 6,
    n_candidates: int = 20,
    radius_frac: float = 0.3,
    seed: int = 42,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Group-LNS phase. Each iter: random focal + K-NN, greedy reinsert each
    macro on a local candidate grid. Iterates until budget exhausted.
    """
    macro_sizes_np = evaluator.macro_sizes.cpu().numpy()
    n_hard = benchmark.num_hard_macros
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if log_fn is not None:
        log_fn(
            f"  group-LNS budget={time_budget_s:.0f}s, K={K}, "
            f"n_candidates={n_candidates}, radius_frac={radius_frac:.2f}, "
            f"hard_movable={len(hard_movable)}"
        )

    t_start = time.perf_counter()
    iters = 0
    total_accepts = 0
    total_improvement = 0.0
    no_improve_streak = 0
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:
            break
        iters += 1
        iter_t0 = time.perf_counter()
        # Snapshot positions for K-NN (cpu numpy)
        positions = evaluator.placement[:n_hard].detach().cpu().numpy().astype(np.float64)
        focal_idx = int(np_rng.choice(np.asarray(hard_movable, dtype=np.int64)))
        focal_xy = (float(positions[focal_idx, 0]), float(positions[focal_idx, 1]))
        # K-NN among hard_movable (always includes focal at distance 0)
        cluster = _k_nearest(focal_xy, hard_movable, positions, K)
        # Re-insert: greedy best-first by canonical Δ.
        # Order: greedy improves; we re-order each step by recomputing cost
        # of moving each remaining macro to center (cheap heuristic).
        iter_delta = 0.0
        iter_accepts = 0
        for m_idx in cluster:
            if time.perf_counter() - t_start >= time_budget_s:
                break
            d = _greedy_reinsert(
                evaluator, m_idx, canvas_w, canvas_h,
                macro_sizes_np, n_hard, n_candidates, rng, radius_frac,
            )
            iter_delta += d
            if d < 0:
                iter_accepts += 1
        total_improvement += iter_delta
        total_accepts += iter_accepts
        iter_wall = time.perf_counter() - iter_t0
        cur_proxy = evaluator.current_cost()["proxy"]
        if log_fn is not None:
            log_fn(
                f"  LNS iter {iters}: focal={focal_idx}, K={len(cluster)}, "
                f"Δ={iter_delta:+.5f} (accepts={iter_accepts}/{len(cluster)}), "
                f"proxy={cur_proxy:.5f}, iter_wall={iter_wall:.1f}s, "
                f"elapsed={time.perf_counter()-t_start:.1f}s"
            )
        if abs(iter_delta) < 1e-7:
            no_improve_streak += 1
            if no_improve_streak >= 5:
                if log_fn is not None:
                    log_fn(f"  group-LNS converged at iter {iters} (5x no improvement)")
                break
        else:
            no_improve_streak = 0

    return {
        "iters": iters,
        "accepts": total_accepts,
        "total_improvement": total_improvement,
        "wall_total_s": time.perf_counter() - t_start,
    }


# ── Placer ──────────────────────────────────────────────────────────────────


def _cd_polish(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    budget_s: float,
    plateau_threshold: float = 0.001,
):
    """Run CD-adaptive polish under a hard wall budget."""
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement)
    movable = [
        i for i in range(benchmark.num_macros)
        if not bool(benchmark.macro_fixed[i])
    ]
    run_cd_adaptive(
        evaluator, benchmark, plc, movable,
        min_time_s=min(30.0, budget_s * 0.5),
        hard_cap_s=budget_s,
        patience=5,
        plateau_threshold=plateau_threshold,
        log_fn=None,
    )
    return evaluator.placement.detach().clone().to(torch.float32)


class E141GroupLNSPlacer:
    """v2 basin -> CD1 -> group LNS -> CD2."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 1500.0,
        # Descent params (mirror v2)
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        # CD params
        cd1_budget_s: float = 300.0,
        cd2_budget_s: float = 300.0,
        cd_plateau_threshold: float = 0.001,
        # LNS params
        lns_budget_s: float = 200.0,
        lns_K: int = 6,
        lns_n_candidates: int = 20,
        lns_radius_frac: float = 0.3,
        lns_seed: int = 42,
        rng_seed: int = 42,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.num_steps = num_steps
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.init = init
        self.cd1_budget_s = cd1_budget_s
        self.cd2_budget_s = cd2_budget_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.lns_budget_s = lns_budget_s
        self.lns_K = lns_K
        self.lns_n_candidates = lns_n_candidates
        self.lns_radius_frac = lns_radius_frac
        self.lns_seed = lns_seed
        self.rng_seed = rng_seed
        self.verbose = verbose
        self.last_phase_walls: dict = {}
        self.last_phase_proxies: dict = {}
        self.last_lns_stats: dict = {}

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E141GroupLNSPlacer ({benchmark.name}) ===")
        self._log(
            f"  budget={self.budget_seconds}s; descent + CD1={self.cd1_budget_s}s "
            f"+ LNS={self.lns_budget_s}s + CD2={self.cd2_budget_s}s"
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        device = _best_device()
        self._log(f"  device={device}")
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)

        # ------- Phase 1: V4 + Gaussian descent + legalize + project_overlaps
        t_descent = time.time()
        pos = None
        for attempt, cfg in enumerate([
            dict(overlap_lambda_end=self.overlap_lambda_end),
            dict(overlap_lambda_end=50.0),
            dict(overlap_lambda_end=100.0, num_steps=300),
        ]):
            try:
                num_steps = cfg.get("num_steps", self.num_steps)
                ovl_end = cfg["overlap_lambda_end"]
                self._log(
                    f"  Phase 1 attempt {attempt+1}: ovl_end={ovl_end} steps={num_steps}"
                )
                descender = SmoothGlobalPlacerV4Gaussian(
                    num_steps=num_steps,
                    lr_frac=self.lr_frac,
                    gamma_start_frac=self.gamma_start_frac,
                    gamma_end_frac=self.gamma_end_frac,
                    overlap_lambda_end=ovl_end,
                    overlap_ramp_pct=self.overlap_ramp_pct,
                    init=self.init,
                    device=device,
                    rng_seed=self.rng_seed + attempt * 100,
                    verbose=False,
                )
                pos_try = descender.place(benchmark)
                ovl_try = compute_overlap_metrics(pos_try, benchmark)["overlap_count"]
                if ovl_try == 0:
                    pos = pos_try
                    self._log(f"    ok: ovl=0")
                    break
                pos_try2, _ = project_overlaps(pos_try, benchmark)
                if compute_overlap_metrics(pos_try2, benchmark)["overlap_count"] == 0:
                    pos = pos_try2
                    self._log(f"    ok: ovl=0 after project_overlaps")
                    break
                self._log(f"    still {ovl_try} overlaps; retrying")
            except Exception as exc:
                self._log(f"    attempt {attempt+1} EXCEPTION: {exc}")
                continue
            if deadline is not None and time.time() > deadline - 600:
                break
        if pos is None:
            self._log("  Phase 1 fallback: SDF + project_overlaps")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)
        descent_wall = time.time() - t_descent
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self._log(
            f"  Phase 1 done: proxy={descent_proxy:.5f} wall={descent_wall:.0f}s"
        )

        # ------- Phase 2: CD polish 1
        if deadline is not None:
            remaining = deadline - time.time()
            reserve = self.lns_budget_s + self.cd2_budget_s + 60.0
            cd1_budget = max(30.0, min(self.cd1_budget_s, remaining - reserve))
        else:
            cd1_budget = self.cd1_budget_s
        t_cd1 = time.time()
        self._log(f"  Phase 2: CD1 budget={cd1_budget:.0f}s")
        pos = _cd_polish(pos, benchmark, plc, cd1_budget, self.cd_plateau_threshold)
        cd1_wall = time.time() - t_cd1
        cd1_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        cd1_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        self._log(
            f"  Phase 2 done: proxy={cd1_proxy:.5f} (Δ={cd1_proxy - descent_proxy:+.5f}) "
            f"ovl={cd1_ovl} wall={cd1_wall:.0f}s"
        )
        if cd1_ovl > 0:
            self._log(f"  WARN: CD1 produced {cd1_ovl} overlaps; project_overlaps")
            pos, _ = project_overlaps(pos, benchmark)

        # ------- Phase 3: group LNS
        if deadline is not None:
            remaining = deadline - time.time()
            lns_budget = max(30.0, min(self.lns_budget_s, remaining - self.cd2_budget_s - 30.0))
        else:
            lns_budget = self.lns_budget_s
        t_lns = time.time()
        lns_proxy_pre = cd1_proxy
        lns_info: dict = {}
        if lns_budget < 30.0:
            self._log(f"  Phase 3 SKIPPED (budget {lns_budget:.0f}s)")
        else:
            self._log(f"  Phase 3: group-LNS budget={lns_budget:.0f}s")
            try:
                # Build evaluator on cd1 placement (f64) for fast Δ
                pos_f64 = pos.detach().clone().to(torch.float64)
                evaluator = IncrementalProxyEvaluator(benchmark, plc, pos_f64)
                hard_movable = [
                    i for i in range(benchmark.num_hard_macros)
                    if not bool(benchmark.macro_fixed[i])
                ]
                lns_info = run_group_lns(
                    evaluator, benchmark, canvas_w, canvas_h,
                    hard_movable, lns_budget,
                    K=self.lns_K,
                    n_candidates=self.lns_n_candidates,
                    radius_frac=self.lns_radius_frac,
                    seed=self.lns_seed,
                    log_fn=self._log if self.verbose else None,
                )
                # Pull placement back
                lns_pos_f64 = evaluator.placement.detach().clone().cpu()
                fixed_mask = benchmark.macro_fixed
                original_positions = benchmark.macro_positions.to(torch.float64)
                if fixed_mask.any():
                    lns_pos_f64[fixed_mask] = original_positions[fixed_mask]
                lns_pos = lns_pos_f64.to(torch.float32)
                lns_ovl = compute_overlap_metrics(lns_pos, benchmark)["overlap_count"]
                lns_proxy = float(compute_proxy_cost(lns_pos, benchmark, plc)["proxy_cost"])
                if lns_ovl == 0 and lns_proxy < cd1_proxy:
                    pos = lns_pos
                    self._log(
                        f"  Phase 3 ACCEPT: {cd1_proxy:.5f} -> {lns_proxy:.5f}"
                    )
                else:
                    self._log(
                        f"  Phase 3 REJECT: ovl={lns_ovl} proxy={lns_proxy:.5f}"
                    )
            except Exception as exc:
                self._log(f"  Phase 3 EXCEPTION: {exc}; skipping")
        lns_wall = time.time() - t_lns
        lns_proxy_after = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self._log(f"  Phase 3 done: proxy={lns_proxy_after:.5f} wall={lns_wall:.0f}s")

        # ------- Phase 4: CD polish 2
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd2_budget = max(30.0, min(self.cd2_budget_s, remaining))
        else:
            cd2_budget = self.cd2_budget_s
        t_cd2 = time.time()
        self._log(f"  Phase 4: CD2 budget={cd2_budget:.0f}s")
        pos = _cd_polish(pos, benchmark, plc, cd2_budget, self.cd_plateau_threshold)
        cd2_wall = time.time() - t_cd2
        final = pos.detach().clone().to(torch.float32)
        final_proxy = float(compute_proxy_cost(final, benchmark, plc)["proxy_cost"])
        final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
        total_wall = time.time() - t0
        self._log(
            f"  Phase 4 done: proxy={final_proxy:.5f} ovl={final_ovl} "
            f"wall={cd2_wall:.0f}s"
        )
        self._log(
            f"  TOTAL: descent={descent_wall:.0f}s CD1={cd1_wall:.0f}s "
            f"LNS={lns_wall:.0f}s CD2={cd2_wall:.0f}s total={total_wall:.0f}s "
            f"final_proxy={final_proxy:.5f}"
        )

        self.last_phase_walls = {
            "descent": descent_wall,
            "cd1": cd1_wall,
            "lns": lns_wall,
            "cd2": cd2_wall,
            "total": total_wall,
        }
        self.last_phase_proxies = {
            "descent": descent_proxy,
            "cd1": cd1_proxy,
            "lns_pre": lns_proxy_pre,
            "lns_after": lns_proxy_after,
            "final": final_proxy,
        }
        self.last_lns_stats = lns_info or {}

        if final_ovl > 0:
            raise RuntimeError(f"Final placement has {final_ovl} overlaps")
        return final


# Aliases for harness
Placer = E141GroupLNSPlacer
