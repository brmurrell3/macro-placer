"""E17 — CD + LNS + SA-v2 with uniform-random legal init (multi-basin probe).

Identical to E25 (`submissions/cd_lns_sa/placer.py`, champion candidate at
avg `--all` 1.0954) except step 1 "SDF init" is replaced by
`random_legal_init`. Hypothesis: SDF may be a contractive basin from
which CD plateau + grid-bin LNS + SA-v2 cannot escape on some
benchmarks. A different legal starting point exposes multi-basin
structure if any per-bench score beats E25.

Pipeline (per benchmark, total ≤ 3600 s legal cap):

  1. ``random_legal_init`` (shuffled-greedy rejection sampling on hard
     movable macros; soft macros take SDF positions; fixed macros stay).
  2. Project overlaps (defensive — random init *should* be already legal,
     but float drift around boundaries can produce sub-eps overlaps).
  3. Build IncrementalProxyEvaluator (full-proxy: WL + density + congestion).
  4. CD phase (≤ 2400 s, plateau detection at threshold 0.001).
  5. LNS phase (≤ 600 s, grid-bin destroy/reinsert, cost-aware destroy).
  6. SA-v2 phase (≤ 600 s, Metropolis on per-axis breakpoints, T₀ = 5e-4,
     T_f = 1e-6, geometric cooling on elapsed/budget). Best-so-far
     tracking ensures non-regression vs LNS state.
  7. Validate (zero overlaps), preserve fixed macros, return.

All hyperparameters global. No per-benchmark tuning.

Random init details:
  * For each hard movable macro (shuffled order, NumPy default_rng with
    ``random_init_seed``): sample (cx, cy) ∼ uniform on
    [half_w, canvas_w − half_w] × [half_h, canvas_h − half_h]. Accept iff
    no overlap with any already-placed hard macro (``_is_legal_2d``).
  * Up to 1000 attempts per macro; if exhausted, fall back to that
    macro's SDF position.
  * Fixed macros stay at ``benchmark.macro_positions``.
  * Soft macros (indices ``[num_hard_macros, num_macros)``) take SDF
    positions — they never block legality, so they don't need rejection.

References:
- E25 ``submissions/cd_lns_sa/placer.py`` — the reference pipeline.
- E12 ``submissions/cd_lns_gridbin/placer.py`` — LNS overlay primitives.
- ``macro_place/cd_core.py`` — ``run_cd_adaptive``, ``project_overlaps``,
  ``legal_axis_range``, ``axis_breakpoints``, ``_grid_lines``, ``sdf_init``.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

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
    run_cd_adaptive,
    sdf_init,
)
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics


# ── Random legal init (E17-specific) ───────────────────────────────────────


def _is_legal_2d(
    idx: int,
    x: float,
    y: float,
    placement: torch.Tensor,
    macro_sizes_np: np.ndarray,
    n_hard: int,
    placed_mask: Optional[np.ndarray] = None,
    eps: float = 1e-4,
) -> bool:
    """True iff placing hard macro idx at (x, y) overlaps no other hard
    macro at its current position. Soft macros never block.

    If ``placed_mask`` is provided, only macros with ``placed_mask[i]=True``
    can act as blockers (used during shuffled-greedy random init, before
    every macro has a position).
    """
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
    blockers[idx] = False
    if placed_mask is not None:
        blockers = blockers & placed_mask[:n_hard]
    return not bool(np.any(blockers))


def random_legal_init(
    benchmark: Benchmark,
    seed: int = 42,
    max_attempts_per_macro: int = 1000,
    log_fn: Optional[Callable[[str], None]] = None,
) -> torch.Tensor:
    """Uniform-random legal placement for hard movable macros.

    Strategy (shuffled-greedy rejection sampling):
      1. Start from ``sdf_init`` (fallback for any macro that exhausts
         its attempt budget; also gives soft macros their starting
         positions).
      2. For each hard movable macro (in shuffled order, seeded by
         ``seed``), reset its position via uniform sampling on
         [half_w, canvas_w − half_w] × [half_h, canvas_h − half_h] until
         the proposed center is legal (no overlap with already-placed
         hard macros).
      3. Up to ``max_attempts_per_macro`` per macro; on budget
         exhaustion, keep the SDF position for that macro.

    Fixed macros remain at ``benchmark.macro_positions``. Soft macros
    take their SDF positions and are not subject to rejection (they
    never block legality).

    Returns a torch.Tensor [num_macros, 2] of (x, y) centers, dtype
    float32 (matches ``sdf_init`` output dtype).
    """
    rng = np.random.default_rng(seed=seed)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    n_hard = int(benchmark.num_hard_macros)
    n_macros = int(benchmark.num_macros)
    macro_sizes_np = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed_np = benchmark.macro_fixed.cpu().numpy().astype(bool)

    # Start from SDF — gives us soft positions and fallback for hard macros
    # whose rejection-sampling budget is exhausted.
    placement = sdf_init(benchmark).detach().clone()

    # Hard movable indices in shuffled order.
    hard_movable = [i for i in range(n_hard) if not bool(fixed_np[i])]
    rng.shuffle(hard_movable)

    # ``placed_mask`` flags hard macros whose position is "committed" for
    # rejection-test purposes. Initially: only fixed hard macros are
    # placed (they block); movable hard macros will be placed in
    # shuffled order, each becoming a blocker for subsequent macros.
    placed_mask = np.zeros(n_macros, dtype=bool)
    for i in range(n_hard):
        if bool(fixed_np[i]):
            placed_mask[i] = True

    fallback_count = 0
    accept_attempts: List[int] = []

    for idx in hard_movable:
        half_w = float(macro_sizes_np[idx, 0]) / 2.0
        half_h = float(macro_sizes_np[idx, 1]) / 2.0
        lo_x = half_w
        hi_x = cw - half_w
        lo_y = half_h
        hi_y = ch - half_h
        # Degenerate ranges: macro is wider/taller than canvas. Fall
        # back to SDF and continue (the same problem the SDF init
        # would face — caller should treat this as data error
        # downstream if it persists).
        if hi_x <= lo_x or hi_y <= lo_y:
            placed_mask[idx] = True
            fallback_count += 1
            continue

        accepted = False
        for attempt in range(max_attempts_per_macro):
            cx = float(rng.uniform(lo_x, hi_x))
            cy = float(rng.uniform(lo_y, hi_y))
            # Temporarily write the candidate position and test
            # legality against already-placed hard macros only.
            placement[idx, 0] = cx
            placement[idx, 1] = cy
            if _is_legal_2d(
                idx, cx, cy, placement, macro_sizes_np, n_hard,
                placed_mask=placed_mask,
            ):
                accepted = True
                accept_attempts.append(attempt + 1)
                break

        if accepted:
            placed_mask[idx] = True
        else:
            # Budget exhausted: fall back to SDF position for this
            # macro. The SDF position was the starting state of
            # ``placement[idx]`` before we started overwriting; rather
            # than restoring it (we've stomped on it), recompute SDF
            # and copy that single index back.
            sdf_fallback = sdf_init(benchmark)
            placement[idx, 0] = sdf_fallback[idx, 0]
            placement[idx, 1] = sdf_fallback[idx, 1]
            placed_mask[idx] = True
            fallback_count += 1

    if log_fn is not None:
        avg_attempts = (
            float(np.mean(accept_attempts)) if accept_attempts else 0.0
        )
        max_attempts_seen = (
            int(np.max(accept_attempts)) if accept_attempts else 0
        )
        log_fn(
            f"  random_legal_init: seed={seed}, hard_movable="
            f"{len(hard_movable)}, accepted={len(accept_attempts)} "
            f"(avg attempts={avg_attempts:.1f}, max={max_attempts_seen}), "
            f"fallback_to_sdf={fallback_count}"
        )

    return placement


# ── Grid-bin LNS primitives (inlined from E25 / E12 cd_lns_gridbin) ────────


def _cost_aware_destroy(
    evaluator: IncrementalProxyEvaluator,
    hard_movable: List[int],
    K: int,
) -> List[int]:
    """Pick K hard movables to destroy by move-to-center proxy delta."""
    cw = evaluator.width / 2.0
    ch = evaluator.height / 2.0
    baseline_p = evaluator.current_cost()["proxy"]
    scores: List[tuple] = []
    for idx in hard_movable:
        try:
            evaluator.move(idx, (cw, ch))
            p = evaluator.current_cost()["proxy"]
            evaluator.revert()
            scores.append((idx, p - baseline_p))
        except Exception:
            scores.append((idx, 0.0))
    scores.sort(key=lambda s: s[1])
    return [s[0] for s in scores[:K]]


def _gridbin_reinsert(
    evaluator: IncrementalProxyEvaluator,
    macro_idx: int,
    plc,
    n_hard: int,
    macro_sizes_np: np.ndarray,
    deadline_s: float,
    t_start: float,
) -> float:
    """Search every legal grid-bin center, commit the best."""
    cw = float(plc.width)
    ch = float(plc.height)
    grid_w = cw / plc.grid_col
    grid_h = ch / plc.grid_row
    half_w = float(macro_sizes_np[macro_idx, 0]) / 2.0
    half_h = float(macro_sizes_np[macro_idx, 1]) / 2.0

    cur_cost = evaluator.current_cost()["proxy"]
    cur_xy = (
        float(evaluator.placement[macro_idx, 0]),
        float(evaluator.placement[macro_idx, 1]),
    )
    best_cost = cur_cost
    best_xy = cur_xy

    for col in range(int(plc.grid_col)):
        if time.perf_counter() - t_start >= deadline_s:
            break
        cx = (col + 0.5) * grid_w
        if cx < half_w or cx > cw - half_w:
            continue
        for row in range(int(plc.grid_row)):
            if time.perf_counter() - t_start >= deadline_s:
                break
            cy = (row + 0.5) * grid_h
            if cy < half_h or cy > ch - half_h:
                continue
            if not _is_legal_2d(
                macro_idx, cx, cy, evaluator.placement, macro_sizes_np, n_hard
            ):
                continue
            evaluator.move(macro_idx, (cx, cy))
            cost = evaluator.current_cost()["proxy"]
            evaluator.revert()
            if cost < best_cost - 1e-9:
                best_cost = cost
                best_xy = (cx, cy)

    if best_cost < cur_cost - 1e-9 and (
        abs(best_xy[0] - cur_xy[0]) > 1e-7 or abs(best_xy[1] - cur_xy[1]) > 1e-7
    ):
        evaluator.move(macro_idx, best_xy)
        return best_cost - cur_cost
    return 0.0


def run_lns_gridbin(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    hard_movable: List[int],
    time_budget_s: float,
    destroy_frac: float = 0.05,
    destroy_cap: int = 30,
    seed: int = 42,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Grid-bin LNS phase: cost-aware destroy + grid-bin reinsert.

    Iterates destroy/reinsert until budget exhausted or a full sample
    produces no improvement.
    """
    K = max(1, min(destroy_cap, int(destroy_frac * len(hard_movable))))
    macro_sizes_np = evaluator.macro_sizes.cpu().numpy()
    n_hard = benchmark.num_hard_macros

    if log_fn is not None:
        log_fn(
            f"  LNS budget={time_budget_s:.0f}s, destroy K={K} "
            f"(={destroy_frac*100:.1f}% of {len(hard_movable)} hard movables, "
            f"capped at {destroy_cap}), strategy=cost_aware"
        )

    sample = 0
    total_improvement = 0.0
    t_start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:
            break
        sample += 1
        sample_t0 = time.perf_counter()

        destroy = _cost_aware_destroy(evaluator, hard_movable, K)

        sample_delta = 0.0
        moves_this_sample = 0
        for idx in destroy:
            if time.perf_counter() - t_start >= time_budget_s:
                break
            delta = _gridbin_reinsert(
                evaluator, idx, plc, n_hard, macro_sizes_np,
                deadline_s=time_budget_s, t_start=t_start,
            )
            sample_delta += delta
            if delta < 0:
                moves_this_sample += 1

        total_improvement += sample_delta
        sample_wall = time.perf_counter() - sample_t0
        elapsed = time.perf_counter() - t_start
        cur_proxy = evaluator.current_cost()["proxy"]
        if log_fn is not None:
            log_fn(
                f"  LNS sample {sample}: K={K}, Δ={sample_delta:+.5f} "
                f"(moves={moves_this_sample}/{K}), proxy={cur_proxy:.5f}, "
                f"sample_wall={sample_wall:.1f}s, total elapsed={elapsed:.1f}s"
            )

        if abs(sample_delta) < 1e-7:
            if log_fn is not None:
                log_fn(f"  LNS converged at sample {sample} (no improvement)")
            break

    return {
        "samples": sample,
        "total_improvement": total_improvement,
        "wall_total_s": time.perf_counter() - t_start,
    }


# ── SA polish v2 primitive (inlined from E25 / E24 cd_sa_polish_v2) ────────


def run_sa_polish_v2(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    hard_movable: List[int],
    time_budget_s: float,
    T0: float = 5e-4,
    Tf: float = 1e-6,
    seed: int = 42,
    breakpoint_budget: int = 12,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """SA polish v2 — best-so-far tracking, low T₀ on per-axis breakpoints."""
    rng = np.random.default_rng(seed=seed)
    grid_lines_x, grid_lines_y = _grid_lines(plc)
    n_hard = benchmark.num_hard_macros

    if not hard_movable:
        if log_fn is not None:
            log_fn("  SA: no hard movable macros; skipping")
        return {
            "proposed": 0, "accepted_better": 0, "accepted_worse": 0,
            "rejected": 0, "skipped": 0,
            "init_proxy": float("nan"), "best_proxy": float("nan"),
            "final_proxy": float("nan"), "improvement_vs_init": 0.0,
            "best_found_at_t": 0.0, "wall_total_s": 0.0,
            "best_restored": False,
        }

    if log_fn is not None:
        log_fn(
            f"  SA v2 budget={time_budget_s:.0f}s, T0={T0:.2e}, Tf={Tf:.2e}, "
            f"|H|={len(hard_movable)}, seed={seed} (best-so-far tracking ON)"
        )

    proposed = 0
    accepted_better = 0
    accepted_worse = 0
    rejected = 0
    skipped = 0
    init_proxy = evaluator.current_cost()["proxy"]
    cur_proxy = init_proxy
    best_proxy = init_proxy
    best_placement = evaluator.placement.detach().clone()
    best_found_at_t = 0.0

    log_ratio = math.log(Tf / T0)
    t_start = time.perf_counter()
    last_log_t = t_start

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:
            break

        frac = elapsed / time_budget_s
        T = T0 * math.exp(log_ratio * frac)

        idx = int(hard_movable[rng.integers(0, len(hard_movable))])
        axis = int(rng.integers(0, 2))

        lo, hi = legal_axis_range(
            idx, evaluator.placement, evaluator.macro_sizes,
            benchmark.macro_fixed, n_hard, axis=axis,
            canvas_w=benchmark.canvas_width,
            canvas_h=benchmark.canvas_height,
        )
        if hi - lo < 1e-5:
            skipped += 1
            continue

        cur_axis_val = float(evaluator.placement[idx, axis])
        grid_lines = grid_lines_x if axis == 0 else grid_lines_y
        cands = axis_breakpoints(
            idx, axis, evaluator, grid_lines, lo, hi,
            max_breakpoints=breakpoint_budget, cur_axis=cur_axis_val,
        )
        if len(cands) > 0:
            mask = np.abs(cands - cur_axis_val) > 1e-6
            cands = cands[mask]
        if len(cands) == 0:
            skipped += 1
            continue

        new_axis_val = float(cands[rng.integers(0, len(cands))])
        cur_xy = (float(evaluator.placement[idx, 0]),
                  float(evaluator.placement[idx, 1]))
        new_xy = list(cur_xy)
        new_xy[axis] = new_axis_val

        proposed += 1
        new_proxy = evaluator.move(idx, tuple(new_xy))["proxy"]
        delta = new_proxy - cur_proxy
        accepted = False

        if delta <= 0.0:
            cur_proxy = new_proxy
            accepted_better += 1
            accepted = True
        else:
            accept_p = math.exp(-delta / T) if T > 0 else 0.0
            if rng.random() < accept_p:
                cur_proxy = new_proxy
                accepted_worse += 1
                accepted = True
            else:
                evaluator.revert()
                rejected += 1

        if accepted and cur_proxy < best_proxy - 1e-12:
            best_proxy = cur_proxy
            best_placement = evaluator.placement.detach().clone()
            best_found_at_t = time.perf_counter() - t_start

        now = time.perf_counter()
        if log_fn is not None and (now - last_log_t) >= 30.0:
            last_log_t = now
            log_fn(
                f"  SA v2 t={now - t_start:6.1f}s T={T:.2e} "
                f"proposed={proposed} better={accepted_better} "
                f"worse={accepted_worse} rejected={rejected} "
                f"skipped={skipped} cur={cur_proxy:.5f} best={best_proxy:.5f}"
            )

    wall = time.perf_counter() - t_start
    final_proxy_chain = evaluator.current_cost()["proxy"]

    best_restored = False
    if best_proxy < final_proxy_chain - 1e-12:
        n_macros = int(evaluator.placement.shape[0])
        for i in range(n_macros):
            tx = float(best_placement[i, 0])
            ty = float(best_placement[i, 1])
            cx = float(evaluator.placement[i, 0])
            cy = float(evaluator.placement[i, 1])
            if abs(tx - cx) > 1e-9 or abs(ty - cy) > 1e-9:
                evaluator.move(i, (tx, ty))
        best_restored = True

    final_proxy = evaluator.current_cost()["proxy"]
    if log_fn is not None:
        log_fn(
            f"  SA v2 done: proposed={proposed}, better={accepted_better}, "
            f"worse={accepted_worse}, rejected={rejected}, skipped={skipped}, "
            f"init={init_proxy:.5f}, best={best_proxy:.5f} "
            f"(found at t={best_found_at_t:.1f}s), "
            f"chain-final={final_proxy_chain:.5f}, "
            f"restored={best_restored}, evaluator-final={final_proxy:.5f}"
        )

    return {
        "proposed": proposed,
        "accepted_better": accepted_better,
        "accepted_worse": accepted_worse,
        "rejected": rejected,
        "skipped": skipped,
        "init_proxy": init_proxy,
        "best_proxy": best_proxy,
        "final_proxy": final_proxy,
        "improvement_vs_init": init_proxy - best_proxy,
        "best_found_at_t": best_found_at_t,
        "wall_total_s": wall,
        "best_restored": best_restored,
    }


# ── Placer (E17) ───────────────────────────────────────────────────────────


class CDLNSSARandomInitPlacer:
    """E17 placer: random legal init + CD + LNS + SA-v2.

    Identical to E25 ``CDLNSSAPlacer`` except step 1 is replaced by
    ``random_legal_init`` (uniform sampling on the legal canvas with
    shuffled-greedy rejection on hard movable macros).

    Time budget per benchmark (≤ 3600 s legal cap):
      * CD phase: 2400 s.
      * LNS phase: 600 s.
      * SA-v2 phase: 600 s.

    All hyperparameters global. No per-benchmark tuning.
    """

    def __init__(
        self,
        cd_hard_cap_s: float = 2400.0,
        cd_min_time_s: float = 300.0,
        cd_patience: int = 3,
        cd_plateau_threshold: float = 0.001,
        lns_budget_s: float = 600.0,
        lns_destroy_frac: float = 0.05,
        lns_destroy_cap: int = 30,
        lns_seed: int = 42,
        sa_budget_s: float = 600.0,
        sa_T0: float = 5e-4,
        sa_Tf: float = 1e-6,
        sa_seed: int = 42,
        sa_breakpoint_budget: int = 12,
        random_init_seed: int = 42,
        random_init_max_attempts: int = 1000,
        verbose: bool = True,
    ) -> None:
        self.cd_hard_cap_s = float(cd_hard_cap_s)
        self.cd_min_time_s = float(cd_min_time_s)
        self.cd_patience = int(cd_patience)
        self.cd_plateau_threshold = float(cd_plateau_threshold)
        self.lns_budget_s = float(lns_budget_s)
        self.lns_destroy_frac = float(lns_destroy_frac)
        self.lns_destroy_cap = int(lns_destroy_cap)
        self.lns_seed = int(lns_seed)
        self.sa_budget_s = float(sa_budget_s)
        self.sa_T0 = float(sa_T0)
        self.sa_Tf = float(sa_Tf)
        self.sa_seed = int(sa_seed)
        self.sa_breakpoint_budget = int(sa_breakpoint_budget)
        self.random_init_seed = int(random_init_seed)
        self.random_init_max_attempts = int(random_init_max_attempts)
        self.verbose = bool(verbose)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _load_plc_for(self, benchmark: Benchmark):
        bench_dir = find_benchmark_dir(benchmark.name)
        return load_benchmark_from_dir(str(bench_dir))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_total0 = time.perf_counter()
        self._log(
            f"=== CDLNSSARandomInitPlacer ({benchmark.name}): "
            f"CD cap={self.cd_hard_cap_s:.0f}s, LNS={self.lns_budget_s:.0f}s, "
            f"SA={self.sa_budget_s:.0f}s, random_seed={self.random_init_seed} ==="
        )
        self._log(
            f"  num_macros={benchmark.num_macros}, "
            f"n_hard={benchmark.num_hard_macros}, "
            f"canvas={benchmark.canvas_width:.1f}x{benchmark.canvas_height:.1f}"
        )

        # 1. Random legal init (E17-specific replacement for sdf_init)
        t_init0 = time.perf_counter()
        placement = random_legal_init(
            benchmark,
            seed=self.random_init_seed,
            max_attempts_per_macro=self.random_init_max_attempts,
            log_fn=self._log if self.verbose else None,
        )
        self._log(
            f"  random_legal_init wall = "
            f"{time.perf_counter() - t_init0:.1f} s"
        )

        # 2. Project overlaps (defensive — random init *should* be legal,
        #    but float drift around boundaries can produce sub-eps overlaps
        #    that ``project_overlaps`` cleans up cheaply).
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

        # 4. CD phase
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        hard_movable = [
            i for i in range(benchmark.num_hard_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        self._log(f"  starting CD phase (cap={self.cd_hard_cap_s:.0f}s)")
        cd_stats = run_cd_adaptive(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            movable=movable,
            min_time_s=self.cd_min_time_s,
            hard_cap_s=self.cd_hard_cap_s,
            patience=self.cd_patience,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=self._log if self.verbose else None,
        )
        cd_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  CD done: exit={cd_stats['exit_reason']}, "
            f"sweeps={cd_stats['sweeps']}, accepts={cd_stats['total_moves']}, "
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
            f"  TOTAL lifts: CD={init_cost['proxy'] - cd_proxy:+.5f}, "
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
                f"CDLNSSARandomInitPlacer produced "
                f"{overlaps['overlap_count']} overlaps "
                f"(area {overlaps['total_overlap_area']:.4f}) "
                f"on '{benchmark.name}'"
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(rand_init + project + CD + LNS + SA + validate)"
        )
        return final_placement
