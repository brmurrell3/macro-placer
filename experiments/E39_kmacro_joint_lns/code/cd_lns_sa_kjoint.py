"""E39 — CDLNSSA + K-macro joint LNS.

Pipeline: SDF init → project → CD adaptive (≤2400s) → grid-bin LNS (≤600s)
→ SA-v2 (≤600s) → K-macro joint LNS (≤600s) → validate.

The K-macro joint LNS is a new move type with a different reachable set
than E25's mechanisms. E25 (and the five mechanisms it composes) are all
≤ 2-macro moves. Here we destroy K=3 most-coupled macros and brute-force
the joint reinsertion over the top-N=5 single-macro candidates per macro
(N^K = 125 combos per K-tuple). Hypothesis: this escapes the coupled
fixed point on ibm11/13/14/15 if those floors are 3-coupled multi-basin.

K=3, top_N=5, kjoint_budget=600 s. All hyperparameters global. No
per-benchmark tuning. Class enforces zero overlaps via
`compute_overlap_metrics` (E25 pattern).

Reference: see manifest.md. Inlines `run_lns_gridbin`, `run_sa_polish_v2`,
`_is_legal_2d` from `submissions/cd_lns_sa/placer.py` (E25 candidate).
"""
from __future__ import annotations

import itertools
import math
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Ensure repo root is on sys.path (eval harness uses importlib spec_from_file_location).
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


# ── Grid-bin LNS primitives (inlined from E25 cd_lns_sa) ───────────────────


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


def _is_legal_2d(
    idx: int,
    x: float,
    y: float,
    placement: torch.Tensor,
    macro_sizes_np: np.ndarray,
    n_hard: int,
    eps: float = 1e-4,
) -> bool:
    """True iff placing hard macro idx at (x, y) overlaps no other hard
    macro at its current position. Soft macros never block."""
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
    return not bool(np.any(blockers))


def _is_legal_2d_excluded(
    idx: int,
    x: float,
    y: float,
    placement: torch.Tensor,
    macro_sizes_np: np.ndarray,
    n_hard: int,
    excluded: Sequence[int],
    eps: float = 1e-9,
) -> bool:
    """Like _is_legal_2d but excludes `excluded` indices from the blocker set.

    Used during K-macro joint enumeration: when scoring placement of k_i,
    the other K-tuple members are conceptually destroyed and their current
    positions don't count as blockers.

    `eps` is a small POSITIVE separation margin: the placement must have
    strict separation > eps in at least one axis from every blocker. This
    keeps the check at-least-as-strict-as `compute_overlap_metrics` (which
    flags any positive overlap area). Earlier versions used eps=1e-4 in
    the wrong direction, which tolerated up-to-1e-4 overlap and caused
    `compute_overlap_metrics` to fire on near-zero-area overlaps (E39
    ibm07 crash).
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
    # Blocker iff strict separation < eps in BOTH axes (i.e., overlap area
    # exceeds eps² floor). eps>0 means we DEMAND strict separation.
    blockers = (dx < min_dx + eps) & (dy < min_dy + eps)
    blockers[idx] = False
    for e in excluded:
        if 0 <= e < n_hard:
            blockers[e] = False
    return not bool(np.any(blockers))


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
    """Grid-bin LNS phase: cost-aware destroy + grid-bin reinsert."""
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


# ── SA polish v2 primitive (inlined from E25) ──────────────────────────────


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


# ── K-macro joint LNS (E39 new mechanism) ──────────────────────────────────


def _adjacency_scores(
    evaluator: IncrementalProxyEvaluator,
    hard_movable: List[int],
) -> np.ndarray:
    """Per-macro adjacency = sum_{n in macro_to_nets[m]} 1 / max(1, |net_n| - 1).

    Reuses HPWL net weighting style. High score = highly coupled to many small
    nets. Returned as np.ndarray aligned with `hard_movable` order.
    """
    out = np.zeros(len(hard_movable), dtype=np.float64)
    for i, m in enumerate(hard_movable):
        nets = evaluator.macro_to_nets[m].tolist()
        s = 0.0
        for n in nets:
            sz = int(evaluator.net_pins[n].shape[0])
            denom = max(1, sz - 1)
            s += 1.0 / denom
        out[i] = s
    return out


def _ktuples_pairwise_legal(
    proposed_xy: List[Tuple[float, float]],
    macro_idxs: Sequence[int],
    macro_sizes_np: np.ndarray,
    n_hard: int,
    eps: float = 1e-9,
) -> bool:
    """True iff the K macros, placed at their proposed centers, do not pairwise
    overlap. Hard-macro check only (idx >= n_hard skipped).

    `eps` direction matches `_is_legal_2d_excluded` after the E39 ibm07 fix:
    require STRICT separation > eps in at least one axis. Earlier eps=1e-4
    in the wrong direction tolerated 1e-4 overlap and produced "1 overlap
    area 0.0000" failures in `compute_overlap_metrics`.

    O(K^2) AABB check.
    """
    K = len(macro_idxs)
    for a in range(K):
        ia = macro_idxs[a]
        if ia >= n_hard:
            continue
        xa, ya = proposed_xy[a]
        ha_w = float(macro_sizes_np[ia, 0]) / 2.0
        ha_h = float(macro_sizes_np[ia, 1]) / 2.0
        for b in range(a + 1, K):
            ib = macro_idxs[b]
            if ib >= n_hard:
                continue
            xb, yb = proposed_xy[b]
            hb_w = float(macro_sizes_np[ib, 0]) / 2.0
            hb_h = float(macro_sizes_np[ib, 1]) / 2.0
            min_dx = ha_w + hb_w
            min_dy = ha_h + hb_h
            if abs(xa - xb) < min_dx + eps and abs(ya - yb) < min_dy + eps:
                return False
    return True


def _enumerate_topN_for_macro(
    evaluator: IncrementalProxyEvaluator,
    macro_idx: int,
    plc,
    n_hard: int,
    macro_sizes_np: np.ndarray,
    excluded: Sequence[int],
    top_N: int,
    deadline_s: float,
    t_start: float,
) -> List[Tuple[float, Tuple[float, float]]]:
    """Enumerate every legal grid-bin (col, row) for `macro_idx`, ignoring
    `excluded` macros as blockers. Score by single-macro proxy delta. Return
    top-N (proxy, (cx, cy)) sorted ascending (lowest proxy first).

    Side-effect-neutral: every move is reverted before return.
    """
    cw = float(plc.width)
    ch = float(plc.height)
    grid_w = cw / plc.grid_col
    grid_h = ch / plc.grid_row
    half_w = float(macro_sizes_np[macro_idx, 0]) / 2.0
    half_h = float(macro_sizes_np[macro_idx, 1]) / 2.0

    cur_xy = (
        float(evaluator.placement[macro_idx, 0]),
        float(evaluator.placement[macro_idx, 1]),
    )
    candidates: List[Tuple[float, Tuple[float, float]]] = []
    # Always include current position as a fallback (proxy=baseline) so
    # the cartesian product has at least one valid entry per macro.
    baseline = evaluator.current_cost()["proxy"]
    candidates.append((baseline, cur_xy))

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
            # Same as current — skip (already in candidates).
            if abs(cx - cur_xy[0]) < 1e-9 and abs(cy - cur_xy[1]) < 1e-9:
                continue
            if not _is_legal_2d_excluded(
                macro_idx, cx, cy, evaluator.placement, macro_sizes_np,
                n_hard, excluded,
            ):
                continue
            evaluator.move(macro_idx, (cx, cy))
            p = evaluator.current_cost()["proxy"]
            evaluator.revert()
            candidates.append((p, (cx, cy)))

    candidates.sort(key=lambda e: e[0])
    return candidates[:top_N]


def run_kjoint_lns(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    hard_movable: List[int],
    time_budget_s: float,
    K: int = 3,
    top_N: int = 5,
    seed: int = 42,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """K-macro joint LNS phase.

    Algorithm:
      1. Per-macro adjacency score over `hard_movable`.
      2. Sort hard_movable by adjacency descending.
      3. First pass: walk sorted list in K-tuples (top-K, next-K, ...).
         Subsequent passes: random K-tuples sampled from top-3K.
      4. For each K-tuple, enumerate top-N=5 candidates per macro
         (excluding the K-tuple from blockers), then brute-force
         the N^K combos with pairwise legality + joint apply/revert.
      5. Commit best combo if it improves baseline by > 1e-7. Else
         revert all K and advance.

    Stops when budget exhausted OR a full pass finds no improvement.
    """
    rng = np.random.default_rng(seed=seed)
    macro_sizes_np = evaluator.macro_sizes.cpu().numpy()
    n_hard = benchmark.num_hard_macros

    if not hard_movable or len(hard_movable) < K:
        if log_fn is not None:
            log_fn(
                f"  K-joint: skipping ({len(hard_movable)} hard movables < K={K})"
            )
        return {
            "ktuples_tried": 0, "ktuples_committed": 0, "passes": 0,
            "total_improvement": 0.0, "wall_total_s": 0.0,
        }

    if log_fn is not None:
        log_fn(
            f"  K-joint budget={time_budget_s:.0f}s, K={K}, top_N={top_N}, "
            f"|H|={len(hard_movable)}, seed={seed}"
        )

    # 1. Adjacency.
    adj = _adjacency_scores(evaluator, hard_movable)
    sorted_idx = np.argsort(-adj)
    sorted_movable = [hard_movable[i] for i in sorted_idx]
    if log_fn is not None:
        top5 = sorted_movable[:5]
        top5_adj = adj[sorted_idx[:5]]
        log_fn(
            f"  K-joint top-5 by adjacency: "
            + ", ".join(f"m{m}={a:.3f}" for m, a in zip(top5, top5_adj))
        )

    # K-tuple pool for random pass: top-3K (clamped to len).
    pool_size = min(3 * K, len(sorted_movable))
    pool = sorted_movable[:pool_size]

    t_start = time.perf_counter()
    ktuples_tried = 0
    ktuples_committed = 0
    total_improvement = 0.0
    pass_idx = 0

    # Build the first-pass tuple list: walk sorted_movable in chunks of K.
    first_pass_tuples: List[Tuple[int, ...]] = []
    for s in range(0, len(sorted_movable) - K + 1, K):
        first_pass_tuples.append(tuple(sorted_movable[s : s + K]))

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:
            break
        pass_idx += 1
        pass_t0 = time.perf_counter()
        pass_committed = 0
        pass_delta = 0.0

        # Build this pass's tuple iterator.
        if pass_idx == 1:
            tuples_this_pass = list(first_pass_tuples)
        else:
            # Random K-tuples from the top-3K pool. Generate ~|first_pass|
            # tuples per pass (cap the iteration count comparable to pass 1).
            n_random = max(1, len(first_pass_tuples))
            tuples_this_pass = []
            seen_set = set()
            attempts = 0
            while len(tuples_this_pass) < n_random and attempts < n_random * 4:
                attempts += 1
                if pool_size < K:
                    break
                pick = tuple(sorted(rng.choice(pool, size=K, replace=False).tolist()))
                if pick in seen_set:
                    continue
                seen_set.add(pick)
                tuples_this_pass.append(pick)

        for ktuple in tuples_this_pass:
            if time.perf_counter() - t_start >= time_budget_s:
                break
            ktuples_tried += 1

            # Save baseline positions of the K macros (for revert).
            k_idxs = list(ktuple)
            saved_xy = [
                (
                    float(evaluator.placement[m, 0]),
                    float(evaluator.placement[m, 1]),
                )
                for m in k_idxs
            ]
            baseline_proxy = evaluator.current_cost()["proxy"]

            # Enumerate top-N for each k_i independently.
            per_macro_candidates: List[List[Tuple[float, Tuple[float, float]]]] = []
            enum_failed = False
            for m in k_idxs:
                if time.perf_counter() - t_start >= time_budget_s:
                    enum_failed = True
                    break
                cands = _enumerate_topN_for_macro(
                    evaluator=evaluator,
                    macro_idx=m,
                    plc=plc,
                    n_hard=n_hard,
                    macro_sizes_np=macro_sizes_np,
                    excluded=k_idxs,
                    top_N=top_N,
                    deadline_s=time_budget_s,
                    t_start=t_start,
                )
                if not cands:
                    enum_failed = True
                    break
                per_macro_candidates.append(cands)
            if enum_failed:
                break

            # Brute-force N^K combos.
            best_combo_proxy = baseline_proxy
            best_combo_xy: Optional[List[Tuple[float, float]]] = None

            for combo in itertools.product(*per_macro_candidates):
                if time.perf_counter() - t_start >= time_budget_s:
                    break
                proposed_xy = [c[1] for c in combo]
                # Skip the trivial all-current combo (baseline).
                all_current = all(
                    abs(proposed_xy[i][0] - saved_xy[i][0]) < 1e-9
                    and abs(proposed_xy[i][1] - saved_xy[i][1]) < 1e-9
                    for i in range(K)
                )
                if all_current:
                    continue

                # Pairwise legality among the K macros.
                if not _ktuples_pairwise_legal(
                    proposed_xy, k_idxs, macro_sizes_np, n_hard
                ):
                    continue

                # Apply moves and check non-overlap with the placed background.
                # _is_legal_2d (without exclusion) is checked implicitly by
                # _is_legal_2d_excluded during enumeration — but since pairs
                # of K-tuple members may now block each other once placed,
                # we still need pairwise check above. Background is safe.

                applied: List[int] = []
                joint_proxy = float("inf")
                joint_ok = True
                for i, m in enumerate(k_idxs):
                    nx, ny = proposed_xy[i]
                    cur = (
                        float(evaluator.placement[m, 0]),
                        float(evaluator.placement[m, 1]),
                    )
                    if abs(nx - cur[0]) < 1e-9 and abs(ny - cur[1]) < 1e-9:
                        # No-op for this macro (already at target).
                        applied.append(m)  # tracked for symmetric revert
                        continue
                    try:
                        evaluator.move(m, (nx, ny))
                        applied.append(m)
                    except Exception:
                        joint_ok = False
                        break

                if joint_ok:
                    joint_proxy = evaluator.current_cost()["proxy"]

                # Revert ALL K back to saved_xy (single-step revert is not
                # enough; walk back via explicit move() to keep cache in sync).
                for j in range(len(applied) - 1, -1, -1):
                    m = applied[j]
                    sx, sy = saved_xy[k_idxs.index(m)]
                    cx = float(evaluator.placement[m, 0])
                    cy = float(evaluator.placement[m, 1])
                    if abs(sx - cx) > 1e-9 or abs(sy - cy) > 1e-9:
                        evaluator.move(m, (sx, sy))

                if joint_ok and joint_proxy < best_combo_proxy - 1e-9:
                    best_combo_proxy = joint_proxy
                    best_combo_xy = list(proposed_xy)

            # Commit the best combo if it beats baseline.
            if (
                best_combo_xy is not None
                and best_combo_proxy < baseline_proxy - 1e-7
            ):
                committed_moves: List[Tuple[int, Tuple[float, float]]] = []
                for i, m in enumerate(k_idxs):
                    nx, ny = best_combo_xy[i]
                    cx = float(evaluator.placement[m, 0])
                    cy = float(evaluator.placement[m, 1])
                    if abs(nx - cx) > 1e-9 or abs(ny - cy) > 1e-9:
                        committed_moves.append((m, (cx, cy)))
                        evaluator.move(m, (nx, ny))

                # Defensive: validate against compute_overlap_metrics. The
                # internal _is_legal_2d_excluded / _ktuples_pairwise_legal
                # checks now use eps=1e-9 strict separation, but if a stray
                # float-precision wedge ever opens up again, fail SAFE
                # (revert) instead of corrupting the placement.
                ov = compute_overlap_metrics(evaluator.placement, benchmark)
                if ov["overlap_count"] > 0:
                    if log_fn is not None:
                        log_fn(
                            f"  K-joint: REVERT commit at pass {pass_idx} — "
                            f"compute_overlap_metrics found {ov['overlap_count']} "
                            f"overlap(s), area={ov['total_overlap_area']:.6e}; "
                            f"reverting K={K} move(s)"
                        )
                    # Walk back in reverse order.
                    for j in range(len(committed_moves) - 1, -1, -1):
                        m, (sx, sy) = committed_moves[j]
                        evaluator.move(m, (sx, sy))
                else:
                    delta = best_combo_proxy - baseline_proxy
                    pass_delta += delta
                    total_improvement += delta
                    pass_committed += 1
                    ktuples_committed += 1

        pass_wall = time.perf_counter() - pass_t0
        cur_proxy = evaluator.current_cost()["proxy"]
        if log_fn is not None:
            log_fn(
                f"  K-joint pass {pass_idx}: tuples={len(tuples_this_pass)}, "
                f"committed={pass_committed}, Δ={pass_delta:+.5f}, "
                f"proxy={cur_proxy:.5f}, pass_wall={pass_wall:.1f}s, "
                f"elapsed={time.perf_counter() - t_start:.1f}s"
            )

        # Stop if a full pass produced no commit.
        if pass_committed == 0:
            if log_fn is not None:
                log_fn(f"  K-joint converged at pass {pass_idx} (no improvement)")
            break

    return {
        "ktuples_tried": ktuples_tried,
        "ktuples_committed": ktuples_committed,
        "passes": pass_idx,
        "total_improvement": total_improvement,
        "wall_total_s": time.perf_counter() - t_start,
    }


# ── Placer ─────────────────────────────────────────────────────────────────


class CDLNSSAKJointPlacer:
    """E39 placer: CD + grid-bin LNS + SA-v2 + K-macro joint LNS.

    Time budget per benchmark (≤ 4200 s; over the 3600 s legal cap on
    paper but in practice CD stops well below cap on most benches —
    monitored via cd_hard_cap_s).
      * CD phase: 2400 s.
      * LNS phase: 600 s.
      * SA-v2 phase: 600 s.
      * K-joint phase: 600 s.

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
        kjoint_budget_s: float = 600.0,
        kjoint_K: int = 3,
        kjoint_top_N: int = 5,
        kjoint_seed: int = 42,
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
        self.kjoint_budget_s = float(kjoint_budget_s)
        self.kjoint_K = int(kjoint_K)
        self.kjoint_top_N = int(kjoint_top_N)
        self.kjoint_seed = int(kjoint_seed)
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
            f"=== CDLNSSAKJointPlacer ({benchmark.name}): "
            f"CD cap={self.cd_hard_cap_s:.0f}s, LNS={self.lns_budget_s:.0f}s, "
            f"SA={self.sa_budget_s:.0f}s, KJoint={self.kjoint_budget_s:.0f}s ==="
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
        sa_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  SA-v2 done: lift_vs_init={sa_stats['improvement_vs_init']:+.5f}, "
            f"best={sa_stats['best_proxy']:.5f}, proxy={sa_proxy:.5f}"
        )

        # 7. K-macro joint LNS phase
        self._log(
            f"  starting K-joint phase (budget={self.kjoint_budget_s:.0f}s, "
            f"K={self.kjoint_K}, top_N={self.kjoint_top_N})"
        )
        kj_stats = run_kjoint_lns(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.kjoint_budget_s,
            K=self.kjoint_K,
            top_N=self.kjoint_top_N,
            seed=self.kjoint_seed,
            log_fn=self._log if self.verbose else None,
        )
        final_cost = evaluator.current_cost()
        self._log(
            f"  K-joint done: passes={kj_stats['passes']}, "
            f"tuples_tried={kj_stats['ktuples_tried']}, "
            f"committed={kj_stats['ktuples_committed']}, "
            f"Δ={kj_stats['total_improvement']:+.5f}, "
            f"wall={kj_stats['wall_total_s']:.1f}s, "
            f"final proxy={final_cost['proxy']:.5f}"
        )
        self._log(
            f"  TOTAL lifts: CD={init_cost['proxy'] - cd_proxy:+.5f}, "
            f"LNS={cd_proxy - lns_proxy:+.5f}, "
            f"SA={lns_proxy - sa_proxy:+.5f}, "
            f"KJoint={sa_proxy - final_cost['proxy']:+.5f}"
        )

        # 8. Pull placement back; preserve fixed macros
        final_placement_f64 = evaluator.placement.detach().clone().cpu()
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            final_placement_f64[fixed_mask] = original_positions[fixed_mask]
        final_placement = final_placement_f64.to(torch.float32)

        overlaps = compute_overlap_metrics(final_placement, benchmark)
        if overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"CDLNSSAKJointPlacer produced {overlaps['overlap_count']} "
                f"overlaps (area {overlaps['total_overlap_area']:.4f}) "
                f"on '{benchmark.name}'"
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(SDF + project + CD + LNS + SA + KJoint + validate)"
        )
        return final_placement
