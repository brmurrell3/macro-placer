"""E68 — work-bounded refactor of the E48 hybrid pipeline.

Implements `docs/research_principles_for_walls.md` §4.5 verbatim:

| Phase   | New primary termination                              | New soft wall cap |
|---------|------------------------------------------------------|-------------------|
| CD      | plateau: sweep delta < 0.001 for 3 consecutive sweeps | 3000 s (was 2400) |
| LNS     | 5 consecutive non-improving samples                  | 900 s (was 600)   |
| SA-v2   | no improvement (best-so-far) in last 1000 moves      | 900 s (was 600)   |
| K-joint | no commits in last 30 K-tuples                       | 900 s (was 600)   |

The CD primary is unchanged in semantics — `run_cd_adaptive` already
plateau-exits on `delta < threshold` for `patience` sweeps with
`patience=3, plateau_threshold=0.001`, exactly the §4.5 spec. Only its
soft cap loosens.

LNS / SA / K-joint get genuinely new saturation counters. The bodies
below are flat copies of the parents (E25 `submissions/cd_lns_sa/placer.py`
and E39 `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py`)
with the four termination loci edited in place. Side-by-side diffs
are recorded in `notes.md`.

This file does NOT modify the champion. The champion at
`submissions/cd_lns_sa_hybrid/placer.py` remains the source of truth
until E68 is validated separately.

Reference:
- E48 hybrid (champion): `submissions/cd_lns_sa_hybrid/placer.py`.
- E25 lane (SDF init):    `submissions/cd_lns_sa/placer.py`.
- E41 lane (DPO + Kjoint): `experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`.
- E39 inner LNS/SA/Kjoint primitives:
    `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py`.
- §4.5 mitigation rationale: `docs/research_principles_for_walls.md`.
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

# The eval harness loads placers via importlib.spec_from_file_location, which
# does NOT add the repo root to sys.path. Add it so `macro_place.*` and the
# cross-experiment imports below resolve. This file lives at
# experiments/E68_workbounded_refactor/code/<this>.py — three parents up to repo root.
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
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# DPO best-of-v2 init (lane 2 init step). E18's module registers the repo
# root on import.
from experiments.E18_dpo_init.code.cd_lns_sa_dpo_init import _best_of_v2_init


# ── Harness entry point ────────────────────────────────────────────────────
# evaluate.py picks the first class in vars(mod).values() with a .place()
# method (= module-definition order in Python 3.7+). This tiny wrapper sits
# at the top so the hybrid is selected regardless of where the impl lives.
# Lazy delegation: __init__ is zero-arg; the impl is constructed on first
# place() call, after all helpers and lane classes have been defined.

class CDLNSSAHybridWBPlacer:
    """E68 work-bounded hybrid (entry point).

    Delegates to `_CDLNSSAHybridWBImpl` defined below. See that class for
    the actual lane composition; see file header for the §4.5 termination
    spec applied to each lane.
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def place(self, benchmark):
        return _CDLNSSAHybridWBImpl(**self._kwargs).place(benchmark)


# ── Grid-bin LNS primitives (inlined verbatim from E39 / E25) ──────────────


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
    """True iff placing hard macro idx at (x, y) overlaps no other hard macro."""
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
    """Like _is_legal_2d but excludes `excluded` indices from blockers.

    Used during K-joint enumeration; eps direction matches the post-E39-ibm07
    fix (require strict separation > eps).
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


# ── §4.5 EDIT: LNS termination — 5 consecutive non-improving samples ───────


def run_lns_gridbin_wb(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    hard_movable: List[int],
    time_budget_s: float,
    destroy_frac: float = 0.05,
    destroy_cap: int = 30,
    seed: int = 42,
    lns_patience: int = 5,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Grid-bin LNS — work-bounded variant.

    Original (E39 / E25): break on a single sample where |sample_delta| < 1e-7.
    §4.5: break after `lns_patience=5` consecutive non-improving samples;
    treat the time_budget as a soft secondary cap.

    The "non-improving sample" definition is unchanged: |sample_delta| < 1e-7.
    The change is only that we now require this to hold five times in a row
    before exiting, not once.
    """
    K = max(1, min(destroy_cap, int(destroy_frac * len(hard_movable))))
    macro_sizes_np = evaluator.macro_sizes.cpu().numpy()
    n_hard = benchmark.num_hard_macros

    if log_fn is not None:
        log_fn(
            f"  LNS-WB budget={time_budget_s:.0f}s (soft), patience={lns_patience}, "
            f"destroy K={K} (={destroy_frac*100:.1f}% of {len(hard_movable)} hard "
            f"movables, capped at {destroy_cap}), strategy=cost_aware"
        )

    sample = 0
    total_improvement = 0.0
    consecutive_no_improve = 0  # §4.5: saturation counter
    t_start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:  # §4.5: soft secondary cap
            if log_fn is not None:
                log_fn(f"  LNS-WB soft cap hit at sample {sample} (wall {elapsed:.1f}s)")
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

        # §4.5: track consecutive non-improving samples (was: break on first).
        if abs(sample_delta) < 1e-7:
            consecutive_no_improve += 1
        else:
            consecutive_no_improve = 0

        if log_fn is not None:
            log_fn(
                f"  LNS-WB sample {sample}: K={K}, Δ={sample_delta:+.5f} "
                f"(moves={moves_this_sample}/{K}), proxy={cur_proxy:.5f}, "
                f"no-improve-streak={consecutive_no_improve}/{lns_patience}, "
                f"sample_wall={sample_wall:.1f}s, total elapsed={elapsed:.1f}s"
            )

        if consecutive_no_improve >= lns_patience:
            if log_fn is not None:
                log_fn(
                    f"  LNS-WB saturated at sample {sample} "
                    f"({lns_patience} consecutive non-improving samples)"
                )
            break

    return {
        "samples": sample,
        "total_improvement": total_improvement,
        "wall_total_s": time.perf_counter() - t_start,
    }


# ── §4.5 EDIT: SA-v2 termination — no improvement in last 1000 moves ───────


def run_sa_polish_v2_wb(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    hard_movable: List[int],
    time_budget_s: float,
    T0: float = 5e-4,
    Tf: float = 1e-6,
    seed: int = 42,
    breakpoint_budget: int = 12,
    sa_no_improve_moves: int = 1000,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """SA polish v2 — work-bounded variant.

    Original (E39 / E25): single termination on `elapsed >= time_budget_s`.
    §4.5: terminate when no best-so-far improvement has occurred in the last
    `sa_no_improve_moves=1000` PROPOSED moves; treat time_budget as soft cap.

    "Move" = a proposed Metropolis step (counted regardless of acceptance,
    matching the chain semantics; skipped iterations don't count). The
    counter resets whenever `cur_proxy < best_proxy - 1e-12` improves the
    best-so-far. This matches the existing best-tracking code path.
    """
    rng = np.random.default_rng(seed=seed)
    grid_lines_x, grid_lines_y = _grid_lines(plc)
    n_hard = benchmark.num_hard_macros

    if not hard_movable:
        if log_fn is not None:
            log_fn("  SA-WB: no hard movable macros; skipping")
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
            f"  SA-WB v2 budget={time_budget_s:.0f}s (soft), T0={T0:.2e}, "
            f"Tf={Tf:.2e}, |H|={len(hard_movable)}, seed={seed}, "
            f"no-improve-moves={sa_no_improve_moves} (best-so-far tracking ON)"
        )

    proposed = 0
    accepted_better = 0
    accepted_worse = 0
    rejected = 0
    skipped = 0
    moves_since_best = 0  # §4.5: saturation counter on best-so-far
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
        if elapsed >= time_budget_s:  # §4.5: soft secondary cap
            if log_fn is not None:
                log_fn(f"  SA-WB soft cap hit (wall {elapsed:.1f}s)")
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

        # §4.5: track moves since last best-so-far improvement.
        if accepted and cur_proxy < best_proxy - 1e-12:
            best_proxy = cur_proxy
            best_placement = evaluator.placement.detach().clone()
            best_found_at_t = time.perf_counter() - t_start
            moves_since_best = 0
        else:
            moves_since_best += 1

        if moves_since_best >= sa_no_improve_moves:
            if log_fn is not None:
                log_fn(
                    f"  SA-WB saturated: {moves_since_best} proposed moves since "
                    f"last best-so-far improvement (best={best_proxy:.5f}); exiting "
                    f"at proposed={proposed}, t={time.perf_counter() - t_start:.1f}s"
                )
            break

        now = time.perf_counter()
        if log_fn is not None and (now - last_log_t) >= 30.0:
            last_log_t = now
            log_fn(
                f"  SA-WB v2 t={now - t_start:6.1f}s T={T:.2e} "
                f"proposed={proposed} better={accepted_better} "
                f"worse={accepted_worse} rejected={rejected} "
                f"skipped={skipped} cur={cur_proxy:.5f} best={best_proxy:.5f} "
                f"moves_since_best={moves_since_best}"
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
            f"  SA-WB v2 done: proposed={proposed}, better={accepted_better}, "
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


# ── K-joint LNS primitives (inlined verbatim from E39) ─────────────────────


def _adjacency_scores(
    evaluator: IncrementalProxyEvaluator,
    hard_movable: List[int],
) -> np.ndarray:
    """Per-macro adjacency = sum_{n in macro_to_nets[m]} 1 / max(1, |net_n| - 1)."""
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
    """O(K^2) AABB pairwise check on the K proposed positions."""
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
    """Per-macro top-N legal grid-bin enumeration (excluded macros ignored)."""
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


# ── §4.5 EDIT: K-joint termination — no commits in last 30 K-tuples ────────


def run_kjoint_lns_wb(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    hard_movable: List[int],
    time_budget_s: float,
    K: int = 3,
    top_N: int = 5,
    seed: int = 42,
    kjoint_no_commit_tuples: int = 30,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """K-macro joint LNS — work-bounded variant.

    Original (E39): break the outer loop when a full pass produces zero
    commits, OR when the soft cap fires.
    §4.5: break when `kjoint_no_commit_tuples=30` consecutive K-tuples
    have been tried without any commit (sliding window, doesn't reset
    on pass boundaries); soft cap secondary.

    Per-pass build (deterministic first pass walking the adjacency-sorted
    list, then random tuples from the top-3K pool) is unchanged.
    """
    rng = np.random.default_rng(seed=seed)
    macro_sizes_np = evaluator.macro_sizes.cpu().numpy()
    n_hard = benchmark.num_hard_macros

    if not hard_movable or len(hard_movable) < K:
        if log_fn is not None:
            log_fn(
                f"  K-joint-WB: skipping ({len(hard_movable)} hard movables < K={K})"
            )
        return {
            "ktuples_tried": 0, "ktuples_committed": 0, "passes": 0,
            "total_improvement": 0.0, "wall_total_s": 0.0,
        }

    if log_fn is not None:
        log_fn(
            f"  K-joint-WB budget={time_budget_s:.0f}s (soft), K={K}, top_N={top_N}, "
            f"|H|={len(hard_movable)}, seed={seed}, "
            f"no-commit-tuples={kjoint_no_commit_tuples}"
        )

    # 1. Adjacency.
    adj = _adjacency_scores(evaluator, hard_movable)
    sorted_idx = np.argsort(-adj)
    sorted_movable = [hard_movable[i] for i in sorted_idx]
    if log_fn is not None:
        top5 = sorted_movable[:5]
        top5_adj = adj[sorted_idx[:5]]
        log_fn(
            f"  K-joint-WB top-5 by adjacency: "
            + ", ".join(f"m{m}={a:.3f}" for m, a in zip(top5, top5_adj))
        )

    pool_size = min(3 * K, len(sorted_movable))
    pool = sorted_movable[:pool_size]

    t_start = time.perf_counter()
    ktuples_tried = 0
    ktuples_committed = 0
    total_improvement = 0.0
    pass_idx = 0
    consecutive_no_commit = 0  # §4.5: sliding saturation counter (across passes)
    saturated = False

    # Build the first-pass tuple list: walk sorted_movable in chunks of K.
    first_pass_tuples: List[Tuple[int, ...]] = []
    for s in range(0, len(sorted_movable) - K + 1, K):
        first_pass_tuples.append(tuple(sorted_movable[s : s + K]))

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:  # §4.5: soft secondary cap
            if log_fn is not None:
                log_fn(
                    f"  K-joint-WB soft cap hit at pass {pass_idx} "
                    f"(wall {elapsed:.1f}s)"
                )
            break
        pass_idx += 1
        pass_t0 = time.perf_counter()
        pass_committed = 0
        pass_delta = 0.0

        # Build this pass's tuple iterator.
        if pass_idx == 1:
            tuples_this_pass = list(first_pass_tuples)
        else:
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

            # Save baseline positions.
            k_idxs = list(ktuple)
            saved_xy = [
                (
                    float(evaluator.placement[m, 0]),
                    float(evaluator.placement[m, 1]),
                )
                for m in k_idxs
            ]
            baseline_proxy = evaluator.current_cost()["proxy"]

            # Enumerate top-N for each k_i.
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
                all_current = all(
                    abs(proposed_xy[i][0] - saved_xy[i][0]) < 1e-9
                    and abs(proposed_xy[i][1] - saved_xy[i][1]) < 1e-9
                    for i in range(K)
                )
                if all_current:
                    continue

                if not _ktuples_pairwise_legal(
                    proposed_xy, k_idxs, macro_sizes_np, n_hard
                ):
                    continue

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
                        applied.append(m)
                        continue
                    try:
                        evaluator.move(m, (nx, ny))
                        applied.append(m)
                    except Exception:
                        joint_ok = False
                        break

                if joint_ok:
                    joint_proxy = evaluator.current_cost()["proxy"]

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
            committed_this_tuple = False
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

                ov = compute_overlap_metrics(evaluator.placement, benchmark)
                if ov["overlap_count"] > 0:
                    if log_fn is not None:
                        log_fn(
                            f"  K-joint-WB: REVERT commit at pass {pass_idx} — "
                            f"compute_overlap_metrics found {ov['overlap_count']} "
                            f"overlap(s), area={ov['total_overlap_area']:.6e}; "
                            f"reverting K={K} move(s)"
                        )
                    for j in range(len(committed_moves) - 1, -1, -1):
                        m, (sx, sy) = committed_moves[j]
                        evaluator.move(m, (sx, sy))
                else:
                    delta = best_combo_proxy - baseline_proxy
                    pass_delta += delta
                    total_improvement += delta
                    pass_committed += 1
                    ktuples_committed += 1
                    committed_this_tuple = True

            # §4.5: sliding saturation across passes (counter does NOT reset
            # on pass boundary; only on commit).
            if committed_this_tuple:
                consecutive_no_commit = 0
            else:
                consecutive_no_commit += 1

            if consecutive_no_commit >= kjoint_no_commit_tuples:
                if log_fn is not None:
                    log_fn(
                        f"  K-joint-WB saturated at pass {pass_idx}, tuple "
                        f"{ktuples_tried}: {kjoint_no_commit_tuples} consecutive "
                        f"K-tuples without commit"
                    )
                saturated = True
                break

        pass_wall = time.perf_counter() - pass_t0
        cur_proxy = evaluator.current_cost()["proxy"]
        if log_fn is not None:
            log_fn(
                f"  K-joint-WB pass {pass_idx}: tuples={len(tuples_this_pass)}, "
                f"committed={pass_committed}, Δ={pass_delta:+.5f}, "
                f"proxy={cur_proxy:.5f}, pass_wall={pass_wall:.1f}s, "
                f"elapsed={time.perf_counter() - t_start:.1f}s, "
                f"no-commit-streak={consecutive_no_commit}/{kjoint_no_commit_tuples}"
            )

        if saturated:
            break

    return {
        "ktuples_tried": ktuples_tried,
        "ktuples_committed": ktuples_committed,
        "passes": pass_idx,
        "total_improvement": total_improvement,
        "wall_total_s": time.perf_counter() - t_start,
    }


# ── Lane 1: SDF-init pipeline (E25-style, work-bounded) ────────────────────


class CDLNSSAPlacerWB:
    """E25 lane with §4.5 work-bounded termination.

    Pipeline: SDF init → project → CD → LNS-WB → SA-WB.
    Soft caps: CD 3000s, LNS 900s, SA 900s.
    """

    def __init__(
        self,
        cd_hard_cap_s: float = 3000.0,        # §4.5: was 2400.0 (soft cap)
        cd_min_time_s: float = 300.0,
        cd_patience: int = 3,                  # §4.5 spec: 3 consecutive sweeps
        cd_plateau_threshold: float = 0.001,   # §4.5 spec: delta < 0.001
        lns_budget_s: float = 900.0,           # §4.5: was 600.0 (soft cap)
        lns_destroy_frac: float = 0.05,
        lns_destroy_cap: int = 30,
        lns_seed: int = 42,
        lns_patience: int = 5,                 # §4.5 spec: 5 consecutive non-improving
        sa_budget_s: float = 900.0,            # §4.5: was 600.0 (soft cap)
        sa_T0: float = 5e-4,
        sa_Tf: float = 1e-6,
        sa_seed: int = 42,
        sa_breakpoint_budget: int = 12,
        sa_no_improve_moves: int = 1000,       # §4.5 spec: 1000 moves
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
        self.lns_patience = int(lns_patience)
        self.sa_budget_s = float(sa_budget_s)
        self.sa_T0 = float(sa_T0)
        self.sa_Tf = float(sa_Tf)
        self.sa_seed = int(sa_seed)
        self.sa_breakpoint_budget = int(sa_breakpoint_budget)
        self.sa_no_improve_moves = int(sa_no_improve_moves)
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
            f"=== CDLNSSAPlacerWB ({benchmark.name}): "
            f"CD soft cap={self.cd_hard_cap_s:.0f}s, "
            f"LNS={self.lns_budget_s:.0f}s, SA={self.sa_budget_s:.0f}s ==="
        )
        self._log(
            f"  num_macros={benchmark.num_macros}, "
            f"n_hard={benchmark.num_hard_macros}, "
            f"canvas={benchmark.canvas_width:.1f}x{benchmark.canvas_height:.1f}"
        )

        # 1. SDF init.
        t_init0 = time.perf_counter()
        placement = sdf_init(benchmark)
        self._log(f"  SDF init wall = {time.perf_counter() - t_init0:.1f} s")

        # 2. Project overlaps.
        placement, proj_iters = project_overlaps(placement, benchmark)
        init_overlaps = compute_overlap_metrics(placement, benchmark)
        self._log(
            f"  overlap projection: {proj_iters} iters, "
            f"residual count={init_overlaps['overlap_count']}"
        )

        # 3. Build evaluator.
        _, plc = self._load_plc_for(benchmark)
        placement_f64 = placement.detach().clone().to(torch.float64)
        evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
        init_cost = evaluator.current_cost()
        self._log(
            f"  init proxy={init_cost['proxy']:.5f} "
            f"[wl={init_cost['wl']:.4f} d={init_cost['density']:.4f} "
            f"c={init_cost['congestion']:.4f}]"
        )

        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        hard_movable = [
            i for i in range(benchmark.num_hard_macros)
            if not bool(benchmark.macro_fixed[i])
        ]

        # 4. CD phase. Plateau check unchanged (matches §4.5); only cap loosens.
        self._log(f"  starting CD phase (soft cap={self.cd_hard_cap_s:.0f}s)")
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

        # 5. LNS-WB phase.
        self._log(
            f"  starting LNS-WB phase (soft budget={self.lns_budget_s:.0f}s, "
            f"patience={self.lns_patience})"
        )
        lns_stats = run_lns_gridbin_wb(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.lns_budget_s,
            destroy_frac=self.lns_destroy_frac,
            destroy_cap=self.lns_destroy_cap,
            seed=self.lns_seed,
            lns_patience=self.lns_patience,
            log_fn=self._log if self.verbose else None,
        )
        lns_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  LNS-WB done: samples={lns_stats['samples']}, "
            f"Δ={lns_stats['total_improvement']:+.5f}, "
            f"wall={lns_stats['wall_total_s']:.1f}s, proxy={lns_proxy:.5f}"
        )

        # 6. SA-WB v2 phase.
        self._log(
            f"  starting SA-WB v2 phase (soft budget={self.sa_budget_s:.0f}s, "
            f"no-improve-moves={self.sa_no_improve_moves})"
        )
        sa_stats = run_sa_polish_v2_wb(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.sa_budget_s,
            T0=self.sa_T0,
            Tf=self.sa_Tf,
            seed=self.sa_seed,
            breakpoint_budget=self.sa_breakpoint_budget,
            sa_no_improve_moves=self.sa_no_improve_moves,
            log_fn=self._log if self.verbose else None,
        )
        final_cost = evaluator.current_cost()
        self._log(
            f"  SA-WB v2 done: lift_vs_init={sa_stats['improvement_vs_init']:+.5f}, "
            f"best={sa_stats['best_proxy']:.5f}, "
            f"final proxy={final_cost['proxy']:.5f}"
        )
        self._log(
            f"  TOTAL lifts: CD={init_cost['proxy'] - cd_proxy:+.5f}, "
            f"LNS={cd_proxy - lns_proxy:+.5f}, "
            f"SA={lns_proxy - final_cost['proxy']:+.5f}"
        )

        # 7. Pull placement back; preserve fixed macros.
        final_placement_f64 = evaluator.placement.detach().clone().cpu()
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            final_placement_f64[fixed_mask] = original_positions[fixed_mask]
        final_placement = final_placement_f64.to(torch.float32)

        overlaps = compute_overlap_metrics(final_placement, benchmark)
        if overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"CDLNSSAPlacerWB produced {overlaps['overlap_count']} overlaps "
                f"(area {overlaps['total_overlap_area']:.4f}) on '{benchmark.name}'"
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(SDF + project + CD + LNS-WB + SA-WB + validate)"
        )
        return final_placement


# ── Lane 2: DPO-init + K-joint pipeline (E41-style, work-bounded) ──────────


class CDLNSSADPOKJointPlacerWB:
    """E41 lane with §4.5 work-bounded termination.

    Pipeline: DPO best_of_v2 init → project → CD → LNS-WB → SA-WB → KJoint-WB.
    Soft caps: CD 3000s, LNS 900s, SA 900s, K-joint 900s.
    """

    def __init__(
        self,
        cd_hard_cap_s: float = 3000.0,        # §4.5
        cd_min_time_s: float = 300.0,
        cd_patience: int = 3,
        cd_plateau_threshold: float = 0.001,
        lns_budget_s: float = 900.0,          # §4.5
        lns_destroy_frac: float = 0.05,
        lns_destroy_cap: int = 30,
        lns_seed: int = 42,
        lns_patience: int = 5,                # §4.5
        sa_budget_s: float = 900.0,           # §4.5
        sa_T0: float = 5e-4,
        sa_Tf: float = 1e-6,
        sa_seed: int = 42,
        sa_breakpoint_budget: int = 12,
        sa_no_improve_moves: int = 1000,      # §4.5
        kjoint_budget_s: float = 900.0,       # §4.5
        kjoint_K: int = 3,
        kjoint_top_N: int = 5,
        kjoint_seed: int = 42,
        kjoint_no_commit_tuples: int = 30,    # §4.5
        seed: int = 42,
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
        self.lns_patience = int(lns_patience)
        self.sa_budget_s = float(sa_budget_s)
        self.sa_T0 = float(sa_T0)
        self.sa_Tf = float(sa_Tf)
        self.sa_seed = int(sa_seed)
        self.sa_breakpoint_budget = int(sa_breakpoint_budget)
        self.sa_no_improve_moves = int(sa_no_improve_moves)
        self.kjoint_budget_s = float(kjoint_budget_s)
        self.kjoint_K = int(kjoint_K)
        self.kjoint_top_N = int(kjoint_top_N)
        self.kjoint_seed = int(kjoint_seed)
        self.kjoint_no_commit_tuples = int(kjoint_no_commit_tuples)
        self.seed = int(seed)
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
            f"=== CDLNSSADPOKJointPlacerWB ({benchmark.name}): "
            f"DPO -> CD soft cap={self.cd_hard_cap_s:.0f}s -> "
            f"LNS={self.lns_budget_s:.0f}s -> SA={self.sa_budget_s:.0f}s -> "
            f"KJoint={self.kjoint_budget_s:.0f}s ==="
        )
        self._log(
            f"  num_macros={benchmark.num_macros}, "
            f"n_hard={benchmark.num_hard_macros}, "
            f"canvas={benchmark.canvas_width:.1f}x{benchmark.canvas_height:.1f}"
        )

        # 1. DPO best_of_v2 init.
        _, plc = self._load_plc_for(benchmark)
        t_init0 = time.perf_counter()
        placement = _best_of_v2_init(
            benchmark, plc, seed=self.seed,
            log_fn=self._log if self.verbose else None,
        )
        self._log(f"  DPO init wall = {time.perf_counter() - t_init0:.1f} s")

        # 2. Project overlaps.
        placement, proj_iters = project_overlaps(placement, benchmark)
        init_overlaps = compute_overlap_metrics(placement, benchmark)
        self._log(
            f"  overlap projection: {proj_iters} iters, "
            f"residual count={init_overlaps['overlap_count']}"
        )

        # 3. Build evaluator.
        placement_f64 = placement.detach().clone().to(torch.float64)
        evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
        init_cost = evaluator.current_cost()
        self._log(
            f"  init proxy={init_cost['proxy']:.5f} "
            f"[wl={init_cost['wl']:.4f} d={init_cost['density']:.4f} "
            f"c={init_cost['congestion']:.4f}]"
        )

        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        hard_movable = [
            i for i in range(benchmark.num_hard_macros)
            if not bool(benchmark.macro_fixed[i])
        ]

        # 4. CD phase.
        self._log(f"  starting CD phase (soft cap={self.cd_hard_cap_s:.0f}s)")
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

        # 5. LNS-WB phase.
        self._log(
            f"  starting LNS-WB phase (soft budget={self.lns_budget_s:.0f}s, "
            f"patience={self.lns_patience})"
        )
        lns_stats = run_lns_gridbin_wb(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.lns_budget_s,
            destroy_frac=self.lns_destroy_frac,
            destroy_cap=self.lns_destroy_cap,
            seed=self.lns_seed,
            lns_patience=self.lns_patience,
            log_fn=self._log if self.verbose else None,
        )
        lns_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  LNS-WB done: samples={lns_stats['samples']}, "
            f"Δ={lns_stats['total_improvement']:+.5f}, "
            f"wall={lns_stats['wall_total_s']:.1f}s, proxy={lns_proxy:.5f}"
        )

        # 6. SA-WB v2 phase.
        self._log(
            f"  starting SA-WB v2 phase (soft budget={self.sa_budget_s:.0f}s, "
            f"no-improve-moves={self.sa_no_improve_moves})"
        )
        sa_stats = run_sa_polish_v2_wb(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.sa_budget_s,
            T0=self.sa_T0,
            Tf=self.sa_Tf,
            seed=self.sa_seed,
            breakpoint_budget=self.sa_breakpoint_budget,
            sa_no_improve_moves=self.sa_no_improve_moves,
            log_fn=self._log if self.verbose else None,
        )
        sa_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  SA-WB v2 done: lift_vs_init={sa_stats['improvement_vs_init']:+.5f}, "
            f"best={sa_stats['best_proxy']:.5f}, proxy={sa_proxy:.5f}"
        )

        # 7. K-joint-WB phase.
        self._log(
            f"  starting K-joint-WB phase (soft budget={self.kjoint_budget_s:.0f}s, "
            f"K={self.kjoint_K}, top_N={self.kjoint_top_N}, "
            f"no-commit-tuples={self.kjoint_no_commit_tuples})"
        )
        kj_stats = run_kjoint_lns_wb(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.kjoint_budget_s,
            K=self.kjoint_K,
            top_N=self.kjoint_top_N,
            seed=self.kjoint_seed,
            kjoint_no_commit_tuples=self.kjoint_no_commit_tuples,
            log_fn=self._log if self.verbose else None,
        )
        final_cost = evaluator.current_cost()
        self._log(
            f"  K-joint-WB done: passes={kj_stats['passes']}, "
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

        # 8. Pull placement back; preserve fixed macros.
        final_placement_f64 = evaluator.placement.detach().clone().cpu()
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            final_placement_f64[fixed_mask] = original_positions[fixed_mask]
        final_placement = final_placement_f64.to(torch.float32)

        overlaps = compute_overlap_metrics(final_placement, benchmark)
        if overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"CDLNSSADPOKJointPlacerWB produced {overlaps['overlap_count']} "
                f"overlaps (area {overlaps['total_overlap_area']:.4f}) on "
                f"'{benchmark.name}'"
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(DPO + project + CD + LNS-WB + SA-WB + KJoint-WB + validate)"
        )
        return final_placement


# ── Hybrid wrapper (E48-style) ─────────────────────────────────────────────


class _CDLNSSAHybridWBImpl:
    """E68 hybrid impl: run E25-WB and E41-WB lanes, return per-bench best.

    Mirrors `submissions/cd_lns_sa_hybrid/placer.py` exactly except both
    lanes use the §4.5 work-bounded termination. The selection is by
    proxy value from `compute_proxy_cost` against zero-overlap outputs;
    no per-benchmark hardcoded logic.

    Wrapped by `CDLNSSAHybridWBPlacer` (top of file) so the harness can
    pick the hybrid class without depending on definition order.
    """

    def __init__(self, **kwargs):
        # Per-lane kwargs split: kjoint_* go to E41 only; the rest are common.
        e41_only_keys = {
            "kjoint_K", "kjoint_top_N", "kjoint_budget_s", "kjoint_seed",
            "kjoint_no_commit_tuples",
        }
        e41_kwargs = {k: v for k, v in kwargs.items() if k in e41_only_keys}
        common_kwargs = {k: v for k, v in kwargs.items() if k not in e41_only_keys}
        self._e25 = CDLNSSAPlacerWB(**common_kwargs)
        self._e41 = CDLNSSADPOKJointPlacerWB(**common_kwargs, **e41_kwargs)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True)
        log("=== E68 HYBRID-WB placer: running E25-WB then E41-WB, "
            "returning lower-proxy output ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Run E25-WB lane.
        t0 = time.perf_counter()
        e25_placement = self._e25.place(benchmark)
        e25_wall = time.perf_counter() - t0
        e25_proxy = compute_proxy_cost(e25_placement, benchmark, plc)
        e25_ov = compute_overlap_metrics(e25_placement, benchmark)
        log(
            f"  E25-WB done: proxy={e25_proxy['proxy_cost']:.5f} "
            f"overlaps={e25_ov['overlap_count']} wall={e25_wall:.1f}s"
        )

        # Run E41-WB lane (re-runs DPO + CD + LNS + SA + K-joint from scratch).
        t1 = time.perf_counter()
        e41_placement = self._e41.place(benchmark)
        e41_wall = time.perf_counter() - t1
        e41_proxy = compute_proxy_cost(e41_placement, benchmark, plc)
        e41_ov = compute_overlap_metrics(e41_placement, benchmark)
        log(
            f"  E41-WB done: proxy={e41_proxy['proxy_cost']:.5f} "
            f"overlaps={e41_ov['overlap_count']} wall={e41_wall:.1f}s"
        )

        # Pick the lower-proxy zero-overlap output.
        e25_valid = e25_ov["overlap_count"] == 0
        e41_valid = e41_ov["overlap_count"] == 0

        if e25_valid and e41_valid:
            if e25_proxy["proxy_cost"] <= e41_proxy["proxy_cost"]:
                winner = "E25-WB"
                final = e25_placement
                final_proxy = e25_proxy["proxy_cost"]
            else:
                winner = "E41-WB"
                final = e41_placement
                final_proxy = e41_proxy["proxy_cost"]
        elif e25_valid:
            winner = "E25-WB (E41-WB had overlaps)"
            final = e25_placement
            final_proxy = e25_proxy["proxy_cost"]
        elif e41_valid:
            winner = "E41-WB (E25-WB had overlaps)"
            final = e41_placement
            final_proxy = e41_proxy["proxy_cost"]
        else:
            raise RuntimeError(
                f"E68 hybrid: BOTH pipelines produced overlapping placements "
                f"(E25-WB {e25_ov['overlap_count']} overlaps, "
                f"E41-WB {e41_ov['overlap_count']} overlaps)"
            )
        total_wall = time.perf_counter() - t0
        log(
            f"  E68 winner: {winner} proxy={final_proxy:.5f} "
            f"(E25-WB={e25_proxy['proxy_cost']:.5f}, "
            f"E41-WB={e41_proxy['proxy_cost']:.5f}) "
            f"total_wall={total_wall:.1f}s"
        )
        return final
