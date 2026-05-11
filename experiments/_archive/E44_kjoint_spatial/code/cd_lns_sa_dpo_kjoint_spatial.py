"""E44 — CDLNSSA + DPO + K=3 K-joint with SPATIAL-adjacency K-tuple selection.

E41 (champion candidate, 1.0848 --all) selects K-tuples by walking a list
sorted by *netlist* adjacency (sum of 1/(net_size-1) over each macro's
nets). E44 replaces this with *spatial* adjacency: K-tuples are formed
from spatially-clustered macros — for each macro, take it plus its K-1
nearest neighbors in the current placement.

Hypothesis: K-joint enumeration's brute-force inner loop (N^K combos
per K-tuple, pairwise non-overlap check) is most productive when the K
macros are spatially proximate. Otherwise the cartesian product of
their independent top-N candidate cells doesn't include many good
combinations. E27 placement-cluster analysis confirmed that ibm14/15
(where E41 K-joint lift was modest) have DPO and SDF in the same
spatial cluster, while ibm11 (where E41 K-joint was -4 %) has DPO and
SDF spatially distinct.

Same pipeline as E41: DPO best_of_v2 init -> CD plateau -> grid-bin
LNS -> SA-v2 polish -> K-joint LNS (K=3, top_N=5, 600 s budget) ->
validate. Only the K-tuple SELECTION inside K-joint differs.

Implementation: monkey-patches the K-tuple builder via a thin wrapper
around CDLNSSADPOKJointPlacer. The patch operates on E41's module
attribute `run_kjoint_lns` (overriding it on the imported module
namespace) so that when E41's `place()` calls `run_kjoint_lns`, our
spatial version runs instead. The patch is applied per-instance and
restored on `__del__`, so other E41-using placers in the same process
are unaffected.

Reference:
- E41 — netlist-adjacency K-tuple parent.
- E39 — K-joint primitive.
- E27 — basin-cluster diagnostic motivating the spatial mechanism.
"""
from __future__ import annotations

import sys
import time
import itertools
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics

# Reuse all helper functions from E39's K-joint module.
from experiments.E39_kmacro_joint_lns.code.cd_lns_sa_kjoint import (
    _enumerate_topN_for_macro,
    _is_legal_2d_excluded,
    _ktuples_pairwise_legal,
)
from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import (
    CDLNSSADPOKJointPlacer,
)
import experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint as _e41_mod


def _spatial_density_scores(
    evaluator: IncrementalProxyEvaluator,
    hard_movable: List[int],
    k_neighbors: int = 10,
) -> np.ndarray:
    """Per-macro spatial density: sum of 1/(d²+ε) to k=10 nearest neighbors
    in the current placement. High score = clustered with many neighbors.
    """
    n = len(hard_movable)
    pos = evaluator.placement.cpu().numpy().astype(np.float64)
    out = np.zeros(n, dtype=np.float64)
    eps = 1e-9
    for i in range(n):
        m = hard_movable[i]
        d2 = np.zeros(n, dtype=np.float64)
        for j in range(n):
            if j == i:
                d2[j] = np.inf
                continue
            other = hard_movable[j]
            dx = pos[m, 0] - pos[other, 0]
            dy = pos[m, 1] - pos[other, 1]
            d2[j] = dx * dx + dy * dy
        # k smallest distances (excluding self).
        if n - 1 <= k_neighbors:
            sel = d2[d2 < np.inf]
        else:
            sel = np.partition(d2, k_neighbors)[:k_neighbors]
        out[i] = float(np.sum(1.0 / (sel + eps)))
    return out


def _spatial_first_pass_tuples(
    evaluator: IncrementalProxyEvaluator,
    hard_movable: List[int],
    K: int,
) -> List[Tuple[int, ...]]:
    """First-pass K-tuples: for each macro, form tuple with its K-1 nearest
    neighbors by L2 in current placement. Deduplicate via sorted-tuple set.
    """
    pos = evaluator.placement.cpu().numpy().astype(np.float64)
    n = len(hard_movable)
    seen = set()
    tuples: List[Tuple[int, ...]] = []
    for i in range(n):
        m = hard_movable[i]
        # distance from m to all other hard movables
        ds = []
        for j in range(n):
            if j == i:
                continue
            other = hard_movable[j]
            dx = pos[m, 0] - pos[other, 0]
            dy = pos[m, 1] - pos[other, 1]
            ds.append((dx * dx + dy * dy, other))
        ds.sort()
        nearest = [other for _, other in ds[: K - 1]]
        ktuple = tuple(sorted([m] + nearest))
        if ktuple not in seen:
            seen.add(ktuple)
            tuples.append(ktuple)
    return tuples


def run_kjoint_lns_spatial(
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
    """K-macro joint LNS phase with SPATIAL K-tuple selection.

    Same outer loop and inner enumeration as E39's run_kjoint_lns; only
    the K-tuple selection differs:
    - First pass: spatial-nearest-neighbor K-tuples per macro.
    - Subsequent passes: random K-tuples from the top-3K macros by
      *spatial density* (instead of net adjacency).

    eps direction matches E39's post-fix (eps=1e-9, strict separation in
    `dx < min + eps`); post-commit compute_overlap_metrics defensive
    revert preserved.
    """
    rng = np.random.default_rng(seed=seed)
    macro_sizes_np = evaluator.macro_sizes.cpu().numpy()
    n_hard = benchmark.num_hard_macros

    if not hard_movable or len(hard_movable) < K:
        if log_fn is not None:
            log_fn(
                f"  K-joint (spatial): skipping ({len(hard_movable)} hard movables < K={K})"
            )
        return {
            "ktuples_tried": 0, "ktuples_committed": 0, "passes": 0,
            "total_improvement": 0.0, "wall_total_s": 0.0,
        }

    if log_fn is not None:
        log_fn(
            f"  K-joint (spatial) budget={time_budget_s:.0f}s, K={K}, top_N={top_N}, "
            f"|H|={len(hard_movable)}, seed={seed}"
        )

    # 1. Spatial density scores (for the random-pass pool ordering).
    density = _spatial_density_scores(evaluator, hard_movable, k_neighbors=10)
    sorted_idx = np.argsort(-density)
    sorted_movable = [hard_movable[i] for i in sorted_idx]
    if log_fn is not None:
        top5 = sorted_movable[:5]
        top5_density = density[sorted_idx[:5]]
        log_fn(
            f"  K-joint (spatial) top-5 by density: "
            + ", ".join(f"m{m}={d:.3f}" for m, d in zip(top5, top5_density))
        )

    pool_size = min(3 * K, len(sorted_movable))
    pool = sorted_movable[:pool_size]

    t_start = time.perf_counter()
    ktuples_tried = 0
    ktuples_committed = 0
    total_improvement = 0.0
    pass_idx = 0

    # First-pass tuples: spatial-nearest-neighbors per macro.
    first_pass_tuples = _spatial_first_pass_tuples(evaluator, hard_movable, K)
    if log_fn is not None:
        log_fn(f"  K-joint (spatial) first-pass: {len(first_pass_tuples)} unique K-tuples")

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:
            break
        pass_idx += 1
        pass_t0 = time.perf_counter()
        pass_committed = 0
        pass_delta = 0.0

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

            k_idxs = list(ktuple)
            saved_xy = [
                (
                    float(evaluator.placement[m, 0]),
                    float(evaluator.placement[m, 1]),
                )
                for m in k_idxs
            ]
            baseline_proxy = evaluator.current_cost()["proxy"]

            per_macro_candidates: List[List[Tuple[float, Tuple[float, float]]]] = []
            enum_failed = False
            for m in k_idxs:
                if time.perf_counter() - t_start >= time_budget_s:
                    enum_failed = True
                    break
                cands = _enumerate_topN_for_macro(
                    evaluator,
                    m,
                    plc,
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
                            f"  K-joint (spatial): REVERT commit at pass {pass_idx} — "
                            f"compute_overlap_metrics found {ov['overlap_count']} "
                            f"overlap(s); reverting"
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

        pass_wall = time.perf_counter() - pass_t0
        cur_proxy = evaluator.current_cost()["proxy"]
        if log_fn is not None:
            log_fn(
                f"  K-joint (spatial) pass {pass_idx}: tuples={len(tuples_this_pass)}, "
                f"committed={pass_committed}, Δ={pass_delta:+.5f}, "
                f"proxy={cur_proxy:.5f}, pass_wall={pass_wall:.1f}s, "
                f"elapsed={time.perf_counter() - t_start:.1f}s"
            )

        if pass_committed == 0:
            if log_fn is not None:
                log_fn(f"  K-joint (spatial) converged at pass {pass_idx} (no improvement)")
            break

    return {
        "ktuples_tried": ktuples_tried,
        "ktuples_committed": ktuples_committed,
        "passes": pass_idx,
        "total_improvement": total_improvement,
        "wall_total_s": time.perf_counter() - t_start,
    }


class CDLNSSADPOKJointSpatialPlacer:
    """E44 placer: same as E41 but K-joint phase uses spatial K-tuple
    selection. Patches E41's module-level `run_kjoint_lns` reference per
    instance.
    """

    def __init__(self, **kwargs):
        # Save original reference for restoration.
        self._original_runner = _e41_mod.run_kjoint_lns
        # Swap to spatial version on E41's module namespace.
        _e41_mod.run_kjoint_lns = run_kjoint_lns_spatial
        self._inner = CDLNSSADPOKJointPlacer(**kwargs)

    def __del__(self):
        # Restore original on instance destruction.
        try:
            _e41_mod.run_kjoint_lns = self._original_runner
        except Exception:
            pass

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # Re-patch on each call in case another placer reset it.
        _e41_mod.run_kjoint_lns = run_kjoint_lns_spatial
        try:
            return self._inner.place(benchmark)
        finally:
            _e41_mod.run_kjoint_lns = self._original_runner
