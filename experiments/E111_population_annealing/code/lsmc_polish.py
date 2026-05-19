"""Large-Step Markov Chain (LSMC) / Iterated Local Search polish.

Canonical TSP escape from SA-saturation plateaus (Martin, Otto, Felten
1991; Lourenço-Martin-Stützle ILS framework). Adapted to macro placement:

  Outer loop:
    1. Pick K=4 random macros forming an approximate spatial cluster.
    2. Cyclically rotate their positions (a "K-cycle kick" — non-sequential,
       unreachable by SA's per-macro Metropolis moves).
    3. Run SA-v2 inner polish for a short budget from the kicked state.
    4. Outer Metropolis: accept new local-min if better, or with prob
       exp(-Δ/T_outer) — kick is rejected by reverting placement.
    5. Repeat until time budget exhausted.

Empirically on Lin-Kernighan-saturated TSP, double-bridge LSMC lifts 1.3-1.6%
where direct SA can't move. Our hypothesis: SA-v2 on cascade output is
similarly saturated; structural kicks escape its single-macro-move neighborhood.

Signature compatible with run_sa_polish_v2 for monkey-patching.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]


_ORIGINAL_SA = None


def _get_original_sa():
    global _ORIGINAL_SA
    if _ORIGINAL_SA is None:
        _spec = importlib.util.spec_from_file_location(
            "_orig_e25_for_lsmc",
            str(_ROOT / "submissions" / "cd_lns_sa" / "placer.py"),
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _ORIGINAL_SA = _mod.run_sa_polish_v2
    return _ORIGINAL_SA


def _pick_kick_tuple(
    placement: torch.Tensor,
    hard_movable: List[int],
    K: int,
    rng: np.random.Generator,
    spatial: bool = True,
) -> List[int]:
    """Select K hard-movable macros for a K-cycle kick.

    spatial=True: pick a seed macro at random, then find its K-1 spatial
                  nearest neighbors among hard_movable (preserves spatial
                  structure on rotation).
    spatial=False: pick K macros uniformly at random.
    """
    if len(hard_movable) < K:
        return list(hard_movable)
    if not spatial:
        idx = rng.choice(len(hard_movable), size=K, replace=False)
        return [int(hard_movable[i]) for i in idx]

    # Pick a seed; find K-1 nearest neighbors by L2 distance among movables.
    seed_idx = int(hard_movable[rng.integers(0, len(hard_movable))])
    seed_xy = placement[seed_idx].numpy().astype(np.float64)
    movable_xy = placement[hard_movable].numpy().astype(np.float64)
    dx = movable_xy[:, 0] - seed_xy[0]
    dy = movable_xy[:, 1] - seed_xy[1]
    dist = np.sqrt(dx * dx + dy * dy)
    order = np.argsort(dist)
    # First in order is seed itself (dist 0). Take seed + K-1 next-nearest.
    picks: List[int] = []
    for j in order[:K]:
        picks.append(int(hard_movable[int(j)]))
    return picks


def _apply_cycle(
    evaluator,
    benchmark,
    macros: List[int],
    rng: np.random.Generator,
) -> Optional[List[Tuple[int, Tuple[float, float]]]]:
    """Cyclically rotate positions of `macros`. Returns the move history
    (used to revert if needed), or None if the rotation produces an
    out-of-bounds placement (macro too big for the destination spot).

    Permutation: macros[i] → macros[i+1]'s original position (cyclic).
    """
    K = len(macros)
    if K < 2:
        return None

    # Snapshot current positions.
    orig: List[Tuple[float, float]] = [
        (float(evaluator.placement[m, 0]), float(evaluator.placement[m, 1]))
        for m in macros
    ]

    # Optionally permute non-trivially (random non-identity permutation of K).
    # For now: simple cyclic shift by +1.
    new_positions = [orig[(i - 1) % K] for i in range(K)]

    # Sanity: ensure new positions are within canvas given each macro's size.
    sizes = evaluator.macro_sizes
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    for m, (nx, ny) in zip(macros, new_positions):
        half_w = float(sizes[m, 0]) / 2.0
        half_h = float(sizes[m, 1]) / 2.0
        if nx - half_w < -1e-3 or nx + half_w > cw + 1e-3:
            return None
        if ny - half_h < -1e-3 or ny + half_h > ch + 1e-3:
            return None

    # Apply via evaluator.move() — incremental cache updates correct.
    # Intermediate states have overlaps among the cyclic set, but the final
    # state is consistent.
    moves: List[Tuple[int, Tuple[float, float]]] = []
    for m, (nx, ny) in zip(macros, new_positions):
        try:
            evaluator.move(m, (nx, ny))
            moves.append((m, (nx, ny)))
        except Exception:
            # Roll back any partial moves.
            for (m_r, _) in reversed(moves):
                # Restore from orig.
                idx_r = macros.index(m_r)
                try:
                    evaluator.move(m_r, orig[idx_r])
                except Exception:
                    pass
            return None
    # Snapshot for later revert.
    return [(m, p) for m, p in zip(macros, orig)]


def _revert_to(evaluator, snapshot: List[Tuple[int, Tuple[float, float]]]):
    """Restore placement of macros in snapshot to their snapshot positions."""
    for m, (px, py) in snapshot:
        try:
            evaluator.move(m, (px, py))
        except Exception:
            pass


def _overlap_count_hard(evaluator, benchmark) -> int:
    """Quick hard-macro overlap count via the canonical metric."""
    from macro_place.objective import compute_overlap_metrics

    pl = evaluator.placement.detach().clone().to(torch.float32)
    return int(compute_overlap_metrics(pl, benchmark)["overlap_count"])


def run_lsmc_polish(
    evaluator,
    benchmark,
    plc,
    hard_movable: List[int],
    time_budget_s: float,
    T0: float = 5e-4,
    Tf: float = 1e-6,
    seed: int = 42,
    breakpoint_budget: int = 12,
    log_fn: Optional[Callable[[str], None]] = None,
    *,
    K_cycle: int = 4,
    inner_budget_s: float = 30.0,
    T_outer_frac: float = 0.005,
    max_outer_iters: int = 999,
    spatial_kick: bool = True,
    fallback_min_budget_s: float = 60.0,
    **_unused,
) -> dict:
    """LSMC/ILS polish — kick → inner SA-v2 → outer Metropolis on local minima.

    Falls back to single SA-v2 if budget is too tight for at least 2 outer
    iterations.
    """
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    init_proxy = float(evaluator.current_cost()["proxy"])

    if not hard_movable:
        if log_fn:
            log_fn("  LSMC: no hard movable macros; skipping")
        return _empty_stats(init_proxy)

    if time_budget_s < fallback_min_budget_s:
        if log_fn:
            log_fn(f"  LSMC: budget {time_budget_s:.0f}s < "
                   f"{fallback_min_budget_s:.0f}s; falling back to SA-v2")
        return _get_original_sa()(
            evaluator, benchmark, plc, hard_movable, time_budget_s,
            T0=T0, Tf=Tf, seed=seed,
            breakpoint_budget=breakpoint_budget, log_fn=log_fn,
        )

    sa_fn = _get_original_sa()
    cur_proxy = init_proxy
    best_proxy = init_proxy
    best_placement = evaluator.placement.detach().clone()
    T_outer = max(init_proxy * T_outer_frac, 1e-6)

    if log_fn:
        log_fn(f"  LSMC budget={time_budget_s:.0f}s, K={K_cycle}, "
               f"inner={inner_budget_s:.0f}s, T_outer={T_outer:.4e}, "
               f"spatial={spatial_kick}, seed={seed}")

    n_outer = 0
    n_accept_better = 0
    n_accept_worse = 0
    n_reject_proxy = 0
    n_reject_overlap = 0
    n_reject_kick = 0
    n_total_overlaps = 0

    # Phase 0 — initial SA-v2 from cascade output (no kick first).
    init_budget = min(inner_budget_s, time_budget_s * 0.25)
    if log_fn:
        log_fn(f"  LSMC phase 0: initial SA-v2 ({init_budget:.0f}s)")
    sa_fn(
        evaluator, benchmark, plc, hard_movable,
        time_budget_s=init_budget,
        T0=T0, Tf=Tf, seed=seed,
        breakpoint_budget=breakpoint_budget, log_fn=None,
    )
    cur_proxy = float(evaluator.current_cost()["proxy"])
    if cur_proxy < best_proxy:
        best_proxy = cur_proxy
        best_placement = evaluator.placement.detach().clone()

    # ── Outer ILS loop ──
    while n_outer < max_outer_iters:
        elapsed = time.perf_counter() - t_start
        remaining = time_budget_s - elapsed
        if remaining <= inner_budget_s + 5.0:
            break

        # Snapshot pre-kick state for full revert.
        pre_kick_placement = evaluator.placement.detach().clone()
        pre_kick_proxy = cur_proxy

        # 1. Pick K-tuple and apply cycle.
        kick_macros = _pick_kick_tuple(
            evaluator.placement, hard_movable, K_cycle, rng,
            spatial=spatial_kick,
        )
        snapshot = _apply_cycle(evaluator, benchmark, kick_macros, rng)
        if snapshot is None:
            n_reject_kick += 1
            n_outer += 1
            continue

        # 2. Inner SA-v2 polish.
        sa_fn(
            evaluator, benchmark, plc, hard_movable,
            time_budget_s=inner_budget_s,
            T0=T0, Tf=Tf, seed=seed + 17 * (n_outer + 1),
            breakpoint_budget=breakpoint_budget, log_fn=None,
        )
        new_proxy = float(evaluator.current_cost()["proxy"])

        # 3. Overlap check — kick may have created hard-macro overlap that
        #    SA can't always recover from.
        ovl = _overlap_count_hard(evaluator, benchmark)

        # 4. Outer Metropolis on local minima.
        accepted = False
        if ovl > 0:
            n_reject_overlap += 1
            n_total_overlaps += ovl
        else:
            delta = new_proxy - cur_proxy
            if delta <= 0:
                accepted = True
                n_accept_better += 1
                cur_proxy = new_proxy
                if new_proxy < best_proxy:
                    best_proxy = new_proxy
                    best_placement = evaluator.placement.detach().clone()
            else:
                p_accept = math.exp(-delta / max(T_outer, 1e-12))
                if rng.random() < p_accept:
                    accepted = True
                    n_accept_worse += 1
                    cur_proxy = new_proxy
                else:
                    n_reject_proxy += 1

        if not accepted:
            # Revert to pre-kick placement entirely. We snapshot the full
            # tensor before kick; restore by per-macro moves to keep
            # evaluator's caches consistent.
            n_total = int(evaluator.placement.shape[0])
            for i in range(n_total):
                tx = float(pre_kick_placement[i, 0])
                ty = float(pre_kick_placement[i, 1])
                cx = float(evaluator.placement[i, 0])
                cy = float(evaluator.placement[i, 1])
                if abs(tx - cx) > 1e-9 or abs(ty - cy) > 1e-9:
                    try:
                        evaluator.move(i, (tx, ty))
                    except Exception:
                        pass
            cur_proxy = pre_kick_proxy

        n_outer += 1
        if log_fn and (n_outer <= 3 or n_outer % 5 == 0):
            log_fn(f"  LSMC iter {n_outer:3d}  cur={cur_proxy:.5f}  "
                   f"best={best_proxy:.5f}  "
                   f"acc_b={n_accept_better} acc_w={n_accept_worse} "
                   f"rej_proxy={n_reject_proxy} rej_ovl={n_reject_overlap} "
                   f"rej_kick={n_reject_kick}  "
                   f"elapsed={time.perf_counter()-t_start:.0f}s")

    # Restore evaluator to best placement.
    n_total = int(evaluator.placement.shape[0])
    for i in range(n_total):
        tx = float(best_placement[i, 0])
        ty = float(best_placement[i, 1])
        cx = float(evaluator.placement[i, 0])
        cy = float(evaluator.placement[i, 1])
        if abs(tx - cx) > 1e-9 or abs(ty - cy) > 1e-9:
            try:
                evaluator.move(i, (tx, ty))
            except Exception:
                pass

    final_proxy = float(evaluator.current_cost()["proxy"])
    if log_fn:
        log_fn(f"  LSMC done: init={init_proxy:.5f} best={best_proxy:.5f} "
               f"final={final_proxy:.5f}  outer_iters={n_outer}  "
               f"acc_b={n_accept_better} acc_w={n_accept_worse}")

    return {
        "proposed": n_outer,
        "accepted_better": n_accept_better,
        "accepted_worse": n_accept_worse,
        "rejected": n_reject_proxy + n_reject_overlap + n_reject_kick,
        "skipped": 0,
        "init_proxy": init_proxy,
        "best_proxy": best_proxy,
        "final_proxy": final_proxy,
        "improvement_vs_init": init_proxy - best_proxy,
        "best_found_at_t": 0.0,
        "wall_total_s": time.perf_counter() - t_start,
        "best_restored": True,
        "lsmc_n_outer": n_outer,
        "lsmc_n_reject_overlap": n_reject_overlap,
        "lsmc_n_reject_kick": n_reject_kick,
        "lsmc_total_overlaps_seen": n_total_overlaps,
    }


def _empty_stats(init_proxy: float) -> dict:
    return {
        "proposed": 0, "accepted_better": 0, "accepted_worse": 0,
        "rejected": 0, "skipped": 0,
        "init_proxy": init_proxy, "best_proxy": init_proxy,
        "final_proxy": init_proxy, "improvement_vs_init": 0.0,
        "best_found_at_t": 0.0, "wall_total_s": 0.0,
        "best_restored": False,
        "lsmc_n_outer": 0, "lsmc_n_reject_overlap": 0,
        "lsmc_n_reject_kick": 0, "lsmc_total_overlaps_seen": 0,
    }
