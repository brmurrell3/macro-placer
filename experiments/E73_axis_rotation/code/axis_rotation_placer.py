"""Axis-rotation transplant placer.

For axis-rotating disagreement pairs between SP_E25 and SP_E41,
TRANSPLANT both macros to the OTHER basin's positions. Project_overlaps
to absorb local disruption. Accept iff proxy strictly improves.

Key difference from E69 swap: a swap exchanges (i, j) positions in
place, preserving their relative arrangement; a transplant MOVES them
to new positions defined by another basin. Swap can only rotate
LEFT↔RIGHT or BELOW↔ABOVE (axis-preserving); transplant can achieve
LEFT↔BELOW (axis-rotating) — the 85-93% structural majority of
disagreements that E69 left unaddressed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_E69 = _ROOT / "experiments" / "E69_sequence_pair_search" / "code"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_E69) not in sys.path:
    sys.path.insert(0, str(_E69))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sp_inverter import encode

from macro_place.benchmark import Benchmark
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def _spatial_dist(pos: np.ndarray, i: int, j: int) -> float:
    return float(np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]))


def _rel_axis(p: bool, m: bool) -> str:
    """Return 'H' for LEFT/RIGHT, 'V' for BELOW/ABOVE."""
    if p == m:
        return "H"  # LEFT (T,T) or RIGHT (F,F)
    return "V"      # BELOW (T,F) or ABOVE (F,T)


def _build_axis_rotating_set(sp_25, sp_41, n_hard, pos_np):
    """Return (i, j, transplant_dist) where SP relations are axis-rotating.

    transplant_dist is the L2 distance from current i+j positions to the
    target (other basin) positions, used to order attempts (smallest disruption first).
    """
    rp25 = sp_25.rank_plus(); rm25 = sp_25.rank_minus()
    rp41 = sp_41.rank_plus(); rm41 = sp_41.rank_minus()
    out = []
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            p25 = rp25[i] < rp25[j]; m25 = rm25[i] < rm25[j]
            p41 = rp41[i] < rp41[j]; m41 = rm41[i] < rm41[j]
            if (p25 == p41) and (m25 == m41):
                continue  # same relation, no disagreement
            ax25 = _rel_axis(p25, m25)
            ax41 = _rel_axis(p41, m41)
            if ax25 != ax41:
                # Axis-rotating: H↔V.
                out.append((i, j, _spatial_dist(pos_np, i, j)))
    return out


def axis_rotation_search(
    start_placement: torch.Tensor,
    other_placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    max_attempts: int = 2000,
    budget_seconds: float = 1800.0,
    log: Optional[Callable[[str], None]] = None,
    seed: int = 42,
    transplant_order: str = "spatial_close_first",
    use_evaluator: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """Axis-rotation transplant search.

    For each axis-rotating disagreement (i, j):
      candidate[i] = other_placement[i]
      candidate[j] = other_placement[j]
      project_overlaps; check feasibility; check proxy improvement.

    Args:
      start_placement: initial Cartesian placement (mutated copy).
      other_placement: target basin's placement (look-up table for transplants).
      benchmark, plc: standard.
      max_attempts: cap on transplant attempts.
      budget_seconds: wall cap.
      transplant_order:
        'spatial_close_first' — close pairs first (less disruption).
        'spatial_far_first' — far pairs first.
        'random_seeded' — shuffle with seed.

    Returns:
      (final_placement, stats_dict)
    """
    if log is None:
        log = lambda s: print(s, flush=True)

    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes

    current = start_placement.detach().clone()
    other_np = other_placement.detach().cpu().numpy().astype(np.float64)
    pos_np = current[:n_hard].detach().cpu().numpy().astype(np.float64).copy()

    # Encode SP for both basins.
    log("[axis-rot] encoding SP for start and other basins...")
    sp_start = encode(start_placement, sizes, n_hard)
    sp_other = encode(other_placement, sizes, n_hard)
    rotating = _build_axis_rotating_set(sp_start, sp_other, n_hard, pos_np)
    n_pairs = n_hard * (n_hard - 1) // 2
    log(f"[axis-rot] axis-rotating disagreement set: "
        f"{len(rotating)} / {n_pairs} pairs ({100 * len(rotating) / n_pairs:.1f}%)")
    if not rotating:
        log("[axis-rot] no axis-rotating pairs (basins agree on adjacency type)")
        return current, {
            "rotating_count": 0, "transplants_tried": 0, "transplants_accepted": 0,
            "starting_proxy": float(compute_proxy_cost(current, benchmark, plc)["proxy_cost"]),
            "final_proxy": None,
        }

    rng = np.random.default_rng(seed)
    if transplant_order == "spatial_close_first":
        rotating.sort(key=lambda t: t[2])
    elif transplant_order == "spatial_far_first":
        rotating.sort(key=lambda t: -t[2])
    elif transplant_order == "random_seeded":
        rng.shuffle(rotating)
    else:
        raise ValueError(f"unknown transplant_order: {transplant_order}")

    starting_proxy = float(compute_proxy_cost(current, benchmark, plc)["proxy_cost"])
    log(f"[axis-rot] starting proxy: {starting_proxy:.5f}")

    current_proxy = starting_proxy
    accepted = 0
    feas_failed = 0
    proxy_failed = 0
    proxy_history = [starting_proxy]
    t0 = time.time()
    attempt_idx = 0

    for attempt_idx, (i, j, d_spatial) in enumerate(rotating):
        if attempt_idx >= max_attempts:
            log(f"[axis-rot] reached max_attempts={max_attempts}")
            break
        if (time.time() - t0) > budget_seconds:
            log(f"[axis-rot] reached budget {budget_seconds:.0f}s")
            break

        candidate = current.clone()
        candidate[i, 0] = float(other_np[i, 0])
        candidate[i, 1] = float(other_np[i, 1])
        candidate[j, 0] = float(other_np[j, 0])
        candidate[j, 1] = float(other_np[j, 1])

        candidate, n_iters = project_overlaps(candidate, benchmark)
        ovl_metrics = compute_overlap_metrics(candidate, benchmark)
        if ovl_metrics["overlap_count"] > 0:
            feas_failed += 1
            continue

        cand_proxy = float(compute_proxy_cost(candidate, benchmark, plc)["proxy_cost"])

        if cand_proxy < current_proxy - 1e-7:
            current = candidate
            current_proxy = cand_proxy
            accepted += 1
            proxy_history.append(current_proxy)
            if accepted <= 5 or accepted % 10 == 0:
                log(f"[axis-rot] [{attempt_idx + 1}/{len(rotating)}] transplant ({i},{j}) "
                    f"d={d_spatial:.2f} ACCEPT → proxy {current_proxy:.5f} "
                    f"(Δ={cand_proxy - proxy_history[-2]:+.5f}, accepted={accepted})")
        else:
            proxy_failed += 1

    wall = time.time() - t0
    attempts = attempt_idx + 1 if rotating else 0
    log(
        f"[axis-rot] done: attempts={attempts} accepted={accepted} "
        f"feas_failed={feas_failed} proxy_failed={proxy_failed} wall={wall:.0f}s"
    )
    log(f"[axis-rot] final proxy: {current_proxy:.5f} "
        f"(vs starting {starting_proxy:.5f}, Δ={current_proxy - starting_proxy:+.5f})")

    return current, {
        "rotating_count": len(rotating),
        "transplants_tried": attempts,
        "transplants_accepted": accepted,
        "feas_failed": feas_failed,
        "proxy_failed": proxy_failed,
        "starting_proxy": starting_proxy,
        "final_proxy": current_proxy,
        "improvement": starting_proxy - current_proxy,
        "improvement_frac": (starting_proxy - current_proxy) / starting_proxy
                             if starting_proxy > 0 else 0.0,
        "wall_seconds": wall,
        "proxy_history": proxy_history,
    }


def cd_polish_placement(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    budget_seconds: float = 600.0,
    log: Optional[Callable[[str], None]] = None,
) -> torch.Tensor:
    """Short CD-adaptive polish; returns mutated placement."""
    if log is None:
        log = lambda s: print(s, flush=True)
    log(f"[cd-polish] starting (budget {budget_seconds:.0f}s)...")
    placement = placement.detach().clone()
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement)
    n_hard = benchmark.num_hard_macros
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable = [i for i in range(n_hard) if not bool(fixed[i])]
    info = run_cd_adaptive(
        evaluator, benchmark, plc, movable,
        min_time_s=60.0,
        hard_cap_s=budget_seconds,
        patience=3,
        plateau_threshold=0.001,
        log_fn=log,
    )
    log(f"[cd-polish] done: {info.get('exit_reason')} after {info.get('total_moves', '?')} moves")
    return evaluator.placement.detach().clone().to(torch.float32)
