"""Pair-swap polish (E114) — atomic 2-macro swaps.

Algorithm: for random pairs of hard macros, attempt swapping their
positions atomically. Accept if:
  - Post-swap proxy < pre-swap proxy
  - No overlap created

This is a NON-LOCAL move set that SA-v2 cannot reach: SA's per-axis
breakpoint search blocks during the intermediate "move m1 first" state
because m1+m2 overlap.

Implementation: uses ev.move() twice (m1→xy2 then m2→xy1). On reject,
ev.revert() restores m2, then a manual ev.move(m1, xy1) restores m1.
Single-step snapshot is consumed by the second revert; safe because
no further revert is requested.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics


def _try_swap(
    ev: IncrementalProxyEvaluator, benchmark,
    m1: int, m2: int, pre_proxy: float,
) -> Tuple[bool, float, int]:
    """Try atomic swap of m1 and m2. Returns (accepted, new_proxy, overlap_count).

    State after call:
      accepted=True: m1 at old-m2, m2 at old-m1.
      accepted=False: state restored (m1 at old-m1, m2 at old-m2).
    """
    xy1 = (float(ev.placement[m1, 0]), float(ev.placement[m1, 1]))
    xy2 = (float(ev.placement[m2, 0]), float(ev.placement[m2, 1]))

    # Two-step swap
    ev.move(m1, xy2)
    ev.move(m2, xy1)
    post_proxy = float(ev.current_cost()["proxy"])
    ovl = int(compute_overlap_metrics(
        ev.placement.detach().clone().to(torch.float32), benchmark
    )["overlap_count"])

    if post_proxy < pre_proxy - 1e-7 and ovl == 0:
        return True, post_proxy, ovl

    # Reject: restore. revert() rolls back m2's last move (m2 back to xy2).
    # Then manually move m1 back to xy1.
    ev.revert()
    ev.move(m1, xy1)
    return False, post_proxy, ovl


def pair_swap_polish(
    ev: IncrementalProxyEvaluator,
    benchmark,
    movable_idx: List[int],
    *,
    n_pairs_per_pass: int = 500,
    max_passes: int = 3,
    seed: int = 42,
    time_budget_s: Optional[float] = None,
    pair_selection: str = "criticality",  # 'random' or 'criticality'
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    """Multi-pass pair-swap polish.

    Args:
      n_pairs_per_pass: number of random pairs to attempt per pass.
      pair_selection: 'random' = uniform; 'criticality' = bias to top-WL macros.
    """
    if log is None:
        log = lambda s: None
    t0 = time.perf_counter()
    deadline = (t0 + time_budget_s) if time_budget_s else None
    rng = np.random.default_rng(seed)

    init_proxy = float(ev.current_cost()["proxy"])
    log(f"pair_swap_polish: init={init_proxy:.5f} n_per_pass={n_pairs_per_pass} "
        f"max_passes={max_passes} pair_selection={pair_selection}")

    arr = np.array(movable_idx)

    # Compute criticality weights for pair selection if needed.
    if pair_selection == "criticality":
        scores = np.zeros(len(arr))
        for i, m in enumerate(arr):
            nets = ev.macro_to_nets[int(m)].tolist()
            if len(nets) > 0:
                scores[i] = float(ev.net_hpwl[nets].sum())
        # Bias toward higher-WL macros
        p = scores + 1e-3
        p = p / p.sum()
    else:
        p = None

    pass_logs = []
    cumulative_accepted = 0
    cumulative_overlapped = 0

    for pass_id in range(max_passes):
        if deadline is not None and time.perf_counter() > deadline:
            log(f"  pass {pass_id+1}: budget exhausted")
            break
        pass_t0 = time.perf_counter()
        pre_pass_proxy = float(ev.current_cost()["proxy"])
        n_accepted = 0
        n_overlapped = 0
        n_proxy_reject = 0

        for _ in range(n_pairs_per_pass):
            if deadline is not None and time.perf_counter() > deadline:
                break
            if pair_selection == "criticality":
                pair = rng.choice(arr, 2, replace=False, p=p)
            else:
                pair = rng.choice(arr, 2, replace=False)
            m1, m2 = int(pair[0]), int(pair[1])
            cur_proxy = float(ev.current_cost()["proxy"])
            accepted, _, ovl = _try_swap(ev, benchmark, m1, m2, cur_proxy)
            if accepted:
                n_accepted += 1
            elif ovl > 0:
                n_overlapped += 1
            else:
                n_proxy_reject += 1

        post_pass_proxy = float(ev.current_cost()["proxy"])
        pass_wall = time.perf_counter() - pass_t0
        cumulative_accepted += n_accepted
        cumulative_overlapped += n_overlapped
        pass_logs.append({
            "pass": pass_id + 1,
            "pre": pre_pass_proxy, "post": post_pass_proxy,
            "delta": post_pass_proxy - pre_pass_proxy,
            "n_accepted": n_accepted, "n_overlapped": n_overlapped,
            "n_proxy_reject": n_proxy_reject,
            "wall": pass_wall,
        })
        log(f"  pass {pass_id+1}: pre={pre_pass_proxy:.5f} post={post_pass_proxy:.5f} "
            f"acc={n_accepted}/{n_pairs_per_pass} ovl_rej={n_overlapped} "
            f"proxy_rej={n_proxy_reject} wall={pass_wall:.0f}s")

        if post_pass_proxy >= pre_pass_proxy - 1e-5:
            log(f"  pass {pass_id+1}: no improvement; stopping")
            break

    final_proxy = float(ev.current_cost()["proxy"])
    wall = time.perf_counter() - t0
    log(f"pair_swap done: init={init_proxy:.5f} final={final_proxy:.5f} "
        f"Δ={final_proxy-init_proxy:+.5f} ({100*(final_proxy-init_proxy)/init_proxy:+.3f}%) "
        f"acc_total={cumulative_accepted} ovl_total={cumulative_overlapped} "
        f"passes={len(pass_logs)} wall={wall:.0f}s")
    return {
        "init_proxy": init_proxy, "final_proxy": final_proxy,
        "delta_proxy": final_proxy - init_proxy,
        "delta_pct": 100 * (final_proxy - init_proxy) / init_proxy,
        "n_accepted_total": cumulative_accepted,
        "n_overlapped_total": cumulative_overlapped,
        "passes_run": len(pass_logs),
        "pass_logs": pass_logs,
        "wall_seconds": wall,
    }
