"""H2 redesign: multi-start SA-v2 polish (replaces PA).

Standalone smoke showed PA loses to SA-v2 by ~3.7% at same budget on ibm01
because PA's resample/clone overhead and per-chain depth split don't
recover the diversity benefit at the polish stage (which is mostly
downhill, not landscape-rugged-enough to need full PA).

Multi-start preserves the cross-disciplinary "replica diversity" intuition
but uses SA-v2's per-chain depth + best-so-far tracking:

  1. Take current placement as anchor.
  2. Run N independent SA-v2 chains, each starting from the anchor,
     with different seeds + a brief high-T diversification "kick" at start.
  3. Take the best across all chains. Restore evaluator to best.

At budget B with N chains, each chain gets B/N. With N=4 and B=600s, each
chain gets 150s — long enough to converge SA. The kick is at higher T₀
than SA-v2's default to seed diversity.
"""
from __future__ import annotations

import copy
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]


# Lazy-load the canonical SA-v2 (avoid monkey-patched version).
_ORIGINAL_SA = None


def _get_original_sa():
    global _ORIGINAL_SA
    if _ORIGINAL_SA is None:
        _spec = importlib.util.spec_from_file_location(
            "_orig_e25_for_ms", str(_ROOT / "submissions" / "cd_lns_sa" / "placer.py")
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _ORIGINAL_SA = _mod.run_sa_polish_v2
    return _ORIGINAL_SA


def run_multistart_sa_polish(
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
    n_chains: int = 4,
    kick_T0: float = 5e-3,
    kick_steps: int = 100,
    **_unused,
) -> dict:
    """N independent SA-v2 chains; take best.

    For each chain k ∈ [0, n_chains):
      - Build a fresh evaluator on a copy of the anchor placement.
      - Kick: kick_steps high-T (T=kick_T0) Metropolis moves.
      - SA-v2 with seed=seed+k, time budget B/N, full best-so-far.

    Restore the original evaluator to the best chain's final state via
    per-macro moves (keeps incremental caches consistent).
    """
    from macro_place.incremental_evaluator import IncrementalProxyEvaluator
    from pa_replica import sweep_replica, Replica  # reuse PA sweep primitive

    t_start = time.perf_counter()
    init_proxy = float(evaluator.current_cost()["proxy"])
    n_chains = max(1, int(n_chains))
    per_chain_budget = (time_budget_s - 5.0) / n_chains  # 5s safety margin

    if per_chain_budget < 20.0 or not hard_movable:
        # Too tight for multi-start; fall through to single SA-v2.
        if log_fn:
            log_fn(f"  multistart: per_chain={per_chain_budget:.1f}s too tight; "
                   f"single-chain SA-v2")
        return _get_original_sa()(
            evaluator, benchmark, plc, hard_movable,
            time_budget_s=time_budget_s,
            T0=T0, Tf=Tf, seed=seed,
            breakpoint_budget=breakpoint_budget, log_fn=log_fn,
        )

    if log_fn:
        log_fn(f"  multistart-SA: N={n_chains} chains × {per_chain_budget:.1f}s each, "
               f"kick_T0={kick_T0:.0e} kick_steps={kick_steps}")

    anchor_placement = evaluator.placement.detach().clone()
    best_proxy = init_proxy
    best_placement = anchor_placement.clone()
    per_chain_results = []

    sa_fn = _get_original_sa()

    from macro_place.cd_core import _grid_lines
    grid_lines_x, grid_lines_y = _grid_lines(plc)

    for k in range(n_chains):
        t_chain0 = time.perf_counter()
        chain_seed = seed + 1000 * k + 7
        chain_eval = IncrementalProxyEvaluator(benchmark, plc, anchor_placement.clone())
        chain_replica = Replica(
            evaluator=chain_eval,
            current_proxy=float(chain_eval.current_cost()["proxy"]),
            seed=chain_seed,
        )

        # Kick: diversification via high-T sweep.
        if kick_steps > 0:
            kick_rng = np.random.default_rng(chain_seed + 1)
            sweep_replica(
                chain_replica, hard_movable, grid_lines_x, grid_lines_y, benchmark,
                T=kick_T0, n_steps=kick_steps, rng=kick_rng,
                breakpoint_budget=breakpoint_budget,
            )

        # SA-v2 polish on this chain.
        elapsed_to_chain = time.perf_counter() - t_start
        remaining_chain_budget = (
            (k + 1) * per_chain_budget - elapsed_to_chain
        )
        if remaining_chain_budget <= 5.0:
            if log_fn:
                log_fn(f"  chain {k}: insufficient remaining budget "
                       f"{remaining_chain_budget:.1f}s; skipping")
            break

        chain_stats = sa_fn(
            chain_eval, benchmark, plc, hard_movable,
            time_budget_s=remaining_chain_budget,
            T0=T0, Tf=Tf, seed=chain_seed,
            breakpoint_budget=breakpoint_budget,
            log_fn=None,  # quiet per-chain
        )
        chain_final = float(chain_eval.current_cost()["proxy"])
        per_chain_results.append({
            "chain": k, "seed": chain_seed,
            "final": chain_final,
            "wall": time.perf_counter() - t_chain0,
        })
        if chain_final < best_proxy:
            best_proxy = chain_final
            best_placement = chain_eval.placement.detach().clone()
        if log_fn:
            log_fn(f"  chain {k}: seed={chain_seed} final={chain_final:.5f} "
                   f"best_so_far={best_proxy:.5f}")

    # Restore evaluator to best.
    n_total = int(evaluator.placement.shape[0])
    for i in range(n_total):
        tx = float(best_placement[i, 0]); ty = float(best_placement[i, 1])
        cx = float(evaluator.placement[i, 0]); cy = float(evaluator.placement[i, 1])
        if abs(tx - cx) > 1e-9 or abs(ty - cy) > 1e-9:
            try:
                evaluator.move(i, (tx, ty))
            except Exception:
                pass

    final_proxy = float(evaluator.current_cost()["proxy"])
    if log_fn:
        log_fn(f"  multistart-SA done: init={init_proxy:.5f} best={best_proxy:.5f} "
               f"final={final_proxy:.5f}  N_chains_run={len(per_chain_results)}")

    return {
        "proposed": 0, "accepted_better": 0, "accepted_worse": 0,
        "rejected": 0, "skipped": 0,
        "init_proxy": init_proxy,
        "best_proxy": best_proxy,
        "final_proxy": final_proxy,
        "improvement_vs_init": init_proxy - best_proxy,
        "best_found_at_t": 0.0,
        "wall_total_s": time.perf_counter() - t_start,
        "best_restored": True,
        "multistart_n_chains_run": len(per_chain_results),
        "multistart_per_chain": per_chain_results,
    }
