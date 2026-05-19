"""Alternating SA-v2 + WireMask polish.

Hypothesis: SA-v2 finds local minima via micro-moves; WireMask finds
non-local atomic positions via 2D grid search. Alternating them lets
each remove the local minima of the other.

Architecture:
  while budget remaining:
    SA-v2 for fraction_sa * remaining
    WireMask for fraction_wm * remaining
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics
from wiremask_polish import wiremask_polish

_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_alt", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
run_sa_polish_v2 = _E25_MOD.run_sa_polish_v2


def alternating_sa_wm_polish(
    ev: IncrementalProxyEvaluator,
    benchmark,
    plc,
    hard_movable: List[int],
    all_movable: List[int],
    *,
    n_rounds: int = 3,
    sa_fraction: float = 0.55,
    wm_fraction: float = 0.45,
    wm_n_per_axis: int = 5,
    wm_local_radius_frac: float = 0.15,
    wm_top_k: Optional[int] = None,
    sa_seed: int = 42,
    time_budget_s: Optional[float] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    """Alternate SA-v2 and WireMask passes for n_rounds.

    Each round divides the round-budget between SA and WM in
    sa_fraction:wm_fraction.
    """
    if log is None:
        log = lambda s: None
    t0 = time.perf_counter()
    init_proxy = float(ev.current_cost()["proxy"])
    log(f"alternating_sa_wm: init={init_proxy:.5f} rounds={n_rounds} "
        f"budget={time_budget_s}")

    round_logs = []
    for r in range(n_rounds):
        elapsed = time.perf_counter() - t0
        remaining = (time_budget_s - elapsed) if time_budget_s else None
        if remaining is not None and remaining < 30.0:
            log(f"  round {r+1}: budget exhausted (remaining={remaining:.0f}s)")
            break
        per_round = (remaining / (n_rounds - r)) if remaining else 120.0
        sa_budget = per_round * sa_fraction
        wm_budget = per_round * wm_fraction
        log(f"  round {r+1}/{n_rounds}: SA={sa_budget:.0f}s WM={wm_budget:.0f}s")

        pre_proxy = float(ev.current_cost()["proxy"])
        # SA-v2 phase
        if sa_budget >= 10.0:
            sa_stats = run_sa_polish_v2(
                ev, benchmark, plc, hard_movable,
                time_budget_s=sa_budget,
                seed=sa_seed + r,  # different seed each round for diversity
                log_fn=lambda s: None,
            )
        # WireMask phase
        if wm_budget >= 30.0:
            wm_stats = wiremask_polish(
                ev, benchmark, all_movable,
                n_per_axis=wm_n_per_axis,
                local_radius_frac=wm_local_radius_frac,
                top_k=wm_top_k,
                max_passes=10,
                time_budget_s=wm_budget,
                log=lambda s: None,
            )
        post_proxy = float(ev.current_cost()["proxy"])
        round_logs.append({"round": r, "pre": pre_proxy, "post": post_proxy,
                           "delta": post_proxy - pre_proxy})
        log(f"    round {r+1}: {pre_proxy:.5f} → {post_proxy:.5f} "
            f"(Δ={post_proxy-pre_proxy:+.5f})")

    final_proxy = float(ev.current_cost()["proxy"])
    wall = time.perf_counter() - t0
    log(f"alternating done: init={init_proxy:.5f} final={final_proxy:.5f} "
        f"Δ={final_proxy-init_proxy:+.5f} ({100*(final_proxy-init_proxy)/init_proxy:+.3f}%) "
        f"rounds={len(round_logs)} wall={wall:.0f}s")
    return {
        "init_proxy": init_proxy, "final_proxy": final_proxy,
        "delta_proxy": final_proxy - init_proxy,
        "delta_pct": 100 * (final_proxy - init_proxy) / init_proxy,
        "rounds_run": len(round_logs), "round_logs": round_logs,
        "wall_seconds": wall,
    }
