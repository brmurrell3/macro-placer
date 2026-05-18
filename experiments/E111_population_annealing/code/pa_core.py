"""Population annealing polish — drop-in replacement for run_sa_polish_v2.

Hukushima & Iba (2003) PRE 67 056111 + Baumgartner & Wang (2013)
systematic resampling. Maintains N replicas at descending temperatures,
resampling weights at each step kill high-energy replicas and clone
low-energy ones. Canonical proxy only — NO smooth-RUDY gradient (which
is the documented failure mode of E88/E95/E98).

Signature is intentionally compatible with `run_sa_polish_v2`; SA-only
kwargs are swallowed by **_unused so callers don't need to change.
Returns the same stats dict shape.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from macro_place.cd_core import _grid_lines
from macro_place.incremental_evaluator import IncrementalProxyEvaluator

from pa_replica import Replica, build_replicas, sweep_replica
from pa_resample import systematic_resample


def run_pa_polish(
    evaluator: IncrementalProxyEvaluator,
    benchmark,
    plc,
    hard_movable: List[int],
    time_budget_s: float,
    T0: float = 1e-3,
    Tf: float = 1e-6,
    seed: int = 42,
    breakpoint_budget: int = 12,
    log_fn: Optional[Callable[[str], None]] = None,
    *,
    n_replicas: int = 24,
    n_ladder: int = 40,
    sweeps_per_step: int = 500,
    pre_ladder_diversify_steps: int = 100,
    **_unused,
) -> dict:
    """Population-annealing polish on canonical proxy.

    Parameters mirror `run_sa_polish_v2` for monkey-patch compatibility.
    PA-specific knobs are keyword-only (n_replicas, n_ladder, sweeps_per_step,
    pre_ladder_diversify_steps).
    """
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    init_proxy = float(evaluator.current_cost()["proxy"])

    if not hard_movable:
        if log_fn:
            log_fn("  PA: no hard movable macros; skipping")
        return _empty_stats(init_proxy)

    if log_fn:
        log_fn(
            f"  PA budget={time_budget_s:.0f}s, N={n_replicas}, ladder={n_ladder} "
            f"× sweeps={sweeps_per_step}, T0={T0:.2e}, Tf={Tf:.2e}, seed={seed}"
        )

    grid_lines_x, grid_lines_y = _grid_lines(plc)

    # 1. Build replicas (each owns its own evaluator + placement copy).
    t_build0 = time.perf_counter()
    replicas: List[Replica] = build_replicas(
        evaluator, benchmark, plc,
        n_replicas=n_replicas, base_seed=seed,
    )
    t_build = time.perf_counter() - t_build0
    if log_fn:
        log_fn(f"  PA: built {n_replicas} replicas in {t_build:.1f}s "
               f"(per-replica {t_build/max(1,n_replicas):.2f}s)")

    # 2. Pre-ladder diversification at T0.
    t_div0 = time.perf_counter()
    for i, r in enumerate(replicas):
        r_rng = np.random.default_rng(r.seed)
        sweep_replica(r, hard_movable, grid_lines_x, grid_lines_y, benchmark,
                      T=T0, n_steps=pre_ladder_diversify_steps,
                      rng=r_rng, breakpoint_budget=breakpoint_budget)
    t_div = time.perf_counter() - t_div0
    if log_fn:
        energies0 = [r.current_proxy for r in replicas]
        log_fn(f"  PA: diversified ({pre_ladder_diversify_steps} steps/repl) in {t_div:.1f}s "
               f"energy_range=[{min(energies0):.5f}, {max(energies0):.5f}] "
               f"mean={np.mean(energies0):.5f}")

    best_proxy = min(r.current_proxy for r in replicas)
    best_placement = next(r.placement.detach().clone() for r in replicas
                          if r.current_proxy == best_proxy)
    best_found_at_t = time.perf_counter() - t_start

    log_ratio = math.log(Tf / T0)
    n_resamples = 0
    accepted_total = 0
    proposed_total = 0
    clones_total = 0
    last_step = 0

    # 3. Temperature ladder.
    for k in range(n_ladder):
        elapsed = time.perf_counter() - t_start
        # Reserve ~5s for best-restore at end of ladder.
        if elapsed >= time_budget_s - 5.0:
            if log_fn:
                log_fn(f"  PA: time budget at step {k}/{n_ladder} "
                       f"(elapsed={elapsed:.1f}s/{time_budget_s:.0f}s)")
            break

        denom = max(1, n_ladder - 1)
        T_k = T0 * math.exp(log_ratio * k / denom)
        T_next = T0 * math.exp(log_ratio * min(k + 1, denom) / denom)

        # 3a. Sweep each replica at T_k.
        step_proposed = 0
        step_accepted = 0
        for r in replicas:
            r_rng = np.random.default_rng(r.seed + k * 13)
            acc = sweep_replica(r, hard_movable, grid_lines_x, grid_lines_y, benchmark,
                                T=T_k, n_steps=sweeps_per_step,
                                rng=r_rng, breakpoint_budget=breakpoint_budget)
            step_accepted += acc
            step_proposed += sweeps_per_step
            if r.current_proxy < best_proxy:
                best_proxy = r.current_proxy
                best_placement = r.placement.detach().clone()
                best_found_at_t = time.perf_counter() - t_start

        proposed_total += step_proposed
        accepted_total += step_accepted

        # 3b. Resample at T_k -> T_next.
        energies = np.array([r.current_proxy for r in replicas], dtype=np.float64)
        e_min = float(energies.min())
        beta_diff = 1.0 / max(T_next, 1e-12) - 1.0 / max(T_k, 1e-12)
        w_unnorm = np.exp(-beta_diff * (energies - e_min))
        # Cap to avoid overflow for extreme energy spreads.
        w_unnorm = np.clip(w_unnorm, 0.0, 1e12)
        counts = systematic_resample(w_unnorm, n_replicas, rng)
        new_replicas: List[Replica] = []
        clones_this_step = 0
        for r, c in zip(replicas, counts):
            if c == 0:
                continue
            new_replicas.append(r)  # original survives
            for _ in range(int(c) - 1):
                new_replicas.append(r.clone(benchmark, plc))
                clones_this_step += 1
        replicas = new_replicas
        clones_total += clones_this_step
        n_resamples += 1
        last_step = k + 1

        if log_fn:
            ene_range = (energies.min(), energies.max())
            log_fn(
                f"  PA step {k+1:3d}/{n_ladder} T={T_k:.2e} -> {T_next:.2e}  "
                f"best={best_proxy:.5f}  "
                f"energy=[{ene_range[0]:.5f}, {ene_range[1]:.5f}]  "
                f"acc/step={step_accepted/max(1,step_proposed)*100:.1f}%  "
                f"clones={clones_this_step}  elapsed={time.perf_counter()-t_start:.1f}s"
            )

    # 4. Restore evaluator to best placement via per-macro moves (keeps
    #    incremental caches consistent — direct tensor mutation would desync).
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
        log_fn(
            f"  PA done: init={init_proxy:.5f} best={best_proxy:.5f} "
            f"final={final_proxy:.5f}  ladder_steps={last_step}/{n_ladder}  "
            f"resamples={n_resamples}  clones={clones_total}  "
            f"wall={time.perf_counter()-t_start:.1f}s"
        )

    return {
        "proposed": proposed_total,
        "accepted_better": accepted_total,
        "accepted_worse": 0,  # PA doesn't split better/worse stats
        "rejected": proposed_total - accepted_total,
        "skipped": 0,
        "init_proxy": init_proxy,
        "best_proxy": best_proxy,
        "final_proxy": final_proxy,
        "improvement_vs_init": init_proxy - best_proxy,
        "best_found_at_t": best_found_at_t,
        "wall_total_s": time.perf_counter() - t_start,
        "best_restored": True,
        # PA-specific extras (callers may ignore).
        "pa_n_replicas": n_replicas,
        "pa_n_ladder_steps": last_step,
        "pa_n_resamples": n_resamples,
        "pa_n_clones": clones_total,
    }


def _empty_stats(init_proxy: float) -> dict:
    return {
        "proposed": 0, "accepted_better": 0, "accepted_worse": 0,
        "rejected": 0, "skipped": 0,
        "init_proxy": init_proxy, "best_proxy": init_proxy,
        "final_proxy": init_proxy, "improvement_vs_init": 0.0,
        "best_found_at_t": 0.0, "wall_total_s": 0.0,
        "best_restored": False,
        "pa_n_replicas": 0, "pa_n_ladder_steps": 0,
        "pa_n_resamples": 0, "pa_n_clones": 0,
    }
