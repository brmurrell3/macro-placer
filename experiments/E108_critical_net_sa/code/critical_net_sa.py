"""E108 — critical-net-weighted SA polish.

Drop-in replacement for `run_sa_polish_v2` from `submissions/cd_lns_sa/placer.py`,
but samples macros by `w[i] = Σ_j∈nets(i) deg(j)²` instead of uniformly.

Mechanism: HPWL is bench-wide dominated by high-degree nets. Quadratic
weighting concentrates SA moves on macros connected to high-leverage nets.
The choice is bench-agnostic — same heuristic everywhere; per-bench weight
distribution emerges from each bench's net topology.

Compute once at SA entry (O(num_nets) build, O(num_macros) for weights).
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.cd_core import _grid_lines, axis_breakpoints, legal_axis_range
from macro_place.incremental_evaluator import IncrementalProxyEvaluator


def compute_critical_net_weights(
    benchmark: Benchmark,
    hard_movable: List[int],
    *,
    weight_power: float = 2.0,
    use_net_weights: bool = True,
) -> np.ndarray:
    """Per-macro weight = Σ_{j ∈ nets(i)} net_weight[j] * deg(j)^weight_power.

    Returns weights for the `hard_movable` subset only (in their listed order).
    Falls back to uniform if all weights are zero (no nets reach hard_movable).
    """
    n_macros = benchmark.num_macros
    macro_w = np.zeros(n_macros, dtype=np.float64)

    nets = benchmark.net_nodes
    net_weights = None
    if use_net_weights and hasattr(benchmark, "net_weights"):
        net_weights = benchmark.net_weights.cpu().numpy()

    for j, net in enumerate(nets):
        nodes = net.cpu().numpy().astype(np.int64)
        nodes = nodes[nodes < n_macros]  # filter ports
        if nodes.size < 2:
            continue
        deg = float(nodes.size)
        nw = float(net_weights[j]) if net_weights is not None else 1.0
        contrib = nw * (deg ** weight_power)
        np.add.at(macro_w, nodes, contrib)

    sub = macro_w[np.asarray(hard_movable, dtype=np.int64)]
    total = float(sub.sum())
    if total <= 0.0:
        return np.full(len(hard_movable), 1.0 / max(1, len(hard_movable)), dtype=np.float64)
    return sub / total


def run_sa_polish_v2_critnet(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    hard_movable: List[int],
    time_budget_s: float,
    *,
    T0: float = 5e-4,
    Tf: float = 1e-6,
    seed: int = 42,
    breakpoint_budget: int = 12,
    weight_power: float = 2.0,
    mix_uniform: float = 0.0,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """SA polish v2 with net-degree²-weighted macro sampling.

    `weight_power`: exponent on net degree in per-macro weight (2.0 default).
    `mix_uniform`: probability of falling back to uniform sampling on a step
        (0.0 = pure critical-net; 1.0 = pure uniform; 0.1 = 90% critnet + 10% uniform).
        Insurance against under-sampling low-degree but critical macros.
    """
    rng = np.random.default_rng(seed=seed)
    grid_lines_x, grid_lines_y = _grid_lines(plc)
    n_hard = benchmark.num_hard_macros

    if not hard_movable:
        if log_fn is not None:
            log_fn("  SA-critnet: no hard movable macros; skipping")
        return {
            "proposed": 0, "accepted_better": 0, "accepted_worse": 0,
            "rejected": 0, "skipped": 0,
            "init_proxy": float("nan"), "best_proxy": float("nan"),
            "final_proxy": float("nan"), "improvement_vs_init": 0.0,
            "best_found_at_t": 0.0, "wall_total_s": 0.0,
            "best_restored": False,
        }

    weights = compute_critical_net_weights(
        benchmark, hard_movable, weight_power=weight_power, use_net_weights=True
    )
    hard_arr = np.asarray(hard_movable, dtype=np.int64)
    uniform_w = np.full_like(weights, 1.0 / len(weights))

    if log_fn is not None:
        wmax = float(weights.max())
        wmin = float(weights[weights > 0].min()) if (weights > 0).any() else 0.0
        log_fn(
            f"  SA-critnet budget={time_budget_s:.0f}s, T0={T0:.2e}, Tf={Tf:.2e}, "
            f"|H|={len(hard_movable)}, weight_power={weight_power:.1f}, "
            f"mix_uniform={mix_uniform:.2f}, w_min={wmin:.2e}, w_max={wmax:.2e}, "
            f"seed={seed}"
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

        # Critical-net weighted sampling (with optional uniform mix).
        if mix_uniform > 0.0 and rng.random() < mix_uniform:
            idx = int(rng.choice(hard_arr, p=uniform_w))
        else:
            idx = int(rng.choice(hard_arr, p=weights))
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
                f"  SA-critnet t={now - t_start:6.1f}s T={T:.2e} "
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
            f"  SA-critnet done: proposed={proposed}, better={accepted_better}, "
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
