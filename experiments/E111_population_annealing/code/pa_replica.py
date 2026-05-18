"""Population annealing replica = (IncrementalProxyEvaluator on its own
placement copy) + cached current_proxy.

`clone()` deep-copies the placement and rebuilds a fresh evaluator. The
evaluator's per-net/per-cell dicts are rebuilt in __init__; cloning is
O(num_nets) which is the dominant overhead in PA (mitigated by only
cloning replicas with resample_count ≥ 2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from macro_place.cd_core import axis_breakpoints, legal_axis_range
from macro_place.incremental_evaluator import IncrementalProxyEvaluator


@dataclass
class Replica:
    evaluator: IncrementalProxyEvaluator
    current_proxy: float
    seed: int

    @property
    def placement(self) -> torch.Tensor:
        return self.evaluator.placement

    def clone(self, benchmark, plc) -> "Replica":
        new_placement = self.evaluator.placement.detach().clone()
        new_eval = IncrementalProxyEvaluator(benchmark, plc, new_placement)
        return Replica(
            evaluator=new_eval,
            current_proxy=self.current_proxy,
            seed=self.seed,
        )


def build_replicas(
    parent_evaluator,
    benchmark,
    plc,
    *,
    n_replicas: int,
    base_seed: int,
) -> List[Replica]:
    """N independent evaluators on copies of parent_evaluator's placement."""
    base_placement = parent_evaluator.placement.detach().clone()
    replicas: List[Replica] = []
    for i in range(n_replicas):
        ev = IncrementalProxyEvaluator(benchmark, plc, base_placement.clone())
        replicas.append(Replica(
            evaluator=ev,
            current_proxy=float(ev.current_cost()["proxy"]),
            seed=base_seed + 1000 * i + 7,
        ))
    return replicas


def sweep_replica(
    replica: Replica,
    hard_movable: List[int],
    grid_lines_x: np.ndarray,
    grid_lines_y: np.ndarray,
    benchmark,
    *,
    T: float,
    n_steps: int,
    rng: np.random.Generator,
    breakpoint_budget: int = 12,
) -> int:
    """Metropolis sweep on canonical proxy. Returns count of accepted moves.

    Move set is `axis_breakpoints` — identical to SA-v2. Accepts on
    Δproxy ≤ 0, or with probability exp(-Δ/T). Reverts on reject.
    """
    ev = replica.evaluator
    n_hard = benchmark.num_hard_macros
    cur = replica.current_proxy
    accepted = 0

    for _ in range(n_steps):
        idx = int(hard_movable[rng.integers(0, len(hard_movable))])
        axis = int(rng.integers(0, 2))
        try:
            lo, hi = legal_axis_range(
                idx, ev.placement, ev.macro_sizes, benchmark.macro_fixed,
                n_hard, axis=axis,
                canvas_w=benchmark.canvas_width,
                canvas_h=benchmark.canvas_height,
            )
        except Exception:
            continue
        if hi - lo < 1e-5:
            continue
        cur_axis_val = float(ev.placement[idx, axis])
        gl = grid_lines_x if axis == 0 else grid_lines_y
        try:
            cands = axis_breakpoints(
                idx, axis, ev, gl, lo, hi,
                max_breakpoints=breakpoint_budget,
                cur_axis=cur_axis_val,
            )
        except Exception:
            continue
        if len(cands) == 0:
            continue
        new_v = float(cands[rng.integers(0, len(cands))])
        new_xy = [float(ev.placement[idx, 0]), float(ev.placement[idx, 1])]
        new_xy[axis] = new_v
        try:
            new_proxy = ev.move(idx, tuple(new_xy))["proxy"]
        except Exception:
            continue
        d = new_proxy - cur
        if d <= 0.0:
            cur = new_proxy
            accepted += 1
        else:
            # Metropolis acceptance.
            p_accept = float(np.exp(-d / max(T, 1e-12)))
            if rng.random() < p_accept:
                cur = new_proxy
                accepted += 1
            else:
                ev.revert()
    replica.current_proxy = cur
    return accepted
