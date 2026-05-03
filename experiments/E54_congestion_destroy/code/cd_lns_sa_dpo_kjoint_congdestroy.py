"""E54 — E41 pipeline with congestion-targeted LNS destroy ranking.

Replaces E39's `_cost_aware_destroy` (rank by move-to-center total
Δproxy) with `_congestion_aware_destroy` (rank by direct contribution
to abu-top-5 % cells, which is what the proxy's congestion term
actually measures). All other phases (DPO init, CD, SA-v2, K-joint K=3
top_N=5) match E41.

Hypothesis: 74 % of proxy is congestion (E8 diagnostic), but no destroy
heuristic in the pipeline explicitly targets congestion. A destroy
ranking aligned with the dominant cost component should find slack the
total-Δproxy ranking misses.

Reference:
- E41 — parent pipeline.
- E39 — LNS-gridbin reinsert + K-joint primitives.
- E18 — DPO init.
- E8 (`analysis/lp_hpwl_diagnostic/`) — proxy decomposition motivating
  the destroy direction.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics

from experiments.E18_dpo_init.code.cd_lns_sa_dpo_init import _best_of_v2_init
from experiments.E39_kmacro_joint_lns.code.cd_lns_sa_kjoint import (
    _gridbin_reinsert,
    run_kjoint_lns,
    run_sa_polish_v2,
)


# ── Congestion-aware destroy ───────────────────────────────────────────────


def _compute_top_cong_cells(
    evaluator: IncrementalProxyEvaluator,
    top_pct: float = 0.05,
):
    """Replicate the abu-top-5 % calculation in `_congestion_cost`.

    Returns ``(top_V_set, top_H_set)`` where each is a set of cell
    indices that fall in the abu top-fraction of (V_total | H_total).
    Matches the cells whose congestion drives the proxy's congestion
    term.
    """
    H_macro_norm = evaluator.H_macro_cong / evaluator.grid_h_routes
    V_macro_norm = evaluator.V_macro_cong / evaluator.grid_v_routes
    H_net_norm = evaluator.H_net_cong / evaluator.grid_h_routes
    V_net_norm = evaluator.V_net_cong / evaluator.grid_v_routes
    H_smoothed = evaluator._smooth(H_net_norm, vertical=False)
    V_smoothed = evaluator._smooth(V_net_norm, vertical=True)
    H_total = H_smoothed + H_macro_norm
    V_total = V_smoothed + V_macro_norm
    combined = torch.cat([V_total, H_total])
    cnt = max(1, int(math.floor(combined.numel() * top_pct)))
    _, top_indices = torch.topk(combined, cnt)
    n_cells = evaluator.num_cells
    top_V_set = set()
    top_H_set = set()
    for i in top_indices.tolist():
        if i < n_cells:
            top_V_set.add(i)
        else:
            top_H_set.add(i - n_cells)
    return top_V_set, top_H_set


def _congestion_aware_destroy(
    evaluator: IncrementalProxyEvaluator,
    hard_movable: List[int],
    K: int,
    top_pct: float = 0.05,
    include_net_share: bool = True,
) -> List[int]:
    """Pick K hard movables most contributing to abu-top-5 % congested cells.

    For each macro M, score = direct macro routing into top cells +
    sum over connected nets of (net's contribution to top cells / n_pins).
    The 1/n_pins weighting is a defensible default — a macro on a 10-pin
    net is one of 10 contributors to that net's bbox. Both terms are
    O(per-macro contrib dict size) reads from cached state; no proxy
    evaluations needed.

    Convention from `IncrementalProxyEvaluator`:
        macro_cong_contrib[m] = {(orient, cell): val} where
            orient=0 → H, orient=1 → V.
    """
    top_V_set, top_H_set = _compute_top_cong_cells(evaluator, top_pct=top_pct)

    scores: List[tuple] = []
    for idx in hard_movable:
        score = 0.0
        for (orient, cell), val in evaluator.macro_cong_contrib[idx].items():
            if orient == 0 and cell in top_H_set:
                score += val
            elif orient == 1 and cell in top_V_set:
                score += val
        if include_net_share:
            for n in evaluator.macro_to_nets[idx].tolist():
                pins = evaluator.net_pins[n]
                n_pins = int(pins.shape[0]) if hasattr(pins, "shape") else len(pins)
                if n_pins == 0:
                    continue
                pin_share = 1.0 / n_pins
                for (orient, cell), val in evaluator.net_cong_contrib[n].items():
                    if orient == 0 and cell in top_H_set:
                        score += val * pin_share
                    elif orient == 1 and cell in top_V_set:
                        score += val * pin_share
        scores.append((idx, score))
    scores.sort(key=lambda s: -s[1])
    return [s[0] for s in scores[:K]]


# ── LNS-gridbin with congestion-aware destroy ──────────────────────────────


def run_lns_gridbin_congestion(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    hard_movable: List[int],
    time_budget_s: float,
    destroy_frac: float = 0.05,
    destroy_cap: int = 30,
    top_pct: float = 0.05,
    include_net_share: bool = True,
    seed: int = 42,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Grid-bin LNS phase with congestion-aware destroy.

    Mirrors `E39.run_lns_gridbin` but swaps `_cost_aware_destroy` for
    `_congestion_aware_destroy`. Reinsert step (`_gridbin_reinsert`)
    is unchanged.
    """
    K = max(1, min(destroy_cap, int(destroy_frac * len(hard_movable))))
    macro_sizes_np = evaluator.macro_sizes.cpu().numpy()
    n_hard = benchmark.num_hard_macros

    if log_fn is not None:
        log_fn(
            f"  LNS-cong budget={time_budget_s:.0f}s, destroy K={K} "
            f"(={destroy_frac*100:.1f}% of {len(hard_movable)} hard movables, "
            f"capped at {destroy_cap}), strategy=congestion_aware "
            f"(top_pct={top_pct}, net_share={include_net_share})"
        )

    sample = 0
    total_improvement = 0.0
    t_destroy_total = 0.0
    t_start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:
            break
        sample += 1
        sample_t0 = time.perf_counter()

        td0 = time.perf_counter()
        destroy = _congestion_aware_destroy(
            evaluator, hard_movable, K,
            top_pct=top_pct, include_net_share=include_net_share,
        )
        t_destroy_total += time.perf_counter() - td0

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
        if log_fn is not None:
            log_fn(
                f"  LNS-cong sample {sample}: K={K}, Δ={sample_delta:+.5f} "
                f"(moves={moves_this_sample}/{K}), proxy={cur_proxy:.5f}, "
                f"sample_wall={sample_wall:.1f}s, elapsed={elapsed:.1f}s"
            )

        if abs(sample_delta) < 1e-7:
            if log_fn is not None:
                log_fn(f"  LNS-cong converged at sample {sample} (no improvement)")
            break

    return {
        "samples": sample,
        "total_improvement": total_improvement,
        "wall_total_s": time.perf_counter() - t_start,
        "destroy_wall_s": t_destroy_total,
    }


# ── Placer ─────────────────────────────────────────────────────────────────


class CDLNSSADPOKJointCongDestroyPlacer:
    """E54 placer: E41 pipeline with congestion-aware LNS destroy.

    Identical to E41 except step 5 (LNS-gridbin) uses
    `_congestion_aware_destroy` in place of `_cost_aware_destroy`.
    """

    def __init__(
        self,
        cd_hard_cap_s: float = 2400.0,
        cd_min_time_s: float = 300.0,
        cd_patience: int = 3,
        cd_plateau_threshold: float = 0.001,
        lns_budget_s: float = 600.0,
        lns_destroy_frac: float = 0.05,
        lns_destroy_cap: int = 30,
        lns_top_pct: float = 0.05,
        lns_include_net_share: bool = True,
        lns_seed: int = 42,
        sa_budget_s: float = 600.0,
        sa_T0: float = 5e-4,
        sa_Tf: float = 1e-6,
        sa_seed: int = 42,
        sa_breakpoint_budget: int = 12,
        kjoint_budget_s: float = 600.0,
        kjoint_K: int = 3,
        kjoint_top_N: int = 5,
        kjoint_seed: int = 42,
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
        self.lns_top_pct = float(lns_top_pct)
        self.lns_include_net_share = bool(lns_include_net_share)
        self.lns_seed = int(lns_seed)
        self.sa_budget_s = float(sa_budget_s)
        self.sa_T0 = float(sa_T0)
        self.sa_Tf = float(sa_Tf)
        self.sa_seed = int(sa_seed)
        self.sa_breakpoint_budget = int(sa_breakpoint_budget)
        self.kjoint_budget_s = float(kjoint_budget_s)
        self.kjoint_K = int(kjoint_K)
        self.kjoint_top_N = int(kjoint_top_N)
        self.kjoint_seed = int(kjoint_seed)
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
            f"=== CDLNSSADPOKJointCongDestroyPlacer ({benchmark.name}): "
            f"DPO -> CD -> LNS-CONG -> SA -> KJoint ==="
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

        movable = [i for i in range(benchmark.num_macros)
                   if not bool(benchmark.macro_fixed[i])]
        hard_movable = [i for i in range(benchmark.num_hard_macros)
                        if not bool(benchmark.macro_fixed[i])]

        # 4. CD phase.
        self._log(f"  starting CD phase (cap={self.cd_hard_cap_s:.0f}s)")
        cd_stats = run_cd_adaptive(
            evaluator=evaluator, benchmark=benchmark, plc=plc,
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

        # 5. LNS-cong phase (the change).
        self._log(f"  starting LNS-cong phase (budget={self.lns_budget_s:.0f}s)")
        lns_stats = run_lns_gridbin_congestion(
            evaluator=evaluator, benchmark=benchmark, plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.lns_budget_s,
            destroy_frac=self.lns_destroy_frac,
            destroy_cap=self.lns_destroy_cap,
            top_pct=self.lns_top_pct,
            include_net_share=self.lns_include_net_share,
            seed=self.lns_seed,
            log_fn=self._log if self.verbose else None,
        )
        lns_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  LNS-cong done: samples={lns_stats['samples']}, "
            f"Δ={lns_stats['total_improvement']:+.5f}, "
            f"wall={lns_stats['wall_total_s']:.1f}s "
            f"(destroy_wall={lns_stats['destroy_wall_s']:.2f}s), "
            f"proxy={lns_proxy:.5f}"
        )

        # 6. SA-v2 phase.
        self._log(f"  starting SA-v2 phase (budget={self.sa_budget_s:.0f}s)")
        sa_stats = run_sa_polish_v2(
            evaluator=evaluator, benchmark=benchmark, plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.sa_budget_s,
            T0=self.sa_T0, Tf=self.sa_Tf,
            seed=self.sa_seed,
            breakpoint_budget=self.sa_breakpoint_budget,
            log_fn=self._log if self.verbose else None,
        )
        sa_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  SA-v2 done: lift_vs_init={sa_stats['improvement_vs_init']:+.5f}, "
            f"best={sa_stats['best_proxy']:.5f}, proxy={sa_proxy:.5f}"
        )

        # 7. K-joint LNS phase.
        self._log(
            f"  starting K-joint phase (budget={self.kjoint_budget_s:.0f}s, "
            f"K={self.kjoint_K}, top_N={self.kjoint_top_N})"
        )
        kj_stats = run_kjoint_lns(
            evaluator=evaluator, benchmark=benchmark, plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.kjoint_budget_s,
            K=self.kjoint_K,
            top_N=self.kjoint_top_N,
            seed=self.kjoint_seed,
            log_fn=self._log if self.verbose else None,
        )
        final_cost = evaluator.current_cost()
        self._log(
            f"  K-joint done: passes={kj_stats['passes']}, "
            f"tuples_tried={kj_stats['ktuples_tried']}, "
            f"committed={kj_stats['ktuples_committed']}, "
            f"Δ={kj_stats['total_improvement']:+.5f}, "
            f"wall={kj_stats['wall_total_s']:.1f}s, "
            f"final proxy={final_cost['proxy']:.5f}"
        )
        self._log(
            f"  TOTAL lifts: CD={init_cost['proxy'] - cd_proxy:+.5f}, "
            f"LNS-cong={cd_proxy - lns_proxy:+.5f}, "
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
                f"CDLNSSADPOKJointCongDestroyPlacer produced "
                f"{overlaps['overlap_count']} overlaps "
                f"(area {overlaps['total_overlap_area']:.4f}) on "
                f"'{benchmark.name}'"
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(DPO + project + CD + LNS-cong + SA + KJoint + validate)"
        )
        return final_placement
