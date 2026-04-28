"""E12 — CDAdaptive + grid-bin LNS escape phase.

Hypothesis: today's failed LNS test (subset-CD reinsertion, 0/24 samples improved
baseline) used the WRONG move type. CD's per-axis line-search rediscovers the
same per-axis fixed point. The actual LNS recipe from the docs uses **grid-bin
search**: for each destroyed macro, evaluate the proxy at every (col, row)
grid cell center, pick the proxy-minimizing legal position. Different move
type — can find non-axis-aligned optima that single-axis CD can't reach.

Algorithm (per benchmark):
  1. Run CDAdaptive (threshold=0.001, cap=3000s — leaves 600s for LNS)
  2. LNS phase: until budget exhausted OR no-improvement sample:
       a. Score all hard movables by per-macro cost contribution
          (= proxy delta when temporarily moved to canvas center)
       b. Pick top-K (K = max(1, min(destroy_cap, int(destroy_frac * |H|))))
       c. For each destroyed macro in score order:
            - Enumerate grid bin centers (col, row) for legal candidates
            - Move there, query proxy via incremental evaluator, revert
            - Commit move to the best legal position if it improves proxy
  3. Validate (zero overlaps), return.

All hyperparameters are global. No per-benchmark tuning. Adaptive criteria
(K from |H|, K threshold from contest legal limit) ensure generality.

Hard limit: total wall ≤ 3600s/benchmark (contest legal limit).
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

# This file lives at submissions/cd_lns_gridbin/, so repo root is 3 levels up.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics

from macro_place.bench_paths import find_benchmark_dir
from submissions.cd_adaptive.placer import (
    CDAdaptivePlacer,
    run_cd_adaptive,
)


_THIS_FILE = Path(__file__).resolve()
_DIAGNOSTIC_PATH = _ROOT / "scripts" / "cd_ibm10_diagnostic.py"


def _import_diagnostic():
    if "cd_ibm10_diagnostic" in sys.modules:
        return sys.modules["cd_ibm10_diagnostic"]
    spec = importlib.util.spec_from_file_location(
        "cd_ibm10_diagnostic", str(_DIAGNOSTIC_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cd_ibm10_diagnostic"] = mod
    spec.loader.exec_module(mod)
    return mod


_diag = _import_diagnostic()
sdf_init = _diag.sdf_init
project_overlaps = _diag.project_overlaps


# ── LNS primitives ──────────────────────────────────────────────────────────


def _cost_aware_destroy(
    evaluator: IncrementalProxyEvaluator,
    hard_movable: List[int],
    K: int,
) -> List[int]:
    """Pick K hard movable macros to destroy.

    Score each macro by the proxy delta produced by temporarily moving it to
    canvas center. A negative delta means moving-to-center improves proxy →
    that macro is currently in a costly position → good destroy candidate.
    Pick the K macros with the smallest (most negative) deltas.
    """
    cw = evaluator.width / 2.0
    ch = evaluator.height / 2.0
    baseline_p = evaluator.current_cost()["proxy"]
    scores: List[tuple] = []
    for idx in hard_movable:
        try:
            evaluator.move(idx, (cw, ch))
            p = evaluator.current_cost()["proxy"]
            evaluator.revert()
            scores.append((idx, p - baseline_p))
        except Exception:
            scores.append((idx, 0.0))
    scores.sort(key=lambda s: s[1])
    return [s[0] for s in scores[:K]]


def _random_destroy(
    hard_movable: List[int],
    K: int,
    rng: np.random.Generator,
) -> List[int]:
    """Pick K hard movable macros uniformly at random (ablation baseline)."""
    if K >= len(hard_movable):
        return list(hard_movable)
    return sorted(rng.choice(hard_movable, size=K, replace=False).tolist())


def _is_legal_2d(
    idx: int,
    x: float,
    y: float,
    placement: torch.Tensor,
    macro_sizes_np: np.ndarray,
    n_hard: int,
    eps: float = 1e-4,
) -> bool:
    """True iff placing hard macro `idx` at (x, y) does not overlap any other
    hard macro at its current position. Soft macros never block."""
    if idx >= n_hard:
        return True
    half_w = float(macro_sizes_np[idx, 0]) / 2.0
    half_h = float(macro_sizes_np[idx, 1]) / 2.0
    pos = placement[:n_hard].cpu().numpy().astype(np.float64)
    sz = macro_sizes_np[:n_hard]
    dx = np.abs(pos[:, 0] - x)
    dy = np.abs(pos[:, 1] - y)
    min_dx = half_w + sz[:, 0] / 2.0
    min_dy = half_h + sz[:, 1] / 2.0
    blockers = (dx < min_dx - eps) & (dy < min_dy - eps)
    blockers[idx] = False  # exclude self
    return not bool(np.any(blockers))


def _gridbin_reinsert(
    evaluator: IncrementalProxyEvaluator,
    macro_idx: int,
    plc,
    n_hard: int,
    macro_sizes_np: np.ndarray,
    deadline_s: float,
    t_start: float,
) -> float:
    """Search every legal grid-bin center for macro_idx, commit the best.

    Returns the proxy delta (negative = improvement, 0.0 if no improving move).
    """
    cw = float(plc.width)
    ch = float(plc.height)
    grid_w = cw / plc.grid_col
    grid_h = ch / plc.grid_row
    half_w = float(macro_sizes_np[macro_idx, 0]) / 2.0
    half_h = float(macro_sizes_np[macro_idx, 1]) / 2.0

    cur_cost = evaluator.current_cost()["proxy"]
    cur_xy = (
        float(evaluator.placement[macro_idx, 0]),
        float(evaluator.placement[macro_idx, 1]),
    )
    best_cost = cur_cost
    best_xy = cur_xy

    for col in range(int(plc.grid_col)):
        if time.perf_counter() - t_start >= deadline_s:
            break
        cx = (col + 0.5) * grid_w
        if cx < half_w or cx > cw - half_w:
            continue
        for row in range(int(plc.grid_row)):
            if time.perf_counter() - t_start >= deadline_s:
                break
            cy = (row + 0.5) * grid_h
            if cy < half_h or cy > ch - half_h:
                continue
            if not _is_legal_2d(
                macro_idx, cx, cy, evaluator.placement, macro_sizes_np, n_hard
            ):
                continue
            evaluator.move(macro_idx, (cx, cy))
            cost = evaluator.current_cost()["proxy"]
            evaluator.revert()
            if cost < best_cost - 1e-9:
                best_cost = cost
                best_xy = (cx, cy)

    if best_cost < cur_cost - 1e-9 and (
        abs(best_xy[0] - cur_xy[0]) > 1e-7 or abs(best_xy[1] - cur_xy[1]) > 1e-7
    ):
        evaluator.move(macro_idx, best_xy)
        return best_cost - cur_cost
    return 0.0


def run_lns_gridbin(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    hard_movable: List[int],
    time_budget_s: float,
    destroy_frac: float = 0.05,
    destroy_cap: int = 30,
    destroy_strategy: str = "cost_aware",
    seed: int = 42,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Grid-bin LNS phase. Iterates destroy + reinsert until budget exhausted
    or a full sample produces no improvement.

    destroy_strategy: 'cost_aware' (default) or 'random' (ablation).
    """
    if destroy_strategy not in ("cost_aware", "random"):
        raise ValueError(f"unknown destroy_strategy: {destroy_strategy}")

    K = max(1, min(destroy_cap, int(destroy_frac * len(hard_movable))))
    macro_sizes_np = evaluator.macro_sizes.cpu().numpy()
    n_hard = benchmark.num_hard_macros
    rng = np.random.default_rng(seed=seed)

    if log_fn is not None:
        log_fn(
            f"  LNS budget={time_budget_s:.0f}s, destroy K={K} "
            f"(={destroy_frac*100:.1f}% of {len(hard_movable)} hard movables, "
            f"capped at {destroy_cap}), strategy={destroy_strategy}"
        )

    sample = 0
    total_improvement = 0.0
    t_start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:
            break
        sample += 1
        sample_t0 = time.perf_counter()

        if destroy_strategy == "cost_aware":
            destroy = _cost_aware_destroy(evaluator, hard_movable, K)
        else:
            destroy = _random_destroy(hard_movable, K, rng)

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
                f"  LNS sample {sample}: K={K}, Δ={sample_delta:+.5f} "
                f"(moves={moves_this_sample}/{K}), proxy={cur_proxy:.5f}, "
                f"sample_wall={sample_wall:.1f}s, total elapsed={elapsed:.1f}s"
            )

        if abs(sample_delta) < 1e-7:
            if log_fn is not None:
                log_fn(f"  LNS converged at sample {sample} (no improvement)")
            break

    return {
        "samples": sample,
        "total_improvement": total_improvement,
        "wall_total_s": time.perf_counter() - t_start,
    }


# ── Placer ──────────────────────────────────────────────────────────────────


class CDLNSGridBinPlacer:
    """E12 placer: CDAdaptive (with E16's tighter threshold) + grid-bin LNS phase.

    Time budget allocation per benchmark:
      * CD phase: hard_cap_s = 3000s (legal limit minus LNS reservation)
      * LNS phase: lns_budget_s = 600s
      * Total: ≤ 3600s (contest legal limit)
    """

    def __init__(
        self,
        cd_hard_cap_s: float = 3000.0,
        cd_min_time_s: float = 300.0,
        cd_patience: int = 3,
        cd_plateau_threshold: float = 0.001,
        lns_budget_s: float = 600.0,
        lns_destroy_frac: float = 0.05,
        lns_destroy_cap: int = 30,
        lns_destroy_strategy: str = "cost_aware",
        lns_seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.cd_hard_cap_s = float(cd_hard_cap_s)
        self.cd_min_time_s = float(cd_min_time_s)
        self.cd_patience = int(cd_patience)
        self.cd_plateau_threshold = float(cd_plateau_threshold)
        self.lns_budget_s = float(lns_budget_s)
        self.lns_destroy_frac = float(lns_destroy_frac)
        self.lns_destroy_cap = int(lns_destroy_cap)
        self.lns_destroy_strategy = str(lns_destroy_strategy)
        self.lns_seed = int(lns_seed)
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _load_plc_for(self, benchmark: Benchmark):
        bench_dir = find_benchmark_dir(benchmark.name)
        return load_benchmark_from_dir(str(bench_dir))

    # ------------------------------------------------------------------

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_total0 = time.perf_counter()
        self._log(
            f"=== CDLNSGridBinPlacer ({benchmark.name}): "
            f"CD cap={self.cd_hard_cap_s:.0f}s, LNS budget={self.lns_budget_s:.0f}s ==="
        )
        self._log(
            f"  num_macros={benchmark.num_macros}, "
            f"n_hard={benchmark.num_hard_macros}, "
            f"canvas={benchmark.canvas_width:.1f}x{benchmark.canvas_height:.1f}"
        )

        # 1. SDF init
        t_init0 = time.perf_counter()
        placement = sdf_init(benchmark)
        self._log(f"  SDF init wall = {time.perf_counter() - t_init0:.1f} s")

        # 2. Project overlaps
        placement, proj_iters = project_overlaps(placement, benchmark)
        init_overlaps = compute_overlap_metrics(placement, benchmark)
        self._log(
            f"  overlap projection: {proj_iters} iters, "
            f"residual count={init_overlaps['overlap_count']}"
        )

        # 3. Build evaluator
        _, plc = self._load_plc_for(benchmark)
        placement_f64 = placement.detach().clone().to(torch.float64)
        evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
        init_cost = evaluator.current_cost()
        self._log(
            f"  init proxy={init_cost['proxy']:.5f} "
            f"[wl={init_cost['wl']:.4f} d={init_cost['density']:.4f} "
            f"c={init_cost['congestion']:.4f}]"
        )

        # 4. CD phase
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        hard_movable = [
            i for i in range(benchmark.num_hard_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        self._log(f"  starting CD phase (cap={self.cd_hard_cap_s:.0f}s)")
        cd_stats = run_cd_adaptive(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
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

        # 5. LNS phase
        self._log(f"  starting LNS phase (budget={self.lns_budget_s:.0f}s)")
        lns_stats = run_lns_gridbin(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.lns_budget_s,
            destroy_frac=self.lns_destroy_frac,
            destroy_cap=self.lns_destroy_cap,
            destroy_strategy=self.lns_destroy_strategy,
            seed=self.lns_seed,
            log_fn=self._log if self.verbose else None,
        )
        final_cost = evaluator.current_cost()
        self._log(
            f"  LNS done: samples={lns_stats['samples']}, "
            f"Δ={lns_stats['total_improvement']:+.5f}, "
            f"wall={lns_stats['wall_total_s']:.1f}s, "
            f"final proxy={final_cost['proxy']:.5f}"
        )

        # 6. Pull placement back; preserve fixed macros
        final_placement_f64 = evaluator.placement.detach().clone().cpu()
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            final_placement_f64[fixed_mask] = original_positions[fixed_mask]
        final_placement = final_placement_f64.to(torch.float32)

        overlaps = compute_overlap_metrics(final_placement, benchmark)
        if overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"CDLNSGridBinPlacer produced {overlaps['overlap_count']} overlaps "
                f"(area {overlaps['total_overlap_area']:.4f}) on '{benchmark.name}'"
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(SDF + project + CD + LNS + validate)"
        )
        return final_placement
