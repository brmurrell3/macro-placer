"""
CDAdaptivePlacer — E9: per-benchmark plateau detection with a 1hr hard cap.

CDOnlyPlacer (E2 productionized) uses a fixed wall-clock budget per benchmark.
That's one-size-fits-all: easy benchmarks (e.g. ibm09) plateau at ~3 min and
leave the remaining time on the table; hard ones (ibm17/18) cap out mid-descent.

E9 replaces the static budget with **per-benchmark plateau detection**: each
benchmark runs until either its proxy delta has been below `plateau_threshold`
for `patience` consecutive sweeps (and we've spent at least `min_time_s`), or
the absolute `hard_cap_s` (1 hr by default — matches the competition rule) is
reached. Easy benchmarks exit early; hard ones use the full hour if still
improving. Transfers to NG45 hidden test without per-bench tuning.

CD primitives and the adaptive sweep loop (``run_cd_adaptive``) live in
``macro_place.cd_core``; this file is just the placer-class wrapper.

Constructor args:
    min_time_s: float — don't check plateau before this much wall clock
        (default 300 = 5 min).
    hard_cap_s: float — absolute upper bound on wall clock (default 3600 = 1 hr).
    patience: int — last N sweeps must all be below threshold (default 3).
    plateau_threshold: float — absolute proxy delta per sweep (default 0.005).
    init_strategy: 'sdf' (default).
    verbose: print per-sweep progress (default True).

Returned placement:
    [num_macros, 2] tensor (float32, on CPU). Hard macros legalized to zero
    overlaps; fixed macros never moved; canvas bounds respected. Validated
    via `compute_overlap_metrics` before return.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import torch

# Eval harness loads us via importlib.spec_from_file_location and does NOT add
# the repo root to sys.path; do it ourselves so `macro_place.*` imports resolve.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir as _find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics


# ── Placer class ────────────────────────────────────────────────────────────


class CDAdaptivePlacer:
    """Adaptive CD placer — E9.

    Pipeline per call to `place(benchmark)`:
      1. SDF init (via SDFPlacer in macro_place/sdf_init.py)
      2. Iterative push-apart projection to clean any residual overlaps
      3. Build IncrementalProxyEvaluator (full-proxy: WL + density + congestion)
      4. Adaptive CD sweeps — exit on plateau or 1hr hard cap
      5. Validate: zero overlaps, fixed macros not moved
    """

    def __init__(
        self,
        min_time_s: float = 300.0,
        hard_cap_s: float = 3600.0,
        patience: int = 3,
        plateau_threshold: float = 0.005,
        init_strategy: str = "sdf",
        verbose: bool = True,
    ) -> None:
        self.min_time_s = float(min_time_s)
        self.hard_cap_s = float(hard_cap_s)
        self.patience = int(patience)
        self.plateau_threshold = float(plateau_threshold)
        if init_strategy != "sdf":
            raise ValueError(
                f"Only init_strategy='sdf' is currently supported "
                f"(got '{init_strategy}')"
            )
        self.init_strategy = init_strategy
        self.verbose = bool(verbose)

    # ----------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _load_plc_for(self, benchmark: Benchmark):
        """Load a fresh PlacementCost for this benchmark."""
        bench_dir = _find_benchmark_dir(benchmark.name)
        return load_benchmark_from_dir(str(bench_dir))

    # ----------------------------------------------------------------------

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """Run the adaptive CD pipeline and return final placement."""
        t_total0 = time.perf_counter()
        self._log(
            f"=== CDAdaptivePlacer ({benchmark.name}): "
            f"min={self.min_time_s:.0f}s, cap={self.hard_cap_s:.0f}s, "
            f"patience={self.patience}, threshold={self.plateau_threshold}, "
            f"init={self.init_strategy} ==="
        )
        self._log(
            f"  num_macros={benchmark.num_macros}, "
            f"n_hard={benchmark.num_hard_macros}, "
            f"canvas={benchmark.canvas_width:.1f}x{benchmark.canvas_height:.1f}"
        )

        # ── 1. SDF init ──
        t_init0 = time.perf_counter()
        placement = sdf_init(benchmark)
        t_init = time.perf_counter() - t_init0
        self._log(f"  SDF init wall = {t_init:.1f} s")

        # ── 2. Project overlaps if any ──
        placement, proj_iters = project_overlaps(placement, benchmark)
        init_overlaps = compute_overlap_metrics(placement, benchmark)
        self._log(
            f"  overlap projection: {proj_iters} iters, "
            f"residual count={init_overlaps['overlap_count']}"
        )
        if init_overlaps["overlap_count"] > 0:
            self._log(
                "  ! WARNING: SDF init still has overlaps after projection; "
                "CD legality enforcement may pin some macros."
            )

        # ── 3. Build evaluator (fresh plc; SDFPlacer mutates its own copy) ──
        t_eval0 = time.perf_counter()
        _, plc = self._load_plc_for(benchmark)
        placement_f64 = placement.detach().clone().to(torch.float64)
        evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
        init_cost = evaluator.current_cost()
        t_eval = time.perf_counter() - t_eval0
        self._log(
            f"  evaluator init = {t_eval:.1f} s, "
            f"init proxy={init_cost['proxy']:.5f} "
            f"[wl={init_cost['wl']:.4f} d={init_cost['density']:.4f} "
            f"c={init_cost['congestion']:.4f}]"
        )

        # ── 4. Adaptive CD ──
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        self._log(
            f"  movable macros: {len(movable)}; starting adaptive CD "
            f"(min={self.min_time_s:.0f}s, cap={self.hard_cap_s:.0f}s)"
        )
        cd_stats = run_cd_adaptive(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            movable=movable,
            min_time_s=self.min_time_s,
            hard_cap_s=self.hard_cap_s,
            patience=self.patience,
            plateau_threshold=self.plateau_threshold,
            log_fn=self._log if self.verbose else None,
        )
        final_cost = evaluator.current_cost()
        deltas_str = ",".join(f"{d:+.5f}" for d in cd_stats["final_deltas"])
        self._log(
            f"=== CD done: exit={cd_stats['exit_reason']}, "
            f"sweeps={cd_stats['sweeps']}, "
            f"accepts={cd_stats['total_moves']}, "
            f"gs-fallbacks={cd_stats['total_gs_fallbacks']}, "
            f"wall={cd_stats['wall_total_s']:.1f}s, "
            f"final_deltas=[{deltas_str}] ==="
        )
        self._log(
            f"  final proxy={final_cost['proxy']:.5f} "
            f"[wl={final_cost['wl']:.4f} d={final_cost['density']:.4f} "
            f"c={final_cost['congestion']:.4f}]"
        )

        # ── 5. Pull placement back; ensure fixed macros unchanged + no overlaps ──
        final_placement_f64 = evaluator.placement.detach().clone().cpu()

        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            final_placement_f64[fixed_mask] = original_positions[fixed_mask]

        final_placement = final_placement_f64.to(torch.float32)

        overlaps = compute_overlap_metrics(final_placement, benchmark)
        if overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"CDAdaptivePlacer produced {overlaps['overlap_count']} overlaps "
                f"(area {overlaps['total_overlap_area']:.4f}) on "
                f"'{benchmark.name}' — this is a bug in CD legality enforcement."
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(init+CD+validate)"
        )
        return final_placement
