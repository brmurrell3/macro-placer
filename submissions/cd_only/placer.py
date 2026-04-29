# STATUS: SUPERSEDED 2026-04-27 by CDAdaptive (1.1055). Was champion at
# 1.1193 --all (matched leaderboard within 0.18%). Replaced by adding
# per-benchmark plateau detection (see cd_adaptive_placer.py). Still the
# fixed-budget reference; `run_cd` is reused by CDLNSPlacer.
"""
CDOnlyPlacer — productionizes the E2 result.

E2 (40-min full-proxy CD on incremental evaluator, SDF init) on ibm10 reached
**proxy 1.0632** (full-eval cross-check 1.0724), beating the DPO champion (1.254)
by 15.2%. This placer wraps the same machinery into a `place(benchmark) -> Tensor`
class so it can run via the `evaluate` harness.

The CD primitives and per-sweep loop live in ``macro_place.cd_core``; this
file is just the placer-class wrapper.

Constructor args:
    time_budget_s: CD wall-clock budget (default 600 = 10 min). SDF init and
        evaluator construction are NOT counted against this — they run before
        the CD timer starts.
    init_strategy: 'sdf' (default). Other choices reserved for future work.
    verbose: print per-sweep progress to stdout (default True).

Returned placement:
    [num_macros, 2] tensor (float32, on CPU). Hard macros legalized to zero
    overlaps; fixed macros never moved; canvas bounds respected. Validated
    via `compute_overlap_metrics` before return; `RuntimeError` raised if
    overlaps remain (should never happen — the CD legality enforcement is
    strict).
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
from macro_place.cd_core import project_overlaps, run_cd, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics


# ── Placer class ────────────────────────────────────────────────────────────


class CDOnlyPlacer:
    """Coordinate-descent-only placer (E2 productionized).

    Pipeline per call to `place(benchmark)`:
      1. SDF init (via the SDFPlacer in macro_place/sdf_init.py)
      2. Iterative push-apart projection to clean any residual overlaps
      3. Build IncrementalProxyEvaluator (full-proxy: WL + density + congestion)
      4. CD sweeps (closed-form breakpoint enumeration per axis) for
         `time_budget_s` seconds
      5. Validate: zero overlaps, fixed macros not moved
    """

    def __init__(
        self,
        time_budget_s: float = 600.0,
        init_strategy: str = "sdf",
        verbose: bool = True,
    ) -> None:
        self.time_budget_s = float(time_budget_s)
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
        """Load a fresh PlacementCost for this benchmark.

        The harness only hands us the Benchmark dataclass — we need the plc to
        feed IncrementalProxyEvaluator. SDFPlacer mutates plc internally, so we
        also reload after init to get a clean plc for the evaluator.
        """
        bench_dir = _find_benchmark_dir(benchmark.name)
        return load_benchmark_from_dir(str(bench_dir))

    # ----------------------------------------------------------------------

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """Run the CD-only pipeline and return final placement."""
        t_total0 = time.perf_counter()
        self._log(
            f"=== CDOnlyPlacer ({benchmark.name}): "
            f"budget={self.time_budget_s:.0f}s, init={self.init_strategy} ==="
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

        # ── 4. CD ──
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        self._log(
            f"  movable macros: {len(movable)}; starting CD "
            f"({self.time_budget_s:.0f}s budget)"
        )
        cd_stats = run_cd(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            movable=movable,
            time_budget_s=self.time_budget_s,
            log_fn=self._log if self.verbose else None,
        )
        final_cost = evaluator.current_cost()
        self._log(
            f"=== CD done: {cd_stats['sweeps']} sweeps, "
            f"{cd_stats['total_moves']} accepts, "
            f"{cd_stats['total_gs_fallbacks']} gs-fallbacks, "
            f"{cd_stats['wall_total_s']:.1f}s wall ==="
        )
        self._log(
            f"  final proxy={final_cost['proxy']:.5f} "
            f"[wl={final_cost['wl']:.4f} d={final_cost['density']:.4f} "
            f"c={final_cost['congestion']:.4f}]"
        )

        # ── 5. Pull placement back; ensure fixed macros unchanged + no overlaps ──
        final_placement_f64 = evaluator.placement.detach().clone().cpu()

        # Restore fixed macros to their input positions verbatim (defensive —
        # the evaluator and CD legality should never have moved them since
        # `movable` excludes them).
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            final_placement_f64[fixed_mask] = original_positions[fixed_mask]

        final_placement = final_placement_f64.to(torch.float32)

        overlaps = compute_overlap_metrics(final_placement, benchmark)
        if overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"CDOnlyPlacer produced {overlaps['overlap_count']} overlaps "
                f"(area {overlaps['total_overlap_area']:.4f}) on "
                f"'{benchmark.name}' — this is a bug in CD legality enforcement."
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(init+CD+validate)"
        )
        return final_placement
