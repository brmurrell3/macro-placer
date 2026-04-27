"""
CDLNSPlacer — CDOnly + LNS rip-up-and-reinsert phase (experiment E3).

Pipeline (per ``place(benchmark)`` call):

    1. SDF init (via SDFPlacer in submissions/polyhedra/init/sdf.py)
    2. Iterative push-apart projection (clears any residual SDF overlaps)
    3. Build IncrementalProxyEvaluator (full-proxy: WL + density + congestion)
    4. CD sweeps for ``cd_time_budget_s`` (default 600 s) — same machinery
       as ``CDOnlyPlacer`` via the shared ``run_cd`` function.
    5. LNS phase for ``lns_time_budget_s`` (default 600 s) — destroy a *k*-set
       of high-cost macros, reinsert each via full-canvas grid-bin search,
       accept iff total proxy improves.
    6. Validate: zero overlaps + fixed macros unchanged, raise otherwise.

The total wall-clock target on a hard benchmark (ibm17/18) is ~1200 s, in
line with E3's "1200 s on hard benchmarks" budget. Easy benchmarks tend to
saturate CD long before 600 s; LNS gets the slack.

Constructor args mirror CDOnly + add LNS knobs:

    cd_time_budget_s   : float  CD wall-clock budget (default 600.0)
    lns_time_budget_s  : float  LNS wall-clock budget (default 600.0)
    k_schedule         : tuple  Destroy-set sizes cycled per LNS iter
                                (default (5, 10, 20, 30))
    init_strategy      : str    'sdf' (only currently supported)
    verbose            : bool   per-sweep / per-iteration progress to stdout

Returned placement: float32 [num_macros, 2], CPU. Hard macros legalized to
zero overlaps; fixed macros never moved; canvas bounds respected. Validated
via ``compute_overlap_metrics`` before return; ``RuntimeError`` raised if
overlaps remain.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Sequence

import torch

from macro_place.benchmark import Benchmark
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics

# The evaluate harness loads placers via importlib.spec_from_file_location, so
# the repo root is not on sys.path. Add it so `submissions.cd.*` resolves as an
# implicit namespace package.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse the CDOnly internals — DO NOT re-implement.
from submissions.cd.cd_only_placer import (  # noqa: E402
    CDOnlyPlacer,  # for _load_plc_for + _log helpers we reuse via subclass
    project_overlaps,
    run_cd,
    sdf_init,
)
from submissions.cd.lns import LNSDestroyAndReinsert  # noqa: E402


class CDLNSPlacer:
    """CD-followed-by-LNS placer for experiment E3."""

    def __init__(
        self,
        cd_time_budget_s: float = 600.0,
        lns_time_budget_s: float = 600.0,
        k_schedule: Sequence[int] = (5, 10, 20, 30),
        init_strategy: str = "sdf",
        verbose: bool = True,
        window_radius: int = 5,
    ) -> None:
        """
        Args:
            cd_time_budget_s: CD wall-clock budget.
            lns_time_budget_s: LNS wall-clock budget.
            k_schedule: cycle of LNS destroy-set sizes.
            init_strategy: only 'sdf' is currently supported.
            verbose: per-sweep / per-iteration progress to stdout.
            window_radius: LNS reinsert local-window radius (in grid cells).
                5x5 window around current grid cell. Default 5 → ±5 cells =
                11x11 candidates. Replaces the previous full-canvas grid
                sweep so reinsert cost is bounded independent of grid size.
        """
        self.cd_time_budget_s = float(cd_time_budget_s)
        self.lns_time_budget_s = float(lns_time_budget_s)
        if init_strategy != "sdf":
            raise ValueError(
                f"Only init_strategy='sdf' is currently supported "
                f"(got '{init_strategy}')"
            )
        self.init_strategy = init_strategy
        self.verbose = bool(verbose)
        self.k_schedule = tuple(int(k) for k in k_schedule)
        if not self.k_schedule:
            raise ValueError("k_schedule must be non-empty")
        if any(k <= 0 for k in self.k_schedule):
            raise ValueError(
                f"k_schedule must have all-positive entries "
                f"(got {self.k_schedule})"
            )
        if window_radius < 1:
            raise ValueError(
                f"window_radius must be >= 1 (got {window_radius})"
            )
        self.window_radius = int(window_radius)

        # Borrow CDOnlyPlacer's _load_plc_for helper without subclassing all
        # of it. Constructing one here just to reuse the method is cheap.
        self._cd_only = CDOnlyPlacer(
            time_budget_s=self.cd_time_budget_s,
            init_strategy=self.init_strategy,
            verbose=self.verbose,
        )

    # ----------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ----------------------------------------------------------------------

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """Run SDF -> CD -> LNS pipeline and return final placement."""
        t_total0 = time.perf_counter()
        self._log(
            f"=== CDLNSPlacer ({benchmark.name}): "
            f"cd_budget={self.cd_time_budget_s:.0f}s, "
            f"lns_budget={self.lns_time_budget_s:.0f}s, "
            f"k_schedule={self.k_schedule}, init={self.init_strategy} ==="
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

        # ── 3. Build evaluator ──
        t_eval0 = time.perf_counter()
        _, plc = self._cd_only._load_plc_for(benchmark)
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
            f"({self.cd_time_budget_s:.0f}s budget)"
        )
        cd_stats = run_cd(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            movable=movable,
            time_budget_s=self.cd_time_budget_s,
            log_fn=self._log if self.verbose else None,
        )
        cd_cost = evaluator.current_cost()
        self._log(
            f"=== CD done: {cd_stats['sweeps']} sweeps, "
            f"{cd_stats['total_moves']} accepts, "
            f"{cd_stats['total_gs_fallbacks']} gs-fallbacks, "
            f"{cd_stats['wall_total_s']:.1f}s wall ==="
        )
        self._log(
            f"  post-CD proxy={cd_cost['proxy']:.5f} "
            f"[wl={cd_cost['wl']:.4f} d={cd_cost['density']:.4f} "
            f"c={cd_cost['congestion']:.4f}]"
        )

        # ── 5. LNS ──
        if self.lns_time_budget_s > 0.0 and movable:
            self._log(
                f"  starting LNS ({self.lns_time_budget_s:.0f}s budget, "
                f"k_schedule={self.k_schedule}, "
                f"window_radius={self.window_radius})"
            )
            lns = LNSDestroyAndReinsert(
                k_schedule=self.k_schedule,
                window_radius=self.window_radius,
            )
            lns_stats = lns.run(
                evaluator=evaluator,
                benchmark=benchmark,
                plc=plc,
                movable=movable,
                time_budget_s=self.lns_time_budget_s,
                log_fn=self._log if self.verbose else None,
            )
            lns_cost = evaluator.current_cost()
            self._log(
                f"=== LNS done: {lns_stats['iterations']} iters, "
                f"{lns_stats['accepts']} accepts / {lns_stats['rejects']} rejects, "
                f"{lns_stats['total_reinserts']} reinserts, "
                f"{lns_stats['total_moves']} moves, "
                f"{lns_stats['wall_total_s']:.1f}s wall ==="
            )
            self._log(
                f"  post-LNS proxy={lns_cost['proxy']:.5f} "
                f"[wl={lns_cost['wl']:.4f} d={lns_cost['density']:.4f} "
                f"c={lns_cost['congestion']:.4f}]"
            )
        else:
            self._log("  skipping LNS phase (budget=0 or no movable macros)")

        # ── 6. Pull placement back; ensure fixed macros unchanged + no overlaps ──
        final_placement_f64 = evaluator.placement.detach().clone().cpu()

        # Restore fixed macros to their input positions verbatim (defensive —
        # neither CD nor LNS should have touched them since `movable` excludes
        # them, but this is a hard-constraint guarantee).
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            final_placement_f64[fixed_mask] = original_positions[fixed_mask]

        final_placement = final_placement_f64.to(torch.float32)

        overlaps = compute_overlap_metrics(final_placement, benchmark)
        if overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"CDLNSPlacer produced {overlaps['overlap_count']} overlaps "
                f"(area {overlaps['total_overlap_area']:.4f}) on "
                f"'{benchmark.name}' — bug in CD/LNS legality enforcement."
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(init+CD+LNS+validate)"
        )
        return final_placement
