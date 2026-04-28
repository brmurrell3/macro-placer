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

The CD primitives (legal_axis_range, search_axis, sdf_init, project_overlaps)
are reused from `scripts/cd_ibm10_diagnostic.py`. The per-sweep loop body is
identical to `run_cd` in `cd_only_placer.py`; this file adds a NEW function
`run_cd_adaptive(...)` so we can keep the original intact for ablations.

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

import importlib.util
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics


# ── Repo-relative paths ─────────────────────────────────────────────────────

_THIS_FILE = Path(__file__).resolve()
_ROOT = _THIS_FILE.parent.parent.parent
_TESTCASE_ROOT = _ROOT / "external/MacroPlacement/Testcases/ICCAD04"
_DIAGNOSTIC_PATH = _ROOT / "scripts" / "cd_ibm10_diagnostic.py"


def _import_diagnostic():
    """Import scripts/cd_ibm10_diagnostic.py as a module without modifying it."""
    if "cd_ibm10_diagnostic" in sys.modules:
        return sys.modules["cd_ibm10_diagnostic"]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
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
legal_axis_range = _diag.legal_axis_range
search_axis = _diag.search_axis


# ── CD adaptive sweep loop ──────────────────────────────────────────────────


def run_cd_adaptive(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    movable: List[int],
    min_time_s: float = 300.0,
    hard_cap_s: float = 3600.0,
    patience: int = 3,
    plateau_threshold: float = 0.005,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Adaptive coordinate-descent: per-benchmark plateau detection + 1hr cap.

    Visits each `movable` macro in randomized order per sweep (X then Y axis,
    closed-form breakpoint enumeration; falls back to golden section when the
    candidate set is too dense). Identical per-sweep body to `run_cd`. After
    each sweep, records `delta = previous_proxy - current_proxy` (always
    non-negative since CD only accepts improving moves) into a deque of length
    `patience`.

    Exit conditions, checked at the bottom of each sweep:
      1. `elapsed >= hard_cap_s`              → exit_reason = "cap"
      2. `elapsed >= min_time_s` AND
         `len(deltas) == patience` AND
         all `d < plateau_threshold`          → exit_reason = "plateau"

    Args:
        evaluator: an IncrementalProxyEvaluator already initialized with a
            valid (zero-overlap) placement.
        benchmark: source benchmark — used for canvas bounds and fixed mask.
        plc: PlacementCost — used for grid-line geometry only.
        movable: list of macro indices that may be moved (i.e. not fixed).
        min_time_s: don't check plateau before this many seconds of wall clock.
        hard_cap_s: absolute upper bound on wall clock.
        patience: last `patience` deltas must all be below threshold.
        plateau_threshold: absolute proxy delta per sweep below which a sweep
            counts as "stalled".
        log_fn: optional `print`-like callable; called once per sweep.

    Returns:
        dict with sweep count, accepted moves, GS fallbacks, total wall, plus
        `exit_reason` ("cap" or "plateau") and `final_deltas` (list of last
        `patience` deltas observed at exit).
    """
    n_hard = benchmark.num_hard_macros

    gw = float(plc.width / plc.grid_col)
    gh = float(plc.height / plc.grid_row)
    grid_lines_x = np.arange(plc.grid_col + 1, dtype=np.float64) * gw
    grid_lines_y = np.arange(plc.grid_row + 1, dtype=np.float64) * gh

    cur_cost = evaluator.current_cost()["proxy"]

    sweep_idx = 0
    total_moves = 0
    total_probes = 0
    total_gs_fallbacks = 0
    deltas: deque = deque(maxlen=patience)
    exit_reason = "cap"  # default if while loop terminates via cap

    t_start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= hard_cap_s:
            exit_reason = "cap"
            break
        sweep_idx += 1
        sweep_t0 = time.perf_counter()
        sweep_accepted = 0
        sweep_probes = 0
        sweep_gs = 0
        prev_cost = cur_cost  # snapshot for delta computation at sweep end

        rng = np.random.default_rng(seed=sweep_idx)
        order = list(movable)
        rng.shuffle(order)

        for macro_idx in order:
            if time.perf_counter() - t_start >= hard_cap_s:
                break

            # ── X-axis ──
            lo_x, hi_x = legal_axis_range(
                macro_idx, evaluator.placement, evaluator.macro_sizes,
                benchmark.macro_fixed, n_hard, axis=0,
                canvas_w=benchmark.canvas_width, canvas_h=benchmark.canvas_height,
            )
            cur_xy = (float(evaluator.placement[macro_idx, 0]),
                      float(evaluator.placement[macro_idx, 1]))
            best_x, best_c, mode = search_axis(
                macro_idx, 0, evaluator, grid_lines_x,
                lo_x, hi_x, cur_cost, cur_xy[1],
            )
            if mode == "gs":
                sweep_gs += 1
                total_gs_fallbacks += 1
            sweep_probes += 1
            if best_c < cur_cost - 1e-9 and abs(best_x - cur_xy[0]) > 1e-7:
                evaluator.move(macro_idx, (float(best_x), cur_xy[1]))
                cur_cost = best_c
                sweep_accepted += 1
                total_moves += 1

            # ── Y-axis ──
            cur_xy = (float(evaluator.placement[macro_idx, 0]),
                      float(evaluator.placement[macro_idx, 1]))
            lo_y, hi_y = legal_axis_range(
                macro_idx, evaluator.placement, evaluator.macro_sizes,
                benchmark.macro_fixed, n_hard, axis=1,
                canvas_w=benchmark.canvas_width, canvas_h=benchmark.canvas_height,
            )
            best_y, best_c, mode = search_axis(
                macro_idx, 1, evaluator, grid_lines_y,
                lo_y, hi_y, cur_cost, cur_xy[0],
            )
            if mode == "gs":
                sweep_gs += 1
                total_gs_fallbacks += 1
            sweep_probes += 1
            if best_c < cur_cost - 1e-9 and abs(best_y - cur_xy[1]) > 1e-7:
                evaluator.move(macro_idx, (cur_xy[0], float(best_y)))
                cur_cost = best_c
                sweep_accepted += 1
                total_moves += 1

        sweep_wall = time.perf_counter() - sweep_t0
        elapsed = time.perf_counter() - t_start
        cost_break = evaluator.current_cost()
        cur_cost = cost_break["proxy"]
        total_probes += sweep_probes

        # Δproxy = prev - cur ≥ 0 (CD only accepts improving moves).
        delta = prev_cost - cur_cost
        deltas.append(delta)

        if log_fn is not None:
            log_fn(
                f"  sweep {sweep_idx:3d}  elapsed={elapsed:7.1f}s  "
                f"sweep_t={sweep_wall:6.1f}s  proxy={cost_break['proxy']:.5f}  "
                f"Δ={delta:+.5f}  "
                f"accepted={sweep_accepted}/{sweep_probes}  gs={sweep_gs}  "
                f"[wl={cost_break['wl']:.4f} d={cost_break['density']:.4f} "
                f"c={cost_break['congestion']:.4f}]"
            )

        # ── Exit checks ──
        if elapsed >= hard_cap_s:
            exit_reason = "cap"
            break
        if (
            elapsed >= min_time_s
            and len(deltas) == patience
            and all(d < plateau_threshold for d in deltas)
        ):
            exit_reason = "plateau"
            break

    return {
        "sweeps": sweep_idx,
        "total_moves": total_moves,
        "total_probes": total_probes,
        "total_gs_fallbacks": total_gs_fallbacks,
        "wall_total_s": time.perf_counter() - t_start,
        "exit_reason": exit_reason,
        "final_deltas": list(deltas),
    }


# ── Placer class ────────────────────────────────────────────────────────────


class CDAdaptivePlacer:
    """Adaptive CD placer — E9.

    Pipeline per call to `place(benchmark)`:
      1. SDF init (via SDFPlacer in submissions/cd/sdf_init.py)
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
        bench_dir = _TESTCASE_ROOT / benchmark.name
        if not bench_dir.exists():
            raise FileNotFoundError(
                f"Benchmark dir not found for '{benchmark.name}': {bench_dir}"
            )
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
