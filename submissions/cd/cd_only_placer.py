"""
CDOnlyPlacer — productionizes the E2 result.

E2 (40-min full-proxy CD on incremental evaluator, SDF init) on ibm10 reached
**proxy 1.0632** (full-eval cross-check 1.0724), beating the DPO champion (1.254)
by 15.2%. This placer wraps the same machinery into a `place(benchmark) -> Tensor`
class so it can run via the `evaluate` harness.

The CD primitives (legal_axis_range, search_axis, sdf_init, project_overlaps) are
imported from `scripts/cd_ibm10_diagnostic.py` rather than copy-pasted. The CD
sweep loop itself is factored into `run_cd(...)` defined here so that anything
that wants CD on top of an existing evaluator can just call this function.

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

import importlib.util
import sys
import time
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
    """Import scripts/cd_ibm10_diagnostic.py as a module without modifying it.

    Reuses the module's helpers: sdf_init, project_overlaps, legal_axis_range,
    search_axis, axis_breakpoints, golden_section.
    """
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


# ── CD sweep loop (extracted; reusable) ─────────────────────────────────────


def run_cd(
    evaluator: IncrementalProxyEvaluator,
    benchmark: Benchmark,
    plc,
    movable: List[int],
    time_budget_s: float,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Coordinate-descent sweeps on the incremental evaluator.

    Visits each `movable` macro in randomized order per sweep. For each macro,
    searches the legal range on each axis (x then y) using `search_axis`
    (closed-form breakpoint enumeration; falls back to golden section when the
    candidate set is too dense). Commits any improving move; reverts otherwise.
    Stops when wall clock crosses `time_budget_s`.

    Args:
        evaluator: an IncrementalProxyEvaluator already initialized with a
            valid (zero-overlap) placement.
        benchmark: source benchmark — used for canvas bounds and fixed mask.
        plc: PlacementCost — used for grid-line geometry only.
        movable: list of macro indices that may be moved (i.e. not fixed).
        time_budget_s: wall-clock budget for CD only (init excluded).
        log_fn: optional `print`-like callable; called once per sweep.

    Returns:
        dict with sweep count, accepted moves, GS fallbacks, total wall.
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

    t_start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_budget_s:
            break
        sweep_idx += 1
        sweep_t0 = time.perf_counter()
        sweep_accepted = 0
        sweep_probes = 0
        sweep_gs = 0

        rng = np.random.default_rng(seed=sweep_idx)
        order = list(movable)
        rng.shuffle(order)

        for macro_idx in order:
            if time.perf_counter() - t_start >= time_budget_s:
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
        if log_fn is not None:
            log_fn(
                f"  sweep {sweep_idx:3d}  elapsed={elapsed:7.1f}s  "
                f"sweep_t={sweep_wall:6.1f}s  proxy={cost_break['proxy']:.5f}  "
                f"accepted={sweep_accepted}/{sweep_probes}  gs={sweep_gs}  "
                f"[wl={cost_break['wl']:.4f} d={cost_break['density']:.4f} "
                f"c={cost_break['congestion']:.4f}]"
            )

    return {
        "sweeps": sweep_idx,
        "total_moves": total_moves,
        "total_probes": total_probes,
        "total_gs_fallbacks": total_gs_fallbacks,
        "wall_total_s": time.perf_counter() - t_start,
    }


# ── Placer class ────────────────────────────────────────────────────────────


class CDOnlyPlacer:
    """Coordinate-descent-only placer (E2 productionized).

    Pipeline per call to `place(benchmark)`:
      1. SDF init (via the SDFPlacer in submissions/polyhedra/init/sdf.py)
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
        bench_dir = _TESTCASE_ROOT / benchmark.name
        if not bench_dir.exists():
            raise FileNotFoundError(
                f"Benchmark dir not found for '{benchmark.name}': {bench_dir}"
            )
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
