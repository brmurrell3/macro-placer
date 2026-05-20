"""CDLNSSAXplacePatchedPlacer — Xplace-style optimizer (Adam + adaptive λ +
multi-stage γ + margin-based overlap) on E111 per-net-trace proxy, then
CD polish.

"Patched DREAMPlace" stand-in: replace DREAMPlace's HPWL +
density_weight*eDensity loss with our exact challenge proxy (HPWL +
0.5*density + 0.5*congestion), keeping the optimization recipe spirit:
multi-stage descent, gradual overlap-penalty (density weight) ramp,
gamma annealing.

Pipeline:
  1. SmoothGlobalPlacerV5 with margin-based overlap penalty
     - Multi-stage γ: 5e-3 → 1e-3 (250 steps), 1e-3 → 5e-4 (200 steps),
       5e-4 → 5e-5 (150 steps)
     - λ_ovl: 0 → 10 (Stage A), 10 → 30 (Stage B), 50 → 200 (Stage C)
     - Margin = 0.003·canvas_width: pushes touching macros slightly apart
       so legalize moves fewer macros (key for ibm09 robustness — vanilla
       V3 leaves 1 stuck overlap on ibm09; V5 + margin → 0 overlaps)
     - Adam (proven beats Nesterov-BB on macro placement; see manifest)
  2. greedy_macro_legalize (zero canonical overlaps)
  3. CD polish (60s default) on the legal placement
  Multi-restart: defaults to n_restarts=2, keeps best-by-canonical-proxy.
  Falls back to SDF + project_overlaps if all restarts fail (rare).

Verified ibm01 (M3):
  - Raw       : ~0.87 (vs V3 raw 0.895, -3%)
  - +CD60s    : ~0.84 (vs V3 +CD60s 0.846, -0.7%)
  - vs Option C ibm01 0.85 cascade: ties or wins at 1/30 the wall

For comparison:
  - Option C (cd_lns_sa_cascade_stacked_periphery) ibm01: 0.85 (~50min)
  - This (~5 min/bench)                                  : ~0.84

Fallback: graceful degradation. If V5 produces overlaps that legalize
can't resolve, falls back to project_overlaps + a short CD recovery.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# Pull in V5 from E113 experiment dir
_E113_CODE = _ROOT / "experiments" / "E113_xplace_recipe" / "code"
if str(_E113_CODE) not in sys.path:
    sys.path.insert(0, str(_E113_CODE))
# Need all the chained imports too
for p in (
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from smooth_global_placer_v5 import SmoothGlobalPlacerV5


class CDLNSSAXplacePatchedPlacer:
    """Xplace-style multi-stage Adam descent on V3 proxy + margin, then CD.

    Default budget: 5 min/bench (smooth descent ~30s, CD ~270s).
    All knobs exposed; defaults tuned on ibm01.
    """

    def __init__(
        self,
        # Total budget per bench (default 5 min)
        budget_seconds: Optional[float] = 300.0,
        # V5 multi-stage descent params
        stage_steps=(250, 200, 150),
        base_lr_frac: float = 0.005,
        gamma_stage_A=(5e-3, 1e-3),
        gamma_stage_B=(1e-3, 5e-4),
        gamma_stage_C=(5e-4, 5e-5),
        overlap_lambda_stage_A=(0.0, 10.0),
        overlap_lambda_stage_B=(10.0, 30.0),
        overlap_lambda_stage_C=(50.0, 200.0),
        overlap_margin_frac: float = 0.003,  # sweet spot from ibm01 multi-seed sweep
        use_adaptive_lambda: bool = False,
        boundary_lambda: float = 50.0,
        # Init
        init: str = "sdf",
        rng_seed: int = 42,
        # Legalize
        legalize_radius_steps: int = 80,
        legalize_step_frac: float = 0.02,
        # CD polish
        cd_min_polish_s: float = 60.0,
        cd_plateau_threshold: float = 1e-4,
        cd_patience: int = 5,
        # Multi-restart (DREAMPlace two_stage_density_scaler equivalent).
        # Default 2: gives one retry on flaky benches (ibm09 V3 baseline
        # fails with 1 stuck overlap; second seed usually resolves).
        n_restarts: int = 2,
        restart_seeds: Optional[list] = None,
        # SDF+CD fallback if all restarts fail
        sdf_fallback: bool = True,
        # Misc
        device: str = "cpu",
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.stage_steps = stage_steps
        self.base_lr_frac = base_lr_frac
        self.gamma_stage_A = gamma_stage_A
        self.gamma_stage_B = gamma_stage_B
        self.gamma_stage_C = gamma_stage_C
        self.overlap_lambda_stage_A = overlap_lambda_stage_A
        self.overlap_lambda_stage_B = overlap_lambda_stage_B
        self.overlap_lambda_stage_C = overlap_lambda_stage_C
        self.overlap_margin_frac = overlap_margin_frac
        self.use_adaptive_lambda = use_adaptive_lambda
        self.boundary_lambda = boundary_lambda
        self.init = init
        self.rng_seed = rng_seed
        self.legalize_radius_steps = legalize_radius_steps
        self.legalize_step_frac = legalize_step_frac
        self.cd_min_polish_s = cd_min_polish_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.cd_patience = cd_patience
        self.n_restarts = max(1, int(n_restarts))
        self.restart_seeds = restart_seeds or [rng_seed, rng_seed + 1, rng_seed + 7]
        self.sdf_fallback = bool(sdf_fallback)
        self.device = device
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(
            f"=== CDLNSSAXplacePatchedPlacer ({benchmark.name}) "
            f"budget={self.budget_seconds:.0f}s ==="
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Phase 1: Multi-restart V5 smooth descent + legalize, keep best.
        best_pos = None
        best_proxy = float("inf")
        seeds = self.restart_seeds[: self.n_restarts]
        for i, seed in enumerate(seeds):
            v5 = SmoothGlobalPlacerV5(
                stage_steps=self.stage_steps,
                gamma_stage_A=self.gamma_stage_A,
                gamma_stage_B=self.gamma_stage_B,
                gamma_stage_C=self.gamma_stage_C,
                overlap_lambda_stage_A=self.overlap_lambda_stage_A,
                overlap_lambda_stage_B=self.overlap_lambda_stage_B,
                overlap_lambda_stage_C=self.overlap_lambda_stage_C,
                overlap_margin_frac=self.overlap_margin_frac,
                use_adaptive_lambda=self.use_adaptive_lambda,
                base_lr_frac=self.base_lr_frac,
                boundary_lambda=self.boundary_lambda,
                init=self.init,
                rng_seed=seed,
                legalize_radius_steps=self.legalize_radius_steps,
                legalize_step_frac=self.legalize_step_frac,
                device=self.device,
                verbose=False,
            )
            pos_try = v5.place(benchmark)
            ovl_try = compute_overlap_metrics(pos_try, benchmark)["overlap_count"]
            if ovl_try > 0:
                self._log(f"  restart{i} (seed={seed}): legalize left {ovl_try} overlaps; trying project_overlaps")
                pos_try, _ = project_overlaps(pos_try, benchmark)
                ovl_try = compute_overlap_metrics(pos_try, benchmark)["overlap_count"]
                if ovl_try > 0:
                    self._log(f"  restart{i} (seed={seed}): SKIP (could not resolve)")
                    continue
            proxy_try = float(compute_proxy_cost(pos_try, benchmark, plc)["proxy_cost"])
            self._log(
                f"  restart{i} (seed={seed}): proxy={proxy_try:.5f} ovl={ovl_try} "
                f"wall_so_far={time.time()-t0:.0f}s"
            )
            if proxy_try < best_proxy:
                best_pos = pos_try
                best_proxy = proxy_try
            # Bail out if running near deadline (leave room for CD polish).
            if deadline is not None and time.time() + self.cd_min_polish_s + 20.0 > deadline:
                self._log(f"  bail-out: deadline {deadline-t0:.0f}s reached after {i+1} restarts")
                break

        if best_pos is None:
            if self.sdf_fallback:
                self._log("  All restarts failed; falling back to SDF + project_overlaps + CD")
                from macro_place.cd_core import sdf_init
                pos = sdf_init(benchmark)
                pos, _ = project_overlaps(pos, benchmark)
                ovl_chk = compute_overlap_metrics(pos, benchmark)["overlap_count"]
                if ovl_chk > 0:
                    raise RuntimeError(f"SDF fallback also failed ({ovl_chk} overlaps)")
                descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
                descent_ovl = 0
                self._log(f"  SDF fallback: proxy={descent_proxy:.5f}")
            else:
                raise RuntimeError("All restarts failed to produce legal placement")
        else:
            pos = best_pos
            descent_proxy = best_proxy
            descent_ovl = 0
            self._log(
                f"  picked best of {min(self.n_restarts, len(seeds))} restarts: "
                f"proxy={descent_proxy:.5f} wall={time.time()-t0:.0f}s"
            )

        # Phase 2: CD polish (cap at deadline)
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd_budget = max(30.0, remaining)
        else:
            cd_budget = max(self.cd_min_polish_s, 60.0)
        self._log(f"  CD polish budget={cd_budget:.0f}s")

        evaluator = IncrementalProxyEvaluator(benchmark, plc, pos.to(torch.float64))
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        run_cd_adaptive(
            evaluator, benchmark, plc, movable,
            min_time_s=min(cd_budget * 0.5, self.cd_min_polish_s),
            hard_cap_s=cd_budget,
            patience=self.cd_patience,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=None,
        )
        final = evaluator.placement.detach().clone().to(torch.float32)
        final_proxy = float(compute_proxy_cost(final, benchmark, plc)["proxy_cost"])
        final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
        self._log(
            f"  Final: proxy={final_proxy:.5f} ovl={final_ovl} "
            f"total_wall={time.time()-t0:.0f}s"
        )

        if final_ovl > 0:
            raise RuntimeError(f"Xplace-patched produced {final_ovl} overlaps")
        return final
