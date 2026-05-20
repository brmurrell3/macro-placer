"""CDLNSSACascadeDPLanePatchedPlacer — DP basin refined by smooth-proxy
gradient descent (Option C from the patched-DREAMPlace investigation).

Architecture (the "patched" idea):
    Stock DREAMPlace uses HPWL + density_weight * eDensity as its loss.
    That objective is mismatched to our challenge proxy
    (HPWL + 0.5*density + 0.5*congestion). Patching DREAMPlace's loss
    internally would require injecting a torch-differentiable RUDY
    congestion gradient into PlaceObj.obj_fn, which DREAMPlace doesn't
    have natively. Even with a clean patch, DREAMPlace requires Docker /
    x86_64 .so binaries that can't run on darwin/arm64.

    The equivalent effect is reachable without modifying DREAMPlace
    source: take DREAMPlace's basin output and refine it with Adam
    descent on OUR proxy (the SmoothGlobalPlacer from E110). This is
    Option C in the task brief — DREAMPlace as a black-box init for
    our gradient-on-the-challenge-proxy placer.

    The hypothesis: DP's electrostatic basin is structurally different
    from SDF/DPO basins (spreads uniformly across the canvas).
    SmoothGlobalPlacer descent from that basin lands somewhere our
    cascade hasn't explored. Plateau-pick against E25/E41/DP+stock
    captures wherever the lift lives.

Lanes (best-of-N):
    1. E25 (SDF init + CD + LNS + SA polish)            -- always
    2. E41 (DPO init + CD + LNS + SA + K-joint polish)  -- always
    3. DP + stock polish (CD-LNS-SA)                    -- if DP available
    4. DP + smooth descent (E110 from DP init)          -- NEW, if DP available
    5. SDF + smooth descent (E110 from SDF init)        -- always, as fallback
                                                            for when DP is
                                                            unavailable (M3 dev)
Then cascading saddle on the plateau winner.

Fallback behavior:
    - DREAMPLACE_ROOT not set or install missing → lanes 3+4 skipped,
      lane 5 (SDF + smooth) still runs. This is the M3 dev path.
    - Lane 5 ALWAYS runs (it's the actual "patched DREAMPlace" stand-in
      we can test locally).

Verified (M3, single-bench smoke 2026-05-19):
    ibm01: SDF+smooth+CD60s → 0.887 (target <0.90 met)
    With DP basin as init, expected: 0.86-0.88 (similar to DP+full polish
    in E91 ibm01 result).
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

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

# Reuse all DP-lane primitives (DP subprocess, polish, legalize).
_DP_LANE_PATH = (
    _ROOT / "submissions" / "cd_lns_sa_cascade_dp_lane" / "placer.py"
)
_DP_LANE_SPEC = importlib.util.spec_from_file_location(
    "dp_lane_placer_orig", str(_DP_LANE_PATH)
)
_DP_LANE_MOD = importlib.util.module_from_spec(_DP_LANE_SPEC)
_DP_LANE_SPEC.loader.exec_module(_DP_LANE_MOD)

CDLNSSAPlacer = _DP_LANE_MOD.CDLNSSAPlacer
run_lns_gridbin = _DP_LANE_MOD.run_lns_gridbin
run_sa_polish_v2 = _DP_LANE_MOD.run_sa_polish_v2
CDLNSSADPOKJointPlacer = _DP_LANE_MOD.CDLNSSADPOKJointPlacer
cascading_saddle_escape = _DP_LANE_MOD.cascading_saddle_escape
greedy_macro_legalize = _DP_LANE_MOD.greedy_macro_legalize
extended_legalize = _DP_LANE_MOD.extended_legalize
_try_run_dreamplace = _DP_LANE_MOD._try_run_dreamplace
_polish_dp_basin = _DP_LANE_MOD._polish_dp_basin

# E110 SmoothGlobalPlacer.
_E110_CODE = _ROOT / "experiments" / "E110_smooth_global_placer" / "code"
if str(_E110_CODE) not in sys.path:
    sys.path.insert(0, str(_E110_CODE))
from smooth_global_placer import SmoothGlobalPlacer  # noqa: E402


def _smooth_descent_from_init(
    init_pos: torch.Tensor,
    benchmark: Benchmark,
    plc,
    log,
    *,
    num_steps: int = 500,
    lr_frac: float = 0.005,
    gamma_start_frac: float = 5e-3,
    gamma_end_frac: float = 5e-5,
    overlap_lambda_end: float = 50.0,
    overlap_ramp_pct: float = 0.7,
    boundary_lambda: float = 50.0,
    legalize_step_frac: float = 0.005,
    legalize_radius_steps: int = 200,
    rng_seed: int = 42,
    cd_polish_budget_s: float = 240.0,
    label: str = "smooth",
) -> Tuple[Optional[torch.Tensor], float, int]:
    """Apply SmoothGlobalPlacer descent + legalize + CD polish, from
    a custom init_pos (NOT the placer's default SDF init).

    Returns (placement, proxy, ovl). Returns (None, inf, -1) on failure.
    """
    t0 = time.time()

    # Construct SmoothGlobalPlacer with init="sdf" (placeholder), then
    # bypass its _init_positions by calling descend()/legalize() with our
    # custom init. The placer reads init_pos from self.descend() which
    # calls self._init_positions(); we override that via a subclass.
    class _CustomInitSmoothPlacer(SmoothGlobalPlacer):
        def __init__(self, _init_pos: torch.Tensor, **kwargs):
            super().__init__(**kwargs)
            self._injected_init = _init_pos.clone().detach()

        def _init_positions(self, benchmark, plc=None):
            # Honor fixed macros (replace any fixed-macro coords with
            # benchmark's fixed positions so the injected init doesn't
            # accidentally violate the fixed constraint).
            pos = self._injected_init.clone()
            fixed_mask = benchmark.macro_fixed.bool()
            if fixed_mask.any():
                pos[fixed_mask] = benchmark.macro_positions[fixed_mask]
            # Project any residual overlaps so descent starts feasible.
            pos, n_iter = project_overlaps(pos, benchmark)
            return pos

    log(f"  [{label}] SmoothGlobalPlacer descent (init injected)")
    try:
        placer = _CustomInitSmoothPlacer(
            init_pos,
            num_steps=num_steps,
            lr_frac=lr_frac,
            gamma_start_frac=gamma_start_frac,
            gamma_end_frac=gamma_end_frac,
            overlap_lambda_end=overlap_lambda_end,
            overlap_ramp_pct=overlap_ramp_pct,
            boundary_lambda=boundary_lambda,
            legalize_step_frac=legalize_step_frac,
            legalize_radius_steps=legalize_radius_steps,
            init="sdf",  # placeholder; overridden by _init_positions above
            rng_seed=rng_seed,
            verbose=False,
        )
        pos = placer.place(benchmark)
    except Exception as exc:
        log(f"  [{label}] descent FAILED: {exc}")
        return None, float("inf"), -1
    descent_wall = time.time() - t0

    # Verify zero overlap; if not, run project_overlaps to clean up.
    ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    if ovl > 0:
        pos, _ = project_overlaps(pos, benchmark)
        ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    if ovl > 0:
        log(f"  [{label}] descent left {ovl} overlaps; lane excluded")
        return None, float("inf"), ovl

    raw_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
    log(
        f"  [{label}] descent done: proxy={raw_proxy:.5f} ovl={ovl} "
        f"wall={descent_wall:.0f}s"
    )

    # CD polish on the smooth-descent output.
    if cd_polish_budget_s > 5.0:
        log(f"  [{label}] CD polish ({cd_polish_budget_s:.0f}s)")
        try:
            evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
            movable = [
                i for i in range(benchmark.num_macros)
                if not bool(benchmark.macro_fixed[i])
            ]
            run_cd_adaptive(
                evaluator, benchmark, plc, movable,
                min_time_s=min(60.0, cd_polish_budget_s * 0.5),
                hard_cap_s=cd_polish_budget_s,
                patience=3, plateau_threshold=0.001, log_fn=None,
            )
            pos = evaluator.placement.detach().clone().to(torch.float32)
        except Exception as exc:
            log(f"  [{label}] CD polish FAILED: {exc}; using descent output")

    final_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
    final_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    log(
        f"  [{label}] FINAL: proxy={final_proxy:.5f} ovl={final_ovl} "
        f"wall={time.time() - t0:.0f}s"
    )
    return pos, final_proxy, final_ovl


class CDLNSSACascadeDPLanePatchedPlacer:
    """Cascade + DP + smooth-proxy gradient lanes.

    Differs from cd_lns_sa_cascade_dp_lane by adding two extra lanes:
      - DP basin → smooth-proxy descent → CD polish
      - SDF basin → smooth-proxy descent → CD polish (always runs;
        equivalent to E110's contribution to stacked_periphery_e110)

    Plateau pick = argmin over {E25, E41, DP+stock, DP+smooth, SDF+smooth}.
    Then cascading saddle.
    """

    def __init__(
        self,
        max_iters: int = 5,
        eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
        polish_budget: float = 180.0,
        budget_seconds: Optional[float] = 3300.0,
        # E110 smooth-descent hyperparameters
        smooth_num_steps: int = 500,
        smooth_lr_frac: float = 0.005,
        smooth_gamma_start_frac: float = 5e-3,
        smooth_gamma_end_frac: float = 5e-5,
        smooth_overlap_lambda_end: float = 50.0,
        smooth_overlap_ramp_pct: float = 0.7,
        smooth_legalize_step_frac: float = 0.005,
        smooth_legalize_radius_steps: int = 200,
        verbose: bool = True,
    ):
        self.max_iters = max_iters
        self.eps_values = eps_values
        self.polish_budget = polish_budget
        self.budget_seconds = budget_seconds
        self.smooth_num_steps = smooth_num_steps
        self.smooth_lr_frac = smooth_lr_frac
        self.smooth_gamma_start_frac = smooth_gamma_start_frac
        self.smooth_gamma_end_frac = smooth_gamma_end_frac
        self.smooth_overlap_lambda_end = smooth_overlap_lambda_end
        self.smooth_overlap_ramp_pct = smooth_overlap_ramp_pct
        self.smooth_legalize_step_frac = smooth_legalize_step_frac
        self.smooth_legalize_radius_steps = smooth_legalize_radius_steps
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSACascadeDPLanePatchedPlacer ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Budget allocation tuned for 5 lanes (vs DP-lane's 3):
        #   E25:           0.10 B (was 0.20)
        #   E41:           0.10 B (was 0.22)
        #   DP+stock:      0.10 B (was 0.20)
        #   DP+smooth:     0.10 B (NEW)
        #   SDF+smooth:    0.08 B (NEW)
        #   Cascade saddle: ~0.30 B (was 0.35)
        # +DP raw/legalize ~0.03 B, post-process ~0.02 B → totals to ~0.83
        # Leaving slack for variance.
        if self.budget_seconds is not None:
            B = self.budget_seconds
            e25_kwargs = dict(
                cd_hard_cap_s=B * 0.10, lns_budget_s=B * 0.03, sa_budget_s=B * 0.03,
            )
            e41_kwargs = dict(
                cd_hard_cap_s=B * 0.10, lns_budget_s=B * 0.03, sa_budget_s=B * 0.03,
                kjoint_budget_s=B * 0.03,
            )
            dp_stock_kwargs = dict(
                cd_hard_cap_s=B * 0.10, lns_budget_s=B * 0.03, sa_budget_s=B * 0.03,
            )
            smooth_cd_budget_s = B * 0.05
            log(
                f"  budget {B:.0f}s; E25={sum(e25_kwargs.values()):.0f}s "
                f"E41={sum(e41_kwargs.values()):.0f}s "
                f"DP+stock={sum(dp_stock_kwargs.values()):.0f}s "
                f"smooth_CD={smooth_cd_budget_s:.0f}s "
                f"saddle ~{B * 0.30:.0f}s"
            )
        else:
            e25_kwargs = e41_kwargs = dp_stock_kwargs = {}
            smooth_cd_budget_s = 240.0

        # Phase 1: E25 (SDF init + polish)
        log("  Phase 1: E25 (SDF + CD + LNS + SA)")
        e25 = CDLNSSAPlacer(**e25_kwargs).place(benchmark)
        e25_proxy = float(compute_proxy_cost(e25, benchmark, plc)["proxy_cost"])
        log(f"  E25 done: proxy={e25_proxy:.5f} (wall={time.time() - t0:.0f}s)")

        # Phase 2: E41 (DPO init + polish)
        e41 = None
        e41_proxy = float("inf")
        if deadline is not None and (deadline - time.time()) < 600.0:
            log(f"  Phase 2 SKIPPED ({deadline - time.time():.0f}s left)")
        else:
            log("  Phase 2: E41 (DPO + CD + LNS + SA + K-joint)")
            try:
                e41 = CDLNSSADPOKJointPlacer(**e41_kwargs).place(benchmark)
                e41_proxy = float(compute_proxy_cost(e41, benchmark, plc)["proxy_cost"])
                log(f"  E41 done: proxy={e41_proxy:.5f} (wall={time.time() - t0:.0f}s)")
            except Exception as exc:
                log(f"  E41 FAILED: {exc}; continuing without E41 lane")

        # Phase 3: DP basin (shared by lanes 3 + 4). Get raw DP placement.
        dp_raw = None
        if self.budget_seconds is not None:
            dp_skip_threshold = self.budget_seconds * 0.20 + 600.0
        else:
            dp_skip_threshold = 1200.0
        if deadline is not None and (deadline - time.time()) < dp_skip_threshold:
            log(f"  Phase 3 DP raw SKIPPED ({deadline - time.time():.0f}s left, "
                f"need {dp_skip_threshold:.0f}s)")
        else:
            log("  Phase 3: DREAMPlace raw basin")
            dp_raw = _try_run_dreamplace(benchmark, plc, log)
            if dp_raw is not None:
                raw_proxy = float(compute_proxy_cost(dp_raw, benchmark, plc)["proxy_cost"])
                raw_ovl = compute_overlap_metrics(dp_raw, benchmark)["overlap_count"]
                log(f"  DP raw: proxy={raw_proxy:.5f} ovl={raw_ovl}")

        # Lane 3: DP + stock polish (E25-equivalent CD-LNS-SA on DP basin)
        dp_stock = None
        dp_stock_proxy = float("inf")
        if dp_raw is not None and (deadline is None or (deadline - time.time()) > 400.0):
            log("  Lane 3: DP + stock polish (CD-LNS-SA)")
            try:
                dp_stock, dp_stock_proxy = _polish_dp_basin(
                    dp_raw, benchmark, plc,
                    cd_hard_cap_s=dp_stock_kwargs["cd_hard_cap_s"],
                    lns_budget_s=dp_stock_kwargs["lns_budget_s"],
                    sa_budget_s=dp_stock_kwargs["sa_budget_s"],
                    log=log,
                )
                log(f"  Lane 3 done: proxy={dp_stock_proxy:.5f} "
                    f"(wall={time.time() - t0:.0f}s)")
            except Exception as exc:
                log(f"  Lane 3 FAILED: {exc}; skipping")
                dp_stock = None

        # Lane 4: DP + smooth-proxy descent (NEW). Use DP basin as init
        # for SmoothGlobalPlacer.
        dp_smooth = None
        dp_smooth_proxy = float("inf")
        if dp_raw is not None and (deadline is None or (deadline - time.time()) > 300.0):
            log("  Lane 4: DP basin + smooth-proxy descent")
            # Pre-legalize the DP basin so smooth descent starts from a
            # feasible-ish init (smooth descent is sensitive to large
            # overlap area in initial position).
            try:
                pre_pos, _ = greedy_macro_legalize(
                    dp_raw, benchmark, step_size_frac=0.01,
                )
                pre_pos, _ = project_overlaps(pre_pos, benchmark)
                dp_smooth, dp_smooth_proxy, dp_smooth_ovl = _smooth_descent_from_init(
                    pre_pos, benchmark, plc, log,
                    num_steps=self.smooth_num_steps,
                    lr_frac=self.smooth_lr_frac,
                    gamma_start_frac=self.smooth_gamma_start_frac,
                    gamma_end_frac=self.smooth_gamma_end_frac,
                    overlap_lambda_end=self.smooth_overlap_lambda_end,
                    overlap_ramp_pct=self.smooth_overlap_ramp_pct,
                    legalize_step_frac=self.smooth_legalize_step_frac,
                    legalize_radius_steps=self.smooth_legalize_radius_steps,
                    cd_polish_budget_s=smooth_cd_budget_s,
                    label="L4 DP+smooth",
                )
                if dp_smooth is not None and dp_smooth_ovl > 0:
                    dp_smooth = None  # exclude lane with residual overlap
                    dp_smooth_proxy = float("inf")
            except Exception as exc:
                log(f"  Lane 4 FAILED: {exc}; skipping")
                dp_smooth = None

        # Lane 5: SDF + smooth-proxy descent (always-on; the DP fallback
        # for when DP can't run locally).
        sdf_smooth = None
        sdf_smooth_proxy = float("inf")
        if deadline is None or (deadline - time.time()) > 300.0:
            log("  Lane 5: SDF basin + smooth-proxy descent")
            try:
                placer = SmoothGlobalPlacer(
                    num_steps=self.smooth_num_steps,
                    lr_frac=self.smooth_lr_frac,
                    gamma_start_frac=self.smooth_gamma_start_frac,
                    gamma_end_frac=self.smooth_gamma_end_frac,
                    overlap_lambda_end=self.smooth_overlap_lambda_end,
                    overlap_ramp_pct=self.smooth_overlap_ramp_pct,
                    legalize_step_frac=self.smooth_legalize_step_frac,
                    legalize_radius_steps=self.smooth_legalize_radius_steps,
                    init="sdf",
                    verbose=False,
                )
                pos = placer.place(benchmark)
                ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
                if ovl > 0:
                    pos, _ = project_overlaps(pos, benchmark)
                    ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]

                if ovl == 0 and smooth_cd_budget_s > 5.0:
                    # CD polish
                    evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
                    movable = [
                        i for i in range(benchmark.num_macros)
                        if not bool(benchmark.macro_fixed[i])
                    ]
                    run_cd_adaptive(
                        evaluator, benchmark, plc, movable,
                        min_time_s=min(60.0, smooth_cd_budget_s * 0.5),
                        hard_cap_s=smooth_cd_budget_s,
                        patience=3, plateau_threshold=0.001, log_fn=None,
                    )
                    pos = evaluator.placement.detach().clone().to(torch.float32)

                ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
                if ovl == 0:
                    sdf_smooth = pos
                    sdf_smooth_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
                    log(f"  Lane 5 done: proxy={sdf_smooth_proxy:.5f} ovl={ovl} "
                        f"(wall={time.time() - t0:.0f}s)")
                else:
                    log(f"  Lane 5 left {ovl} overlaps; excluded")
            except Exception as exc:
                log(f"  Lane 5 FAILED: {exc}; skipping")
                sdf_smooth = None

        # Phase 6: plateau pick across all 5 lanes (zero-overlap candidates only)
        candidates = [(e25_proxy, e25, "E25")]
        if e41 is not None:
            candidates.append((e41_proxy, e41, "E41"))
        if dp_stock is not None:
            candidates.append((dp_stock_proxy, dp_stock, "DP+stock"))
        if dp_smooth is not None:
            candidates.append((dp_smooth_proxy, dp_smooth, "DP+smooth"))
        if sdf_smooth is not None:
            candidates.append((sdf_smooth_proxy, sdf_smooth, "SDF+smooth"))
        candidates.sort(key=lambda c: c[0])
        plateau_proxy, plateau, plateau_label = candidates[0]
        log(f"  plateau pick: {plateau_label} ({plateau_proxy:.5f}) "
            f"[{', '.join(f'{n}={p:.5f}' for p, _, n in candidates)}]")

        # Phase 7: cascading saddle escape
        cascade_state = plateau
        cascade_proxy = plateau_proxy
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining < 400.0:
                log(f"  Phase 7 SADDLE SKIPPED ({remaining:.0f}s left)")
            else:
                cascade_budget = remaining - 60.0
                log(f"  Phase 7: cascading saddle (budget {cascade_budget:.0f}s)")
                try:
                    cascade_state, stats = cascading_saddle_escape(
                        plateau, benchmark, plc,
                        max_iters=self.max_iters,
                        eps_values=self.eps_values,
                        polish_budget=self.polish_budget,
                        total_budget_s=cascade_budget,
                        log=lambda s: None,
                    )
                    cascade_proxy = float(compute_proxy_cost(
                        cascade_state, benchmark, plc)["proxy_cost"])
                    log(f"  cascade done: {cascade_proxy:.5f} "
                        f"iters={stats.get('iters_run', '?')}")
                except Exception as exc:
                    log(f"  cascade FAILED: {exc}; falling back to plateau")
                    cascade_state = plateau
                    cascade_proxy = plateau_proxy

        # Phase 8: best of all (cascade vs each lane individually)
        all_candidates = list(candidates) + [(cascade_proxy, cascade_state, "cascade")]
        all_candidates.sort(key=lambda c: c[0])
        best_proxy, best_placement, best_name = all_candidates[0]
        log(f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"({', '.join(f'{n}={p:.5f}' for p, _, n in all_candidates)}) "
            f"total wall={time.time() - t0:.0f}s")

        # Validate.
        ovl = compute_overlap_metrics(best_placement, benchmark)["overlap_count"]
        if ovl > 0:
            raise RuntimeError(
                f"CascadeDPLanePatched winner ({best_name}) has {ovl} hard-macro overlaps"
            )
        return best_placement
