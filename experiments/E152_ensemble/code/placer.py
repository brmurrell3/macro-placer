"""E152 — In-process two-lane ensemble (v2-extCD + E138 saddle).

Hypothesis: offline ensemble of v2-extCD (avg 0.98387) and E138 (avg 0.98379)
shows -0.33% per-bench MIN lift (avg 0.98064). 7 benches won by E138, 10 by
v2-extCD. They find STRUCTURALLY DIFFERENT basins. Running both lanes
in-process and selecting canonical-best preserves the ensemble win without
per-bench tuning.

Architecture:
  Lane A (v2-extCD): V4+Gauss descent + legalize + CD polish 700s
                     (rng_seed=42)
  Lane B (E138):     V4+Gauss descent + legalize + CD polish 400s
                     + bounded cascade saddle 240s + CD polish 200s
                     (rng_seed=142)
  Pick lane with lower canonical proxy.

Total budget: ~28 min/bench, fits within 60-min/bench partcl cap.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _ROOT,
    _HERE,
    _ROOT / "experiments" / "E74_hessian_saddle" / "code",
    _ROOT / "experiments" / "E84_cascading_saddle" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E127_v4_gaussian" / "code",
    _ROOT / "experiments" / "E138_bounded_saddle" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v4_gaussian import SmoothGlobalPlacerV4Gaussian  # noqa: E402
from bounded_saddle import bounded_cascading_saddle_escape  # noqa: E402


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class E152EnsemblePlacer:
    """Two-lane in-process ensemble: v2-extCD + E138 saddle. Pick canonical-best."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 1800.0,  # 30 min/bench total cap
        # Shared descent params (matches thinkorplace-v2 / E138)
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        # Lane A: v2-extCD style
        lane_a_cd_polish_s: float = 700.0,
        lane_a_seed: int = 42,
        # Lane B: E138 saddle style
        lane_b_cd1_polish_s: float = 400.0,
        lane_b_saddle_budget_s: float = 240.0,
        lane_b_saddle_max_iters: int = 2,
        lane_b_eigsh_maxiter: int = 50,
        lane_b_eigsh_tol: float = 1e-2,
        lane_b_cd2_polish_s: float = 200.0,
        lane_b_seed: int = 142,
        # Shared CD plateau threshold
        cd_plateau_threshold: float = 0.001,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.num_steps = num_steps
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.init = init
        self.lane_a_cd_polish_s = lane_a_cd_polish_s
        self.lane_a_seed = lane_a_seed
        self.lane_b_cd1_polish_s = lane_b_cd1_polish_s
        self.lane_b_saddle_budget_s = lane_b_saddle_budget_s
        self.lane_b_saddle_max_iters = lane_b_saddle_max_iters
        self.lane_b_eigsh_maxiter = lane_b_eigsh_maxiter
        self.lane_b_eigsh_tol = lane_b_eigsh_tol
        self.lane_b_cd2_polish_s = lane_b_cd2_polish_s
        self.lane_b_seed = lane_b_seed
        self.cd_plateau_threshold = cd_plateau_threshold
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    # ---------- helpers ----------

    def _do_descent(self, benchmark: Benchmark, device: str, seed: int) -> torch.Tensor:
        """V4+Gauss descent with 3-attempt fallback ladder (mirrors v2)."""
        pos = None
        for attempt, cfg in enumerate([
            dict(overlap_lambda_end=self.overlap_lambda_end),
            dict(overlap_lambda_end=50.0),
            dict(overlap_lambda_end=100.0, num_steps=300),
        ]):
            try:
                num_steps = cfg.get("num_steps", self.num_steps)
                ovl_end = cfg["overlap_lambda_end"]
                self._log(f"    descent attempt {attempt+1}: ovl_end={ovl_end} steps={num_steps}")
                descender = SmoothGlobalPlacerV4Gaussian(
                    num_steps=num_steps,
                    lr_frac=self.lr_frac,
                    gamma_start_frac=self.gamma_start_frac,
                    gamma_end_frac=self.gamma_end_frac,
                    overlap_lambda_end=ovl_end,
                    overlap_ramp_pct=self.overlap_ramp_pct,
                    init=self.init,
                    device=device,
                    rng_seed=seed + attempt * 100,
                    verbose=False,
                )
                pos_try = descender.place(benchmark)
                ovl_try = compute_overlap_metrics(pos_try, benchmark)["overlap_count"]
                if ovl_try == 0:
                    pos = pos_try
                    self._log(f"    descent attempt {attempt+1}: ovl=0")
                    break
                pos_try2, _ = project_overlaps(pos_try, benchmark)
                if compute_overlap_metrics(pos_try2, benchmark)["overlap_count"] == 0:
                    pos = pos_try2
                    self._log(f"    descent attempt {attempt+1}: ovl=0 after project_overlaps")
                    break
                self._log(f"    descent attempt {attempt+1}: still {ovl_try} overlaps, retrying...")
            except Exception as exc:
                self._log(f"    descent attempt {attempt+1} EXCEPTION: {exc}")
                continue

        if pos is None:
            self._log("    descent fallback: SDF + project_overlaps")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)
        return pos

    def _do_cd_polish(self, pos, benchmark, plc, budget_s: float):
        evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        run_cd_adaptive(
            evaluator, benchmark, plc, movable,
            min_time_s=budget_s * 0.5,
            hard_cap_s=budget_s,
            patience=5,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=None,
        )
        out = evaluator.placement.detach().clone().to(torch.float32)
        return out

    # ---------- lanes ----------

    def _run_lane_a(self, benchmark, plc, device, deadline) -> Tuple[torch.Tensor, float, float]:
        """Lane A: V4+Gauss descent + legalize + CD polish ~700s. RNG=42."""
        t0 = time.time()
        self._log(f"  -- Lane A (v2-extCD, seed={self.lane_a_seed}) --")
        pos = self._do_descent(benchmark, device, self.lane_a_seed)
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self._log(f"    basin: proxy={descent_proxy:.5f} wall={time.time()-t0:.0f}s")

        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd_budget = max(30.0, min(remaining, self.lane_a_cd_polish_s))
        else:
            cd_budget = self.lane_a_cd_polish_s
        self._log(f"    CD polish budget={cd_budget:.0f}s")

        pos = self._do_cd_polish(pos, benchmark, plc, cd_budget)
        proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        wall = time.time() - t0
        self._log(f"  Lane A: proxy={proxy:.5f} ovl={ovl} wall={wall:.0f}s")
        if ovl > 0:
            return None, float("inf"), wall
        return pos, proxy, wall

    def _run_lane_b(self, benchmark, plc, device, deadline) -> Tuple[torch.Tensor, float, float]:
        """Lane B: V4+Gauss descent + legalize + CD polish 400s + bounded
        cascade saddle + CD polish 200s. RNG=142."""
        t0 = time.time()
        self._log(f"  -- Lane B (E138 saddle, seed={self.lane_b_seed}) --")
        pos = self._do_descent(benchmark, device, self.lane_b_seed)
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self._log(f"    basin: proxy={descent_proxy:.5f} wall={time.time()-t0:.0f}s")

        # CD polish #1
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd1_budget = max(30.0, min(remaining, self.lane_b_cd1_polish_s))
        else:
            cd1_budget = self.lane_b_cd1_polish_s
        self._log(f"    CD1 budget={cd1_budget:.0f}s")
        pos = self._do_cd_polish(pos, benchmark, plc, cd1_budget)
        cd1_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        cd1_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        self._log(f"    after CD1: proxy={cd1_proxy:.5f} ovl={cd1_ovl} wall={time.time()-t0:.0f}s")
        if cd1_ovl > 0:
            self._log("    Lane B: CD1 produced overlaps, aborting")
            return None, float("inf"), time.time() - t0

        # Bounded cascade saddle escape
        t_saddle = time.time()
        if deadline is not None:
            remaining = deadline - time.time() - self.lane_b_cd2_polish_s - 30.0
            sb = max(30.0, min(remaining, self.lane_b_saddle_budget_s))
        else:
            sb = self.lane_b_saddle_budget_s
        self._log(f"    saddle budget={sb:.0f}s")
        try:
            saddle_pos, saddle_info = bounded_cascading_saddle_escape(
                pos, benchmark, plc,
                max_iters=self.lane_b_saddle_max_iters,
                total_budget_s=sb,
                eigsh_maxiter=self.lane_b_eigsh_maxiter,
                eigsh_tol=self.lane_b_eigsh_tol,
                log=lambda s: self._log(f"    [saddle] {s}"),
            )
            saddle_ovl = compute_overlap_metrics(saddle_pos, benchmark)["overlap_count"]
            if saddle_ovl == 0:
                saddle_proxy = float(compute_proxy_cost(saddle_pos, benchmark, plc)["proxy_cost"])
                if saddle_proxy <= cd1_proxy:
                    pos = saddle_pos
                    self._log(f"    saddle accept: {cd1_proxy:.5f} -> {saddle_proxy:.5f}")
                else:
                    self._log(f"    saddle reject: {saddle_proxy:.5f} > {cd1_proxy:.5f}")
            else:
                self._log(f"    saddle output has {saddle_ovl} overlaps; rejecting")
        except Exception as exc:
            self._log(f"    saddle EXCEPTION: {exc}; skipping")
        self._log(f"    saddle wall={time.time()-t_saddle:.0f}s")

        # CD polish #2
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd2_budget = max(30.0, min(remaining, self.lane_b_cd2_polish_s))
        else:
            cd2_budget = self.lane_b_cd2_polish_s
        self._log(f"    CD2 budget={cd2_budget:.0f}s")
        pos = self._do_cd_polish(pos, benchmark, plc, cd2_budget)
        proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        wall = time.time() - t0
        self._log(f"  Lane B: proxy={proxy:.5f} ovl={ovl} wall={wall:.0f}s")
        if ovl > 0:
            return None, float("inf"), wall
        return pos, proxy, wall

    # ---------- top-level ----------

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline_total = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E152 ensemble ({benchmark.name}) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        device = _best_device()
        self._log(f"  device={device}")

        # Per-lane budgeting: split remaining wall in half (approximately)
        # Lane A target ~13 min, Lane B target ~17 min (saddle costs extra)
        # We use absolute deadlines so each lane is self-bounded.
        if self.budget_seconds is not None:
            # Lane A gets ~45% of total, Lane B gets the rest
            lane_a_deadline = t0 + self.budget_seconds * 0.45
            lane_b_deadline = deadline_total
        else:
            lane_a_deadline = None
            lane_b_deadline = None

        # Lane A
        pos_a, proxy_a, wall_a = self._run_lane_a(benchmark, plc, device, lane_a_deadline)

        # Lane B
        pos_b, proxy_b, wall_b = self._run_lane_b(benchmark, plc, device, lane_b_deadline)

        # Pick the better lane
        if pos_a is None and pos_b is None:
            raise RuntimeError("Both lanes failed to produce a legal placement")
        if pos_a is None:
            picked = "B"
            final = pos_b
            final_proxy = proxy_b
        elif pos_b is None:
            picked = "A"
            final = pos_a
            final_proxy = proxy_a
        elif proxy_a <= proxy_b:
            picked = "A"
            final = pos_a
            final_proxy = proxy_a
        else:
            picked = "B"
            final = pos_b
            final_proxy = proxy_b

        total_wall = time.time() - t0
        self._log(
            f"=== E152 PICK={picked}: proxy_a={proxy_a:.5f} proxy_b={proxy_b:.5f} "
            f"-> final={final_proxy:.5f} total_wall={total_wall:.0f}s ==="
        )

        # Final sanity
        final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
        if final_ovl > 0:
            raise RuntimeError(f"Final placement has {final_ovl} overlaps")
        return final
