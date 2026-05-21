"""E132 — V4Gauss → CD → warm-start re-descent → CD.

Adds a bridge step between two CD polishes: an 80-step Adam warm-restart
from the CD-polished position with tiny γ (5e-5) and strong overlap
penalty (50). Hypothesis: Adam can co-move groups of macros that CD's
single-axis sweep cannot, escaping the CD plateau without the Hessian
machinery E128 relied on.

Pipeline:
  1. V4Gaussian descent (500 steps, init="sdf")
  2. greedy legalize + project_overlaps
  3. CD polish (cd_polish_s budget)            → pos_polished_1
  4. V4Gaussian warm-restart from pos_polished_1
     (80 steps, lr_frac=0.0008, γ flat 5e-5, λ_ovl_end=50, "warmstart")
  5. project_overlaps
  6. CD polish (cd_polish_s budget)            → final

Total wall budget: ~720s.

Each CD polish is half the total CD budget (200s default).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E127_v4_gaussian" / "code",
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
from smooth_global_placer_v4_gaussian_warmstart import (  # noqa: E402
    SmoothGlobalPlacerV4GaussianWarmstart,
)


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class E132PolishRelaxPlacer:
    """V4Gauss → CD → warm-restart re-descent → CD."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 720.0,
        # Stage-1 descent
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        # CD polish (each leg)
        cd_polish_s: float = 200.0,
        cd_plateau_threshold: float = 0.001,
        # Warm-restart re-descent
        warmstart_steps: int = 80,
        warmstart_lr_frac: float = 0.0008,
        warmstart_gamma_frac: float = 5e-5,
        warmstart_overlap_lambda_end: float = 50.0,
        warmstart_overlap_ramp_pct: float = 0.5,
        rng_seed: int = 42,
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
        self.cd_polish_s = cd_polish_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.warmstart_steps = warmstart_steps
        self.warmstart_lr_frac = warmstart_lr_frac
        self.warmstart_gamma_frac = warmstart_gamma_frac
        self.warmstart_overlap_lambda_end = warmstart_overlap_lambda_end
        self.warmstart_overlap_ramp_pct = warmstart_overlap_ramp_pct
        self.rng_seed = rng_seed
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _cd_polish(self, pos, benchmark, plc, budget_s, label):
        self._log(f"  {label}: CD polish budget={budget_s:.0f}s")
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
        result = evaluator.placement.detach().clone().to(torch.float32)
        proxy = float(compute_proxy_cost(result, benchmark, plc)["proxy_cost"])
        ovl = compute_overlap_metrics(result, benchmark)["overlap_count"]
        self._log(f"  {label}: proxy={proxy:.5f} ovl={ovl}")
        return result, proxy, ovl

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E132PolishRelaxPlacer ({benchmark.name}) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        device = _best_device()
        self._log(f"  device={device}")

        # === Stage 1: V4Gaussian descent ===
        pos = None
        for attempt, cfg in enumerate([
            dict(overlap_lambda_end=self.overlap_lambda_end),
            dict(overlap_lambda_end=50.0),
            dict(overlap_lambda_end=100.0, num_steps=300),
        ]):
            try:
                num_steps = cfg.get("num_steps", self.num_steps)
                ovl_end = cfg["overlap_lambda_end"]
                self._log(
                    f"  stage1 attempt {attempt+1}: ovl_end={ovl_end} steps={num_steps}"
                )
                descender = SmoothGlobalPlacerV4Gaussian(
                    num_steps=num_steps,
                    lr_frac=self.lr_frac,
                    gamma_start_frac=self.gamma_start_frac,
                    gamma_end_frac=self.gamma_end_frac,
                    overlap_lambda_end=ovl_end,
                    overlap_ramp_pct=self.overlap_ramp_pct,
                    init=self.init,
                    device=device,
                    rng_seed=self.rng_seed + attempt * 100,
                    verbose=False,
                )
                pos_try = descender.place(benchmark)
                ovl_try = compute_overlap_metrics(pos_try, benchmark)["overlap_count"]
                if ovl_try == 0:
                    pos = pos_try
                    self._log(f"  stage1 attempt {attempt+1}: ovl=0")
                    break
                pos_try2, _ = project_overlaps(pos_try, benchmark)
                if compute_overlap_metrics(pos_try2, benchmark)["overlap_count"] == 0:
                    pos = pos_try2
                    self._log(f"  stage1 attempt {attempt+1}: ovl=0 after project_overlaps")
                    break
                self._log(
                    f"  stage1 attempt {attempt+1}: still {ovl_try} overlaps, retrying..."
                )
            except Exception as exc:
                self._log(f"  stage1 attempt {attempt+1} EXCEPTION: {exc}")
                continue
            if deadline is not None and time.time() > deadline - 60:
                break

        if pos is None:
            self._log("  fallback: SDF + project_overlaps")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)

        descent_wall = time.time() - t0
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self._log(f"  basin: proxy={descent_proxy:.5f} wall={descent_wall:.0f}s")

        # === Stage 2: first CD polish ===
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            # Reserve room for re-descent (~30s on smaller benches, more on bigger)
            # and second CD polish.
            cd1_budget = max(30.0, min(remaining * 0.4, self.cd_polish_s))
        else:
            cd1_budget = self.cd_polish_s
        pos_polished_1, proxy_polished_1, ovl_p1 = self._cd_polish(
            pos, benchmark, plc, cd1_budget, "CD1"
        )
        if ovl_p1 > 0:
            raise RuntimeError(f"CD1 left {ovl_p1} overlaps")

        # === Stage 3: warm-restart re-descent ===
        t_wr0 = time.time()
        try:
            self._log(
                f"  warmstart: {self.warmstart_steps} steps lr_frac={self.warmstart_lr_frac} "
                f"gamma={self.warmstart_gamma_frac} ovl_end={self.warmstart_overlap_lambda_end}"
            )
            wr_descender = SmoothGlobalPlacerV4GaussianWarmstart(
                num_steps=self.warmstart_steps,
                lr_frac=self.warmstart_lr_frac,
                gamma_start_frac=self.warmstart_gamma_frac,
                gamma_end_frac=self.warmstart_gamma_frac,
                overlap_lambda_start=0.0,
                overlap_lambda_end=self.warmstart_overlap_lambda_end,
                overlap_ramp_pct=self.warmstart_overlap_ramp_pct,
                init="warmstart",
                warmstart_pos=pos_polished_1,
                device=device,
                rng_seed=self.rng_seed,
                verbose=False,
            )
            pos_wr, wr_stats = wr_descender.descend(benchmark, plc)
            pos_wr_proxy_raw = float(
                compute_proxy_cost(pos_wr, benchmark, plc)["proxy_cost"]
            )
            ovl_wr_raw = compute_overlap_metrics(pos_wr, benchmark)["overlap_count"]
            self._log(
                f"  warmstart raw: proxy={pos_wr_proxy_raw:.5f} ovl={ovl_wr_raw} "
                f"wall={time.time()-t_wr0:.1f}s"
            )
            if ovl_wr_raw > 0:
                pos_wr, _ = project_overlaps(pos_wr, benchmark)
                ovl_wr = compute_overlap_metrics(pos_wr, benchmark)["overlap_count"]
                if ovl_wr > 0:
                    self._log(
                        f"  warmstart projection failed (ovl={ovl_wr}); reverting to CD1 pos"
                    )
                    pos_wr = pos_polished_1
            pos_wr_proxy = float(
                compute_proxy_cost(pos_wr, benchmark, plc)["proxy_cost"]
            )
            self._log(
                f"  warmstart projected: proxy={pos_wr_proxy:.5f} "
                f"(vs CD1 {proxy_polished_1:.5f}, Δ={pos_wr_proxy - proxy_polished_1:+.5f})"
            )
            # If warm-restart made things noticeably worse (>5%), revert.
            # Otherwise CD2 will recover or improve.
            if pos_wr_proxy > proxy_polished_1 * 1.05:
                self._log(
                    "  warmstart diverged >5%; reverting to CD1 pos for CD2"
                )
                pos_wr = pos_polished_1
        except Exception as exc:
            self._log(f"  warmstart EXCEPTION: {exc}; reverting to CD1 pos")
            pos_wr = pos_polished_1

        # === Stage 4: second CD polish ===
        if deadline is not None:
            remaining = deadline - time.time() - 5.0
            cd2_budget = max(30.0, min(remaining, self.cd_polish_s))
        else:
            cd2_budget = self.cd_polish_s
        final, final_proxy, final_ovl = self._cd_polish(
            pos_wr, benchmark, plc, cd2_budget, "CD2"
        )
        if final_ovl > 0:
            raise RuntimeError(f"Final placement has {final_ovl} overlaps")

        # Stat: did warm-restart improve final vs first CD polish?
        delta_vs_cd1 = final_proxy - proxy_polished_1
        self._log(
            f"  COMPARISON: CD1={proxy_polished_1:.5f}, CD2={final_proxy:.5f}, "
            f"Δ(CD2-CD1)={delta_vs_cd1:+.5f}"
        )
        self._log(
            f"  Final: proxy={final_proxy:.5f} ovl={final_ovl} "
            f"total_wall={time.time()-t0:.0f}s"
        )
        return final
