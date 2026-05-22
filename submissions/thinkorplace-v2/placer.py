"""thinkorplace-v2 — V4 FastDiffProxy + Gaussian density + extended CD polish.

Pipeline:
  V4+Gaussian Adam descent → greedy legalize → project_overlaps → CD polish

Three architectural changes vs the v1 cascade plus extended CD polish budget
(CD on hard benches was budget-limited, not plateau-saturated):

  1. Per-net-trace congestion (E111) — matches canonical within 15-25 %.
  2. Gaussian-smeared erf density (E117) — C∞ at cell boundaries; Adam
     finds a lower basin on dense benches (ibm12/14/17/18).
  3. FastDiffProxy backbone (E115) — drops per-net pair_chunk loop, uses
     index_select; 3-16× faster fwd+bwd on GPU.
  4. Extended budget — cd_polish_s 600→900s, budget_seconds 720→1500s,
     +0.73 % EPYC lift verified.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (_ROOT, _HERE / "lib"):
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


# Defensive retry chain — escalate overlap penalty if a previous attempt
# leaves residual overlaps. Each entry overrides the placer's defaults.
_DESCENT_RETRIES = (
    {},                                            # primary cfg (use defaults)
    {"overlap_lambda_end": 50.0},                  # stronger overlap
    {"overlap_lambda_end": 100.0, "num_steps": 300},  # aggressive
)


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Placer:
    """V4 + Gaussian density + extended CD polish.

    ``place(benchmark)`` returns a ``[num_macros, 2]`` float32 tensor of
    macro center coordinates with zero hard-macro overlaps.
    """

    def __init__(
        self,
        budget_seconds: Optional[float] = 1500.0,
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        cd_polish_s: float = 900.0,
        cd_plateau_threshold: float = 0.001,
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
        self.rng_seed = rng_seed
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _try_descent(self, benchmark, device, attempt, cfg):
        """One descent attempt; return clean positions or None."""
        num_steps = cfg.get("num_steps", self.num_steps)
        ovl_end = cfg.get("overlap_lambda_end", self.overlap_lambda_end)
        self._log(f"  attempt {attempt + 1}: ovl_end={ovl_end} steps={num_steps}")
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
        pos = descender.place(benchmark)
        if compute_overlap_metrics(pos, benchmark)["overlap_count"] == 0:
            self._log(f"  attempt {attempt + 1}: ovl=0")
            return pos
        pos, _ = project_overlaps(pos, benchmark)
        if compute_overlap_metrics(pos, benchmark)["overlap_count"] == 0:
            self._log(f"  attempt {attempt + 1}: ovl=0 after project_overlaps")
            return pos
        self._log(f"  attempt {attempt + 1}: still overlapping, retrying")
        return None

    def _safe_init(self, benchmark):
        """SDF init + project_overlaps. Always-legal fallback."""
        self._log("  fallback: SDF + project_overlaps")
        pos = sdf_init(benchmark)
        pos, _ = project_overlaps(pos, benchmark)
        return pos

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== thinkorplace-v2 ({benchmark.name}) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        device = _best_device()
        self._log(f"  device={device}")

        pos = None
        for attempt, cfg in enumerate(_DESCENT_RETRIES):
            try:
                pos = self._try_descent(benchmark, device, attempt, cfg)
                if pos is not None:
                    break
            except Exception as exc:
                self._log(f"  attempt {attempt + 1} EXCEPTION: {exc}")
            if deadline is not None and time.time() > deadline - 60:
                break

        if pos is None:
            pos = self._safe_init(benchmark)

        basin_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self._log(f"  basin: proxy={basin_proxy:.5f} wall={time.time() - t0:.0f}s")

        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd_budget = max(30.0, min(remaining, self.cd_polish_s))
        else:
            cd_budget = self.cd_polish_s
        self._log(f"  CD polish budget={cd_budget:.0f}s")

        evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
        movable = [i for i in range(benchmark.num_macros)
                   if not bool(benchmark.macro_fixed[i])]
        run_cd_adaptive(
            evaluator, benchmark, plc, movable,
            min_time_s=cd_budget * 0.5,
            hard_cap_s=cd_budget,
            patience=5,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=None,
        )

        final = evaluator.placement.detach().clone().to(torch.float32)
        final_proxy = float(compute_proxy_cost(final, benchmark, plc)["proxy_cost"])
        final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
        self._log(
            f"  Final: proxy={final_proxy:.5f} ovl={final_ovl} "
            f"total_wall={time.time() - t0:.0f}s"
        )
        if final_ovl > 0:
            raise RuntimeError(f"Final placement has {final_ovl} overlaps")
        return final
