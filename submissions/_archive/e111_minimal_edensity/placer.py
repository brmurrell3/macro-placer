"""E111MinimalEDensity — V3 + eDensity descent + CD polish (E114).

Drop-in variant of submissions/_archive/e111_minimal_ovl10_480s/ that
swaps the grid-bin density for electrostatic eDensity. Everything else
identical (SDF init, Adam descent, greedy legalize, CD polish).

Hypothesis: eDensity's globally-smooth electric potential gives Adam a
canonical-aligned density gradient (no bin-edge discontinuities),
producing a basin closer to the canonical optimum than grid-bin density.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E114_edensity" / "code",
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

from diff_proxy_v3_edensity import SmoothGlobalPlacerV3eDensity  # noqa: E402


class E111MinimalEDensityPlacer:
    """V3 + eDensity descent + CD polish.

    Args:
        edensity_scale: Per-bench scale calibration. eDensity raw values
            are ~10-20x grid-density at SDF init. Set scale ~0.1 to
            bring them to a comparable magnitude before descent.
        edensity_target: Target density per bin (1.0 = penalize only
            above-full bins; lower → push for more spread).
        edensity_sigma_frac: Gaussian smearing width in cells.
    """

    def __init__(
        self,
        budget_seconds: Optional[float] = 720.0,
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        cd_polish_s: float = 600.0,
        cd_plateau_threshold: float = 0.001,
        rng_seed: int = 42,
        verbose: bool = True,
        # eDensity-specific
        edensity_scale: float = 0.10,
        edensity_target: float = 1.0,
        edensity_sigma_frac: float = 0.5,
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
        self.edensity_kwargs = {
            "scale": edensity_scale,
            "target_density": edensity_target,
            "sigma_frac": edensity_sigma_frac,
        }

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E111MinimalEDensity ({benchmark.name}) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Phase 1: V3+eDensity descent + legalize
        placer = SmoothGlobalPlacerV3eDensity(
            num_steps=self.num_steps,
            lr_frac=self.lr_frac,
            gamma_start_frac=self.gamma_start_frac,
            gamma_end_frac=self.gamma_end_frac,
            overlap_lambda_end=self.overlap_lambda_end,
            overlap_ramp_pct=self.overlap_ramp_pct,
            init=self.init,
            rng_seed=self.rng_seed,
            verbose=False,
            edensity_kwargs=self.edensity_kwargs,
        )
        pos = placer.place(benchmark)
        descent_wall = time.time() - t0
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        descent_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        self._log(
            f"  V3eDensity+legalize: proxy={descent_proxy:.5f} ovl={descent_ovl} "
            f"wall={descent_wall:.0f}s"
        )

        if descent_ovl > 0:
            self._log(f"  Recovery: project_overlaps")
            pos, _ = project_overlaps(pos, benchmark)
            descent_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
            if descent_ovl > 0:
                raise RuntimeError(
                    f"eDensity placer produced {descent_ovl} overlaps despite legalize+project"
                )

        # Phase 2: CD polish (cap at deadline)
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd_budget = max(30.0, min(remaining, self.cd_polish_s))
        else:
            cd_budget = self.cd_polish_s
        self._log(f"  CD polish budget={cd_budget:.0f}s")

        evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
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
            f"total_wall={time.time()-t0:.0f}s"
        )

        if final_ovl > 0:
            raise RuntimeError(f"E111MinimalEDensity produced {final_ovl} overlaps")
        return final


# Match the convention from e111_minimal: also alias the class name
class Placer(E111MinimalEDensityPlacer):
    pass
