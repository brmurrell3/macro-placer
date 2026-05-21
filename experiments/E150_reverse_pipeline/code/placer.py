"""E150 — Reverse pipeline: SDF → CD → Adam → CD.

Inverts the v2 / E127 / E132 lineage's descent-first ordering. Instead of
descending a smooth proxy from SDF and polishing with CD, we polish the
SDF init with CD *first* (canonical anchor), then run a warm-start Adam
descent from that polished position, then a final CD polish.

Pipeline:
  1. sdf_init + project_overlaps          → pos_sdf
  2. run_cd_adaptive (cd1_budget_s)       → pos_polished_1
  3. V4GaussianWarmstart descend
       init="warmstart", warmstart_pos=pos_polished_1
       (300 steps, lr_frac=0.001, γ flat 5e-5, λ_ovl_end=20, 50% ramp)
                                          → pos_mid_raw
     legalize: greedy + project_overlaps  → pos_mid
  4. run_cd_adaptive (cd2_budget_s)       → pos_final

Defaults match the spec: CD₁ 300s, CD₂ 400s, descender ~300 steps,
total budget 1500s. Imports the E132 warmstart class without modifying
shipped files.

Divergence guard: if middle-Adam proxy > 1.05 × CD1 proxy after
legalize, revert to CD1 pos for CD2.
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
    _ROOT / "experiments" / "E132_polish_relax" / "code",
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

from smooth_global_placer_v4_gaussian_warmstart import (  # noqa: E402
    SmoothGlobalPlacerV4GaussianWarmstart,
)


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class E150ReversePipelinePlacer:
    """SDF → CD → Adam (warmstart, tiny lr) → CD."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 1500.0,
        # CD stages
        cd1_budget_s: float = 300.0,
        cd2_budget_s: float = 400.0,
        cd_plateau_threshold: float = 0.001,
        # Middle Adam (warmstart from CD-polished)
        mid_num_steps: int = 300,
        mid_lr_frac: float = 0.001,
        mid_gamma_frac: float = 5e-5,
        mid_overlap_lambda_end: float = 20.0,
        mid_overlap_ramp_pct: float = 0.5,
        mid_divergence_factor: float = 1.05,
        rng_seed: int = 42,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.cd1_budget_s = cd1_budget_s
        self.cd2_budget_s = cd2_budget_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.mid_num_steps = mid_num_steps
        self.mid_lr_frac = mid_lr_frac
        self.mid_gamma_frac = mid_gamma_frac
        self.mid_overlap_lambda_end = mid_overlap_lambda_end
        self.mid_overlap_ramp_pct = mid_overlap_ramp_pct
        self.mid_divergence_factor = mid_divergence_factor
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
        self._log(f"=== E150ReversePipelinePlacer ({benchmark.name}) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        device = _best_device()
        self._log(f"  device={device}")

        # === Stage 1: SDF init + project ===
        t_sdf0 = time.time()
        pos = sdf_init(benchmark)
        pos, n_iter = project_overlaps(pos, benchmark)
        ovl_sdf = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        proxy_sdf = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self._log(
            f"  SDF init: proxy={proxy_sdf:.5f} ovl={ovl_sdf} "
            f"project_iter={n_iter} wall={time.time()-t_sdf0:.1f}s"
        )
        if ovl_sdf > 0:
            raise RuntimeError(f"SDF init left {ovl_sdf} overlaps after project")

        # === Stage 2: CD₁ ===
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd1_budget = max(30.0, min(remaining * 0.35, self.cd1_budget_s))
        else:
            cd1_budget = self.cd1_budget_s
        pos_polished_1, proxy_polished_1, ovl_p1 = self._cd_polish(
            pos, benchmark, plc, cd1_budget, "CD1"
        )
        if ovl_p1 > 0:
            raise RuntimeError(f"CD1 left {ovl_p1} overlaps")

        # === Stage 3: middle Adam (warmstart from CD-polished) ===
        t_mid0 = time.time()
        pos_mid = pos_polished_1
        proxy_mid_raw = proxy_polished_1
        ovl_mid_raw = 0
        try:
            self._log(
                f"  midAdam: {self.mid_num_steps} steps lr_frac={self.mid_lr_frac} "
                f"gamma={self.mid_gamma_frac} ovl_end={self.mid_overlap_lambda_end}"
            )
            mid_descender = SmoothGlobalPlacerV4GaussianWarmstart(
                num_steps=self.mid_num_steps,
                lr_frac=self.mid_lr_frac,
                gamma_start_frac=self.mid_gamma_frac,
                gamma_end_frac=self.mid_gamma_frac,
                overlap_lambda_start=0.0,
                overlap_lambda_end=self.mid_overlap_lambda_end,
                overlap_ramp_pct=self.mid_overlap_ramp_pct,
                init="warmstart",
                warmstart_pos=pos_polished_1,
                device=device,
                rng_seed=self.rng_seed,
                verbose=False,
            )
            pos_mid_raw_t, mid_stats = mid_descender.descend(benchmark, plc)
            proxy_mid_raw = float(
                compute_proxy_cost(pos_mid_raw_t, benchmark, plc)["proxy_cost"]
            )
            ovl_mid_raw = compute_overlap_metrics(pos_mid_raw_t, benchmark)["overlap_count"]
            self._log(
                f"  midAdam raw: proxy={proxy_mid_raw:.5f} ovl={ovl_mid_raw} "
                f"wall={time.time()-t_mid0:.1f}s"
            )

            # Legalize / project
            pos_mid_candidate = pos_mid_raw_t
            if ovl_mid_raw > 0:
                pos_mid_candidate, _ = project_overlaps(pos_mid_candidate, benchmark)
                ovl_mid_proj = compute_overlap_metrics(pos_mid_candidate, benchmark)["overlap_count"]
                if ovl_mid_proj > 0:
                    self._log(
                        f"  midAdam projection failed (ovl={ovl_mid_proj}); reverting to CD1"
                    )
                    pos_mid_candidate = pos_polished_1
            proxy_mid_proj = float(
                compute_proxy_cost(pos_mid_candidate, benchmark, plc)["proxy_cost"]
            )
            self._log(
                f"  midAdam projected: proxy={proxy_mid_proj:.5f} "
                f"(vs CD1 {proxy_polished_1:.5f}, Δ={proxy_mid_proj - proxy_polished_1:+.5f})"
            )
            # Divergence guard
            if proxy_mid_proj > proxy_polished_1 * self.mid_divergence_factor:
                self._log(
                    f"  midAdam diverged > {(self.mid_divergence_factor-1)*100:.0f}%; reverting to CD1"
                )
                pos_mid = pos_polished_1
            else:
                pos_mid = pos_mid_candidate
        except Exception as exc:
            self._log(f"  midAdam EXCEPTION: {exc}; reverting to CD1")
            pos_mid = pos_polished_1

        proxy_mid_used = float(compute_proxy_cost(pos_mid, benchmark, plc)["proxy_cost"])
        ovl_mid_used = compute_overlap_metrics(pos_mid, benchmark)["overlap_count"]
        self._log(
            f"  midAdam USED: proxy={proxy_mid_used:.5f} ovl={ovl_mid_used} "
            f"wall_total={time.time()-t_mid0:.1f}s"
        )

        # === Stage 4: CD₂ ===
        if deadline is not None:
            remaining = deadline - time.time() - 5.0
            cd2_budget = max(30.0, min(remaining, self.cd2_budget_s))
        else:
            cd2_budget = self.cd2_budget_s
        final, final_proxy, final_ovl = self._cd_polish(
            pos_mid, benchmark, plc, cd2_budget, "CD2"
        )
        if final_ovl > 0:
            raise RuntimeError(f"Final placement has {final_ovl} overlaps")

        delta_vs_cd1 = final_proxy - proxy_polished_1
        self._log(
            f"  COMPARISON: SDF={proxy_sdf:.5f} CD1={proxy_polished_1:.5f} "
            f"midAdam_raw={proxy_mid_raw:.5f} midAdam_used={proxy_mid_used:.5f} "
            f"CD2={final_proxy:.5f} Δ(CD2-CD1)={delta_vs_cd1:+.5f}"
        )
        self._log(
            f"  Final: proxy={final_proxy:.5f} ovl={final_ovl} "
            f"total_wall={time.time()-t0:.0f}s"
        )
        return final
