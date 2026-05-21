"""E135 — SDF + DPO two-lane V4+Gaussian placer.

Pipeline per bench:
  Lane A: V4+Gauss descent with init="sdf" (500 steps) → legalize.
  Lane B: V4+Gauss descent with init="dpo" (500 steps) → legalize.
  Pick lane with lower canonical proxy → CD polish.

DPO-init runs E18's `_best_of_v2_init` (returns the better of {raw SDF,
DPO-v2 optimized}). Even when that lane's seed coincides with raw SDF,
the descent trajectory from a different starting point can converge to
a different basin. When DPO is strictly better, this gives a fundamentally
different attractor than Lane A.

Budget: 900s default. Each lane ~250s (descent + legalize) for ibm04-
class, CD polish gets remaining ~400s.
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
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E18_dpo_init" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E127_v4_gaussian" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
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

from smooth_global_placer_v4_gaussian import SmoothGlobalPlacerV4Gaussian
from macro_legalizer import greedy_macro_legalize


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class E135SdfDpoLanePlacer:
    """Two-lane V4+Gaussian: SDF basin vs DPO basin, pick better, CD polish."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 900.0,
        rng_seed: int = 42,
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        cd_polish_s: float = 400.0,
        cd_plateau_threshold: float = 0.001,
        legalize_radius_steps: int = 80,
        legalize_step_frac: float = 0.02,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.rng_seed = rng_seed
        self.num_steps = num_steps
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.cd_polish_s = cd_polish_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.legalize_radius_steps = legalize_radius_steps
        self.legalize_step_frac = legalize_step_frac
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _make_descender(self, init: str, device: str) -> SmoothGlobalPlacerV4Gaussian:
        return SmoothGlobalPlacerV4Gaussian(
            num_steps=self.num_steps,
            lr_frac=self.lr_frac,
            gamma_start_frac=self.gamma_start_frac,
            gamma_end_frac=self.gamma_end_frac,
            overlap_lambda_end=self.overlap_lambda_end,
            overlap_ramp_pct=self.overlap_ramp_pct,
            init=init,
            device=device,
            rng_seed=self.rng_seed,
            verbose=False,
            legalize_radius_steps=self.legalize_radius_steps,
            legalize_step_frac=self.legalize_step_frac,
        )

    def _run_lane(
        self,
        lane_name: str,
        init: str,
        benchmark: Benchmark,
        plc,
        device: str,
    ) -> Optional[Tuple[float, torch.Tensor]]:
        """Run one V4+Gauss lane; return (canonical_proxy, legal_pos) or None on failure."""
        t_lane = time.time()
        try:
            desc = self._make_descender(init, device)
            pos_raw, stats = desc.descend(benchmark, plc)
            ovl_pre = compute_overlap_metrics(pos_raw, benchmark)["overlap_count"]
            legal_pos = self._legalize(pos_raw, benchmark)
            ovl_post = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
            wall = time.time() - t_lane
            if ovl_post > 0:
                self._log(
                    f"  [{lane_name}] init={init}: smooth={stats['final_smooth']:.5f} "
                    f"ovl_pre={ovl_pre} ovl_post={ovl_post} SKIP "
                    f"wall={wall:.0f}s"
                )
                return None
            canonical = float(
                compute_proxy_cost(legal_pos, benchmark, plc)["proxy_cost"]
            )
            self._log(
                f"  [{lane_name}] init={init}: smooth={stats['final_smooth']:.5f} "
                f"canonical={canonical:.5f} ovl_pre={ovl_pre} ovl_post={ovl_post} "
                f"wall={wall:.0f}s"
            )
            return canonical, legal_pos
        except Exception as exc:
            wall = time.time() - t_lane
            self._log(f"  [{lane_name}] init={init}: EXCEPTION {exc} wall={wall:.0f}s")
            return None

    def _legalize(self, positions: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        legal_pos, _ = greedy_macro_legalize(
            positions, benchmark,
            search_radius_steps=self.legalize_radius_steps,
            step_size_frac=self.legalize_step_frac,
            verbose=False,
        )
        ovl = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        if ovl > 0:
            legal_pos, _ = project_overlaps(legal_pos, benchmark)
        return legal_pos

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_global = time.time()
        deadline = (t_global + self.budget_seconds) if self.budget_seconds else None
        self._log(
            f"=== E135 sdf+dpo lane ({benchmark.name}) "
            f"num_steps={self.num_steps} seed={self.rng_seed} "
            f"budget={self.budget_seconds}s ==="
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        device = _best_device()
        self._log(f"  device={device}")

        # ---- Lane A: SDF init ----
        results = []  # list of (canonical, lane_name, legal_pos)
        sdf_result = self._run_lane("A", "sdf", benchmark, plc, device)
        if sdf_result is not None:
            results.append((sdf_result[0], "SDF", sdf_result[1]))

        # Safety: if running out of time, skip Lane B
        if deadline is not None and time.time() > deadline - (self.cd_polish_s + 100):
            self._log(f"  [B] budget tight after Lane A; skip Lane B")
        else:
            # ---- Lane B: DPO init ----
            dpo_result = self._run_lane("B", "dpo", benchmark, plc, device)
            if dpo_result is not None:
                results.append((dpo_result[0], "DPO", dpo_result[1]))

        if not results:
            self._log("  Both lanes failed; SDF fallback")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)
            return self._cd_polish_and_return(
                pos, benchmark, plc, deadline, t_global
            )

        # Pick best canonical
        results.sort(key=lambda r: r[0])
        best_canonical, best_lane, best_pos = results[0]
        per_lane = ", ".join(f"{n}={c:.5f}" for c, n, _ in results)
        self._log(
            f"  [PICK] {best_lane} basin (canonical={best_canonical:.5f}); "
            f"per-lane: {per_lane}"
        )

        return self._cd_polish_and_return(
            best_pos, benchmark, plc, deadline, t_global,
            best_lane=best_lane,
        )

    def _cd_polish_and_return(
        self,
        pos: torch.Tensor,
        benchmark: Benchmark,
        plc,
        deadline: Optional[float],
        t_global: float,
        best_lane: str = "?",
    ) -> torch.Tensor:
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd_budget = max(30.0, min(remaining, self.cd_polish_s))
        else:
            cd_budget = self.cd_polish_s
        self._log(f"  CD polish budget={cd_budget:.0f}s (from {best_lane} basin)")

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
            f"basin={best_lane} total_wall={time.time()-t_global:.0f}s"
        )
        if final_ovl > 0:
            raise RuntimeError(f"Final has {final_ovl} overlaps")
        return final
