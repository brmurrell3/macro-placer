"""E126 — UltimatePlacer: 3-way composition of E115 + E117 + E120.

Three validated wins layered into a single placer:

1. **E115 FastDiffProxy** — `index_select` for pin gather + single-pass
   per-net trace congestion. 3-16x per-Adam-step speedup. Numerically
   identical to V3.
2. **E117 Gaussian density** — C^inf erf-integrated Gaussian replaces the
   piecewise-linear `_grid_density`. Validated -2.71% avg on dense benches.
3. **E120 Continuous Hessian saddle escape** — Hessian smallest-algebraic
   eigvec perturbation after Adam basin convergence, then resume Adam.
   Validated -1 to -2.2% per bench at proper budget.

Composition strategy:
- Build `FastDiffProxyGaussianDensity` that subclasses `FastDiffProxy` and
  swaps the density model for the E117 Gaussian. WL + congestion stay
  on the fast `index_select` path.
- Build `UltimatePlacer` that subclasses `ContinuousHessianSaddlePlacer`
  (from E120) but instantiates the new fast+gaussian proxy. All saddle
  escape mechanics work unchanged since they operate on `proxy.cost()`.
- At 1500s budget with V4 backbone we get 5-50x more Adam steps than the
  E125 (V3-backbone composition). This enables more saddle stages and
  longer resume descents.

Architecture (identical to E125, only the proxy backbone is faster):
  Phase A: V3+Gaussian+Fast Adam descent (300-500 steps)  -> settle in basin
  Phase B: Hessian saddle escape on Gaussian smooth proxy -> escape basin
           Adam resume from each candidate (150-300 steps)
           Pick best resulting smooth proxy
  Phase C: 1-3 more saddle escapes if budget remains
  Phase D: greedy_macro_legalize + CD polish
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
# Make all dependency module trees importable.
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E120_continuous_saddle" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from macro_legalizer import greedy_macro_legalize

from continuous_saddle_placer import (
    ContinuousHessianSaddlePlacer,
    _adam_descend,
)
from fast_proxy import FastDiffProxy, fast_loss_with_penalty
from gaussian_density import GaussianGridDensity


# ============================================================
# Fast + Gaussian-density proxy (composition of E115 + E117)
# ============================================================

class FastDiffProxyGaussianDensity(FastDiffProxy):
    """FastDiffProxy with Gaussian-smoothed density.

    Keeps E115's accelerated index_select WL gather and single-pass
    per-net-trace congestion; swaps the piecewise-linear `_grid_density`
    for the C^inf erf-integrated `GaussianGridDensity` from E117.

    Drop-in for FastDiffProxy: same `cost()` / `overlap_penalty()` /
    `out_of_canvas_penalty()` interface, so it works with both
    `fast_loss_with_penalty` (E115) and the E120 Hessian saddle escape.

    Numerical note: density model differs from FastDiffProxy by design
    (the whole point of E117). WL + congestion are bit-identical to V4.
    """

    def __init__(
        self,
        benchmark,
        plc,
        device: str = "cpu",
        gamma_frac: float = 5e-4,
        trace_kwargs: Optional[Dict] = None,
        # Gaussian density params (from E117)
        sigma_scale: float = 1.0,
        sigma_floor_frac: float = 0.5,
        topk_frac: float = 0.10,
        chunk_macros: Optional[int] = None,
    ):
        super().__init__(
            benchmark, plc, device=device, gamma_frac=gamma_frac,
            trace_kwargs=trace_kwargs,
        )
        # Swap in the Gaussian density model.
        self.gaussian_density = GaussianGridDensity(
            benchmark,
            device=device,
            sigma_scale=sigma_scale,
            sigma_floor_frac=sigma_floor_frac,
            topk_frac=topk_frac,
            grid_rows=benchmark.grid_rows,
            grid_cols=benchmark.grid_cols,
            chunk_macros=chunk_macros,
        )

    def cost(self, positions: torch.Tensor, *, include_congestion: bool = True):
        """Fast WL + Gaussian density + fast trace congestion.

        Same WL + congestion as `FastDiffProxy` (E115 index_select path),
        density swapped for E117 Gaussian-smeared erf-based model.
        """
        if positions.device != self.device:
            positions = positions.to(self.device)
        clamped = self._clamp_to_canvas(positions)
        # WL: reuse FastDiffProxy's fast index_select primitive.
        from fast_proxy import fast_lse_hpwl  # noqa: E402
        wl = fast_lse_hpwl(clamped, self.net_data, self.port_base, self.gamma) / self.wl_norm
        # Density: SWAP for Gaussian-smeared C^inf model.
        density = self.gaussian_density.compute_density(clamped)
        # Congestion: reuse FastPerNetTraceCongestion (single-pass, index_select).
        if include_congestion:
            cong = self.trace.compute_congestion(clamped)
        else:
            cong = torch.tensor(0.0, device=self.device)
        return wl + 0.5 * density + 0.5 * cong, {
            "wl": wl.detach(),
            "density": density.detach(),
            "cong": cong.detach(),
        }


# ============================================================
# UltimatePlacer
# ============================================================

class UltimatePlacer(ContinuousHessianSaddlePlacer):
    """E126: 3-way composition of E115 + E117 + E120.

    Subclass of ContinuousHessianSaddlePlacer that swaps the proxy for
    `FastDiffProxyGaussianDensity`. All saddle escape mechanics work
    unchanged because they only call `proxy.cost()`.

    Default config targets 1500s/bench with the V4 backbone enabling
    more Adam steps + deeper saddle cascades than E125's V3 backbone
    could afford.
    """

    def __init__(
        self,
        # New: Gaussian density params (forwarded to FastDiffProxyGaussianDensity)
        sigma_scale: float = 1.0,
        sigma_floor_frac: float = 0.5,
        topk_frac: float = 0.10,
        chunk_macros: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sigma_scale = sigma_scale
        self.sigma_floor_frac = sigma_floor_frac
        self.topk_frac = topk_frac
        self.chunk_macros = chunk_macros

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """Same pipeline as E120/E125, with FastDiffProxyGaussianDensity."""
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else float("inf")
        self._log(f"=== UltimatePlacer (E126) ({benchmark.name}) ===")
        self._log(
            f"  budget={self.budget_seconds:.0f}s phaseA_steps={self.num_steps_phaseA} "
            f"resume_steps={self.num_steps_resume} max_saddle={self.max_saddle_stages} "
            f"eps_values={self.eps_values}"
        )
        self._log(
            f"  Gaussian density: sigma_scale={self.sigma_scale} "
            f"sigma_floor_frac={self.sigma_floor_frac} topk_frac={self.topk_frac}"
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        cw = float(benchmark.canvas_width)
        lr = self.lr_frac * cw

        # Device handling: CUDA > MPS > CPU.
        if self.device == "cuda" and not torch.cuda.is_available():
            self._log("  CUDA not available, falling back to CPU")
            self.device = "cpu"
        elif self.device == "mps" and not torch.backends.mps.is_available():
            self._log("  MPS not available, falling back to CPU")
            self.device = "cpu"
        device = torch.device(self.device)

        # *** KEY CHANGE vs E125: use FastDiffProxyGaussianDensity ***
        proxy = FastDiffProxyGaussianDensity(
            benchmark, plc, device=str(device),
            gamma_frac=self.gamma_start_frac,
            trace_kwargs=self.trace_kwargs,
            sigma_scale=self.sigma_scale,
            sigma_floor_frac=self.sigma_floor_frac,
            topk_frac=self.topk_frac,
            chunk_macros=self.chunk_macros,
        )

        init_pos = self._init_positions(benchmark, plc=plc)
        fixed_mask = benchmark.macro_fixed.bool().to(device)

        # Anneal across (phaseA + max_saddle_stages * resume_steps) so gamma
        # reaches gamma_end_frac at the very end of the last Adam stage.
        total_anneal_steps = (
            self.num_steps_phaseA
            + self.max_saddle_stages * self.num_steps_resume
        )

        # ----- PHASE A: settle into a basin -----
        self._log(f"\n--- Phase A: Adam descent for {self.num_steps_phaseA} steps ---")
        phaseA_pos, phaseA_stats = _adam_descend(
            proxy, init_pos, fixed_mask,
            num_steps=self.num_steps_phaseA,
            lr=lr,
            gamma_start_frac=self.gamma_start_frac,
            gamma_end_frac=self.gamma_end_frac,
            overlap_lambda_start=self.overlap_lambda_start,
            overlap_lambda_end=self.overlap_lambda_end,
            overlap_ramp_pct=self.overlap_ramp_pct,
            boundary_lambda=self.boundary_lambda,
            log=self._log,
            log_every=self.log_every,
            starting_step=0,
            total_steps_for_anneal=total_anneal_steps,
        )
        self._log(
            f"  Phase A done: smooth={phaseA_stats['final_smooth']:.5f} "
            f"ovl_area={phaseA_stats['final_overlap_area']:.0f} "
            f"wall={time.time() - t0:.0f}s"
        )

        current_pos = phaseA_pos
        with torch.no_grad():
            sc, _ = proxy.cost(current_pos.to(proxy.device), include_congestion=True)
            current_smooth = float(sc.item())
        starting_global_step = self.num_steps_phaseA

        # ----- PHASE B & C: saddle escapes -----
        saddle_stage_stats: List[Dict] = []
        for stage_idx in range(self.max_saddle_stages):
            remaining = deadline - time.time()
            # Conservative budgeting: reserve CD polish budget + eigsh + 6 resumes.
            min_required = (
                self.cd_polish_s
                + 60.0  # eigsh
                + 6 * 30.0  # 6 perturb-and-resume attempts, ~30s each rough
            )
            if remaining < min_required:
                self._log(
                    f"\n[stage {stage_idx + 1}] remaining {remaining:.0f}s "
                    f"< min_required {min_required:.0f}s; skipping further saddles"
                )
                break

            self._log(f"\n--- Stage {stage_idx + 1}: Hessian saddle escape ---")
            cand_pos, stage_stats = self._try_saddle_escape(
                proxy, current_pos, fixed_mask, benchmark,
                stage=stage_idx + 1,
                total_anneal_steps=total_anneal_steps,
                starting_global_step=starting_global_step,
                lr=lr,
                deadline=deadline,
            )
            saddle_stage_stats.append(stage_stats)
            cand_smooth = stage_stats.get("best_smooth_after_resume", current_smooth)

            if cand_smooth < current_smooth - 1e-7:
                self._log(
                    f"[stage {stage_idx + 1}] ACCEPT: smooth "
                    f"{current_smooth:.5f} -> {cand_smooth:.5f} "
                    f"(delta={cand_smooth - current_smooth:+.5f})"
                )
                current_pos = cand_pos
                current_smooth = cand_smooth
                starting_global_step += self.num_steps_resume
            else:
                self._log(
                    f"[stage {stage_idx + 1}] REJECT: no improvement "
                    f"(best resume {cand_smooth:.5f} >= basin {current_smooth:.5f}); "
                    f"stopping saddle cascade"
                )
                break

        # ----- PHASE D: legalize + CD polish -----
        self._log(f"\n--- Phase D: legalize + CD polish ---")
        leg_t0 = time.time()
        legal_pos, leg_stats = greedy_macro_legalize(
            current_pos, benchmark,
            search_radius_steps=self.legalize_radius_steps,
            step_size_frac=self.legalize_step_frac,
            verbose=False,
        )
        ovl = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        if ovl > 0:
            self._log(
                f"  greedy_legalize left {ovl} overlaps "
                f"(moved={leg_stats.get('n_moved')} failed={leg_stats.get('n_failed')}); "
                f"running project_overlaps"
            )
            legal_pos, _ = project_overlaps(legal_pos, benchmark)
            ovl = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        self._log(f"  legalize done: ovl={ovl} wall={time.time() - leg_t0:.0f}s")

        if ovl > 0:
            raise RuntimeError(
                f"E126 produced {ovl} overlaps after legalize+project"
            )

        # CD polish
        remaining = max(30.0, deadline - time.time() - 5.0)
        cd_budget = min(remaining, self.cd_polish_s)
        self._log(f"  CD polish budget = {cd_budget:.0f}s")
        evaluator = IncrementalProxyEvaluator(benchmark, plc, legal_pos)
        movable_idx = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        run_cd_adaptive(
            evaluator, benchmark, plc, movable_idx,
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
            raise RuntimeError(f"E126 final has {final_ovl} overlaps")

        # Save run stats
        self.last_run_stats = {
            "bench": benchmark.name,
            "phaseA_stats": {k: v for k, v in phaseA_stats.items() if k != "history"},
            "saddle_stage_stats": saddle_stage_stats,
            "final_proxy": final_proxy,
            "final_overlap": final_ovl,
            "total_wall_s": time.time() - t0,
            "config": {
                "sigma_scale": self.sigma_scale,
                "sigma_floor_frac": self.sigma_floor_frac,
                "budget_seconds": self.budget_seconds,
                "cd_polish_s": self.cd_polish_s,
                "num_steps_phaseA": self.num_steps_phaseA,
                "num_steps_resume": self.num_steps_resume,
                "max_saddle_stages": self.max_saddle_stages,
                "device": self.device,
            },
        }
        return final


def main_smoke():
    """Smoke test on ibm17.

    Run:
        cd <repo>
        uv run python experiments/E126_fast_gaussian_hessian/code/ultimate_placer.py
    """
    import os
    BENCH = os.environ.get("E126_BENCH", "ibm17")
    BUDGET = float(os.environ.get("E126_BUDGET", "1500"))
    DEVICE = os.environ.get("E126_DEVICE", "cpu")  # cuda, mps, cpu
    bench_dir = find_benchmark_dir(BENCH)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(
        f"Loaded {BENCH}: {benchmark.num_macros} macros "
        f"({benchmark.num_hard_macros} hard), nets={benchmark.num_nets}, "
        f"canvas={benchmark.canvas_width:.0f}x{benchmark.canvas_height:.0f}",
        flush=True,
    )

    placer = UltimatePlacer(
        budget_seconds=BUDGET,
        num_steps_phaseA=500,  # V4 enables more steps
        num_steps_resume=200,  # also more resume steps per saddle
        max_saddle_stages=3,   # try up to 3 saddle stages
        overlap_lambda_end=10.0,
        cd_polish_s=720.0,  # 12 min CD polish at 1500s budget
        sigma_scale=1.0,
        sigma_floor_frac=0.5,
        device=DEVICE,
        verbose=True,
        log_every=100,
    )
    t0 = time.time()
    final = placer.place(benchmark)
    print(
        f"\n=== {BENCH} DONE ===\n"
        f"  final_proxy: {placer.last_run_stats['final_proxy']:.5f}\n"
        f"  wall: {time.time() - t0:.0f}s\n"
        f"  References:\n"
        f"    V3Min ovl10 720s baseline (ibm17): 1.200\n"
        f"    E117 alone @720s (ibm17): 1.181\n"
        f"    E120 alone @720s (ibm17): 1.212\n"
        f"    Target (E126 composed): < 1.13\n",
        flush=True,
    )


if __name__ == "__main__":
    main_smoke()
