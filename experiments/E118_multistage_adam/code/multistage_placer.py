"""E118 — MultiStageAdamPlacer: ePlace/Xplace 3-stage descent + Adam.

Departs from V3 (`SmoothGlobalPlacerV3`)'s single linear γ-anneal
(5e-3 → 5e-5 over 500 Adam steps) by partitioning descent into three
fixed-γ stages, each allowed to converge before γ drops. This mirrors
ePlace (`eplace::Algorithm 2`) and Xplace's multi-stage optimizer.

Departs from V5 (`SmoothGlobalPlacerV5`) by **keeping Adam** rather than
switching to Nesterov-BB. E113/V5 measured Nesterov-BB at +5.8% worse
than Adam on ibm01 raw — the Xplace optimizer recipe doesn't transfer
to our objective geometry. The actual lift in V5 came from its other
two ideas (multi-stage γ + best-tracking). This experiment isolates
those wins under Adam without dragging the worse optimizer along.

Schedule (defaults; configurable per-stage):
  Stage A (200 steps): γ = 5e-3, overlap_lambda ramp 0 → 0.5
  Stage B (200 steps): γ = 5e-4, overlap_lambda = 0.5 fixed
  Stage C (200 steps): γ = 5e-5, overlap_lambda = 0.5 fixed

γ is held CONSTANT within each stage (the canonical ePlace pattern).
Best position by smooth proxy is tracked across all stages and returned
for legalize / CD polish.

The overlap_lambda interpretation: the soft pairwise overlap penalty
coefficient — what ePlace calls the "density weight" — sits separately
from the smooth proxy's density coefficient (which stays at 0.5).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, sdf_init
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v3 import DiffProxyV3, loss_with_penalty_v3
from macro_legalizer import greedy_macro_legalize


class MultiStageAdamPlacer:
    """Adam + 3-stage fixed-γ descent with overlap-lambda ramp in stage A."""

    def __init__(
        self,
        # Steps per stage (A/B/C)
        stage_steps: Tuple[int, int, int] = (200, 200, 200),
        # γ for each stage (FIXED within stage)
        gamma_stage_A: float = 5e-3,
        gamma_stage_B: float = 5e-4,
        gamma_stage_C: float = 5e-5,
        # overlap_lambda schedule (the task spec uses "density_weight" 0 → 0.5)
        overlap_lambda_stage_A: Tuple[float, float] = (0.0, 0.5),
        overlap_lambda_stage_B: float = 0.5,
        overlap_lambda_stage_C: float = 0.5,
        # Optim
        base_lr_frac: float = 0.005,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        # Boundary
        boundary_lambda: float = 50.0,
        # Init
        init: str = "sdf",
        device: str = "cpu",
        rng_seed: int = 42,
        # Logging
        verbose: bool = True,
        log_every: int = 50,
        # Legalize
        legalize_radius_steps: int = 80,
        legalize_step_frac: float = 0.02,
        # V3 proxy kwargs
        trace_kwargs: Optional[Dict] = None,
        include_congestion: bool = True,
    ):
        self.stage_steps = stage_steps
        self.gammas = (gamma_stage_A, gamma_stage_B, gamma_stage_C)
        # Stage A is a ramp tuple; B/C are scalars. Normalize all to ramp tuples.
        self.overlap_lambdas: Tuple[Tuple[float, float], ...] = (
            overlap_lambda_stage_A,
            (overlap_lambda_stage_B, overlap_lambda_stage_B),
            (overlap_lambda_stage_C, overlap_lambda_stage_C),
        )
        self.base_lr_frac = base_lr_frac
        self.adam_betas = adam_betas
        self.boundary_lambda = boundary_lambda
        self.init = init
        self.device = device
        self.rng_seed = rng_seed
        self.verbose = verbose
        self.log_every = log_every
        self.legalize_radius_steps = legalize_radius_steps
        self.legalize_step_frac = legalize_step_frac
        self.trace_kwargs = trace_kwargs
        self.include_congestion = include_congestion

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _init_positions(self, benchmark: Benchmark, plc=None) -> torch.Tensor:
        if self.init == "sdf":
            pos = sdf_init(benchmark)
            pos, n_iter = project_overlaps(pos, benchmark)
            self._log(f"  init=sdf: project_overlaps n_iter={n_iter}")
            return pos
        elif self.init == "dpo":
            sys.path.insert(0, str(_ROOT / "experiments" / "E18_dpo_init" / "code"))
            from cd_lns_sa_dpo_init import _best_of_v2_init
            assert plc is not None
            pos = _best_of_v2_init(benchmark, plc, seed=self.rng_seed)
            pos, n_iter = project_overlaps(pos, benchmark)
            self._log(f"  init=dpo: project_overlaps n_iter={n_iter}")
            return pos
        else:
            raise ValueError(f"Unknown init {self.init!r}")

    def _stage_descend(
        self,
        proxy: DiffProxyV3,
        positions: torch.Tensor,
        fixed_mask: torch.Tensor,
        *,
        stage_name: str,
        num_steps: int,
        gamma: float,
        lambda_range: Tuple[float, float],
        lr: float,
        global_step_offset: int,
    ) -> Tuple[torch.Tensor, float, List[Dict]]:
        """Run one fixed-γ stage of Adam descent.

        γ is held constant at *gamma*; overlap_lambda linearly interpolates
        between lambda_range[0] and lambda_range[1] across num_steps. (For
        stages B/C with a flat lambda, the range collapses to a scalar.)
        Returns (best_positions, best_smooth_cost, history).
        """
        optimizer = torch.optim.Adam([positions], lr=lr, betas=self.adam_betas)
        proxy.set_gamma_frac(gamma)
        lambda_start, lambda_end = lambda_range
        history = []
        best_smooth = float("inf")
        best_pos = positions.detach().clone()

        for k in range(num_steps):
            t = k / max(1, num_steps - 1)
            ovl_lambda = lambda_start + t * (lambda_end - lambda_start)

            optimizer.zero_grad()
            loss, parts = loss_with_penalty_v3(
                proxy, positions, ovl_lambda,
                include_congestion=self.include_congestion,
                boundary_lambda=self.boundary_lambda,
            )
            loss.backward()
            with torch.no_grad():
                if positions.grad is not None:
                    positions.grad[fixed_mask] = 0.0
            optimizer.step()

            smooth_now = float(parts["smooth_cost"].item())
            if smooth_now < best_smooth:
                best_smooth = smooth_now
                best_pos = positions.detach().clone()

            if k % self.log_every == 0 or k == num_steps - 1:
                msg = (
                    f"  [{stage_name}] step {global_step_offset + k:4d}  "
                    f"loss={float(loss.item()):.5f}  smooth={smooth_now:.5f}  "
                    f"wl={float(parts['wl'].item()):.4f}  "
                    f"d={float(parts['density'].item()):.4f}  "
                    f"c={float(parts['cong'].item()):.4f}  "
                    f"ovl_area={float(parts['overlap_area_raw'].item()):.0f}  "
                    f"γ={gamma:.5f}  λ_ovl={ovl_lambda:.3f}"
                )
                if self.verbose:
                    print(msg, flush=True)
                history.append({
                    "step": global_step_offset + k,
                    "stage": stage_name,
                    "loss": float(loss.item()),
                    "smooth": smooth_now,
                    "wl": float(parts["wl"].item()),
                    "density": float(parts["density"].item()),
                    "cong": float(parts["cong"].item()),
                    "overlap_area": float(parts["overlap_area_raw"].item()),
                    "gamma": gamma,
                    "overlap_lambda": ovl_lambda,
                })

        return best_pos, best_smooth, history

    def descend(
        self,
        benchmark: Benchmark,
        plc,
    ) -> Tuple[torch.Tensor, Dict]:
        t0 = time.time()
        init_pos = self._init_positions(benchmark, plc=plc)
        cw = float(benchmark.canvas_width)
        base_lr = self.base_lr_frac * cw

        if self.device == "mps" and not torch.backends.mps.is_available():
            self._log("  MPS not available, falling back to CPU")
            self.device = "cpu"
        device = torch.device(self.device)
        proxy = DiffProxyV3(
            benchmark, plc, device=str(device),
            gamma_frac=self.gammas[0],
            trace_kwargs=self.trace_kwargs,
        )

        positions = init_pos.clone().detach().to(device).requires_grad_(True)
        fixed_mask = benchmark.macro_fixed.bool().to(device)

        history_all = []
        global_offset = 0
        # Best-overall = best smooth across ALL stages (key V5 win we keep).
        best_overall = (float("inf"), positions.detach().clone(), 0)

        for stage_idx, (stage_steps, gamma, lambda_range) in enumerate(zip(
            self.stage_steps, self.gammas, self.overlap_lambdas
        )):
            stage_name = "ABC"[stage_idx]
            self._log(
                f"  === Stage {stage_name}: {stage_steps} steps, γ={gamma:.5f} (FIXED), "
                f"λ_ovl=[{lambda_range[0]:.3f}, {lambda_range[1]:.3f}], lr={base_lr:.2f}"
            )
            best_pos_stage, best_smooth_stage, hist = self._stage_descend(
                proxy, positions, fixed_mask,
                stage_name=stage_name,
                num_steps=stage_steps,
                gamma=gamma,
                lambda_range=lambda_range,
                lr=base_lr,
                global_step_offset=global_offset,
            )
            global_offset += stage_steps
            history_all.extend(hist)
            self._log(
                f"  === Stage {stage_name} done: best_smooth={best_smooth_stage:.5f} "
                f"(global step {global_offset})"
            )

            if best_smooth_stage < best_overall[0]:
                best_overall = (best_smooth_stage, best_pos_stage.clone(), global_offset)

            # Carry positions forward to the next stage. We continue from
            # current-position (NOT best-of-stage) so the next stage sees
            # the natural end-of-stage state. The best-overall snapshot
            # is what we return for legalize.
            # (V5 carried best-of-stage; that worked but is more aggressive.
            # Continuing from current matches ePlace/Xplace more faithfully.)

        positions_final = best_overall[1].cpu()
        stats = {
            "history": history_all,
            "best_smooth": best_overall[0],
            "best_at_step": best_overall[2],
            "descend_wall_s": time.time() - t0,
        }
        return positions_final, stats

    def legalize(
        self,
        positions: torch.Tensor,
        benchmark: Benchmark,
    ) -> Tuple[torch.Tensor, Dict]:
        t0 = time.time()
        legal_pos, leg_stats = greedy_macro_legalize(
            positions, benchmark,
            search_radius_steps=self.legalize_radius_steps,
            step_size_frac=self.legalize_step_frac,
            verbose=False,
        )
        ovl = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        if ovl > 0:
            self._log(
                f"  greedy_legalize left {ovl} overlaps; project_overlaps fallback"
            )
            legal_pos, _ = project_overlaps(legal_pos, benchmark)
            ovl = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        leg_stats["final_overlaps"] = ovl
        leg_stats["legalize_wall_s"] = time.time() - t0
        return legal_pos, leg_stats

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        self._log(f"=== MultiStageAdamPlacer ({benchmark.name}) ===")
        self._log(
            f"  config: stages={self.stage_steps} γ={self.gammas} "
            f"λ_ovl={self.overlap_lambdas} base_lr_frac={self.base_lr_frac} "
            f"init={self.init} device={self.device}"
        )
        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        positions, descend_stats = self.descend(benchmark, plc)
        self._log(
            f"  descent done: best_smooth={descend_stats['best_smooth']:.5f} "
            f"@ global_step {descend_stats['best_at_step']} "
            f"wall={descend_stats['descend_wall_s']:.1f}s"
        )

        legal_pos, leg_stats = self.legalize(positions, benchmark)
        self._log(
            f"  legalize: n_moved={leg_stats.get('n_moved')} "
            f"final_overlaps={leg_stats['final_overlaps']} "
            f"wall={leg_stats['legalize_wall_s']:.1f}s"
        )

        proxy_cost = float(compute_proxy_cost(legal_pos, benchmark, plc)["proxy_cost"])
        self._log(
            f"  Final: proxy={proxy_cost:.5f} ovl={leg_stats['final_overlaps']} "
            f"total_wall={time.time()-t0:.1f}s"
        )
        return legal_pos
