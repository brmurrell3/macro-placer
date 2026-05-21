"""E134 — Checkpointable V4+Gaussian descender (serial-with-resume).

Splits SmoothGlobalPlacerV4Gaussian.descend() into two callable phases so we
can run N seeds for `num_steps_a` steps, snapshot their (positions, optimizer
state, smooth score), pick top-K by smooth score, then resume only those for
`num_steps_b` more steps.

Per-step body is verbatim from V4Gaussian's descend (same gamma schedule, same
overlap ramp, same boundary penalty). Only structural change: split point at
step==num_steps_a; we save full state so resume continues identically to a
single-pass run with same total step count.

Key invariant: descend_partial(N=150) → descend_resume(M=350) is bitwise-
equivalent to a single descend(N+M=500) given the same seed and device, as
long as proxy.set_gamma_frac() schedule uses total_steps=500 in both halves.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
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
from fast_proxy import fast_loss_with_penalty
from v4_gaussian_proxy import FastDiffProxyGaussian
from smooth_global_placer_v4_gaussian import SmoothGlobalPlacerV4Gaussian


class CheckpointableV4Gauss(SmoothGlobalPlacerV4Gaussian):
    """V4+Gaussian descender split into descend_partial + descend_resume.

    Total step count (=num_steps) governs the gamma/overlap schedules so two
    halves stack identically to a single descend. The split point is
    `partial_steps`; the resumer runs `total_steps - partial_steps` more steps.

    descend_partial returns a STATE_DICT we can carry across calls; it holds
      - positions tensor (with grad attached)
      - optimizer state dict
      - proxy (same instance reused across resume)
      - step counter
      - cached fixed_mask
    so resume mutates and returns the final positions.
    """

    def descend_partial(
        self,
        benchmark: Benchmark,
        plc,
        partial_steps: int,
    ) -> Dict:
        """Run `partial_steps` of descent; return state for resume.

        State dict contains everything needed by descend_resume to continue
        the same trajectory.
        """
        t0 = time.time()
        init_pos = self._init_positions(benchmark, plc=plc)
        cw = float(benchmark.canvas_width)
        lr = self.lr_frac * cw

        if self.adaptive_num_steps:
            n_macros = int(benchmark.num_hard_macros)
            total_steps = max(200, int(n_macros * self.steps_per_macro))
        else:
            total_steps = self.num_steps

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        if self.device == "mps" and not torch.backends.mps.is_available():
            self.device = "cpu"
        device = torch.device(self.device)

        # Seed RNG for reproducibility per-restart.
        torch.manual_seed(self.rng_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.rng_seed)

        proxy = FastDiffProxyGaussian(
            benchmark, plc, device=str(device),
            gamma_frac=self.gamma_start_frac,
            trace_kwargs=self.trace_kwargs,
            sigma_scale=self.sigma_scale,
            topk_frac=self.topk_frac,
        )

        positions = init_pos.clone().detach().to(device).requires_grad_(True)
        fixed_mask = benchmark.macro_fixed.bool().to(device)
        optimizer = torch.optim.Adam([positions], lr=lr)

        last_parts = None
        n_steps_run = min(partial_steps, total_steps)
        self._run_steps_inplace(
            positions, optimizer, proxy, fixed_mask,
            step_start=0, step_end=n_steps_run, total_steps=total_steps,
        )
        # Final smooth at step partial_steps for ranking.
        with torch.no_grad():
            _, last_parts = fast_loss_with_penalty(
                proxy, positions, self.overlap_lambda_end,
                include_congestion=self.include_congestion,
                boundary_lambda=self.boundary_lambda,
            )
        smooth_score = float(last_parts["smooth_cost"].item())
        ovl_area = float(last_parts["overlap_area_raw"].item())

        state = {
            "positions": positions,
            "optimizer": optimizer,
            "proxy": proxy,
            "fixed_mask": fixed_mask,
            "step": n_steps_run,
            "total_steps": total_steps,
            "device": device,
            "smooth_score": smooth_score,
            "overlap_area": ovl_area,
            "partial_wall_s": time.time() - t0,
        }
        return state

    def descend_resume(self, state: Dict) -> Tuple[torch.Tensor, Dict]:
        """Continue a partial descent to total_steps; return final positions+stats."""
        t0 = time.time()
        positions = state["positions"]
        optimizer = state["optimizer"]
        proxy = state["proxy"]
        fixed_mask = state["fixed_mask"]
        step_start = state["step"]
        total_steps = state["total_steps"]

        self._run_steps_inplace(
            positions, optimizer, proxy, fixed_mask,
            step_start=step_start, step_end=total_steps, total_steps=total_steps,
        )

        with torch.no_grad():
            _, last_parts = fast_loss_with_penalty(
                proxy, positions, self.overlap_lambda_end,
                include_congestion=self.include_congestion,
                boundary_lambda=self.boundary_lambda,
            )
        positions_final = positions.detach().cpu()
        stats = {
            "final_smooth": float(last_parts["smooth_cost"].item()),
            "final_overlap_area": float(last_parts["overlap_area_raw"].item()),
            "resume_wall_s": time.time() - t0,
            "partial_wall_s": state.get("partial_wall_s", 0.0),
        }
        return positions_final, stats

    def _run_steps_inplace(
        self,
        positions: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        proxy,
        fixed_mask: torch.Tensor,
        step_start: int,
        step_end: int,
        total_steps: int,
    ) -> None:
        """Identical per-step body to V4Gaussian.descend; runs steps [step_start, step_end)."""
        for step in range(step_start, step_end):
            t = step / max(1, total_steps - 1)
            gamma_frac = self.gamma_start_frac + t * (
                self.gamma_end_frac - self.gamma_start_frac
            )
            proxy.set_gamma_frac(gamma_frac)
            ramp_t = min(1.0, step / max(1, total_steps * self.overlap_ramp_pct))
            overlap_lambda = self.overlap_lambda_start + ramp_t * (
                self.overlap_lambda_end - self.overlap_lambda_start
            )

            optimizer.zero_grad()
            loss, _ = fast_loss_with_penalty(
                proxy, positions, overlap_lambda,
                include_congestion=self.include_congestion,
                boundary_lambda=self.boundary_lambda,
            )
            loss.backward()
            with torch.no_grad():
                if positions.grad is not None:
                    positions.grad[fixed_mask] = 0.0
            optimizer.step()
