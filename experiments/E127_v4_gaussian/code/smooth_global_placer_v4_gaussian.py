"""E127 — V4 + Gaussian density placer.

Subclasses SmoothGlobalPlacerV4 and swaps `FastDiffProxy` for
`FastDiffProxyGaussian` (V4 backbone + erf-based smeared density).

Mirrors V4's descend/legalize/place flow; only proxy construction differs.
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
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark
from smooth_global_placer_v4 import SmoothGlobalPlacerV4
from fast_proxy import fast_loss_with_penalty
from v4_gaussian_proxy import FastDiffProxyGaussian


class SmoothGlobalPlacerV4Gaussian(SmoothGlobalPlacerV4):
    """V4 + Gaussian density. Same constructor; descend uses FastDiffProxyGaussian."""

    def __init__(self, *args, sigma_scale: float = 1.0, topk_frac: float = 0.10, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_scale = sigma_scale
        self.topk_frac = topk_frac

    def descend(self, benchmark: Benchmark, plc) -> Tuple[torch.Tensor, Dict]:
        t0 = time.time()
        init_pos = self._init_positions(benchmark, plc=plc)
        cw = float(benchmark.canvas_width)
        lr = self.lr_frac * cw

        if self.adaptive_num_steps:
            n_macros = int(benchmark.num_hard_macros)
            adaptive_steps = max(200, int(n_macros * self.steps_per_macro))
            self._log(
                f"  adaptive_num_steps: {self.num_steps} -> {adaptive_steps} "
                f"(n_hard={n_macros}, spm={self.steps_per_macro})"
            )
            num_steps = adaptive_steps
        else:
            num_steps = self.num_steps

        if self.device == "cuda" and not torch.cuda.is_available():
            self._log("  CUDA not available, falling back to CPU")
            self.device = "cpu"
        if self.device == "mps" and not torch.backends.mps.is_available():
            self._log("  MPS not available, falling back to CPU")
            self.device = "cpu"
        device = torch.device(self.device)

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

        history = []
        last_parts = None
        last_loss = None
        for step in range(num_steps):
            t = step / max(1, num_steps - 1)
            gamma_frac = self.gamma_start_frac + t * (
                self.gamma_end_frac - self.gamma_start_frac
            )
            proxy.set_gamma_frac(gamma_frac)
            ramp_t = min(1.0, step / max(1, num_steps * self.overlap_ramp_pct))
            overlap_lambda = self.overlap_lambda_start + ramp_t * (
                self.overlap_lambda_end - self.overlap_lambda_start
            )

            optimizer.zero_grad()
            loss, parts = fast_loss_with_penalty(
                proxy, positions, overlap_lambda,
                include_congestion=self.include_congestion,
                boundary_lambda=self.boundary_lambda,
            )
            loss.backward()
            with torch.no_grad():
                if positions.grad is not None:
                    positions.grad[fixed_mask] = 0.0
            optimizer.step()

            last_parts = parts
            last_loss = float(loss.item())
            if step % self.log_every == 0 or step == num_steps - 1:
                self._log(
                    f"  step {step:4d}  loss={last_loss:.5f}  "
                    f"smooth={parts['smooth_cost'].item():.5f}  "
                    f"wl={parts['wl'].item():.4f}  "
                    f"d={parts['density'].item():.4f}  "
                    f"c={parts['cong'].item():.4f}  "
                    f"ovl_area={parts['overlap_area_raw'].item():.0f}  "
                    f"g={gamma_frac:.4f}  l_ovl={overlap_lambda:.1f}"
                )
                history.append({
                    "step": step,
                    "loss": last_loss,
                    "smooth": parts["smooth_cost"].item(),
                    "wl": parts["wl"].item(),
                    "density": parts["density"].item(),
                    "cong": parts["cong"].item(),
                    "overlap_area": parts["overlap_area_raw"].item(),
                    "gamma_frac": gamma_frac,
                    "overlap_lambda": overlap_lambda,
                })

        positions_final = positions.detach().cpu()
        return positions_final, {
            "history": history,
            "final_loss": last_loss,
            "final_smooth": last_parts["smooth_cost"].item() if last_parts else None,
            "final_overlap_area": last_parts["overlap_area_raw"].item() if last_parts else None,
            "descend_wall_s": time.time() - t0,
            "final_parts": {k: v.item() if hasattr(v, "item") else v
                           for k, v in (last_parts or {}).items()},
        }
