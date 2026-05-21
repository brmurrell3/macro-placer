"""E115 SmoothGlobalPlacerV5 — V4 + torch.compile(reduce-overhead).

V4 already gives 16× speedup over V3 from index_select + single-pass.
V5 adds `torch.compile(mode="reduce-overhead")` on the per-step loss
closure, which fuses kernels and removes Python dispatch — getting us
~2-3× more on CUDA.

CUDA-only (compile mode is most effective on CUDA via CUDA graphs).
Falls back to V4 behavior on MPS/CPU.
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
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from smooth_global_placer_v4 import SmoothGlobalPlacerV4
from fast_proxy import fast_loss_with_penalty


class SmoothGlobalPlacerV5(SmoothGlobalPlacerV4):
    """V4 + torch.compile on the loss closure.

    Falls back to V4 behavior if device is not CUDA or torch.compile is
    unavailable. Compile overhead (~10s for first call) is amortized
    across the 300+ descent steps.
    """

    def __init__(self, *args, compile_mode: str = "reduce-overhead", **kwargs):
        super().__init__(*args, **kwargs)
        self.compile_mode = compile_mode

    def descend(self, benchmark, plc):
        # If not on CUDA, skip compile and fall back to V4
        if self.device != "cuda" or not torch.cuda.is_available():
            return super().descend(benchmark, plc)
        return self._descend_compiled(benchmark, plc)

    def _descend_compiled(self, benchmark, plc):
        from fast_proxy import FastDiffProxy
        from macro_place.cd_core import project_overlaps, sdf_init

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

        device = torch.device(self.device)
        proxy = FastDiffProxy(
            benchmark, plc, device=str(device),
            gamma_frac=self.gamma_start_frac,
            trace_kwargs=self.trace_kwargs,
        )

        positions = init_pos.clone().detach().to(device).requires_grad_(True)
        fixed_mask = benchmark.macro_fixed.bool().to(device)
        optimizer = torch.optim.Adam([positions], lr=lr)

        # Define the closure once; we only update overlap_lambda (a Python scalar)
        # and proxy.gamma between steps. torch.compile is keyed on tensor
        # shape, so gamma being a Python float and ovl_lambda being a Python
        # float both cause re-compilation. Keep them as scalars (PyTorch
        # promotes them to 0-d tensors at trace time).
        def step_closure(positions, overlap_lambda):
            loss, _ = fast_loss_with_penalty(
                proxy, positions, overlap_lambda,
                include_congestion=self.include_congestion,
                boundary_lambda=self.boundary_lambda,
            )
            return loss

        try:
            compiled_closure = torch.compile(step_closure, mode=self.compile_mode)
            # warmup compile
            for _ in range(2):
                optimizer.zero_grad()
                loss = compiled_closure(positions, 0.0)
                loss.backward()
                with torch.no_grad():
                    if positions.grad is not None:
                        positions.grad[fixed_mask] = 0.0
                optimizer.step()
            self._log(f"  compiled step warmed up")
        except Exception as e:
            self._log(f"  torch.compile failed ({e}); falling back to eager")
            compiled_closure = step_closure

        history = []
        last_loss = None
        last_parts_log = None
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
            loss = compiled_closure(positions, overlap_lambda)
            loss.backward()
            with torch.no_grad():
                if positions.grad is not None:
                    positions.grad[fixed_mask] = 0.0
            optimizer.step()
            last_loss = float(loss.item())

            if step % self.log_every == 0 or step == num_steps - 1:
                # Re-run with uncompiled closure to get parts for logging
                with torch.no_grad():
                    _, parts = fast_loss_with_penalty(
                        proxy, positions, overlap_lambda,
                        include_congestion=self.include_congestion,
                        boundary_lambda=self.boundary_lambda,
                    )
                last_parts_log = parts
                self._log(
                    f"  step {step:4d}  loss={last_loss:.5f}  "
                    f"wl={parts['wl'].item():.4f}  "
                    f"d={parts['density'].item():.4f}  "
                    f"c={parts['cong'].item():.4f}  "
                    f"ovl_area={parts['overlap_area_raw'].item():.0f}  "
                    f"g={gamma_frac:.4f}  l_ovl={overlap_lambda:.1f}"
                )
                history.append({
                    "step": step,
                    "loss": last_loss,
                    "wl": parts["wl"].item(),
                    "density": parts["density"].item(),
                    "cong": parts["cong"].item(),
                    "overlap_area": parts["overlap_area_raw"].item(),
                    "gamma_frac": gamma_frac,
                    "overlap_lambda": overlap_lambda,
                })

        positions_final = positions.detach().cpu()
        stats = {
            "history": history,
            "final_loss": last_loss,
            "final_smooth": last_parts_log["smooth_cost"].item() if last_parts_log else None,
            "final_overlap_area": last_parts_log["overlap_area_raw"].item() if last_parts_log else None,
            "descend_wall_s": time.time() - t0,
        }
        return positions_final, stats
