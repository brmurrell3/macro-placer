"""Helper module for E111 margin placer — kept separate so the placer
loader doesn't pick the helper as the main placer.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch

_ROOT = Path(__file__).resolve().parents[2]
_E111_CODE = _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code"
_E113_CODE = _ROOT / "experiments" / "E113_xplace_recipe" / "code"
_E88_CODE = _ROOT / "experiments" / "E88_diff_proxy" / "code"
_E95_CODE = _ROOT / "experiments" / "E95_diff_proxy_v2" / "code"
_E76_CODE = _ROOT / "experiments" / "E76_dreamplace_integration" / "code"
for p in (_E111_CODE, _E113_CODE, _E88_CODE, _E95_CODE, _E76_CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from smooth_global_placer_v3 import SmoothGlobalPlacerV3
from diff_proxy_v3_margin import DiffProxyV3Margin
from diff_proxy import loss_with_penalty


class SmoothGlobalV3Margin(SmoothGlobalPlacerV3):
    """V3 placer that uses DiffProxyV3Margin for overlap penalty."""

    def __init__(self, overlap_margin_frac: float = 0.003, **kwargs):
        super().__init__(**kwargs)
        self.overlap_margin_frac = overlap_margin_frac

    def descend(self, benchmark, plc):
        t0 = time.time()
        init_pos = self._init_positions(benchmark, plc=plc)
        cw = float(benchmark.canvas_width)
        lr = self.lr_frac * cw

        device = torch.device(self.device)
        proxy = DiffProxyV3Margin(
            benchmark, plc, device=str(device),
            gamma_frac=self.gamma_start_frac,
            overlap_margin_frac=self.overlap_margin_frac,
        )

        positions = init_pos.clone().detach().to(device).requires_grad_(True)
        fixed_mask = benchmark.macro_fixed.bool().to(device)
        optimizer = torch.optim.Adam([positions], lr=lr)

        last_loss = None
        last_parts = None
        num_steps = self.num_steps
        for step in range(num_steps):
            t = step / max(1, num_steps - 1)
            gamma_frac = self.gamma_start_frac + t * (
                self.gamma_end_frac - self.gamma_start_frac
            )
            proxy.gamma = gamma_frac * proxy.cw
            ramp_t = min(1.0, step / max(1, num_steps * self.overlap_ramp_pct))
            overlap_lambda = self.overlap_lambda_start + ramp_t * (
                self.overlap_lambda_end - self.overlap_lambda_start
            )
            optimizer.zero_grad()
            loss, parts = loss_with_penalty(
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

        positions_final = positions.detach().cpu()
        return positions_final, {
            "final_loss": last_loss,
            "final_smooth": last_parts["smooth_cost"].item() if last_parts else None,
            "final_overlap_area": last_parts["overlap_area_raw"].item() if last_parts else None,
            "descend_wall_s": time.time() - t0,
        }
