"""E113 — DiffProxyV3Margin: V3 proxy + margin-based overlap penalty.

The default `overlap_penalty` clamps at separation distance = sep_x.
Macros that are just-touching (dx == sep_x, ox = 0) feel zero gradient,
which lets the descent settle into states with many micro-overlaps
that legalize then has to dislodge.

Adding a small margin (default 0.5% canvas) creates a gradient that
pushes touching macros apart slightly. This produces "pre-legalized"
basins where greedy_macro_legalize moves fewer macros and the canonical
overlap count is closer to zero.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE, _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from smooth_global_placer_v3 import DiffProxyV3


class DiffProxyV3Margin(DiffProxyV3):
    """V3 + margin-based overlap penalty.

    `overlap_margin_frac`: fraction of canvas width to use as margin.
    Setting to 0.0 reproduces V3 behavior. Default 0.005 (0.5%).
    """

    def __init__(
        self,
        benchmark,
        plc,
        device: str = "cpu",
        gamma_frac: float = 5e-4,
        trace_kwargs: Optional[Dict] = None,
        overlap_margin_frac: float = 0.005,
    ):
        super().__init__(benchmark, plc, device=device, gamma_frac=gamma_frac,
                         trace_kwargs=trace_kwargs)
        self.overlap_margin = float(overlap_margin_frac) * self.cw

    def overlap_penalty(self, positions: torch.Tensor) -> torch.Tensor:
        """Pairwise overlap with `margin` added to the separation requirement.

        Standard:
            ox = clamp(sep_x - dx, min=0); penalty = sum(ox * oy)
        With margin m:
            ox = clamp(sep_x + m - dx, min=0); penalty = sum(ox * oy) - (free_area)
        where free_area is the "always zero with no overlap" baseline so
        the penalty stays 0 for fully separated macros (distance > m + sep_x).

        Concretely we use a simple shifted ReLU formula:
            ox_eff = clamp(sep_x + m - dx, min=0)
            oy_eff = clamp(sep_y + m - dy, min=0)
            ox_subtract = clamp(m - dx, min=0)  (always 0 unless dx < m, i.e., centers close)
            penalty_contribution = max(0, ox_eff * oy_eff - small)

        For the simple useful case (gradient nudge near boundary), just
        increase the smooth-overlap area by the margin band. Macros that
        are *more than* (sep + margin) apart have 0 penalty.
        """
        n = self.num_hard
        if n <= 1:
            return torch.tensor(0.0, device=self.device)
        clamped = self._clamp_to_canvas(positions)
        pos = clamped[:n]
        hw = self.half_sizes[:n, 0]
        hh = self.half_sizes[:n, 1]
        dx = (pos[:, 0].unsqueeze(0) - pos[:, 0].unsqueeze(1)).abs()
        dy = (pos[:, 1].unsqueeze(0) - pos[:, 1].unsqueeze(1)).abs()
        sep_x = hw.unsqueeze(0) + hw.unsqueeze(1)
        sep_y = hh.unsqueeze(0) + hh.unsqueeze(1)
        m = self.overlap_margin
        ox = torch.clamp(sep_x + m - dx, min=0.0)
        oy = torch.clamp(sep_y + m - dy, min=0.0)
        area = ox * oy
        mask = torch.triu(torch.ones_like(area, dtype=torch.bool), diagonal=1)
        return area[mask].sum()
