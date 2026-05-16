"""E95 — thin extension of E88's DiffProxy with mutable γ for annealing.

E88's primitives (LSE-HPWL, top-10% density, ABU-5% RUDY congestion) are
already matched to canonical at the formula level. The fixed γ=0.0005·canvas
is what biases smooth-vs-canonical by ~2% at the cascade optimum. Allowing γ
to anneal during descent lets the smooth gradient converge to the canonical
(sub-)gradient as descent settles.

This file is a 30-line wrapper. The heavy lifting still lives in
experiments/E88_diff_proxy/code/diff_proxy.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
_E88_DIR = _ROOT / "experiments" / "E88_diff_proxy" / "code"
for p in (_ROOT, _E88_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from diff_proxy import DiffProxy as _BaseDiffProxy
from diff_proxy import loss_with_penalty as _base_loss_with_penalty


class DiffProxyV2(_BaseDiffProxy):
    """DiffProxy with mutable γ (smoothing temperature) for annealing and
    a normalization fix.

    Inherits primitives unchanged; adds:
      - set_gamma_frac() so the descent driver can shrink γ over the trajectory
      - WL normalization fix: use plc.net_cnt (sum of weights) instead of
        len(plc.nets) to match canonical's get_cost()

    Canonical: total_hpwl_weighted / ((W+H) * plc.net_cnt)
    E88 had:   total_hpwl_weighted / ((W+H) * len(plc.nets))
    Ratio on ibm01: 7269 / 5993 = 1.213 → smooth WL over-estimates by 21%.
    """

    def __init__(self, benchmark, plc, device="cpu", gamma_frac=0.0005):
        super().__init__(benchmark, plc, device=device, gamma_frac=gamma_frac)
        # Override WL normalization with canonical's net_cnt (sum of weights).
        # plc.net_cnt is incremented per net by the driver's weight (1 if none),
        # see plc_client_os.py:282-360. Canonical's get_cost() normalizes by
        # this. Falls back to total_net_count if plc.net_cnt unavailable / 0.
        try:
            canonical_net_cnt = float(plc.net_cnt) if plc.net_cnt else self.net_data.total_net_count
        except AttributeError:
            canonical_net_cnt = self.net_data.total_net_count
        self.wl_norm = (self.cw + self.ch) * max(1.0, canonical_net_cnt)
        self._canonical_net_cnt = canonical_net_cnt

    def set_gamma_frac(self, gamma_frac: float) -> None:
        """Update LSE smoothing temperature in place.

        γ → 0 makes LSE-HPWL approach the exact bbox max; this reduces the
        smooth-vs-canonical bias at converged placements.
        """
        self.gamma = gamma_frac * self.cw


def loss_with_penalty(proxy, positions, overlap_lambda, *,
                       include_congestion=True, boundary_lambda=1.0):
    """Re-export E88's loss_with_penalty unchanged for convenience."""
    return _base_loss_with_penalty(
        proxy, positions, overlap_lambda,
        include_congestion=include_congestion,
        boundary_lambda=boundary_lambda,
    )
