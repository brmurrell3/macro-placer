"""E114 — DiffProxyV3eDensity: V3 + electrostatic density.

A direct subclass of DiffProxyV3 (E111) that swaps the grid-bin density
for the electrostatic eDensity computed via 2D Poisson / DCT.

Hypothesis: the smooth electric potential gives Adam a globally-smooth
density gradient (no step discontinuities at bin boundaries), which
should help escape edge-aligned plateaus on hard benches (ibm17).

Everything else (LSE-HPWL, PerNetTraceCongestion, overlap penalty,
boundary penalty, init, optimizer, legalize) is inherited unchanged
from DiffProxyV3 / SmoothGlobalPlacerV3.
"""
from __future__ import annotations

import sys
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
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from smooth_global_placer_v3 import DiffProxyV3, SmoothGlobalPlacerV3
from edensity import ElectrostaticDensity


class DiffProxyV3eDensity(DiffProxyV3):
    """DiffProxyV3 with electrostatic eDensity replacing grid-bin density."""

    def __init__(
        self,
        benchmark,
        plc,
        device: str = "cpu",
        gamma_frac: float = 5e-4,
        trace_kwargs: Optional[Dict] = None,
        edensity_kwargs: Optional[Dict] = None,
    ):
        super().__init__(
            benchmark, plc, device=device, gamma_frac=gamma_frac,
            trace_kwargs=trace_kwargs,
        )
        edensity_kwargs = edensity_kwargs or {}
        self.edensity = ElectrostaticDensity(
            benchmark, device=device, **edensity_kwargs
        )

    def cost(self, positions: torch.Tensor, *, include_congestion: bool = True):
        """Smooth proxy. WL + 0.5*edensity + 0.5*trace_congestion."""
        if positions.device != self.device:
            positions = positions.to(self.device)
        clamped = self._clamp_to_canvas(positions)
        # Import _lse_hpwl from the base (DPO) primitives
        from diff_proxy import _lse_hpwl
        wl = _lse_hpwl(clamped, self.net_data, self.port_base, self.gamma) / self.wl_norm
        # eDensity (electrostatic) instead of grid-bin density
        density = self.edensity.compute_density(clamped)
        if include_congestion:
            cong = self.trace.compute_congestion(clamped)
        else:
            cong = torch.tensor(0.0, device=self.device)
        return wl + 0.5 * density + 0.5 * cong, {
            "wl": wl.detach(),
            "density": density.detach(),
            "cong": cong.detach(),
        }


def loss_with_penalty_v3_edensity(
    proxy: DiffProxyV3eDensity,
    positions: torch.Tensor,
    overlap_lambda: float,
    *,
    include_congestion: bool = True,
    boundary_lambda: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """Mirror loss_with_penalty_v3 but for the eDensity proxy."""
    smooth_cost, parts = proxy.cost(positions, include_congestion=include_congestion)
    penalty = proxy.overlap_penalty(positions)
    boundary = proxy.out_of_canvas_penalty(positions)
    canvas_area = proxy.cw * proxy.ch
    penalty_norm = penalty / canvas_area
    boundary_norm = boundary / canvas_area
    total = smooth_cost + overlap_lambda * penalty_norm + boundary_lambda * boundary_norm
    parts["overlap_area_raw"] = penalty.detach()
    parts["overlap_area_norm"] = penalty_norm.detach()
    parts["boundary_raw"] = boundary.detach()
    parts["boundary_norm"] = boundary_norm.detach()
    parts["smooth_cost"] = smooth_cost.detach()
    return total, parts


class SmoothGlobalPlacerV3eDensity(SmoothGlobalPlacerV3):
    """SmoothGlobalPlacerV3 with the proxy swapped for eDensity variant."""

    def __init__(self, edensity_kwargs: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.edensity_kwargs = edensity_kwargs

    def descend(self, benchmark, plc):
        """Override descend to instantiate DiffProxyV3eDensity instead."""
        import time

        from macro_place.cd_core import project_overlaps, sdf_init  # noqa: F401

        t0 = time.time()
        init_pos = self._init_positions(benchmark, plc=plc)
        cw = float(benchmark.canvas_width)
        lr = self.lr_frac * cw

        if self.adaptive_num_steps:
            n_macros = int(benchmark.num_hard_macros)
            adaptive_steps = max(200, int(n_macros * self.steps_per_macro))
            self._log(
                f"  adaptive_num_steps: {self.num_steps} → {adaptive_steps} "
                f"(n_hard={n_macros}, spm={self.steps_per_macro})"
            )
            num_steps = adaptive_steps
        else:
            num_steps = self.num_steps

        if self.device == "mps" and not torch.backends.mps.is_available():
            self._log("  MPS not available, falling back to CPU")
            self.device = "cpu"
        device = torch.device(self.device)
        proxy = DiffProxyV3eDensity(
            benchmark, plc, device=str(device),
            gamma_frac=self.gamma_start_frac,
            trace_kwargs=self.trace_kwargs,
            edensity_kwargs=self.edensity_kwargs,
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
            loss, parts = loss_with_penalty_v3_edensity(
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
                    f"γ={gamma_frac:.4f}  λ_ovl={overlap_lambda:.1f}"
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
        stats = {
            "history": history,
            "final_loss": last_loss,
            "final_smooth": last_parts["smooth_cost"].item() if last_parts else None,
            "final_overlap_area": last_parts["overlap_area_raw"].item() if last_parts else None,
            "descend_wall_s": time.time() - t0,
        }
        return positions_final, stats
