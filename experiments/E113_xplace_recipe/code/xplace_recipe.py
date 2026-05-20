"""E113 — Xplace-recipe optimizer on the challenge proxy.

The "patched DREAMPlace" stand-in: replace DREAMPlace's
HPWL + density_weight*eDensity loss with our exact challenge proxy
(HPWL + 0.5*density + 0.5*congestion) but keep DREAMPlace's
optimization recipe:

  1. Nesterov-BB (Algorithm 2 from e-place, with Barzilai-Borwein step)
  2. Adaptive density weight via HPWL feedback (DP's RePlAce mode)
  3. Adaptive gamma annealing via overflow (DP's overflow-based gamma)
  4. Multi-stage descent (high gamma → anneal → low gamma)

This file owns ONLY the optimizer state machine. The proxy primitives
(LSE-HPWL, grid density, per-net trace congestion, overlap penalty,
boundary penalty) come from E111 V3 (PerNetTraceCongestion +
DiffProxyV3).

Why Nesterov over Adam:
- Adam's per-parameter adaptive step shrinks aggressively on the
  long-tail directions; e-place argues macro placement benefits from
  *uniform* step size with line-search (BB) instead.
- Adam has no built-in concept of "improving objective", so density
  weight ramps purely on iteration count. HPWL-feedback in DP detects
  stagnation and ramps faster.

Why HPWL-feedback density weight:
- E110 used linear ramp (0 → 50 over 70% of steps). E110 sweep found
  ovl_lambda_end=10 was optimal on --fast (-4.4% vs SDF), but the
  optimum is bench-dependent. HPWL-feedback ramps the weight to
  whatever level the macros tolerate while still improving HPWL.

Why overflow-based gamma:
- Overflow = total smooth-overlap-area / canvas-area. Early in
  descent overflow is large (macros piled up) and gamma should be
  large (soft LSE = spread macros via long-range gradients). Late in
  descent overflow → 0 and gamma should shrink so smooth WL matches
  canonical bbox HPWL.
- DREAMPlace's exact formula:
    coef = 10^((overflow - 0.1) * 20/9 - 1)
    gamma = base_gamma * coef
  with base_gamma = params.gamma * (bin_x + bin_y). We mirror.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


# DREAMPlace HPWL-mode density-weight update constants (RePlAce/elfPlace defaults).
DP_RePlAce_LOWER_PCOF = 0.95
DP_RePlAce_UPPER_PCOF = 1.05
DP_RePlAce_ref_hpwl = 350000.0  # microns; only used as scale for delta_hpwl normalization


class XplaceRecipeOptimizer:
    """Nesterov-BB descent with adaptive density-weight and adaptive gamma.

    Operates on a (positions, proxy_v3, get_overflow_fn) triplet:
    - `positions`: leaf Tensor, requires_grad. Shape [N, 2].
    - `proxy_v3`: any DiffProxy-like object with .cost() and .set_gamma_frac().
      We assume it's E111 V3 (per-net trace).
    - The loss callable evaluated each step: total = cost + λ*ovl + b*boundary.

    Density-weight = the overlap penalty coefficient (λ_ovl). It ramps
    via DP's HPWL-feedback rule. Boundary penalty is held fixed (we don't
    want to ramp it — boundary violations are categorical, not soft).

    Gamma is annealed via overflow rule each step (rather than linear).
    """

    def __init__(
        self,
        positions: torch.Tensor,
        proxy_v3,
        *,
        fixed_mask: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, float], Tuple[torch.Tensor, dict]],
        base_lr: float,
        num_steps: int = 1000,
        # Density weight
        overlap_lambda_init: float = 0.5,
        overlap_lambda_max: float = 100.0,
        density_weight_upper_pcof: float = DP_RePlAce_UPPER_PCOF,
        density_weight_lower_pcof: float = DP_RePlAce_LOWER_PCOF,
        ref_hpwl: float = DP_RePlAce_ref_hpwl,
        # Gamma annealing
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        gamma_base_frac: float = 1e-3,    # DP's base_gamma = params.gamma * bin_size
        use_overflow_gamma: bool = True,  # if False, use linear anneal
        # Boundary
        boundary_lambda: float = 50.0,
        # Nesterov BB
        nesterov_use_bb: bool = True,
        nesterov_max_backtrack: int = 4,
        # Misc
        log_every: int = 50,
        verbose: bool = True,
    ):
        self.positions = positions   # leaf tensor (the V_k in DP parlance)
        self.proxy = proxy_v3
        self.fixed_mask = fixed_mask
        self.loss_fn = loss_fn
        self.base_lr = float(base_lr)
        self.num_steps = int(num_steps)
        self.overlap_lambda = float(overlap_lambda_init)
        self.overlap_lambda_max = float(overlap_lambda_max)
        self.UPPER_PCOF = float(density_weight_upper_pcof)
        self.LOWER_PCOF = float(density_weight_lower_pcof)
        self.ref_hpwl = float(ref_hpwl)
        self.gamma_start_frac = float(gamma_start_frac)
        self.gamma_end_frac = float(gamma_end_frac)
        self.gamma_base_frac = float(gamma_base_frac)
        self.use_overflow_gamma = bool(use_overflow_gamma)
        self.boundary_lambda = float(boundary_lambda)
        self.nesterov_use_bb = bool(nesterov_use_bb)
        self.nesterov_max_backtrack = int(nesterov_max_backtrack)
        self.log_every = int(log_every)
        self.verbose = bool(verbose)

        # Nesterov state
        self.device = positions.device
        self.dtype = positions.dtype
        self.u_k = positions.detach().clone()                 # major solution
        self.v_k = positions                                  # reference (alias to leaf)
        self.v_k_1 = positions.detach().clone() - self.base_lr * self._initial_grad()
        self.a_k = torch.ones(1, device=self.device, dtype=self.dtype)
        # alpha will be set on first step.
        self.alpha_k = None
        self.prev_hpwl = None
        self.history = []

        # Cache the canvas area for the smooth-overlap "overflow" computation.
        # overflow = sum(pairwise overlap area) / canvas_area, clamped to [0, ~1].
        self.canvas_area = float(self.proxy.cw * self.proxy.ch)

    # ------------------------------------------------------------------- grad
    def _initial_grad(self) -> torch.Tensor:
        """Compute initial gradient (no Nesterov step), used to seed v_{k-1}."""
        self.proxy.set_gamma_frac(self.gamma_start_frac)
        # Compute grad at the current v_k position; copy out the gradient.
        if self.positions.grad is not None:
            self.positions.grad.zero_()
        else:
            self.positions.grad = torch.zeros_like(self.positions)
        loss, parts = self.loss_fn(self.positions, self.overlap_lambda)
        loss.backward()
        g = self.positions.grad.detach().clone()
        # zero fixed-macro gradients
        g[self.fixed_mask] = 0.0
        self.positions.grad.zero_()
        self.prev_hpwl = parts.get("wl", torch.tensor(0.0)).item()
        return g

    def _obj_and_grad(self, pos_eval: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Evaluate (loss, grad, parts) at a *non-leaf* tensor.

        Mirrors DP's `obj_and_grad_fn`. We can't reuse the leaf tensor here
        because Nesterov evaluates at intermediate v_{k+1} positions.
        """
        # Make a fresh requires_grad leaf so we can backprop cleanly.
        p = pos_eval.detach().clone().requires_grad_(True)
        loss, parts = self.loss_fn(p, self.overlap_lambda)
        loss.backward()
        g = p.grad.detach()
        g[self.fixed_mask] = 0.0
        return loss.detach(), g, {k: v.detach() if torch.is_tensor(v) else v for k, v in parts.items()}

    # ----------------------------------------------------------- gamma update
    def _current_overflow(self, parts: dict) -> float:
        """Smooth-overlap-area fraction of canvas. Used as overflow proxy.

        DP overflow is "fraction of bins that exceed target density"; we
        substitute "overlap area / canvas area" which is a stricter
        proxy (canonical zero-overlap target ↔ DP zero-overflow target).
        """
        ovl_raw = float(parts.get("overlap_area_raw", torch.tensor(0.0)).item())
        return max(0.0, ovl_raw / max(self.canvas_area, 1e-9))

    def _update_gamma(self, step: int, parts: dict) -> float:
        """Set proxy gamma. Three modes:

        - `use_overflow_gamma=True`: hybrid: linear anneal as the floor,
          overflow-boost lifts gamma when overflow is high. Specifically:
            g_floor = linear_anneal(gamma_start → gamma_end, t = step/N)
            g_boost = gamma_base * 10^((overflow - 0.1)*20/9 - 1)
            g_frac  = max(g_floor, g_boost), clipped to [gamma_end, gamma_start]
          This way SDF init (overflow=0) still gets the high-gamma early steps
          (long-range WL gradient for global cell spreading), while a basin
          that re-develops overlap (e.g., after density-weight relaxation) gets
          the gamma boost on demand.
        - `use_overflow_gamma=False`: pure linear anneal (no overflow boost).
        """
        if self.use_overflow_gamma:
            overflow = self._current_overflow(parts)
            t = step / max(1, self.num_steps - 1)
            g_floor = self.gamma_start_frac + t * (
                self.gamma_end_frac - self.gamma_start_frac
            )
            coef = pow(10.0, (overflow - 0.1) * 20.0 / 9.0 - 1.0)
            g_boost = self.gamma_base_frac * coef
            g_frac = max(g_floor, g_boost)
            g_frac = max(self.gamma_end_frac, min(self.gamma_start_frac, g_frac))
        else:
            t = step / max(1, self.num_steps - 1)
            g_frac = self.gamma_start_frac + t * (
                self.gamma_end_frac - self.gamma_start_frac
            )
        self.proxy.set_gamma_frac(g_frac)
        return g_frac

    # ---------------------------------------------------- density-weight upd.
    def _update_density_weight(self, parts: dict, step: int) -> None:
        """HPWL-feedback density-weight update (DP `update_density_weight_op_hpwl`).

        Delta = current_wl - prev_wl. If delta < 0 (HPWL improved):
          mu = UPPER_PCOF * max(0.9999^iter, 0.98)
        Otherwise:
          mu = UPPER_PCOF * UPPER_PCOF^(-delta/ref_hpwl) clamped to [LOWER, UPPER]
        density_weight *= mu, clipped to overlap_lambda_max.

        Modification for our challenge: scale mu by *overlap pressure*.
        DP's intent is to ramp the density weight just enough to push macros
        apart. For our problem with a legal SDF init (overlap=0 initially),
        we don't need rapid ramp — only ramp when overlap re-develops.

        If overlap is essentially 0, freeze λ (don't grow). This stops the
        runaway ramp to 100 that wastes 60% of the iterations.
        """
        cur_wl = float(parts.get("wl", torch.tensor(0.0)).item())
        if self.prev_hpwl is None:
            self.prev_hpwl = cur_wl
            return
        # If overlap is essentially 0, freeze λ (no need to ramp). Use a small
        # threshold (1% of canvas area) so we don't freeze for trivial overlap.
        overflow = self._current_overflow(parts)
        if overflow < 1e-4:  # < 0.01% of canvas area — basically zero overlap
            self.prev_hpwl = cur_wl
            return
        # Use WL relative to ref for delta normalization. Our wl is normalized
        # by (W+H)*net_cnt; multiply by that for absolute microns-ish scale.
        wl_norm_factor = float(self.proxy.wl_norm)
        delta_hpwl = (cur_wl - self.prev_hpwl) * wl_norm_factor
        self.prev_hpwl = cur_wl
        if delta_hpwl < 0:
            mu = self.UPPER_PCOF * max(pow(0.9999, float(step)), 0.98)
        else:
            raw = pow(self.UPPER_PCOF, -delta_hpwl / self.ref_hpwl)
            mu = self.UPPER_PCOF * min(max(raw, self.LOWER_PCOF), self.UPPER_PCOF)
        self.overlap_lambda = min(self.overlap_lambda * mu, self.overlap_lambda_max)

    # ----------------------------------------------------- Nesterov-BB step
    def step_bb(self, step: int) -> dict:
        """One BB-Nesterov step. Returns parts dict for logging."""
        # 1. Evaluate (loss, grad) at the current v_k.
        loss_k, g_k, parts_k = self._obj_and_grad(self.v_k)
        # First-iteration setup of v_{k-1} gradient, alpha_k.
        if self.alpha_k is None:
            _loss_km1, g_km1, _ = self._obj_and_grad(self.v_k_1)
            s_k = (self.v_k.detach() - self.v_k_1.detach()).flatten()
            y_k = (g_k - g_km1).flatten()
            yny = float(y_k.dot(y_k).item())
            if yny < 1e-20:
                alpha = self.base_lr
            else:
                alpha = float(s_k.norm(p=2).item() / max(y_k.norm(p=2).item(), 1e-12))
            self.alpha_k = max(alpha, 1e-6)
            self.g_k_1_data = g_km1

        # 2. BB step size: s_k.y_k / y_k.y_k (short) or fallback.
        s_k = (self.v_k.detach() - self.v_k_1.detach()).flatten()
        y_k = (g_k - self.g_k_1_data).flatten()
        sk_yk = float(s_k.dot(y_k).item())
        yk_yk = float(y_k.dot(y_k).item())
        if abs(yk_yk) < 1e-20:
            step_size = self.alpha_k
        else:
            bb_short = sk_yk / yk_yk
            if bb_short > 0:
                step_size = bb_short
            else:
                # Lipschitz fallback: ||s|| / ||y||.
                lip_step = float(s_k.norm(p=2).item() / max(y_k.norm(p=2).item(), 1e-12))
                step_size = min(lip_step, self.alpha_k)
        step_size = max(1e-6, min(step_size, self.base_lr * 10.0))

        # 3. Nesterov coefficient
        a_kp1 = (1.0 + (4.0 * float(self.a_k.item()) ** 2 + 1.0) ** 0.5) / 2.0
        coef = (float(self.a_k.item()) - 1.0) / a_kp1

        # 4. u_{k+1} = v_k - step_size * g_k
        u_kp1 = self.v_k.detach() - step_size * g_k
        # 5. v_{k+1} = u_{k+1} + coef * (u_{k+1} - u_k)
        v_kp1 = u_kp1 + coef * (u_kp1 - self.u_k)
        # 6. Constraints: clamp to canvas (proxy already clamps inside cost(),
        #    but the leaf tensor itself may drift; reapply the strict clamp).
        v_kp1 = self._clamp_to_canvas(v_kp1)
        # Fixed macros: snap back.
        v_kp1[self.fixed_mask] = self.v_k.detach()[self.fixed_mask]

        # 7. Update state in-place on the leaf tensor v_k.
        self.v_k_1 = self.v_k.detach().clone()
        self.g_k_1_data = g_k.clone()
        self.u_k = u_kp1
        with torch.no_grad():
            self.v_k.data.copy_(v_kp1)
        self.a_k.data.fill_(a_kp1)
        self.alpha_k = step_size

        # 8. Update density weight + gamma (using parts from current step).
        self._update_density_weight(parts_k, step)
        gamma_frac = self._update_gamma(step, parts_k)

        parts_k["overlap_lambda"] = self.overlap_lambda
        parts_k["gamma_frac"] = gamma_frac
        parts_k["step_size"] = step_size
        parts_k["loss"] = float(loss_k.item())
        return parts_k

    def step_simple_momentum(self, step: int) -> dict:
        """Heavy-ball momentum step (no line search) — fallback when BB is unstable."""
        loss_k, g_k, parts_k = self._obj_and_grad(self.v_k)
        # Heavy ball: v_{k+1} = v_k - lr*g_k + momentum*(v_k - v_{k-1})
        momentum = 0.9
        lr = self.base_lr
        v_kp1 = self.v_k.detach() - lr * g_k + momentum * (self.v_k.detach() - self.v_k_1.detach())
        v_kp1 = self._clamp_to_canvas(v_kp1)
        v_kp1[self.fixed_mask] = self.v_k.detach()[self.fixed_mask]
        self.v_k_1 = self.v_k.detach().clone()
        with torch.no_grad():
            self.v_k.data.copy_(v_kp1)
        self._update_density_weight(parts_k, step)
        gamma_frac = self._update_gamma(step, parts_k)
        parts_k["overlap_lambda"] = self.overlap_lambda
        parts_k["gamma_frac"] = gamma_frac
        parts_k["loss"] = float(loss_k.item())
        return parts_k

    def _clamp_to_canvas(self, pos: torch.Tensor) -> torch.Tensor:
        """Strict per-axis clamp inside canvas (half-sizes from proxy)."""
        half_w = self.proxy.half_sizes[:, 0]
        half_h = self.proxy.half_sizes[:, 1]
        cw = self.proxy.cw
        ch = self.proxy.ch
        x = pos[:, 0].clamp(min=half_w, max=(cw - half_w))
        y = pos[:, 1].clamp(min=half_h, max=(ch - half_h))
        return torch.stack([x, y], dim=1)

    # -------------------------------------------------------- run main loop
    def descend(self) -> Tuple[torch.Tensor, dict]:
        """Run for `num_steps` Nesterov-BB iters. Returns (final_positions, stats)."""
        t0 = time.time()
        for step in range(self.num_steps):
            if self.nesterov_use_bb:
                parts = self.step_bb(step)
            else:
                parts = self.step_simple_momentum(step)

            if step % self.log_every == 0 or step == self.num_steps - 1:
                msg = (
                    f"  step {step:4d}  loss={parts.get('loss', 0.0):.5f}  "
                    f"wl={float(parts['wl'].item()):.4f}  "
                    f"d={float(parts['density'].item()):.4f}  "
                    f"c={float(parts['cong'].item()):.4f}  "
                    f"ovl_area={float(parts['overlap_area_raw'].item()):.0f}  "
                    f"γ={parts.get('gamma_frac', 0.0):.5f}  "
                    f"λ_ovl={parts.get('overlap_lambda', 0.0):.2f}  "
                    f"step_size={parts.get('step_size', 0.0):.3f}"
                )
                if self.verbose:
                    print(msg, flush=True)
                self.history.append({
                    "step": step,
                    "loss": parts.get("loss", 0.0),
                    "wl": float(parts["wl"].item()),
                    "density": float(parts["density"].item()),
                    "cong": float(parts["cong"].item()),
                    "overlap_area": float(parts["overlap_area_raw"].item()),
                    "gamma_frac": parts.get("gamma_frac", 0.0),
                    "overlap_lambda": parts.get("overlap_lambda", 0.0),
                    "step_size": parts.get("step_size", 0.0),
                })

        # Return u_k (the major solution) per e-place. Often v_k is the better
        # one in practice — try both and let caller pick.
        stats = {
            "history": self.history,
            "final_v_k": self.v_k.detach().clone(),
            "final_u_k": self.u_k.detach().clone(),
            "final_overlap_lambda": self.overlap_lambda,
            "descend_wall_s": time.time() - t0,
        }
        return self.v_k.detach().clone(), stats
