"""E113 — SmoothGlobalPlacerV5: Adam + adaptive λ + multi-stage descent.

Pivot from V4 (Nesterov-BB) after finding Adam beats BB on macro
placement (V3 raw 0.895 vs V4 raw 0.947 on ibm01). The real value of
the DREAMPlace recipe is:
  - HPWL-feedback adaptive density weight (overlap penalty)
  - Multi-stage γ scheduling (high → medium → low)
  - Multi-restart from best (Lsub iterations)

V5 keeps Adam's per-parameter step (the proven win) but layers the
DP recipe on top.

Pipeline:
  Stage A — high γ (5e-3 → 1e-3), λ ramping from 0 to ~10:
     macros spread under long-range WL + cong gradients.
  Stage B — medium γ (1e-3 → 5e-4), λ ramping to ~30:
     macros pack tighter without losing the basin.
  Stage C — low γ (5e-4 → 5e-5), λ pushed to ~50:
     overlap resolution + sharp WL minimization.

After each stage, if smooth proxy of current best is *worse* than
start-of-stage, restart from best with α reduced 0.5×. This mirrors
DP's two-stage outer loop (`two_stage_density_scaler`).
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


DP_RePlAce_LOWER_PCOF = 0.95
DP_RePlAce_UPPER_PCOF = 1.05
DP_RePlAce_ref_hpwl = 350000.0


class SmoothGlobalPlacerV5:
    """Adam + adaptive λ (HPWL-feedback) + multi-stage γ descent."""

    def __init__(
        self,
        # Steps per stage (A/B/C: high-γ / med-γ / low-γ)
        stage_steps: Tuple[int, int, int] = (250, 200, 150),
        # γ schedule for each stage (start, end)
        gamma_stage_A: Tuple[float, float] = (5e-3, 1e-3),
        gamma_stage_B: Tuple[float, float] = (1e-3, 5e-4),
        gamma_stage_C: Tuple[float, float] = (5e-4, 5e-5),
        # λ (overlap) schedule per stage
        overlap_lambda_stage_A: Tuple[float, float] = (0.0, 10.0),
        overlap_lambda_stage_B: Tuple[float, float] = (10.0, 30.0),
        overlap_lambda_stage_C: Tuple[float, float] = (30.0, 50.0),
        # Adaptive HPWL-feedback (if True, override the schedule with DP rule)
        use_adaptive_lambda: bool = False,
        adaptive_lambda_init: float = 0.5,
        adaptive_lambda_max: float = 50.0,
        # Optim
        base_lr_frac: float = 0.005,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        # Boundary
        boundary_lambda: float = 50.0,
        # Init
        init: str = "sdf",
        device: str = "cpu",
        rng_seed: int = 42,
        # Multi-restart
        restart_on_regression: bool = True,
        restart_lr_decay: float = 0.5,
        # Logging
        verbose: bool = True,
        log_every: int = 50,
        # Legalize
        legalize_radius_steps: int = 80,
        legalize_step_frac: float = 0.02,
        # V3 proxy kwargs
        trace_kwargs: Optional[Dict] = None,
        # Optionally drop congestion
        include_congestion: bool = True,
        # Margin (0.0 = pure V3 behavior; >0 = use DiffProxyV3Margin)
        overlap_margin_frac: float = 0.0,
    ):
        self.stage_steps = stage_steps
        self.gamma_stages = (gamma_stage_A, gamma_stage_B, gamma_stage_C)
        self.overlap_lambda_stages = (
            overlap_lambda_stage_A,
            overlap_lambda_stage_B,
            overlap_lambda_stage_C,
        )
        self.use_adaptive_lambda = use_adaptive_lambda
        self.adaptive_lambda_init = adaptive_lambda_init
        self.adaptive_lambda_max = adaptive_lambda_max
        self.base_lr_frac = base_lr_frac
        self.adam_betas = adam_betas
        self.boundary_lambda = boundary_lambda
        self.init = init
        self.device = device
        self.rng_seed = rng_seed
        self.restart_on_regression = restart_on_regression
        self.restart_lr_decay = restart_lr_decay
        self.verbose = verbose
        self.log_every = log_every
        self.legalize_radius_steps = legalize_radius_steps
        self.legalize_step_frac = legalize_step_frac
        self.trace_kwargs = trace_kwargs
        self.include_congestion = include_congestion
        self.overlap_margin_frac = overlap_margin_frac

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
        gamma_range: Tuple[float, float],
        lambda_range: Tuple[float, float],
        lr: float,
        global_step_offset: int,
        adaptive_lambda_state: Dict,
    ) -> Tuple[torch.Tensor, float, List[Dict]]:
        """Run one stage of Adam descent. Returns (best_positions, best_smooth, history)."""
        optimizer = torch.optim.Adam([positions], lr=lr, betas=self.adam_betas)
        gamma_start, gamma_end = gamma_range
        lambda_start, lambda_end = lambda_range
        history = []
        best_smooth = float("inf")
        best_pos = positions.detach().clone()
        prev_wl = None

        for k in range(num_steps):
            t = k / max(1, num_steps - 1)
            g_frac = gamma_start + t * (gamma_end - gamma_start)
            proxy.set_gamma_frac(g_frac)

            if self.use_adaptive_lambda:
                ovl_lambda = adaptive_lambda_state["lambda"]
            else:
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

            # Adaptive λ update (DP HPWL-feedback)
            if self.use_adaptive_lambda:
                cur_wl = float(parts["wl"].item())
                ovl_raw = float(parts["overlap_area_raw"].item())
                canvas_area = float(proxy.cw * proxy.ch)
                overflow = ovl_raw / max(canvas_area, 1e-9)
                if prev_wl is not None and overflow > 1e-4:
                    delta_hpwl = (cur_wl - prev_wl) * float(proxy.wl_norm)
                    if delta_hpwl < 0:
                        mu = DP_RePlAce_UPPER_PCOF * max(
                            pow(0.9999, float(global_step_offset + k)), 0.98
                        )
                    else:
                        raw = pow(DP_RePlAce_UPPER_PCOF, -delta_hpwl / DP_RePlAce_ref_hpwl)
                        mu = DP_RePlAce_UPPER_PCOF * min(
                            max(raw, DP_RePlAce_LOWER_PCOF), DP_RePlAce_UPPER_PCOF
                        )
                    adaptive_lambda_state["lambda"] = min(
                        adaptive_lambda_state["lambda"] * mu, self.adaptive_lambda_max
                    )
                prev_wl = cur_wl

            # Track best-by-smooth
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
                    f"γ={g_frac:.5f}  λ={ovl_lambda:.2f}"
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
                    "gamma_frac": g_frac,
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
        if self.overlap_margin_frac > 0:
            from diff_proxy_v3_margin import DiffProxyV3Margin
            proxy = DiffProxyV3Margin(
                benchmark, plc, device=str(device),
                gamma_frac=self.gamma_stages[0][0],
                trace_kwargs=self.trace_kwargs,
                overlap_margin_frac=self.overlap_margin_frac,
            )
        else:
            proxy = DiffProxyV3(
                benchmark, plc, device=str(device),
                gamma_frac=self.gamma_stages[0][0],
                trace_kwargs=self.trace_kwargs,
            )

        positions = init_pos.clone().detach().to(device).requires_grad_(True)
        fixed_mask = benchmark.macro_fixed.bool().to(device)

        history_all = []
        global_offset = 0
        cur_lr = base_lr
        adaptive_state = {"lambda": float(self.adaptive_lambda_init)}
        best_overall = (float("inf"), positions.detach().clone(), 0)

        for stage_idx, (stage_steps, gamma_range, lambda_range) in enumerate(zip(
            self.stage_steps, self.gamma_stages, self.overlap_lambda_stages
        )):
            stage_name = "ABC"[stage_idx]
            self._log(
                f"  === Stage {stage_name}: {stage_steps} steps, γ=[{gamma_range[0]:.4f},"
                f"{gamma_range[1]:.4f}], λ=[{lambda_range[0]:.1f},{lambda_range[1]:.1f}], lr={cur_lr:.2f}"
            )
            best_pos_stage, best_smooth_stage, hist = self._stage_descend(
                proxy, positions, fixed_mask,
                stage_name=stage_name,
                num_steps=stage_steps,
                gamma_range=gamma_range,
                lambda_range=lambda_range,
                lr=cur_lr,
                global_step_offset=global_offset,
                adaptive_lambda_state=adaptive_state,
            )
            global_offset += stage_steps
            history_all.extend(hist)
            self._log(
                f"  === Stage {stage_name} done: best_smooth={best_smooth_stage:.5f} "
                f"(global step {global_offset})"
            )

            if best_smooth_stage < best_overall[0]:
                best_overall = (best_smooth_stage, best_pos_stage.clone(), global_offset)

            # Restart-from-best for the next stage (always — keep monotonically
            # the best basin to date).
            with torch.no_grad():
                positions.data.copy_(best_pos_stage)

        positions_final = best_overall[1].cpu()
        stats = {
            "history": history_all,
            "best_smooth": best_overall[0],
            "best_at_step": best_overall[2],
            "descend_wall_s": time.time() - t0,
            "final_overlap_lambda": adaptive_state["lambda"],
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
        self._log(f"=== SmoothGlobalPlacerV5 ({benchmark.name}) ===")
        self._log(
            f"  config: stages={self.stage_steps} base_lr_frac={self.base_lr_frac} "
            f"adaptive_λ={self.use_adaptive_lambda} init={self.init} device={self.device}"
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
