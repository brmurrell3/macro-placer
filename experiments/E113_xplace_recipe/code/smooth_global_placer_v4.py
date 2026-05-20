"""E113 — SmoothGlobalPlacerV4: V3 proxy + Xplace-recipe optimizer.

Drop-in replacement for SmoothGlobalPlacerV3. The only behavioral change
is the inner optimization loop: Adam → XplaceRecipeOptimizer (Nesterov-BB
with adaptive density-weight and overflow-based gamma annealing).

Pipeline (identical to V3 except step 2):
  1. SDF init + project_overlaps.
  2. XplaceRecipeOptimizer descent on DiffProxyV3 (per-net trace congestion).
  3. greedy_macro_legalize + project_overlaps fallback.
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
from xplace_recipe import XplaceRecipeOptimizer


class SmoothGlobalPlacerV4:
    """V3 proxy + Xplace-recipe descent (Nesterov-BB + adaptive λ, γ)."""

    def __init__(
        self,
        num_steps: int = 600,
        base_lr_frac: float = 0.005,
        # Density weight (overlap penalty)
        overlap_lambda_init: float = 0.5,
        overlap_lambda_max: float = 100.0,
        # Gamma
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        gamma_base_frac: float = 1e-3,
        use_overflow_gamma: bool = True,
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
        adaptive_num_steps: bool = False,
        steps_per_macro: float = 2.0,
        # V3 trace kwargs
        trace_kwargs: Optional[Dict] = None,
        # Nesterov toggle
        nesterov_use_bb: bool = True,
        # Optionally include congestion (default yes)
        include_congestion: bool = True,
    ):
        self.num_steps = num_steps
        self.base_lr_frac = base_lr_frac
        self.overlap_lambda_init = overlap_lambda_init
        self.overlap_lambda_max = overlap_lambda_max
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.gamma_base_frac = gamma_base_frac
        self.use_overflow_gamma = use_overflow_gamma
        self.boundary_lambda = boundary_lambda
        self.init = init
        self.device = device
        self.rng_seed = rng_seed
        self.verbose = verbose
        self.log_every = log_every
        self.legalize_radius_steps = legalize_radius_steps
        self.legalize_step_frac = legalize_step_frac
        self.adaptive_num_steps = adaptive_num_steps
        self.steps_per_macro = steps_per_macro
        self.trace_kwargs = trace_kwargs
        self.nesterov_use_bb = nesterov_use_bb
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

    def descend(
        self,
        benchmark: Benchmark,
        plc,
    ) -> Tuple[torch.Tensor, Dict]:
        t0 = time.time()
        init_pos = self._init_positions(benchmark, plc=plc)
        cw = float(benchmark.canvas_width)
        base_lr = self.base_lr_frac * cw

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
        proxy = DiffProxyV3(
            benchmark, plc, device=str(device),
            gamma_frac=self.gamma_start_frac,
            trace_kwargs=self.trace_kwargs,
        )

        positions = init_pos.clone().detach().to(device).requires_grad_(True)
        fixed_mask = benchmark.macro_fixed.bool().to(device)

        # Build the loss closure (parts dict matches what XplaceRecipeOptimizer expects).
        def loss_fn(pos: torch.Tensor, ovl_lambda: float):
            return loss_with_penalty_v3(
                proxy, pos, ovl_lambda,
                include_congestion=self.include_congestion,
                boundary_lambda=self.boundary_lambda,
            )

        opt = XplaceRecipeOptimizer(
            positions, proxy,
            fixed_mask=fixed_mask,
            loss_fn=loss_fn,
            base_lr=base_lr,
            num_steps=num_steps,
            overlap_lambda_init=self.overlap_lambda_init,
            overlap_lambda_max=self.overlap_lambda_max,
            gamma_start_frac=self.gamma_start_frac,
            gamma_end_frac=self.gamma_end_frac,
            gamma_base_frac=self.gamma_base_frac,
            use_overflow_gamma=self.use_overflow_gamma,
            boundary_lambda=self.boundary_lambda,
            nesterov_use_bb=self.nesterov_use_bb,
            log_every=self.log_every,
            verbose=self.verbose,
        )
        final_pos, stats = opt.descend()

        # Pick the better of u_k vs v_k vs current leaf (one is often
        # better-positioned for legalize; we score by smooth cost).
        with torch.no_grad():
            scores = {}
            for tag, p in (
                ("v_k", stats["final_v_k"]),
                ("u_k", stats["final_u_k"]),
            ):
                p_test = p.clone()
                p_test[fixed_mask] = positions.detach()[fixed_mask]
                # Use proxy's strict canvas clamp via its internal _clamp_to_canvas.
                p_test = proxy._clamp_to_canvas(p_test)
                lp, lparts = loss_fn(p_test, opt.overlap_lambda)
                scores[tag] = (float(lp.item()), p_test.cpu())
            best_tag = min(scores, key=lambda k: scores[k][0])
            self._log(
                f"  pick={best_tag}  scores: " +
                ", ".join(f"{k}={v[0]:.4f}" for k, v in scores.items())
            )
            positions_final = scores[best_tag][1].clone()

        stats["descend_wall_s"] = time.time() - t0
        stats["pick_tag"] = best_tag
        stats["pick_score"] = scores[best_tag][0]
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
                f"  greedy_legalize left {ovl} overlaps; running project_overlaps"
            )
            legal_pos, _ = project_overlaps(legal_pos, benchmark)
            ovl = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        leg_stats["final_overlaps"] = ovl
        leg_stats["legalize_wall_s"] = time.time() - t0
        return legal_pos, leg_stats

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        self._log(f"=== SmoothGlobalPlacerV4 ({benchmark.name}) ===")
        self._log(
            f"  config: steps={self.num_steps} base_lr_frac={self.base_lr_frac} "
            f"γ={self.gamma_start_frac}→{self.gamma_end_frac} "
            f"λ_ovl={self.overlap_lambda_init}→{self.overlap_lambda_max} "
            f"nesterov_bb={self.nesterov_use_bb} init={self.init} "
            f"device={self.device}"
        )
        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        positions, descend_stats = self.descend(benchmark, plc)
        self._log(
            f"  descent done: wall={descend_stats['descend_wall_s']:.1f}s "
            f"pick={descend_stats['pick_tag']} smooth_score={descend_stats['pick_score']:.4f}"
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
