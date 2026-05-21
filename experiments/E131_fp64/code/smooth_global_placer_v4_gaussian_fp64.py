"""E131 — fp64-aware copy of SmoothGlobalPlacerV4Gaussian (from E127).

Key change: positions are cast to torch.float64 before Adam descent, and
the proxy is built with dtype=torch.float64. Adam state (m, v) inherits
the parameter dtype, so all accumulation runs in fp64. After descent, the
positions are cast back to torch.float32 for downstream legalize / CD
polish (which expect float32; canonical evaluator path is C++).

Inherits all other behavior (init, legalize, place flow) from V4 via
delegation — we don't subclass V4 since the proxy construction differs.
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
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, sdf_init
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from macro_legalizer import greedy_macro_legalize  # noqa: E402

# fp64-aware proxy / loss
from fast_proxy_fp64 import fast_loss_with_penalty  # noqa: E402
from v4_gaussian_proxy_fp64 import FastDiffProxyGaussianFp64  # noqa: E402


class SmoothGlobalPlacerV4GaussianFp64:
    """V4 + Gaussian density with fp64 Adam descent.

    Same constructor as SmoothGlobalPlacerV4Gaussian (E127), plus an
    optional `dtype` (default torch.float64). After descent, the placement
    is cast back to float32 for legalize / canonical evaluation.
    """

    def __init__(
        self,
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-4,
        overlap_lambda_start: float = 0.0,
        overlap_lambda_end: float = 50.0,
        overlap_ramp_pct: float = 0.7,
        boundary_lambda: float = 50.0,
        include_congestion: bool = True,
        init: str = "sdf",
        device: str = "cuda",
        rng_seed: int = 42,
        verbose: bool = True,
        log_every: int = 50,
        legalize_radius_steps: int = 80,
        legalize_step_frac: float = 0.02,
        adaptive_num_steps: bool = False,
        steps_per_macro: float = 2.0,
        trace_kwargs: Optional[Dict] = None,
        sigma_scale: float = 1.0,
        topk_frac: float = 0.10,
        dtype: torch.dtype = torch.float64,
    ):
        self.num_steps = num_steps
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_start = overlap_lambda_start
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.boundary_lambda = boundary_lambda
        self.include_congestion = include_congestion
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
        self.sigma_scale = sigma_scale
        self.topk_frac = topk_frac
        self.dtype = dtype

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
        elif self.init == "random":
            g = torch.Generator().manual_seed(self.rng_seed)
            cw = float(benchmark.canvas_width)
            ch = float(benchmark.canvas_height)
            sizes = benchmark.macro_sizes
            half_w = sizes[:, 0] / 2.0
            half_h = sizes[:, 1] / 2.0
            x = torch.rand(benchmark.num_macros, generator=g) * (cw - 2 * half_w) + half_w
            y = torch.rand(benchmark.num_macros, generator=g) * (ch - 2 * half_h) + half_h
            pos = torch.stack([x, y], dim=1)
            mask = benchmark.macro_fixed.bool()
            pos[mask] = benchmark.macro_positions[mask]
            return pos
        elif self.init == "center":
            cw = float(benchmark.canvas_width)
            ch = float(benchmark.canvas_height)
            pos = torch.zeros(benchmark.num_macros, 2)
            pos[:, 0] = cw / 2.0
            pos[:, 1] = ch / 2.0
            mask = benchmark.macro_fixed.bool()
            pos[mask] = benchmark.macro_positions[mask]
            return pos
        else:
            raise ValueError(f"Unknown init: {self.init!r}")

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
        # MPS does not currently support fp64. Auto-fallback to CPU for fp64 work.
        if self.device == "mps" and self.dtype == torch.float64:
            self._log("  MPS does not support fp64; falling back to CPU for descent")
            self.device = "cpu"
        device = torch.device(self.device)

        self._log(
            f"  fp64-descent: device={device} dtype={self.dtype} "
            f"(positions cast to {self.dtype} for Adam)"
        )

        proxy = FastDiffProxyGaussianFp64(
            benchmark, plc, device=str(device),
            gamma_frac=self.gamma_start_frac,
            trace_kwargs=self.trace_kwargs,
            sigma_scale=self.sigma_scale,
            topk_frac=self.topk_frac,
            dtype=self.dtype,
        )

        # Cast init positions to the descent dtype. requires_grad on fp64 tensor
        # makes the Adam state (exp_avg, exp_avg_sq) also fp64.
        positions = init_pos.clone().detach().to(device=device, dtype=self.dtype).requires_grad_(True)
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

        # Cast back to float32 for the downstream legalize / CD polish path.
        positions_final = positions.detach().to(dtype=torch.float32).cpu()
        return positions_final, {
            "history": history,
            "final_loss": last_loss,
            "final_smooth": last_parts["smooth_cost"].item() if last_parts else None,
            "final_overlap_area": last_parts["overlap_area_raw"].item() if last_parts else None,
            "descend_wall_s": time.time() - t0,
            "final_parts": {k: v.item() if hasattr(v, "item") else v
                           for k, v in (last_parts or {}).items()},
        }

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
                f"  greedy_legalize left {ovl} overlaps "
                f"(moved={leg_stats.get('n_moved')} failed={leg_stats.get('n_failed')}); "
                f"running project_overlaps"
            )
            legal_pos, _ = project_overlaps(legal_pos, benchmark)
            ovl = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        leg_stats["final_overlaps"] = ovl
        leg_stats["legalize_wall_s"] = time.time() - t0
        return legal_pos, leg_stats

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        self._log(f"=== SmoothGlobalPlacerV4GaussianFp64 ({benchmark.name}) ===")
        self._log(
            f"  config: steps={self.num_steps} lr_frac={self.lr_frac} "
            f"gamma={self.gamma_start_frac}->{self.gamma_end_frac} "
            f"l_ovl={self.overlap_lambda_start}->{self.overlap_lambda_end}"
            f" (ramp {self.overlap_ramp_pct*100:.0f}%) init={self.init} "
            f"device={self.device} dtype={self.dtype}"
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        positions, descend_stats = self.descend(benchmark, plc)
        self._log(
            f"  descent done: final_smooth={descend_stats['final_smooth']:.5f} "
            f"ovl_area={descend_stats['final_overlap_area']:.0f} "
            f"wall={descend_stats['descend_wall_s']:.1f}s"
        )

        legal_pos, leg_stats = self.legalize(positions, benchmark)
        self._log(
            f"  legalize: n_moved={leg_stats.get('n_moved')} "
            f"n_failed={leg_stats.get('n_failed')} "
            f"max_disp={leg_stats.get('max_displacement', -1):.1f} "
            f"final_overlaps={leg_stats['final_overlaps']} "
            f"wall={leg_stats['legalize_wall_s']:.1f}s"
        )

        proxy_cost = float(compute_proxy_cost(legal_pos, benchmark, plc)["proxy_cost"])
        self._log(
            f"  Final: proxy={proxy_cost:.5f} ovl={leg_stats['final_overlaps']} "
            f"total_wall={time.time()-t0:.1f}s"
        )

        return legal_pos
