"""E146 — V4 basin + Poisson-augmented Adam refinement + CD polish + cascade.

DCGP-style integration: after V4 produces a basin, run a short Adam
refinement loop that adds a Poisson-FFT congestion penalty to the loss.
Poisson congestion provides a globally-smooth pressure signal that
differs from the local ABU-5% used in canonical congestion. Adam
descending the combined loss may find lower-pressure neighbors.

Then CD polish + cascade saddle escape.

Isolated experiment — uses V4Placer + PerNetTraceCongestion read-only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E84_cascading_saddle" / "code",
    _ROOT / "experiments" / "E74_hessian_saddle" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E127_v4_gaussian" / "code",
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from per_net_trace_proxy import PerNetTraceCongestion
from cascading_saddle import cascading_saddle_escape
from poisson_congestion import poisson_congestion_scalar

# Load V4 production placer read-only
import importlib.util as _ilu
_V4 = _ROOT / "submissions" / "thinkorplace-v2" / "placer.py"
_v4_spec = _ilu.spec_from_file_location("_v4_placer_e146", str(_V4))
_v4_mod = _ilu.module_from_spec(_v4_spec)
_v4_spec.loader.exec_module(_v4_mod)
V4Placer = _v4_mod.Placer


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def poisson_refine(
    init_pos: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    num_steps: int = 100,
    lr_frac: float = 0.001,  # smaller than V4's lr — refinement only
    poisson_alpha: float = 0.5,
    overlap_lambda: float = 50.0,  # high to prevent overlap regression
    device: str = "cpu",
    verbose: bool = False,
) -> torch.Tensor:
    """Refine a placement with Adam + Poisson congestion penalty.

    Loss = original_canonical_proxy + α * poisson_congestion + λ * overlap_penalty

    Uses PerNetTraceCongestion's internals to compute V/H demand grids,
    then computes Poisson scalar from them. All differentiable.

    Returns refined positions.
    """
    log = (lambda s: print(s, flush=True)) if verbose else (lambda s: None)

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    lr = lr_frac * cw

    # Set up congestion proxy (gives us V/H demand grids)
    cong = PerNetTraceCongestion(benchmark, plc, device=device)

    # Macro sizes for overlap penalty
    half_w = (benchmark.macro_sizes[:, 0] / 2).to(device)
    half_h = (benchmark.macro_sizes[:, 1] / 2).to(device)
    n_hard = benchmark.num_hard_macros
    fixed_mask = benchmark.macro_fixed.bool().to(device)

    positions = init_pos.detach().clone().to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([positions], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()

        # Clamp to canvas (no grad)
        with torch.no_grad():
            positions.data[:, 0].clamp_(half_w, cw - half_w)
            positions.data[:, 1].clamp_(half_h, ch - half_h)

        # Compute V/H demand grids (differentiable)
        V_route, H_route = cong._trace_route_congestion(positions)
        V_route = V_route / max(cong.grid_v_routes, 1e-9)
        H_route = H_route / max(cong.grid_h_routes, 1e-9)
        V_macro, H_macro = cong._macro_route_congestion(positions)
        V_macro = V_macro / max(cong.grid_v_routes, 1e-9)
        H_macro = H_macro / max(cong.grid_h_routes, 1e-9)
        V_total = V_route + V_macro
        H_total = H_route + H_macro

        # Poisson scalar
        poisson = poisson_congestion_scalar(V_total, H_total, top_k_frac=0.05)

        # Pairwise overlap penalty (hard-only)
        # Compute overlap area for hard macro pairs
        pos_hard = positions[:n_hard]
        hw = half_w[:n_hard]
        hh = half_h[:n_hard]
        dx = pos_hard.unsqueeze(0)[..., 0] - pos_hard.unsqueeze(1)[..., 0]
        dy = pos_hard.unsqueeze(0)[..., 1] - pos_hard.unsqueeze(1)[..., 1]
        min_dx = hw.unsqueeze(0) + hw.unsqueeze(1)
        min_dy = hh.unsqueeze(0) + hh.unsqueeze(1)
        ov_x = torch.relu(min_dx - dx.abs())
        ov_y = torch.relu(min_dy - dy.abs())
        # Triangular mask to avoid double-counting and self-pairs
        mask = torch.triu(torch.ones(n_hard, n_hard, device=device), diagonal=1).bool()
        overlap_pen = (ov_x * ov_y * mask).sum()

        loss = poisson_alpha * poisson + overlap_lambda * overlap_pen
        loss.backward()
        # Zero grad on fixed macros
        if positions.grad is not None:
            positions.grad[fixed_mask] = 0.0
        optimizer.step()

        if verbose and step % 20 == 0:
            log(
                f"    refine step {step}: poisson={poisson.item():.5f} "
                f"overlap={overlap_pen.item():.3f}"
            )

    with torch.no_grad():
        positions.data[:, 0].clamp_(half_w, cw - half_w)
        positions.data[:, 1].clamp_(half_h, ch - half_h)

    return positions.detach().cpu()


class Placer:
    """V4 basin → Poisson-augmented Adam refinement → CD polish → cascade."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 1500.0,
        refine_steps: int = 100,
        refine_lr_frac: float = 0.001,
        poisson_alpha: float = 0.5,
        refine_overlap_lambda: float = 50.0,
        cascade_reserve_s: float = 300.0,
        cascade_max_iters: int = 2,
        cascade_eps_values: tuple = (0.5, 1.5),
        cascade_polish_budget: float = 100.0,
        cd_polish_s: float = 450.0,
        cd_plateau_threshold: float = 0.001,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.refine_steps = refine_steps
        self.refine_lr_frac = refine_lr_frac
        self.poisson_alpha = poisson_alpha
        self.refine_overlap_lambda = refine_overlap_lambda
        self.cascade_reserve_s = cascade_reserve_s
        self.cascade_max_iters = cascade_max_iters
        self.cascade_eps_values = cascade_eps_values
        self.cascade_polish_budget = cascade_polish_budget
        self.cd_polish_s = cd_polish_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        self._log(f"=== E146 v4_poisson_refine ({benchmark.name}) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Phase 1: V4 basin
        # Allocate ~half budget to V4, rest to refine/CD/cascade
        v4_budget = None
        if self.budget_seconds is not None:
            reserve = 60 + self.cascade_reserve_s + self.cd_polish_s
            v4_budget = max(60.0, self.budget_seconds - reserve - 60.0)
        v4 = V4Placer(budget_seconds=v4_budget, verbose=False, cd_polish_s=200.0)
        basin = v4.place(benchmark)
        basin_proxy = float(compute_proxy_cost(basin, benchmark, plc)["proxy_cost"])
        basin_ovl = compute_overlap_metrics(basin, benchmark)["overlap_count"]
        self._log(
            f"[E146] V4 basin: proxy={basin_proxy:.5f} ovl={basin_ovl} "
            f"wall={time.time()-t0:.0f}s"
        )

        if basin_ovl > 0:
            return basin

        # Phase 2: Poisson-augmented Adam refinement
        device = _best_device()
        self._log(
            f"[E146] Poisson refine: steps={self.refine_steps} α={self.poisson_alpha} device={device}"
        )
        try:
            refined = poisson_refine(
                basin, benchmark, plc,
                num_steps=self.refine_steps,
                lr_frac=self.refine_lr_frac,
                poisson_alpha=self.poisson_alpha,
                overlap_lambda=self.refine_overlap_lambda,
                device=device,
                verbose=self.verbose,
            )
            refined_ovl = compute_overlap_metrics(refined, benchmark)["overlap_count"]
            if refined_ovl > 0:
                refined, _ = project_overlaps(refined, benchmark)
                refined_ovl = compute_overlap_metrics(refined, benchmark)["overlap_count"]
            refined_proxy = float(compute_proxy_cost(refined, benchmark, plc)["proxy_cost"])
            self._log(
                f"[E146] refined: proxy={refined_proxy:.5f} ovl={refined_ovl} "
                f"(basin was {basin_proxy:.5f}, Δ={refined_proxy-basin_proxy:+.5f})"
            )
            if refined_ovl > 0:
                self._log(f"[E146] refine left overlaps; reverting to basin")
                pos_polish = basin
            else:
                pos_polish = refined
        except Exception as exc:
            self._log(f"[E146] Poisson refine EXCEPTION: {exc}; using basin")
            pos_polish = basin

        # Phase 3: CD polish
        cd_budget = self.cd_polish_s
        if self.budget_seconds is not None:
            remaining = (t0 + self.budget_seconds) - time.time() - self.cascade_reserve_s - 30
            cd_budget = max(30.0, min(remaining, self.cd_polish_s))
        self._log(f"[E146] CD polish budget={cd_budget:.0f}s")
        ev = IncrementalProxyEvaluator(benchmark, plc, pos_polish)
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        run_cd_adaptive(
            ev, benchmark, plc, movable,
            min_time_s=cd_budget * 0.5, hard_cap_s=cd_budget,
            patience=5, plateau_threshold=self.cd_plateau_threshold, log_fn=None,
        )
        polished = ev.placement.detach().clone().to(torch.float32)
        polished_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
        polished_ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
        self._log(f"[E146] CD polish: proxy={polished_proxy:.5f} ovl={polished_ovl}")
        if polished_ovl > 0:
            polished, _ = project_overlaps(polished, benchmark)
            polished_ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
            if polished_ovl > 0:
                raise RuntimeError(f"Polished has {polished_ovl} overlaps")
            polished_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])

        # Phase 4: cascade saddle
        cascade_budget = self.cascade_reserve_s
        if self.budget_seconds is not None:
            cascade_budget = max(60.0, (t0 + self.budget_seconds) - time.time() - 30)
        self._log(f"[E146] cascade budget={cascade_budget:.0f}s")
        try:
            saddled, _ = cascading_saddle_escape(
                polished.to(torch.float32), benchmark, plc,
                max_iters=self.cascade_max_iters,
                eps_values=self.cascade_eps_values,
                polish_budget=self.cascade_polish_budget,
                total_budget_s=cascade_budget,
                log=self._log if self.verbose else (lambda s: None),
            )
            sp = float(compute_proxy_cost(saddled, benchmark, plc)["proxy_cost"])
            so = compute_overlap_metrics(saddled, benchmark)["overlap_count"]
            if so == 0 and sp < polished_proxy - 1e-7:
                self._log(
                    f"[E146] cascade: {polished_proxy:.5f} -> {sp:.5f} "
                    f"({100*(polished_proxy-sp)/polished_proxy:+.2f}%)"
                )
                return saddled
            return polished
        except Exception as exc:
            self._log(f"[E146] cascade EXCEPTION: {exc}")
            return polished
