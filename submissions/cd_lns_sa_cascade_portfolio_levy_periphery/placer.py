"""CDLNSSACascadePortfolioLevyPeripheryPlacer — portfolio (4-weight) Lévy + periphery.

Stack:
  1. E25 lane (SDF + CD + LNS + SA)
  2. E41 lane (DPO + CD + LNS + SA + K-joint)
  3. portfolio_saddle (4 weight vectors — canonical, cong-focus,
     density-focus, non-WL — each finding its softest eigvec and
     ε-perturbing with Lévy magnitudes; accepted by CANONICAL proxy)
  4. periphery push α=0.01 + CD polish (strictly conservative)

Expected lift components (cumulative):
  - portfolio_saddle on ibm10: -1.089% vs cascade plateau (E100 spike)
  - periphery wrapper on ariane133: -0.95% vs Lévy cascade (E107 spike)

Strictly conservative: each layer is reverted if it worsens canonical proxy
or introduces overlaps.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# Reuse the portfolio (4-weight) Lévy cascade placer.
_PORTFOLIO_PATH = _ROOT / "submissions" / "cd_lns_sa_cascade_portfolio_levy" / "placer.py"
_spec = importlib.util.spec_from_file_location("portfolio_levy_placer", str(_PORTFOLIO_PATH))
_portfolio = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_portfolio)
CDLNSSACascadePortfolioLevyPlacer = _portfolio.CDLNSSACascadePortfolioLevyPlacer


def periphery_push_and_polish(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    alpha: float = 0.01,
    cd_budget_s: float = 180.0,
    log=None,
):
    """Push each movable macro α fraction toward nearest canvas edge,
    re-legalize, and CD-polish. Returns (polished, proxy, overlap_count).

    Args:
      placement: [N, 2] tensor of macro centers
      benchmark: benchmark with canvas_width/height, macro_sizes, macro_fixed
      plc: TILOS placement client
      alpha: push fraction (0.0 = no change; 0.01 default = tiny push)
      cd_budget_s: CD polish hard cap
    """
    if log is None:
        log = lambda s: None
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    pos = placement.cpu().numpy().copy()
    half_w = (benchmark.macro_sizes[:, 0] / 2).cpu().numpy()
    half_h = (benchmark.macro_sizes[:, 1] / 2).cpu().numpy()
    fixed = benchmark.macro_fixed.cpu().numpy()

    for i in range(pos.shape[0]):
        if fixed[i]:
            continue
        x, y = pos[i, 0], pos[i, 1]
        dx_l = x - half_w[i]; dx_r = (cw - half_w[i]) - x
        dy_b = y - half_h[i]; dy_t = (ch - half_h[i]) - y
        dists = [dx_l, dx_r, dy_b, dy_t]
        nearest = int(np.argmin([abs(d) for d in dists]))
        if nearest == 0:   pos[i, 0] = x - alpha * dx_l
        elif nearest == 1: pos[i, 0] = x + alpha * dx_r
        elif nearest == 2: pos[i, 1] = y - alpha * dy_b
        else:              pos[i, 1] = y + alpha * dy_t
        pos[i, 0] = np.clip(pos[i, 0], half_w[i], cw - half_w[i])
        pos[i, 1] = np.clip(pos[i, 1], half_h[i], ch - half_h[i])

    pushed = torch.tensor(pos, dtype=torch.float32)
    legal, _ = project_overlaps(pushed, benchmark)
    movable_idx = [i for i in range(benchmark.num_macros) if not bool(fixed[i])]
    evaluator = IncrementalProxyEvaluator(benchmark, plc, legal.clone())
    run_cd_adaptive(
        evaluator, benchmark, plc, movable_idx,
        min_time_s=min(30.0, cd_budget_s),
        hard_cap_s=cd_budget_s,
        patience=3,
        plateau_threshold=0.001,
        log_fn=None,
    )
    polished = evaluator.placement.detach().clone().to(torch.float32)
    proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
    ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
    return polished, proxy, ovl


class CDLNSSACascadePortfolioLevyPeripheryPlacer:
    """Portfolio (4-weight) Lévy cascade + periphery-bias post-pass."""

    def __init__(
        self,
        max_iters: int = 5,
        K_eps: int = 3,
        eps_scale: float = 1.0,
        polish_budget: float = 180.0,
        budget_seconds: Optional[float] = 3300.0,
        rng_seed: int = 42,
        periphery_alpha: float = 0.01,
        periphery_polish_s: float = 180.0,
        verbose: bool = True,
    ):
        self.max_iters = max_iters
        self.K_eps = K_eps
        self.eps_scale = eps_scale
        self.polish_budget = polish_budget
        # Reserve some budget for periphery polish.
        self.budget_seconds = budget_seconds
        self.rng_seed = rng_seed
        self.periphery_alpha = periphery_alpha
        self.periphery_polish_s = periphery_polish_s
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSACascadePortfolioLevyPeripheryPlacer ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        # Reserve periphery_polish_s for the post-pass.
        levy_budget = (
            self.budget_seconds - self.periphery_polish_s - 30.0
            if self.budget_seconds is not None else None
        )
        log(f"  budget split: Lévy={levy_budget}s, periphery polish={self.periphery_polish_s}s")

        levy = CDLNSSACascadePortfolioLevyPlacer(
            max_iters=self.max_iters,
            K_eps=self.K_eps,
            eps_scale=self.eps_scale,
            polish_budget=self.polish_budget,
            budget_seconds=levy_budget,
            rng_seed=self.rng_seed,
            verbose=self.verbose,
        )
        base_placement = levy.place(benchmark)

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        base_proxy = float(compute_proxy_cost(base_placement, benchmark, plc)["proxy_cost"])
        base_ovl = compute_overlap_metrics(base_placement, benchmark)["overlap_count"]
        log(f"  Lévy cascade done: proxy={base_proxy:.5f} ovl={base_ovl}"
            f" (wall={time.time() - t0:.0f}s)")

        if base_ovl > 0:
            log(f"  WARNING: Lévy cascade returned ovl={base_ovl}; skipping periphery push")
            return base_placement

        if deadline is not None:
            remaining = deadline - time.time()
            if remaining < 90.0:
                log(f"  SKIPPING periphery polish (only {remaining:.0f}s left)")
                return base_placement
            polish_s = min(self.periphery_polish_s, remaining - 20.0)
        else:
            polish_s = self.periphery_polish_s

        log(f"  Periphery push α={self.periphery_alpha} + CD polish (budget {polish_s:.0f}s)")
        try:
            periph, periph_proxy, periph_ovl = periphery_push_and_polish(
                base_placement, benchmark, plc,
                alpha=self.periphery_alpha,
                cd_budget_s=polish_s,
                log=log,
            )
        except Exception as exc:
            log(f"  periphery push FAILED: {exc}; returning Lévy result")
            return base_placement

        improvement = base_proxy - periph_proxy
        log(f"  periphery done: proxy={periph_proxy:.5f} ({improvement:+.5f}) ovl={periph_ovl}"
            f" (total wall={time.time() - t0:.0f}s)")

        # Strictly conservative: only accept if zero overlaps AND lower proxy.
        if periph_ovl == 0 and periph_proxy < base_proxy:
            log(f"  ACCEPT periphery: {base_proxy:.5f} → {periph_proxy:.5f}"
                f" ({improvement / base_proxy * 100:+.2f}%)")
            return periph
        log(f"  REJECT periphery (ovl={periph_ovl}, Δ={improvement:+.5f}); keeping Lévy")
        return base_placement
