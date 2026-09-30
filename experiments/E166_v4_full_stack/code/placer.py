"""E142 — V4-Gaussian + multi-init/multi-seed + cascade saddle + portfolio saddle.

Full Option C architecture on V4 base. After E141's multi-init multi-seed
basin pick + CD polish + cascade saddle, layer portfolio saddle (E100) for
multi-weight Hessian eigvec escape directions.

Isolated experiment — imports V4, cascade, portfolio primitives read-only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E18_dpo_init" / "code",
    _ROOT / "experiments" / "E84_cascading_saddle" / "code",
    _ROOT / "experiments" / "E97_levy_saddle" / "code",
    _ROOT / "experiments" / "E100_weight_portfolio_saddle" / "code",
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
from macro_place.cd_core import project_overlaps, run_cd_adaptive, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v4_gaussian import SmoothGlobalPlacerV4Gaussian
from cascading_saddle import cascading_saddle_escape
from portfolio_saddle import portfolio_saddle_escape


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEFAULT_LANES = (
    ("sdf42", "sdf", 42),
    ("sdf123", "sdf", 123),
    ("sdf999", "sdf", 999),
    ("dpo42", "dpo", 42),
)

DEFAULT_PORTFOLIO = [
    (1.0, 0.5, 0.5),  # canonical
    (1.0, 0.0, 1.0),  # congestion focus
    (1.0, 1.0, 0.0),  # density focus
    (0.0, 1.0, 1.0),  # non-WL
]


class Placer:
    """V4 multi-init + multi-seed + CD polish + cascade saddle + portfolio saddle."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 2400.0,
        lanes: Tuple[Tuple[str, str, int], ...] = DEFAULT_LANES,
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        cd_polish_s: float = 500.0,
        cd_plateau_threshold: float = 0.001,
        cascade_reserve_s: float = 400.0,
        cascade_max_iters: int = 2,
        cascade_eps_values: Tuple[float, ...] = (0.5, 1.5),
        cascade_polish_budget: float = 100.0,
        portfolio_reserve_s: float = 600.0,
        portfolio_max_iters: int = 2,
        portfolio_K_eps: int = 2,
        portfolio_eps_scale: float = 1.0,
        portfolio_polish_budget: float = 100.0,
        portfolio_rng_seed: int = 42,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.lanes = lanes
        self.num_steps = num_steps
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.cd_polish_s = cd_polish_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.cascade_reserve_s = cascade_reserve_s
        self.cascade_max_iters = cascade_max_iters
        self.cascade_eps_values = cascade_eps_values
        self.cascade_polish_budget = cascade_polish_budget
        self.portfolio_reserve_s = portfolio_reserve_s
        self.portfolio_max_iters = portfolio_max_iters
        self.portfolio_K_eps = portfolio_K_eps
        self.portfolio_eps_scale = portfolio_eps_scale
        self.portfolio_polish_budget = portfolio_polish_budget
        self.portfolio_rng_seed = portfolio_rng_seed
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _run_lane(
        self, label: str, init: str, seed: int,
        benchmark: Benchmark, plc, device: str,
    ) -> Optional[Tuple[float, torch.Tensor]]:
        t_lane = time.time()
        try:
            desc = SmoothGlobalPlacerV4Gaussian(
                num_steps=self.num_steps,
                lr_frac=self.lr_frac,
                gamma_start_frac=self.gamma_start_frac,
                gamma_end_frac=self.gamma_end_frac,
                overlap_lambda_end=self.overlap_lambda_end,
                overlap_ramp_pct=self.overlap_ramp_pct,
                init=init, device=device, rng_seed=seed, verbose=False,
            )
            pos = desc.place(benchmark)
            ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
            if ovl > 0:
                pos, _ = project_overlaps(pos, benchmark)
                ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
            wall = time.time() - t_lane
            if ovl > 0:
                self._log(f"  [{label}] SKIP {ovl} overlaps wall={wall:.0f}s")
                return None
            canonical = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
            self._log(f"  [{label}] basin canonical={canonical:.5f} wall={wall:.0f}s")
            return canonical, pos
        except Exception as exc:
            self._log(f"  [{label}] EXCEPTION {exc}")
            return None

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_global = time.time()
        deadline = (t_global + self.budget_seconds) if self.budget_seconds else None
        self._log(
            f"=== E142 v4_full_stack ({benchmark.name}) budget={self.budget_seconds}s ==="
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        device = _best_device()
        self._log(f"  device={device}")

        # Phase 1: multi-init multi-seed
        results: List[Tuple[float, str, torch.Tensor]] = []
        reserve = self.cd_polish_s + self.cascade_reserve_s + self.portfolio_reserve_s + 60
        for label, init, seed in self.lanes:
            if deadline is not None and time.time() > deadline - reserve:
                self._log(f"  [budget] skip remaining lanes")
                break
            r = self._run_lane(label, init, seed, benchmark, plc, device)
            if r is not None:
                results.append((r[0], label, r[1]))

        if not results:
            self._log("  All lanes failed; SDF fallback")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)
        else:
            results.sort(key=lambda r: r[0])
            best_canonical, best_label, best_pos = results[0]
            per_lane = ", ".join(f"{l}:{c:.5f}" for c, l, _ in results)
            self._log(
                f"  [PICK] {best_label} (canonical={best_canonical:.5f}); per-lane: {per_lane}"
            )
            pos = best_pos

        # Phase 2: CD polish
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd_budget = max(30.0, min(
                remaining - self.cascade_reserve_s - self.portfolio_reserve_s,
                self.cd_polish_s,
            ))
        else:
            cd_budget = self.cd_polish_s
        self._log(f"  CD polish budget={cd_budget:.0f}s")

        evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        run_cd_adaptive(
            evaluator, benchmark, plc, movable,
            min_time_s=cd_budget * 0.5, hard_cap_s=cd_budget,
            patience=5, plateau_threshold=self.cd_plateau_threshold, log_fn=None,
        )
        polished = evaluator.placement.detach().clone().to(torch.float32)
        polished_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
        polished_ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
        self._log(f"  CD polish done: proxy={polished_proxy:.5f} ovl={polished_ovl}")
        if polished_ovl > 0:
            polished, _ = project_overlaps(polished, benchmark)
            polished_ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
            if polished_ovl > 0:
                raise RuntimeError(f"Polished basin has {polished_ovl} overlaps")
            polished_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])

        current = polished
        current_proxy = polished_proxy

        # Phase 3: cascade saddle escape
        if deadline is not None:
            cascade_budget = max(60.0, min(
                deadline - time.time() - self.portfolio_reserve_s - 30,
                self.cascade_reserve_s,
            ))
        else:
            cascade_budget = self.cascade_reserve_s
        self._log(f"  cascade budget={cascade_budget:.0f}s")

        try:
            cascaded, cstats = cascading_saddle_escape(
                current.to(torch.float32), benchmark, plc,
                max_iters=self.cascade_max_iters,
                eps_values=self.cascade_eps_values,
                polish_budget=self.cascade_polish_budget,
                total_budget_s=cascade_budget,
                log=self._log if self.verbose else (lambda s: None),
            )
            cascaded_proxy = float(compute_proxy_cost(cascaded, benchmark, plc)["proxy_cost"])
            cascaded_ovl = compute_overlap_metrics(cascaded, benchmark)["overlap_count"]
            if cascaded_ovl == 0 and cascaded_proxy < current_proxy - 1e-7:
                lift = current_proxy - cascaded_proxy
                self._log(
                    f"  cascade: {current_proxy:.5f} -> {cascaded_proxy:.5f} "
                    f"lift {lift:+.5f} ({100*lift/current_proxy:+.2f}%)"
                )
                current = cascaded
                current_proxy = cascaded_proxy
            else:
                self._log(f"  cascade: no improvement, kept {current_proxy:.5f}")
        except Exception as exc:
            self._log(f"  cascade EXCEPTION: {exc}")

        # Phase 4: portfolio saddle escape
        if deadline is not None:
            portfolio_budget = max(60.0, deadline - time.time() - 30.0)
        else:
            portfolio_budget = self.portfolio_reserve_s
        self._log(f"  portfolio budget={portfolio_budget:.0f}s")

        try:
            ported, pstats = portfolio_saddle_escape(
                current.to(torch.float32), benchmark, plc,
                max_iters=self.portfolio_max_iters,
                portfolio=DEFAULT_PORTFOLIO,
                K_eps=self.portfolio_K_eps,
                eps_scale=self.portfolio_eps_scale,
                polish_budget=self.portfolio_polish_budget,
                total_budget_s=portfolio_budget,
                rng_seed=self.portfolio_rng_seed,
                log=self._log if self.verbose else (lambda s: None),
            )
            ported_proxy = float(compute_proxy_cost(ported, benchmark, plc)["proxy_cost"])
            ported_ovl = compute_overlap_metrics(ported, benchmark)["overlap_count"]
            if ported_ovl == 0 and ported_proxy < current_proxy - 1e-7:
                lift = current_proxy - ported_proxy
                self._log(
                    f"  portfolio: {current_proxy:.5f} -> {ported_proxy:.5f} "
                    f"lift {lift:+.5f} ({100*lift/current_proxy:+.2f}%) "
                    f"total_wall={time.time()-t_global:.0f}s"
                )
                current = ported
                current_proxy = ported_proxy
            else:
                self._log(f"  portfolio: no improvement, kept {current_proxy:.5f}")
        except Exception as exc:
            self._log(f"  portfolio EXCEPTION: {exc}")

        self._log(
            f"  FINAL: proxy={current_proxy:.5f} total_wall={time.time()-t_global:.0f}s"
        )
        return current
