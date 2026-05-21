"""E147 — continuous-relaxation refinement of CD-plateau placement.

Pipeline:
  1. V4+Gaussian descent + legalize (~150 s).
  2. CD polish 1 (~400 s).
  3. L-BFGS-B (PyTorch) on the smooth proxy with FROZEN overlap lambda
     starting from the CD1 plateau (~200 s).
  4. project_overlaps to repair any introduced overlaps.
  5. CD polish 2 (~300 s).

The hypothesis is that CD's axis-aligned greedy moves get stuck at
local minima that are not stationary points of the continuous-relaxation
smooth proxy. A quasi-Newton refinement of the differentiable smooth
proxy (FastDiffProxyGaussian) can move off the lattice toward a globally
better configuration, even if that configuration has some overlaps that
projection then repairs.

DO NOT mutate parent files. Imports follow the E140 pattern.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _ROOT,
    _HERE,
    _ROOT / "experiments" / "E74_hessian_saddle" / "code",
    _ROOT / "experiments" / "E84_cascading_saddle" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E127_v4_gaussian" / "code",
    _ROOT / "experiments" / "E138_bounded_saddle" / "code",
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

from smooth_global_placer_v4_gaussian import SmoothGlobalPlacerV4Gaussian  # noqa: E402
from fast_proxy import fast_loss_with_penalty  # noqa: E402
from v4_gaussian_proxy import FastDiffProxyGaussian  # noqa: E402


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cd_polish(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    budget_s: float,
    plateau_threshold: float = 0.001,
):
    """Run CD-adaptive polish under a hard wall budget."""
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement)
    movable = [
        i for i in range(benchmark.num_macros)
        if not bool(benchmark.macro_fixed[i])
    ]
    run_cd_adaptive(
        evaluator, benchmark, plc, movable,
        min_time_s=min(30.0, budget_s * 0.5),
        hard_cap_s=budget_s,
        patience=5,
        plateau_threshold=plateau_threshold,
        log_fn=None,
    )
    return evaluator.placement.detach().clone().to(torch.float32)


def _lbfgs_refine(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    device: str,
    overlap_lambda: float = 10.0,
    boundary_lambda: float = 1.0,
    gamma_frac: float = 5e-5,
    max_iter: int = 60,
    max_eval: int = 80,
    history_size: int = 20,
    budget_s: float = 200.0,
    include_congestion: bool = True,
    log_fn=None,
) -> Tuple[torch.Tensor, dict]:
    """Quasi-Newton refinement of the smooth proxy via PyTorch L-BFGS.

    Starts from `placement`, optimizes positions of NON-FIXED macros only.
    Returns refined positions (CPU float32) and a stats dict.
    """
    log = log_fn or (lambda s: None)
    t0 = time.time()
    dev = torch.device(device)

    proxy = FastDiffProxyGaussian(
        benchmark, plc, device=str(dev),
        gamma_frac=gamma_frac,
    )

    positions = placement.detach().clone().to(dev).to(torch.float32).requires_grad_(True)
    fixed_mask = benchmark.macro_fixed.bool().to(dev)

    # Cache initial loss for relative-tolerance early stop.
    with torch.no_grad():
        loss0, parts0 = fast_loss_with_penalty(
            proxy, positions, overlap_lambda,
            include_congestion=include_congestion,
            boundary_lambda=boundary_lambda,
        )
        loss0_v = float(loss0.item())
        smooth0_v = float(parts0["smooth_cost"].item())
        ovl0_v = float(parts0["overlap_area_raw"].item())
    log(f"L-BFGS init: smooth={smooth0_v:.5f} ovl_area={ovl0_v:.0f} total={loss0_v:.5f}")

    # PyTorch L-BFGS — strong-Wolfe line search, low memory.
    optimizer = torch.optim.LBFGS(
        [positions],
        max_iter=max_iter,
        max_eval=max_eval,
        history_size=history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    n_calls = {"n": 0}
    last = {"loss": loss0_v, "parts": parts0}
    deadline = t0 + budget_s

    def closure():
        # Deadline guard inside closure (LBFGS may call this many times).
        if time.time() > deadline:
            # Returning current loss is fine; LBFGS will continue or stop.
            optimizer.zero_grad(set_to_none=False)
            with torch.no_grad():
                loss_v, _ = fast_loss_with_penalty(
                    proxy, positions, overlap_lambda,
                    include_congestion=include_congestion,
                    boundary_lambda=boundary_lambda,
                )
            return loss_v
        optimizer.zero_grad(set_to_none=False)
        loss, parts = fast_loss_with_penalty(
            proxy, positions, overlap_lambda,
            include_congestion=include_congestion,
            boundary_lambda=boundary_lambda,
        )
        loss.backward()
        with torch.no_grad():
            if positions.grad is not None:
                positions.grad[fixed_mask] = 0.0
        n_calls["n"] += 1
        last["loss"] = float(loss.item())
        last["parts"] = parts
        if n_calls["n"] % 5 == 0:
            log(
                f"L-BFGS step {n_calls['n']}: total={last['loss']:.5f} "
                f"smooth={parts['smooth_cost'].item():.5f} "
                f"ovl_area={parts['overlap_area_raw'].item():.0f} "
                f"wall={time.time()-t0:.0f}s"
            )
        return loss

    try:
        optimizer.step(closure)
    except Exception as exc:
        log(f"L-BFGS exception: {exc!r}")

    # Snap final
    with torch.no_grad():
        loss_f, parts_f = fast_loss_with_penalty(
            proxy, positions, overlap_lambda,
            include_congestion=include_congestion,
            boundary_lambda=boundary_lambda,
        )
        loss_f_v = float(loss_f.item())
        smooth_f_v = float(parts_f["smooth_cost"].item())
        ovl_f_v = float(parts_f["overlap_area_raw"].item())

    wall = time.time() - t0
    log(
        f"L-BFGS done in {n_calls['n']} closures, wall={wall:.0f}s: "
        f"smooth {smooth0_v:.5f} -> {smooth_f_v:.5f} "
        f"({smooth_f_v - smooth0_v:+.5f}), "
        f"ovl_area {ovl0_v:.0f} -> {ovl_f_v:.0f}, "
        f"total {loss0_v:.5f} -> {loss_f_v:.5f}"
    )

    final_pos = positions.detach().to(torch.float32).cpu()
    stats = {
        "closures": n_calls["n"],
        "smooth_init": smooth0_v,
        "smooth_final": smooth_f_v,
        "smooth_delta": smooth_f_v - smooth0_v,
        "total_init": loss0_v,
        "total_final": loss_f_v,
        "ovl_area_init": ovl0_v,
        "ovl_area_final": ovl_f_v,
        "wall_s": wall,
    }
    return final_pos, stats


class E147LpRefinePlacer:
    """V4+Gauss -> CD1 -> L-BFGS -> project -> CD2."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 1500.0,
        # Descent params (mirror E140)
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        # CD budgets
        cd1_budget_s: float = 400.0,
        cd2_budget_s: float = 300.0,
        cd_plateau_threshold: float = 0.001,
        # L-BFGS params
        lbfgs_budget_s: float = 200.0,
        lbfgs_overlap_lambda: float = 10.0,
        lbfgs_boundary_lambda: float = 1.0,
        lbfgs_gamma_frac: float = 5e-5,
        lbfgs_max_iter: int = 60,
        lbfgs_max_eval: int = 80,
        lbfgs_history_size: int = 20,
        lbfgs_include_congestion: bool = True,
        rng_seed: int = 42,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.num_steps = num_steps
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.init = init
        self.cd1_budget_s = cd1_budget_s
        self.cd2_budget_s = cd2_budget_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.lbfgs_budget_s = lbfgs_budget_s
        self.lbfgs_overlap_lambda = lbfgs_overlap_lambda
        self.lbfgs_boundary_lambda = lbfgs_boundary_lambda
        self.lbfgs_gamma_frac = lbfgs_gamma_frac
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_max_eval = lbfgs_max_eval
        self.lbfgs_history_size = lbfgs_history_size
        self.lbfgs_include_congestion = lbfgs_include_congestion
        self.rng_seed = rng_seed
        self.verbose = verbose
        self.last_phase_walls: dict = {}
        self.last_phase_proxies: dict = {}
        self.last_lbfgs_stats: dict = {}

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E147LpRefinePlacer ({benchmark.name}) ===")
        self._log(
            f"  budget={self.budget_seconds}s; descent + CD1={self.cd1_budget_s}s "
            f"+ LBFGS={self.lbfgs_budget_s}s + CD2={self.cd2_budget_s}s"
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        device = _best_device()
        self._log(f"  device={device}")

        # ---- Phase 1: V4+Gaussian descent + legalize ----
        t_descent = time.time()
        pos = None
        for attempt, cfg in enumerate([
            dict(overlap_lambda_end=self.overlap_lambda_end),
            dict(overlap_lambda_end=50.0),
            dict(overlap_lambda_end=100.0, num_steps=300),
        ]):
            try:
                num_steps = cfg.get("num_steps", self.num_steps)
                ovl_end = cfg["overlap_lambda_end"]
                self._log(
                    f"  Phase 1 attempt {attempt+1}: ovl_end={ovl_end} steps={num_steps}"
                )
                descender = SmoothGlobalPlacerV4Gaussian(
                    num_steps=num_steps,
                    lr_frac=self.lr_frac,
                    gamma_start_frac=self.gamma_start_frac,
                    gamma_end_frac=self.gamma_end_frac,
                    overlap_lambda_end=ovl_end,
                    overlap_ramp_pct=self.overlap_ramp_pct,
                    init=self.init,
                    device=device,
                    rng_seed=self.rng_seed + attempt * 100,
                    verbose=False,
                )
                pos_try = descender.place(benchmark)
                ovl_try = compute_overlap_metrics(pos_try, benchmark)["overlap_count"]
                if ovl_try == 0:
                    pos = pos_try
                    self._log("    ok: ovl=0")
                    break
                pos_try2, _ = project_overlaps(pos_try, benchmark)
                if compute_overlap_metrics(pos_try2, benchmark)["overlap_count"] == 0:
                    pos = pos_try2
                    self._log("    ok: ovl=0 after project_overlaps")
                    break
                self._log(f"    still {ovl_try} overlaps; retrying")
            except Exception as exc:
                self._log(f"    attempt {attempt+1} EXCEPTION: {exc!r}")
                continue
            if deadline is not None and time.time() > deadline - 600:
                break
        if pos is None:
            self._log("  Phase 1 fallback: SDF + project_overlaps")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)
        descent_wall = time.time() - t_descent
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self._log(
            f"  Phase 1 done: proxy={descent_proxy:.5f} wall={descent_wall:.0f}s"
        )

        # ---- Phase 2: CD polish 1 ----
        if deadline is not None:
            remaining = deadline - time.time()
            reserve = self.lbfgs_budget_s + self.cd2_budget_s + 60.0
            cd1_budget = max(30.0, min(self.cd1_budget_s, remaining - reserve))
        else:
            cd1_budget = self.cd1_budget_s
        t_cd1 = time.time()
        self._log(f"  Phase 2: CD1 budget={cd1_budget:.0f}s")
        pos = _cd_polish(pos, benchmark, plc, cd1_budget, self.cd_plateau_threshold)
        cd1_wall = time.time() - t_cd1
        cd1_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        cd1_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        self._log(
            f"  Phase 2 done: proxy={cd1_proxy:.5f} (delta={cd1_proxy - descent_proxy:+.5f}) "
            f"ovl={cd1_ovl} wall={cd1_wall:.0f}s"
        )
        if cd1_ovl > 0:
            self._log(f"  WARN: CD1 produced {cd1_ovl} overlaps; project_overlaps")
            pos, _ = project_overlaps(pos, benchmark)

        cd1_baseline = pos.detach().clone()
        cd1_baseline_proxy = cd1_proxy

        # ---- Phase 3: L-BFGS refinement on smooth proxy ----
        if deadline is not None:
            remaining = deadline - time.time()
            lbfgs_budget = max(
                30.0, min(self.lbfgs_budget_s, remaining - self.cd2_budget_s - 30.0)
            )
        else:
            lbfgs_budget = self.lbfgs_budget_s
        t_lbfgs = time.time()
        lbfgs_accepted = False
        lbfgs_stats: dict = {}
        if lbfgs_budget < 30.0:
            self._log(f"  Phase 3 SKIPPED (budget {lbfgs_budget:.0f}s)")
        else:
            self._log(
                f"  Phase 3: L-BFGS budget={lbfgs_budget:.0f}s "
                f"overlap_lambda={self.lbfgs_overlap_lambda} "
                f"max_iter={self.lbfgs_max_iter}"
            )
            try:
                refined_pos, lbfgs_stats = _lbfgs_refine(
                    pos, benchmark, plc,
                    device=device,
                    overlap_lambda=self.lbfgs_overlap_lambda,
                    boundary_lambda=self.lbfgs_boundary_lambda,
                    gamma_frac=self.lbfgs_gamma_frac,
                    max_iter=self.lbfgs_max_iter,
                    max_eval=self.lbfgs_max_eval,
                    history_size=self.lbfgs_history_size,
                    budget_s=lbfgs_budget,
                    include_congestion=self.lbfgs_include_congestion,
                    log_fn=lambda s: self._log(f"    [lbfgs] {s}"),
                )
                # Project to legality.
                ovl_post = compute_overlap_metrics(refined_pos, benchmark)["overlap_count"]
                self._log(f"    post-LBFGS overlap count={ovl_post}")
                if ovl_post > 0:
                    refined_pos, proj_info = project_overlaps(refined_pos, benchmark)
                    ovl_post2 = compute_overlap_metrics(
                        refined_pos, benchmark)["overlap_count"]
                    self._log(
                        f"    project_overlaps: ovl={ovl_post}->{ovl_post2} "
                        f"info={proj_info}"
                    )
                    ovl_post = ovl_post2
                if ovl_post > 0:
                    self._log(
                        f"  Phase 3 REJECT: cannot legalize ({ovl_post} ovl); "
                        f"reverting to CD1 baseline"
                    )
                else:
                    refined_proxy = float(
                        compute_proxy_cost(refined_pos, benchmark, plc)["proxy_cost"]
                    )
                    self._log(
                        f"    canonical proxy after legalize: {refined_proxy:.5f} "
                        f"(CD1 was {cd1_baseline_proxy:.5f})"
                    )
                    if refined_proxy < cd1_baseline_proxy:
                        pos = refined_pos
                        lbfgs_accepted = True
                        self._log(
                            f"  Phase 3 ACCEPT: {cd1_baseline_proxy:.5f} -> "
                            f"{refined_proxy:.5f}"
                        )
                    else:
                        self._log(
                            f"  Phase 3 REJECT (worse): {refined_proxy:.5f} >= "
                            f"{cd1_baseline_proxy:.5f}"
                        )
            except Exception as exc:
                self._log(f"  Phase 3 EXCEPTION: {exc!r}; skipping")
        lbfgs_wall = time.time() - t_lbfgs
        lbfgs_proxy_after = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self._log(
            f"  Phase 3 done: proxy={lbfgs_proxy_after:.5f} wall={lbfgs_wall:.0f}s"
        )

        # ---- Phase 4: CD polish 2 ----
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd2_budget = max(30.0, min(self.cd2_budget_s, remaining))
        else:
            cd2_budget = self.cd2_budget_s
        t_cd2 = time.time()
        self._log(f"  Phase 4: CD2 budget={cd2_budget:.0f}s")
        pos = _cd_polish(pos, benchmark, plc, cd2_budget, self.cd_plateau_threshold)
        cd2_wall = time.time() - t_cd2
        final = pos.detach().clone().to(torch.float32)
        final_proxy = float(compute_proxy_cost(final, benchmark, plc)["proxy_cost"])
        final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
        total_wall = time.time() - t0
        self._log(
            f"  Phase 4 done: proxy={final_proxy:.5f} ovl={final_ovl} "
            f"wall={cd2_wall:.0f}s"
        )

        # Safety net: if CD2 somehow regressed below CD1 baseline (very rare),
        # fall back to CD1 result (which is guaranteed legal).
        if final_proxy > cd1_baseline_proxy * 1.005:
            self._log(
                f"  WARN: final {final_proxy:.5f} > 1.005 * CD1 baseline "
                f"{cd1_baseline_proxy:.5f}; reverting to CD1 baseline"
            )
            final = cd1_baseline.to(torch.float32)
            final_proxy = cd1_baseline_proxy

        self._log(
            f"  TOTAL: descent={descent_wall:.0f}s CD1={cd1_wall:.0f}s "
            f"LBFGS={lbfgs_wall:.0f}s (accepted={lbfgs_accepted}) "
            f"CD2={cd2_wall:.0f}s total={total_wall:.0f}s "
            f"final_proxy={final_proxy:.5f}"
        )

        self.last_phase_walls = {
            "descent": descent_wall,
            "cd1": cd1_wall,
            "lbfgs": lbfgs_wall,
            "cd2": cd2_wall,
            "total": total_wall,
        }
        self.last_phase_proxies = {
            "descent": descent_proxy,
            "cd1": cd1_proxy,
            "lbfgs_after": lbfgs_proxy_after,
            "final": final_proxy,
            "lbfgs_accepted": lbfgs_accepted,
        }
        self.last_lbfgs_stats = lbfgs_stats

        if final_ovl > 0:
            raise RuntimeError(f"Final placement has {final_ovl} overlaps")
        return final


Placer = E147LpRefinePlacer
