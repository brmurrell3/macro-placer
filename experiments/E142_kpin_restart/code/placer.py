"""E142 — K-pin random restart on top of v2 pipeline.

Architecture:

  1. V4 + Gaussian descent (`SmoothGlobalPlacerV4Gaussian.place()`) — fast
     GPU basin. (same as v2 thinkorplace-v2)
  2. Greedy macro legalize + project_overlaps.
  3. CD polish 1 (~300 s) — settle into the local minimum.
  4. K-pin restart loop (3 iterations, 150s each):
     - Copy current best placement.
     - Perturb K=15 random *hard movable* macros to random valid positions
       (uniform sample within (half_w, canvas_w - half_w) bbox).
     - Re-run `greedy_macro_legalize` to remove the introduced overlaps.
     - Re-run `project_overlaps` for safety.
     - CD polish 150s.
     - Accept iff proxy < current best by canonical compute_proxy_cost.
  5. Final CD polish 2 (~200 s).

Hypothesis: K-pin restart is a smaller perturbation than full multi-restart
(which redoes descent). It keeps the GLOBAL basin from descent but explores
alternative LOCAL arrangements. May find a lower CD plateau than the
single-pass CD that ran in step 3.

Total budget 1500 s/bench. Within the partcl 60 min/bench cap.

DO NOT mutate shipped placer files — this is an experiment.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _ROOT,
    _HERE,
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E127_v4_gaussian" / "code",
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
from macro_legalizer import greedy_macro_legalize  # noqa: E402


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
) -> torch.Tensor:
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


def _perturb_kpin_random(
    placement: torch.Tensor,
    benchmark: Benchmark,
    k: int,
    rng: np.random.RandomState,
) -> tuple[torch.Tensor, list[int]]:
    """Randomly pick K hard *movable* macros, reset each to a uniformly random
    valid position inside its legal-canvas bbox.

    Returns (perturbed_placement_f32, list_of_perturbed_indices).
    """
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.cpu().numpy()
    n_hard = benchmark.num_hard_macros

    movable_hard = [i for i in range(n_hard) if not bool(fixed[i])]
    if len(movable_hard) == 0:
        return placement.detach().clone(), []

    k_eff = min(k, len(movable_hard))
    chosen = rng.choice(len(movable_hard), size=k_eff, replace=False)
    chosen_idx = [int(movable_hard[c]) for c in chosen]

    pos = placement.detach().cpu().numpy().astype(np.float64).copy()
    for i in chosen_idx:
        hw = sizes[i, 0] / 2.0
        hh = sizes[i, 1] / 2.0
        x_lo, x_hi = hw, cw - hw
        y_lo, y_hi = hh, ch - hh
        if x_hi <= x_lo or y_hi <= y_lo:
            continue  # macro larger than canvas: leave alone
        pos[i, 0] = rng.uniform(x_lo, x_hi)
        pos[i, 1] = rng.uniform(y_lo, y_hi)

    return torch.tensor(pos, dtype=torch.float32), chosen_idx


class E142KPinRestartPlacer:
    """V4+Gaussian basin -> CD1 -> 3 K-pin restart iters -> CD2."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 1500.0,
        # Descent params (mirror v2 thinkorplace-v2)
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        # CD params
        cd1_budget_s: float = 300.0,
        cd2_budget_s: float = 200.0,
        cd_plateau_threshold: float = 0.001,
        # K-pin restart params
        kpin_k: int = 15,
        kpin_n_iters: int = 3,
        kpin_polish_budget: float = 150.0,
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
        self.kpin_k = kpin_k
        self.kpin_n_iters = kpin_n_iters
        self.kpin_polish_budget = kpin_polish_budget
        self.rng_seed = rng_seed
        self.verbose = verbose
        # Phase records (populated by place()).
        self.last_phase_walls: dict = {}
        self.last_phase_proxies: dict = {}
        self.last_kpin_iters: list = []

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E142KPinRestartPlacer ({benchmark.name}) ===")
        self._log(
            f"  budget={self.budget_seconds}s; descent + CD1={self.cd1_budget_s}s "
            f"+ K-pin({self.kpin_n_iters}x{self.kpin_polish_budget}s) "
            f"+ CD2={self.cd2_budget_s}s; K={self.kpin_k}"
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        device = _best_device()
        self._log(f"  device={device}")

        # ------- Phase 1: V4 + Gaussian descent + project_overlaps
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
                    self._log(f"    ok: ovl=0")
                    break
                pos_try2, _ = project_overlaps(pos_try, benchmark)
                if compute_overlap_metrics(pos_try2, benchmark)["overlap_count"] == 0:
                    pos = pos_try2
                    self._log(f"    ok: ovl=0 after project_overlaps")
                    break
                self._log(f"    still {ovl_try} overlaps; retrying")
            except Exception as exc:
                self._log(f"    attempt {attempt+1} EXCEPTION: {exc}")
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

        # ------- Phase 2: CD polish 1 — settle into the basin's minimum.
        if deadline is not None:
            remaining = deadline - time.time()
            reserve = (
                self.kpin_n_iters * self.kpin_polish_budget
                + self.cd2_budget_s
                + 60.0
            )
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
            f"  Phase 2 done: proxy={cd1_proxy:.5f} (Δ={cd1_proxy - descent_proxy:+.5f}) "
            f"ovl={cd1_ovl} wall={cd1_wall:.0f}s"
        )
        if cd1_ovl > 0:
            self._log(f"  WARN: CD1 produced {cd1_ovl} overlaps; project_overlaps")
            pos, _ = project_overlaps(pos, benchmark)

        # ------- Phase 3: K-pin restart loop
        best_pos = pos
        best_proxy = cd1_proxy
        rng = np.random.RandomState(self.rng_seed)
        kpin_iters_log: list = []
        t_kpin = time.time()
        for it in range(self.kpin_n_iters):
            t_it = time.time()
            # Budget guard — leave room for CD2 + slack.
            if deadline is not None:
                remaining = deadline - time.time()
                slack = self.cd2_budget_s + 20.0
                iter_budget = max(30.0, min(self.kpin_polish_budget, remaining - slack))
                if iter_budget < 30.0:
                    self._log(
                        f"  Phase 3 iter {it+1} SKIPPED (only {remaining:.0f}s left)"
                    )
                    break
            else:
                iter_budget = self.kpin_polish_budget

            try:
                # Perturb K random hard movable macros.
                perturbed, chosen = _perturb_kpin_random(
                    best_pos, benchmark, self.kpin_k, rng
                )
                pert_ovl_pre = compute_overlap_metrics(perturbed, benchmark)["overlap_count"]
                # Legalize: greedy first (preserves topology better than project alone),
                # then project_overlaps as safety net.
                try:
                    legal, leg_stats = greedy_macro_legalize(
                        perturbed, benchmark,
                        search_radius_steps=50,
                        step_size_frac=0.05,
                        verbose=False,
                    )
                except Exception as exc:
                    self._log(
                        f"    iter {it+1}: greedy_macro_legalize EXCEPTION: {exc}; fallback project"
                    )
                    legal = perturbed
                legal_ovl = compute_overlap_metrics(legal, benchmark)["overlap_count"]
                if legal_ovl > 0:
                    legal, _ = project_overlaps(legal, benchmark)
                    legal_ovl = compute_overlap_metrics(legal, benchmark)["overlap_count"]
                if legal_ovl > 0:
                    self._log(
                        f"    iter {it+1}: still {legal_ovl} ovl after legalize+project; SKIP"
                    )
                    kpin_iters_log.append(
                        dict(iter=it + 1, perturbed_n=len(chosen),
                             pre_ovl=int(pert_ovl_pre), post_ovl=int(legal_ovl),
                             pre_polish_proxy=None, post_polish_proxy=None,
                             accepted=False, wall=time.time() - t_it)
                    )
                    continue
                pre_polish_proxy = float(
                    compute_proxy_cost(legal, benchmark, plc)["proxy_cost"]
                )
                # Polish.
                polished = _cd_polish(
                    legal, benchmark, plc,
                    iter_budget, self.cd_plateau_threshold,
                )
                polished_ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
                if polished_ovl > 0:
                    polished, _ = project_overlaps(polished, benchmark)
                post_proxy = float(
                    compute_proxy_cost(polished, benchmark, plc)["proxy_cost"]
                )
                accepted = post_proxy < best_proxy
                if accepted:
                    self._log(
                        f"  Phase 3 iter {it+1} ACCEPT: K={len(chosen)} "
                        f"pre_polish={pre_polish_proxy:.5f} -> {post_proxy:.5f} "
                        f"(prev best {best_proxy:.5f}) wall={time.time()-t_it:.0f}s"
                    )
                    best_pos = polished
                    best_proxy = post_proxy
                else:
                    self._log(
                        f"  Phase 3 iter {it+1} REJECT: K={len(chosen)} "
                        f"pre_polish={pre_polish_proxy:.5f} -> {post_proxy:.5f} "
                        f"(best {best_proxy:.5f}) wall={time.time()-t_it:.0f}s"
                    )
                kpin_iters_log.append(
                    dict(iter=it + 1, perturbed_n=len(chosen),
                         pre_ovl=int(pert_ovl_pre), post_ovl=int(polished_ovl),
                         pre_polish_proxy=pre_polish_proxy,
                         post_polish_proxy=post_proxy,
                         accepted=bool(accepted),
                         wall=time.time() - t_it)
                )
            except Exception as exc:
                self._log(f"  Phase 3 iter {it+1} EXCEPTION: {exc}; skipping iter")
                kpin_iters_log.append(
                    dict(iter=it + 1, error=str(exc),
                         wall=time.time() - t_it, accepted=False)
                )
                continue
        kpin_wall = time.time() - t_kpin
        self._log(
            f"  Phase 3 done: best_proxy={best_proxy:.5f} "
            f"(Δ vs CD1 {best_proxy - cd1_proxy:+.5f}) wall={kpin_wall:.0f}s"
        )

        pos = best_pos

        # ------- Phase 4: CD polish 2 (final)
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
        self._log(
            f"  TOTAL: descent={descent_wall:.0f}s CD1={cd1_wall:.0f}s "
            f"K-pin={kpin_wall:.0f}s CD2={cd2_wall:.0f}s total={total_wall:.0f}s "
            f"final_proxy={final_proxy:.5f}"
        )

        # Record stats for harness/inspection.
        self.last_phase_walls = {
            "descent": descent_wall,
            "cd1": cd1_wall,
            "kpin": kpin_wall,
            "cd2": cd2_wall,
            "total": total_wall,
        }
        self.last_phase_proxies = {
            "descent": descent_proxy,
            "cd1": cd1_proxy,
            "kpin_best": best_proxy,
            "final": final_proxy,
        }
        self.last_kpin_iters = kpin_iters_log

        if final_ovl > 0:
            raise RuntimeError(f"Final placement has {final_ovl} overlaps")
        return final


# Alias for harness convenience.
Placer = E142KPinRestartPlacer
