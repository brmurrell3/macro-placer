"""E137 — E135 SDF+DPO two-lane V4+Gaussian with unblocked legalize.

Clone of E135 with a jitter-and-retry legalize wrapper so Lane A (SDF
init) doesn't get SKIPped on stable 1-overlap fixed points. Touches no
shared code: imports cd_core.project_overlaps and macro_legalizer.greedy_macro_legalize
read-only, wraps them with retry logic locally.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E18_dpo_init" / "code",
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
from macro_legalizer import greedy_macro_legalize


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _robust_legalize(
    positions: torch.Tensor,
    benchmark: Benchmark,
    radius_steps: int,
    step_frac: float,
    rng_seed: int = 0,
    log=None,
) -> torch.Tensor:
    """Multi-stage legalize: greedy -> project -> jitter+project (5x) -> greedy 2x.

    Returns positions; may still have overlaps if all stages fail (caller
    is responsible for handling that — CD polish will get a chance to
    resolve, then final check will raise).
    """
    if log is None:
        log = lambda s: None
    rng = np.random.default_rng(rng_seed)

    pos, _ = greedy_macro_legalize(
        positions, benchmark,
        search_radius_steps=radius_steps,
        step_size_frac=step_frac,
        verbose=False,
    )
    ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    if ovl == 0:
        return pos

    pos, _ = project_overlaps(pos, benchmark)
    ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
    if ovl == 0:
        return pos
    log(f"    legalize: {ovl} overlaps after greedy+project; entering jitter retry")

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    n_hard = benchmark.num_hard_macros
    fixed = benchmark.macro_fixed.cpu().numpy()
    sizes = benchmark.macro_sizes.cpu().numpy()
    half_w = sizes[:n_hard, 0] / 2
    half_h = sizes[:n_hard, 1] / 2

    best_pos = pos.clone()
    best_ovl = ovl
    for attempt in range(5):
        jitter_scale = (0.005 + 0.005 * attempt) * cw  # 0.5% -> 2.5% canvas
        pos_np = best_pos.cpu().numpy().astype(np.float64).copy()
        jitter = rng.normal(0.0, jitter_scale, size=pos_np[:n_hard].shape)
        for i in range(n_hard):
            if bool(fixed[i]):
                continue
            pos_np[i, 0] += jitter[i, 0]
            pos_np[i, 1] += jitter[i, 1]
        pos_np[:n_hard, 0] = np.clip(pos_np[:n_hard, 0], half_w, cw - half_w)
        pos_np[:n_hard, 1] = np.clip(pos_np[:n_hard, 1], half_h, ch - half_h)
        cand = torch.tensor(pos_np, dtype=positions.dtype)
        cand, _ = project_overlaps(cand, benchmark)
        ovl_cand = compute_overlap_metrics(cand, benchmark)["overlap_count"]
        if ovl_cand == 0:
            log(f"    legalize: jitter retry {attempt+1} resolved")
            return cand
        if ovl_cand < best_ovl:
            best_ovl = ovl_cand
            best_pos = cand
        log(f"    legalize: jitter retry {attempt+1} -> {ovl_cand} overlaps")

    pos2, _ = greedy_macro_legalize(
        best_pos, benchmark,
        search_radius_steps=radius_steps * 2,
        step_size_frac=step_frac,
        verbose=False,
    )
    ovl2 = compute_overlap_metrics(pos2, benchmark)["overlap_count"]
    if ovl2 == 0:
        log(f"    legalize: large-radius greedy fallback resolved")
        return pos2
    log(f"    legalize: all stages exhausted, returning {ovl2} overlaps "
        f"(caller must handle)")
    return pos2


class E137SdfDpoLaneUnblockedPlacer:
    """Two-lane V4+Gaussian with robust legalize so Lane A doesn't SKIP."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 900.0,
        rng_seed: int = 42,
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        cd_polish_s: float = 400.0,
        cd_plateau_threshold: float = 0.001,
        legalize_radius_steps: int = 80,
        legalize_step_frac: float = 0.02,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.rng_seed = rng_seed
        self.num_steps = num_steps
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.cd_polish_s = cd_polish_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.legalize_radius_steps = legalize_radius_steps
        self.legalize_step_frac = legalize_step_frac
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _make_descender(self, init: str, device: str) -> SmoothGlobalPlacerV4Gaussian:
        return SmoothGlobalPlacerV4Gaussian(
            num_steps=self.num_steps,
            lr_frac=self.lr_frac,
            gamma_start_frac=self.gamma_start_frac,
            gamma_end_frac=self.gamma_end_frac,
            overlap_lambda_end=self.overlap_lambda_end,
            overlap_ramp_pct=self.overlap_ramp_pct,
            init=init,
            device=device,
            rng_seed=self.rng_seed,
            verbose=False,
            legalize_radius_steps=self.legalize_radius_steps,
            legalize_step_frac=self.legalize_step_frac,
        )

    def _run_lane(
        self,
        lane_name: str,
        init: str,
        benchmark: Benchmark,
        plc,
        device: str,
    ) -> Optional[Tuple[float, torch.Tensor, int]]:
        """Run one V4+Gauss lane; return (canonical_proxy, pos, residual_ovl).

        Unlike E135, never returns None on legalize residuals — accepts
        residual-overlap placements and lets caller decide via residual_ovl.
        """
        t_lane = time.time()
        try:
            desc = self._make_descender(init, device)
            pos_raw, stats = desc.descend(benchmark, plc)
            ovl_pre = compute_overlap_metrics(pos_raw, benchmark)["overlap_count"]
            legal_pos = _robust_legalize(
                pos_raw, benchmark,
                radius_steps=self.legalize_radius_steps,
                step_frac=self.legalize_step_frac,
                rng_seed=self.rng_seed + (0 if lane_name == "A" else 1000),
                log=self._log,
            )
            ovl_post = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
            wall = time.time() - t_lane
            canonical = float(
                compute_proxy_cost(legal_pos, benchmark, plc)["proxy_cost"]
            )
            self._log(
                f"  [{lane_name}] init={init}: smooth={stats['final_smooth']:.5f} "
                f"canonical={canonical:.5f} ovl_pre={ovl_pre} ovl_post={ovl_post} "
                f"wall={wall:.0f}s"
            )
            return canonical, legal_pos, ovl_post
        except Exception as exc:
            wall = time.time() - t_lane
            self._log(f"  [{lane_name}] init={init}: EXCEPTION {exc} wall={wall:.0f}s")
            return None

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_global = time.time()
        deadline = (t_global + self.budget_seconds) if self.budget_seconds else None
        self._log(
            f"=== E137 sdf+dpo lane UNBLOCKED ({benchmark.name}) "
            f"num_steps={self.num_steps} seed={self.rng_seed} "
            f"budget={self.budget_seconds}s ==="
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        device = _best_device()
        self._log(f"  device={device}")

        results = []  # list of (canonical, lane_name, pos, residual_ovl)
        sdf_result = self._run_lane("A", "sdf", benchmark, plc, device)
        if sdf_result is not None:
            results.append((sdf_result[0], "SDF", sdf_result[1], sdf_result[2]))

        if deadline is not None and time.time() > deadline - (self.cd_polish_s + 100):
            self._log(f"  [B] budget tight after Lane A; skip Lane B")
        else:
            dpo_result = self._run_lane("B", "dpo", benchmark, plc, device)
            if dpo_result is not None:
                results.append((dpo_result[0], "DPO", dpo_result[1], dpo_result[2]))

        if not results:
            self._log("  Both lanes failed; SDF fallback")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)
            return self._cd_polish_and_return(
                pos, benchmark, plc, deadline, t_global
            )

        # Prefer ovl=0 lanes; among equals, lowest canonical wins.
        results.sort(key=lambda r: (r[3] > 0, r[0]))
        best_canonical, best_lane, best_pos, best_ovl = results[0]
        per_lane = ", ".join(
            f"{n}={c:.5f}(ovl={o})" for c, n, _, o in results
        )
        self._log(
            f"  [PICK] {best_lane} basin (canonical={best_canonical:.5f}, "
            f"ovl={best_ovl}); per-lane: {per_lane}"
        )

        return self._cd_polish_and_return(
            best_pos, benchmark, plc, deadline, t_global,
            best_lane=best_lane,
        )

    def _cd_polish_and_return(
        self,
        pos: torch.Tensor,
        benchmark: Benchmark,
        plc,
        deadline: Optional[float],
        t_global: float,
        best_lane: str = "?",
    ) -> torch.Tensor:
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd_budget = max(30.0, min(remaining, self.cd_polish_s))
        else:
            cd_budget = self.cd_polish_s
        self._log(f"  CD polish budget={cd_budget:.0f}s (from {best_lane} basin)")

        evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        run_cd_adaptive(
            evaluator, benchmark, plc, movable,
            min_time_s=cd_budget * 0.5,
            hard_cap_s=cd_budget,
            patience=5,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=None,
        )
        final = evaluator.placement.detach().clone().to(torch.float32)
        final_proxy = float(compute_proxy_cost(final, benchmark, plc)["proxy_cost"])
        final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
        self._log(
            f"  Final: proxy={final_proxy:.5f} ovl={final_ovl} "
            f"basin={best_lane} total_wall={time.time()-t_global:.0f}s"
        )
        if final_ovl > 0:
            # last-ditch: project_overlaps before raising
            final, _ = project_overlaps(final, benchmark)
            final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
            if final_ovl > 0:
                raise RuntimeError(f"Final has {final_ovl} overlaps")
        return final


# Match the evaluator's expected class name
Placer = E137SdfDpoLaneUnblockedPlacer
