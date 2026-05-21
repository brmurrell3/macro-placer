"""E134 — Pruned multi-restart placer.

Pipeline per bench:
  Phase A: 5 seeds × 150 steps of V4+Gaussian descent (serial); record smooth.
  Prune: sort by smooth, keep top-2.
  Phase B: top-2 resume to 500 total steps each.
  Phase C: legalize both, score canonical proxy, pick best, CD polish.

Budget: 720s default. Phase A ~ 150s (5 × 30s), Phase B ~ 140s (2 × 70s),
legalize ~10s, CD polish gets remaining ~420s.
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
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
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

from batched_descender import CheckpointableV4Gauss
from macro_legalizer import greedy_macro_legalize


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class E134PrunedRestartPlacer:
    """thinkorplace-v2 with pruned multi-restart.

    5 seeds run 150 steps each (Phase A); 2 best resume to 500 total
    (Phase B). Best canonical-proxy basin → CD polish.
    """

    def __init__(
        self,
        budget_seconds: Optional[float] = 720.0,
        seeds: Optional[List[int]] = None,
        keep_top_k: int = 2,
        partial_steps: int = 150,
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        cd_polish_s: float = 400.0,
        cd_plateau_threshold: float = 0.001,
        legalize_radius_steps: int = 80,
        legalize_step_frac: float = 0.02,
        verbose: bool = True,
    ):
        self.budget_seconds = budget_seconds
        self.seeds = seeds or [42, 142, 242, 342, 442]
        self.keep_top_k = keep_top_k
        self.partial_steps = partial_steps
        self.num_steps = num_steps
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.init = init
        self.cd_polish_s = cd_polish_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.legalize_radius_steps = legalize_radius_steps
        self.legalize_step_frac = legalize_step_frac
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _make_descender(self, seed: int, device: str) -> CheckpointableV4Gauss:
        return CheckpointableV4Gauss(
            num_steps=self.num_steps,
            lr_frac=self.lr_frac,
            gamma_start_frac=self.gamma_start_frac,
            gamma_end_frac=self.gamma_end_frac,
            overlap_lambda_end=self.overlap_lambda_end,
            overlap_ramp_pct=self.overlap_ramp_pct,
            init=self.init,
            device=device,
            rng_seed=seed,
            verbose=False,
            legalize_radius_steps=self.legalize_radius_steps,
            legalize_step_frac=self.legalize_step_frac,
        )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_global = time.time()
        deadline = t_global + self.budget_seconds if self.budget_seconds else None
        self._log(
            f"=== E134 pruned-restart ({benchmark.name}) "
            f"n_seeds={len(self.seeds)} keep_top={self.keep_top_k} "
            f"partial={self.partial_steps}/{self.num_steps} ==="
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        device = _best_device()
        self._log(f"  device={device}  budget={self.budget_seconds}s")

        # ---- Phase A: 5 seeds × partial_steps ----
        phase_a_t0 = time.time()
        partial_states: List[Tuple[float, int, dict, CheckpointableV4Gauss]] = []
        for seed in self.seeds:
            t_seed = time.time()
            try:
                desc = self._make_descender(seed, device)
                state = desc.descend_partial(benchmark, plc, self.partial_steps)
                wall = time.time() - t_seed
                self._log(
                    f"  [A] seed {seed:3d}: smooth={state['smooth_score']:.5f} "
                    f"ovl_area={state['overlap_area']:.0f} wall={wall:.0f}s"
                )
                partial_states.append((state["smooth_score"], seed, state, desc))
            except Exception as exc:
                wall = time.time() - t_seed
                self._log(f"  [A] seed {seed:3d}: EXCEPTION {exc} wall={wall:.0f}s")
            # Safety: if running out of time, bail to Phase C with what we have
            if deadline is not None and time.time() > deadline - 200:
                self._log(f"  [A] budget tight after {len(partial_states)} seeds; abort Phase A")
                break

        if not partial_states:
            self._log("  All Phase A seeds failed; SDF fallback")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)
            return self._cd_polish_and_return(
                pos, benchmark, plc, deadline, t_global
            )

        # ---- Prune: keep top-K by smooth score ----
        partial_states.sort(key=lambda r: r[0])
        kept = partial_states[: self.keep_top_k]
        killed = partial_states[self.keep_top_k :]
        self._log(
            f"  [PRUNE] kept seeds {[s for _, s, _, _ in kept]} "
            f"killed {[s for _, s, _, _ in killed]} "
            f"(Phase A wall={time.time()-phase_a_t0:.0f}s)"
        )

        # ---- Phase B: top-K resume to num_steps ----
        phase_b_t0 = time.time()
        finished: List[Tuple[float, int, torch.Tensor]] = []  # (canonical, seed, legal_pos)
        for smooth_a, seed, state, desc in kept:
            t_seed = time.time()
            try:
                pos_raw, stats = desc.descend_resume(state)
                wall_b = time.time() - t_seed
                ovl_pre = compute_overlap_metrics(pos_raw, benchmark)["overlap_count"]
                # Legalize using V4's strategy
                legal_pos = self._legalize(pos_raw, benchmark)
                ovl_post = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
                if ovl_post > 0:
                    self._log(
                        f"  [B] seed {seed:3d}: smooth={stats['final_smooth']:.5f} "
                        f"ovl_pre={ovl_pre} ovl_post={ovl_post} SKIP"
                    )
                    continue
                canonical = float(
                    compute_proxy_cost(legal_pos, benchmark, plc)["proxy_cost"]
                )
                self._log(
                    f"  [B] seed {seed:3d}: smooth={stats['final_smooth']:.5f} "
                    f"canonical={canonical:.5f} ovl_pre={ovl_pre} ovl_post={ovl_post} "
                    f"wall_b={wall_b:.0f}s"
                )
                finished.append((canonical, seed, legal_pos))
            except Exception as exc:
                self._log(f"  [B] seed {seed:3d}: EXCEPTION {exc}")
            # Free GPU mem of completed state
            try:
                del state["positions"]
                del state["optimizer"]
                del state["proxy"]
                del state["fixed_mask"]
            except Exception:
                pass
            if device != "cpu":
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        if not finished:
            self._log("  All Phase B seeds left overlaps; falling back to best partial")
            # Fall back: legalize from best partial pos via project_overlaps
            best_smooth, best_seed, best_state, _ = kept[0]
            pos_raw = best_state["positions"].detach().cpu()
            legal_pos, _ = project_overlaps(pos_raw, benchmark)
            ovl = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
            if ovl > 0:
                self._log("  Fallback also has overlaps; SDF fallback")
                legal_pos = sdf_init(benchmark)
                legal_pos, _ = project_overlaps(legal_pos, benchmark)
            return self._cd_polish_and_return(
                legal_pos, benchmark, plc, deadline, t_global
            )

        finished.sort(key=lambda r: r[0])
        best_canonical, best_seed, best_pos = finished[0]
        self._log(
            f"  [PICK] best seed {best_seed}: canonical={best_canonical:.5f} "
            f"(Phase B wall={time.time()-phase_b_t0:.0f}s)"
        )

        return self._cd_polish_and_return(
            best_pos, benchmark, plc, deadline, t_global
        )

    def _legalize(self, positions: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        legal_pos, _ = greedy_macro_legalize(
            positions, benchmark,
            search_radius_steps=self.legalize_radius_steps,
            step_size_frac=self.legalize_step_frac,
            verbose=False,
        )
        ovl = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        if ovl > 0:
            legal_pos, _ = project_overlaps(legal_pos, benchmark)
        return legal_pos

    def _cd_polish_and_return(
        self,
        pos: torch.Tensor,
        benchmark: Benchmark,
        plc,
        deadline: Optional[float],
        t_global: float,
    ) -> torch.Tensor:
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd_budget = max(30.0, min(remaining, self.cd_polish_s))
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
            f"total_wall={time.time()-t_global:.0f}s"
        )
        if final_ovl > 0:
            raise RuntimeError(f"Final has {final_ovl} overlaps")
        return final
