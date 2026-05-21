"""E156: V4+Gauss descent -> fd_legalize -> CD polish.

Same pipeline as thinkorplace-v2 except the greedy spiral-search legalizer
is replaced with a force-directed (FD) legalizer. Hypothesis: FD finds a
DIFFERENT post-legalize basin from greedy. Different starting point for CD
polish => potentially lower local optimum.

Key implementation detail: thinkorplace-v2 calls `descender.place()` which
INTERNALLY runs `greedy_macro_legalize`. To swap the legalizer, we call
`descender.descend()` directly to get the raw descent output, then apply
fd_legalize ourselves.

Safety: if FD leaves residual overlaps, we fall back to the canonical chain
of project_overlaps -> greedy_macro_legalize (same safety net as v2).

Total budget: 1500s. Mirrors thinkorplace-v2.
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
    _ROOT,
    _HERE,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
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

from fd_legalize import fd_legalize  # noqa: E402


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class E156FdLegalizePlacer:
    """V4+Gauss + FD legalize + CD polish; aimed at finding a different basin."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 1500.0,
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        cd_polish_s: float = 900.0,
        cd_plateau_threshold: float = 0.001,
        fd_max_iters: int = 200,
        fd_strength: float = 2.0,
        rng_seed: int = 42,
        verbose: bool = True,
        report_greedy_basin: bool = False,
    ):
        self.budget_seconds = budget_seconds
        self.num_steps = num_steps
        self.lr_frac = lr_frac
        self.gamma_start_frac = gamma_start_frac
        self.gamma_end_frac = gamma_end_frac
        self.overlap_lambda_end = overlap_lambda_end
        self.overlap_ramp_pct = overlap_ramp_pct
        self.init = init
        self.cd_polish_s = cd_polish_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.fd_max_iters = fd_max_iters
        self.fd_strength = fd_strength
        self.rng_seed = rng_seed
        self.verbose = verbose
        self.report_greedy_basin = report_greedy_basin

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E156 fd_legalize ({benchmark.name}) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        device = _best_device()
        self._log(f"  device={device}")

        # --- Phase 1: V4+Gauss descent (RAW, no internal legalize) ---
        pos = None
        for attempt, cfg in enumerate([
            dict(overlap_lambda_end=self.overlap_lambda_end),
            dict(overlap_lambda_end=50.0),
            dict(overlap_lambda_end=100.0, num_steps=300),
        ]):
            try:
                num_steps = cfg.get("num_steps", self.num_steps)
                ovl_end = cfg["overlap_lambda_end"]
                self._log(f"  descent attempt {attempt+1}: ovl_end={ovl_end} steps={num_steps}")
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
                # Call descend() directly to skip the internal greedy legalize.
                pos_raw, descend_stats = descender.descend(benchmark, plc)
                pos_raw = pos_raw.detach().to(torch.float32)
                pre_legal_ovl = compute_overlap_metrics(pos_raw, benchmark)["overlap_count"]
                self._log(
                    f"  descent done: ovl_area={descend_stats.get('final_overlap_area', -1):.0f} "
                    f"pre-legalize ovl_count={pre_legal_ovl}"
                )

                # FD legalize
                t_fd0 = time.time()
                pos_fd, fd_stats = fd_legalize(
                    pos_raw,
                    benchmark,
                    max_iters=self.fd_max_iters,
                    strength=self.fd_strength,
                    verbose=False,
                )
                t_fd = time.time() - t_fd0
                fd_ovl = compute_overlap_metrics(pos_fd, benchmark)["overlap_count"]
                fd_proxy = float(compute_proxy_cost(pos_fd, benchmark, plc)["proxy_cost"]) if fd_ovl == 0 else float("nan")
                self._log(
                    f"  fd_legalize: init_ovl={fd_stats['init_overlaps']} "
                    f"final_ovl={fd_ovl} iters={fd_stats['n_iters']} "
                    f"max_disp={fd_stats['max_displacement']:.2f} "
                    f"wall={t_fd:.2f}s proxy={fd_proxy:.5f}"
                )

                # Comparison: greedy legalize on the SAME descent output
                # (disabled by default; can be slow on hard benches).
                if self.report_greedy_basin and attempt == 0:
                    pass  # see git history for old compare path

                if fd_ovl == 0:
                    pos = pos_fd
                    self._log(f"  attempt {attempt+1}: FD legalize succeeded (ovl=0)")
                    break

                # FD failed: try project_overlaps on FD output as quick cleanup
                self._log(f"  FD left {fd_ovl} overlaps; trying project_overlaps cleanup")
                pos_po, _ = project_overlaps(pos_fd, benchmark)
                ovl_po = compute_overlap_metrics(pos_po, benchmark)["overlap_count"]
                if ovl_po == 0:
                    pos = pos_po
                    self._log(f"  attempt {attempt+1}: project_overlaps cleaned up FD residuals")
                    break

                # FD path failed entirely: fall back to v2's standard path
                # (greedy_macro_legalize on raw descent + project_overlaps).
                self._log(f"  FD path FAILED ({fd_ovl}->{ovl_po}); falling back to v2 standard path (greedy on raw)")
                t_gr0 = time.time()
                pos_greedy_raw, gr_stats = greedy_macro_legalize(pos_raw, benchmark, verbose=False)
                t_gr = time.time() - t_gr0
                ovl_gr_raw = compute_overlap_metrics(pos_greedy_raw, benchmark)["overlap_count"]
                self._log(f"  greedy(raw): ovl={ovl_gr_raw} failed={gr_stats['n_failed']} wall={t_gr:.1f}s")
                if ovl_gr_raw > 0:
                    pos_greedy_raw, _ = project_overlaps(pos_greedy_raw, benchmark)
                    ovl_gr_raw = compute_overlap_metrics(pos_greedy_raw, benchmark)["overlap_count"]
                if ovl_gr_raw == 0:
                    pos = pos_greedy_raw
                    self._log(f"  attempt {attempt+1}: greedy(raw) safety net succeeded")
                    break

                self._log(f"  attempt {attempt+1}: still has overlaps after all legalize attempts, retrying...")
            except Exception as exc:
                self._log(f"  attempt {attempt+1} EXCEPTION: {exc!r}")
                continue

            if deadline is not None and time.time() > deadline - 60:
                break

        if pos is None:
            self._log("  fallback: SDF + project_overlaps + CD polish")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)

        descent_wall = time.time() - t0
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self._log(f"  basin: proxy={descent_proxy:.5f} wall={descent_wall:.0f}s")

        # --- Phase 2: CD polish (same as v2) ---
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
            f"total_wall={time.time()-t0:.0f}s"
        )
        if final_ovl > 0:
            raise RuntimeError(f"Final placement has {final_ovl} overlaps")
        return final


# Harness alias
Placer = E156FdLegalizePlacer
