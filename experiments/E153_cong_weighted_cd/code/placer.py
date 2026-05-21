"""E153 congestion-weighted CD second pass placer.

Pipeline:
  1. V4+Gaussian descent (same as thinkorplace-v2 basin)
  2. CD-A: canonical weights {wl:1, d:0.5, c:0.5} polish (hard_cap 400s)
  3. CD-B: cong-weighted basin nudge {wl:1, d:0.5, c:1.5} (3× canonical, 150s)
  4. CD-C: canonical re-polish (200s)

Hypothesis: The cong-weighted pass biases the search into a different
basin, and re-polishing under canonical weights preserves the lift.
Mechanism verified on M3 ibm04: -2.09% vs canonical-only CD.

Total budget: ~1500s (matches thinkorplace-v2).

Important: switches weights by mutating `evaluator.weights` between
run_cd_adaptive calls (weights dict is mutable on the evaluator).
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


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Canonical proxy weights (matches IncrementalProxyEvaluator default).
CANONICAL_W = {"wirelength": 1.0, "density": 0.5, "congestion": 0.5}


class E153CongWeightedPlacer:
    """V4 + Gaussian + 3-phase CD (A canonical → B cong×3 → C canonical)."""

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
        cd_a_s: float = 400.0,
        cd_b_s: float = 150.0,
        cd_c_s: float = 200.0,
        cong_weight_scale: float = 3.0,
        cd_plateau_threshold: float = 0.001,
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
        self.cd_a_s = cd_a_s
        self.cd_b_s = cd_b_s
        self.cd_c_s = cd_c_s
        self.cong_weight_scale = cong_weight_scale
        self.cd_plateau_threshold = cd_plateau_threshold
        self.rng_seed = rng_seed
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E153CongWeightedPlacer ({benchmark.name}) ===")

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        device = _best_device()
        self._log(f"  device={device}")

        # ── Descent (same robust ladder as thinkorplace-v2) ──
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
                pos_try = descender.place(benchmark)
                ovl_try = compute_overlap_metrics(pos_try, benchmark)["overlap_count"]
                if ovl_try == 0:
                    pos = pos_try
                    self._log(f"  descent attempt {attempt+1}: ovl=0")
                    break
                pos_try2, _ = project_overlaps(pos_try, benchmark)
                if compute_overlap_metrics(pos_try2, benchmark)["overlap_count"] == 0:
                    pos = pos_try2
                    self._log(f"  descent attempt {attempt+1}: ovl=0 after project_overlaps")
                    break
                self._log(f"  descent attempt {attempt+1}: still {ovl_try} overlaps, retrying...")
            except Exception as exc:
                self._log(f"  descent attempt {attempt+1} EXCEPTION: {exc}")
                continue

            if deadline is not None and time.time() > deadline - 60:
                break

        if pos is None:
            self._log("  fallback: SDF + project_overlaps")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)

        descent_wall = time.time() - t0
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self._log(f"  basin: proxy={descent_proxy:.5f} wall={descent_wall:.0f}s")

        # ── Budget allocation for the 3-phase CD ──
        # Reserve 30s safety margin; scale phase budgets to fit remaining time.
        if deadline is not None:
            remaining = deadline - time.time() - 30.0
            requested = self.cd_a_s + self.cd_b_s + self.cd_c_s
            if remaining < requested:
                scale = max(0.2, remaining / requested)
                cd_a = self.cd_a_s * scale
                cd_b = self.cd_b_s * scale
                cd_c = self.cd_c_s * scale
            else:
                cd_a = self.cd_a_s
                cd_b = self.cd_b_s
                cd_c = self.cd_c_s
        else:
            cd_a = self.cd_a_s
            cd_b = self.cd_b_s
            cd_c = self.cd_c_s
        cd_a = max(30.0, cd_a)
        cd_b = max(30.0, cd_b)
        cd_c = max(30.0, cd_c)
        self._log(f"  CD budgets: A={cd_a:.0f}s B={cd_b:.0f}s C={cd_c:.0f}s scale={self.cong_weight_scale:.1f}")

        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]

        # ── CD-A: canonical weights ──
        evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
        # Ensure canonical weights for A (in case evaluator defaults change).
        evaluator.weights = dict(CANONICAL_W)
        t_a = time.time()
        run_cd_adaptive(
            evaluator, benchmark, plc, movable,
            min_time_s=cd_a * 0.5,
            hard_cap_s=cd_a,
            patience=5,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=None,
        )
        posA = evaluator.placement.detach().clone().to(torch.float32)
        proxyA = float(compute_proxy_cost(posA, benchmark, plc)["proxy_cost"])
        detA = compute_proxy_cost(posA, benchmark, plc)
        self._log(
            f"  CD-A canonical: proxy={proxyA:.5f} wl={detA['wirelength_cost']:.4f} "
            f"d={detA['density_cost']:.4f} c={detA['congestion_cost']:.4f} "
            f"wall={time.time()-t_a:.0f}s"
        )

        # ── CD-B: cong-weighted nudge (mutate weights in place) ──
        evaluator.weights = {
            "wirelength": 1.0,
            "density": 0.5,
            "congestion": 0.5 * self.cong_weight_scale,
        }
        # Reset internal cost cache so first sweep recomputes under new weights.
        # IncrementalProxyEvaluator tracks current_cost from accumulated deltas;
        # safest path is to re-init the evaluator on the same placement.
        evaluator = IncrementalProxyEvaluator(benchmark, plc, posA.clone())
        evaluator.weights = {
            "wirelength": 1.0,
            "density": 0.5,
            "congestion": 0.5 * self.cong_weight_scale,
        }
        t_b = time.time()
        run_cd_adaptive(
            evaluator, benchmark, plc, movable,
            min_time_s=cd_b * 0.5,
            hard_cap_s=cd_b,
            patience=5,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=None,
        )
        posB = evaluator.placement.detach().clone().to(torch.float32)
        detB = compute_proxy_cost(posB, benchmark, plc)
        proxyB = float(detB["proxy_cost"])
        delta_b = (proxyB - proxyA) / proxyA * 100.0
        self._log(
            f"  CD-B cong×{self.cong_weight_scale:.1f} (canonical recomputed): "
            f"proxy={proxyB:.5f} c={detB['congestion_cost']:.4f} "
            f"Δ_vs_A={delta_b:+.2f}% wall={time.time()-t_b:.0f}s"
        )

        # ── CD-C: canonical re-polish ──
        evaluator = IncrementalProxyEvaluator(benchmark, plc, posB.clone())
        evaluator.weights = dict(CANONICAL_W)
        t_c = time.time()
        run_cd_adaptive(
            evaluator, benchmark, plc, movable,
            min_time_s=cd_c * 0.5,
            hard_cap_s=cd_c,
            patience=5,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=None,
        )
        posC = evaluator.placement.detach().clone().to(torch.float32)
        detC = compute_proxy_cost(posC, benchmark, plc)
        proxyC = float(detC["proxy_cost"])
        delta_c = (proxyC - proxyA) / proxyA * 100.0
        self._log(
            f"  CD-C canonical re-polish: proxy={proxyC:.5f} "
            f"wl={detC['wirelength_cost']:.4f} d={detC['density_cost']:.4f} "
            f"c={detC['congestion_cost']:.4f} "
            f"Δ_vs_A={delta_c:+.2f}% wall={time.time()-t_c:.0f}s"
        )

        # ── Safety: pick best of {posA, posC} under canonical objective ──
        # The mechanism preserves lift in expectation, but for any individual
        # bench we want to never regress vs canonical-only.
        if proxyC <= proxyA:
            final = posC
            final_proxy = proxyC
            self._log(f"  PICK C ({proxyC:.5f} <= {proxyA:.5f})")
        else:
            final = posA
            final_proxy = proxyA
            self._log(f"  PICK A ({proxyA:.5f} < {proxyC:.5f}) — cong-weighted nudge regressed")

        final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
        self._log(
            f"  Final: proxy={final_proxy:.5f} ovl={final_ovl} "
            f"total_wall={time.time()-t0:.0f}s"
        )
        if final_ovl > 0:
            raise RuntimeError(f"Final placement has {final_ovl} overlaps")
        return final


# Alias for evaluate.py convention (it scans for any callable `Placer`).
Placer = E153CongWeightedPlacer
