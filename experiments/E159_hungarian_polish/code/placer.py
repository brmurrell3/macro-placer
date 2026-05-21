"""E158 — v2-extCD pipeline + K=50 Hungarian joint-permutation polish phase.

Hypothesis: v2-extCD's post-CD basin is plateau-saturated under single-macro,
pair-swap, and N-macro independent perturbations (E139/E141/E142/E149/E151
all falsified at noise floor). It is **NOT** saturated under joint K-macro
permutations — there is no published placer in this repo that has explored
the topology of `swap which-macro-goes-to-which-slot` on this basin.

The Hungarian primitive (E67/E70 archive) solves the K × n_slots rectangular
assignment problem in O(K^2 · n_slots) via scipy. It returns a permutation
that minimizes the sum of per-(macro, slot) proxy deltas, where the deltas
are computed via the IncrementalProxyEvaluator (E1, 4657× speedup).

Pipeline:
  1. V4 + Gaussian descent → legalize → project_overlaps (v2 basin).
  2. CD polish 1 (~600 s) — same as v2-extCD.
  3. **Hungarian polish loop** (~300 s):
       - rotate cluster_seed each iter for cluster diversification
       - K=50 cluster + n_slots=100 candidates
       - sequential commit with per-move legality
       - early stop on 15-rejection streak
  4. CD polish 2 (~300 s).
  5. Validate zero overlaps.

Total ~1500 s/bench, within partcl 60-min cap.

The Hungarian phase is purely additive: per-step revert on no-improvement
or overlap. Worst case, the output is identical to v2 sans the Hungarian
phase, modulo the budget shape (CD1 + CD2 ~900s vs v2's 900s monolithic).
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
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E127_v4_gaussian" / "code",
    _ROOT / "experiments" / "_archive" / "E67_kjoint_hungarian" / "code",
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
from kjoint_hungarian import kjoint_hungarian_step  # noqa: E402


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run_cd_polish(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    budget_s: float,
    plateau_threshold: float = 0.001,
    log_fn=None,
) -> torch.Tensor:
    """Run CD-adaptive polish on `placement`. Returns float32 placement."""
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
        log_fn=log_fn,
    )
    return evaluator.placement.detach().clone().to(torch.float32)


def _run_hungarian_loop(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    budget_s: float,
    k: int = 50,
    n_slots: int = 100,
    reject_patience: int = 15,
    seed_base: int = 42,
    log_fn=None,
) -> tuple[torch.Tensor, dict]:
    """Iterate K=50 Hungarian permutation steps until budget or saturation.

    Returns (best_placement, stats). Each iter advances cluster_seed to
    diversify which K-macro cluster is being re-permuted.
    """
    n_hard = benchmark.num_hard_macros
    fixed = benchmark.macro_fixed.cpu().numpy()
    n_movable = sum(1 for i in range(n_hard) if not bool(fixed[i]))
    k_eff = min(k, n_movable)

    placement_f64 = placement.detach().clone().to(torch.float64)
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
    baseline = evaluator.current_cost()["proxy"]
    best_proxy = baseline
    best_placement = evaluator.placement.detach().clone()
    best_iter = -1

    t_start = time.perf_counter()
    iters = 0
    accepts = 0
    reject_streak = 0
    iter_info_log = []
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= budget_s:
            stop_reason = "budget"
            break
        if reject_streak >= reject_patience:
            stop_reason = "saturation"
            break
        iters += 1
        cluster_seed = seed_base + iters
        iter_t0 = time.perf_counter()
        new_placement, info = kjoint_hungarian_step(
            benchmark, evaluator.placement, plc,
            evaluator=evaluator,
            k=k_eff, n_slots=n_slots,
            mode="adjacency",
            commit_mode="sequential",
            cluster_seed=cluster_seed,
        )
        iter_wall = time.perf_counter() - iter_t0
        accepted = bool(info.get("accepted", False))
        proxy_after = info.get("commit_proxy_after", info.get("proxy_baseline"))
        iter_info_log.append({
            "iter": iters,
            "cluster_seed": cluster_seed,
            "accepted": accepted,
            "proxy_before": info.get("proxy_baseline"),
            "proxy_after": proxy_after,
            "n_moved": info.get("commit_n_moved", 0),
            "reason": info.get("commit_reason", "?"),
            "wall_s": iter_wall,
            "cost_finite_frac": info.get("cost_finite_frac"),
        })
        if accepted:
            accepts += 1
            reject_streak = 0
            cur = evaluator.current_cost()["proxy"]
            if cur < best_proxy - 1e-9:
                best_proxy = cur
                best_placement = evaluator.placement.detach().clone()
                best_iter = iters
        else:
            reject_streak += 1

        if log_fn is not None and (iters <= 3 or iters % 5 == 0):
            log_fn(
                f"  H iter {iters}: seed={cluster_seed} acc={accepted} "
                f"reason={info.get('commit_reason')} "
                f"proxy {info.get('proxy_baseline'):.5f}->{proxy_after if proxy_after is not None else float('nan'):.5f} "
                f"wall={iter_wall:.1f}s elapsed={elapsed:.0f}s streak={reject_streak}"
            )

    stats = {
        "iters": iters,
        "accepts": accepts,
        "best_proxy": best_proxy,
        "baseline_proxy": baseline,
        "delta": best_proxy - baseline,
        "wall_s": time.perf_counter() - t_start,
        "stop_reason": stop_reason,
        "iter_log_head": iter_info_log[:5],
        "iter_log_tail": iter_info_log[-3:] if len(iter_info_log) > 3 else [],
    }
    return best_placement.to(torch.float32), stats


class E158HungarianPolishPlacer:
    """v2-extCD basin + CD1 + Hungarian K=50 + CD2 polish."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 2400.0,
        # Descent params (mirror v2)
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        # CD params — sum CD1 + CD2 = 1200s ≥ v2-extCD's 900s
        cd1_budget_s: float = 600.0,
        cd2_budget_s: float = 600.0,
        cd_plateau_threshold: float = 0.001,
        # Hungarian params
        hung_budget_s: float = 300.0,
        hung_k: int = 50,
        hung_n_slots: int = 100,
        hung_reject_patience: int = 15,
        hung_seed_base: int = 42,
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
        self.hung_budget_s = hung_budget_s
        self.hung_k = hung_k
        self.hung_n_slots = hung_n_slots
        self.hung_reject_patience = hung_reject_patience
        self.hung_seed_base = hung_seed_base
        self.rng_seed = rng_seed
        self.verbose = verbose
        self.last_phase_walls: dict = {}
        self.last_phase_proxies: dict = {}
        self.last_hung_stats: dict = {}

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E158HungarianPolishPlacer ({benchmark.name}) ===")
        self._log(
            f"  budget={self.budget_seconds}s: descent + CD1={self.cd1_budget_s}s "
            f"+ Hungarian={self.hung_budget_s}s + CD2={self.cd2_budget_s}s"
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        device = _best_device()
        self._log(f"  device={device}")

        # ── Phase 1: V4 + Gaussian descent ────────────────────────────────
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
                    self._log(f"  descent attempt {attempt+1}: ovl=0 after project")
                    break
                self._log(f"  descent attempt {attempt+1}: still {ovl_try} ovl, retrying...")
            except Exception as exc:
                self._log(f"  descent attempt {attempt+1} EXC: {exc}")
                continue
            if deadline is not None and time.time() > deadline - 120:
                break

        if pos is None:
            self._log("  fallback: SDF + project_overlaps")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)

        descent_wall = time.time() - t_descent
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self.last_phase_walls["descent"] = descent_wall
        self.last_phase_proxies["descent"] = descent_proxy
        self._log(f"  Phase 1 (descent): proxy={descent_proxy:.5f} wall={descent_wall:.0f}s")

        # ── Phase 2: CD polish 1 ──────────────────────────────────────────
        t_cd1 = time.time()
        # budget bookkeeping
        if deadline is not None:
            remaining = deadline - time.time() - 30.0
            # split: cd1 + hung + cd2 must fit in remaining
            total_planned = self.cd1_budget_s + self.hung_budget_s + self.cd2_budget_s
            if total_planned > remaining:
                # scale down proportionally
                scale = max(0.1, remaining / total_planned)
                cd1_b = max(30.0, self.cd1_budget_s * scale)
                hung_b = max(30.0, self.hung_budget_s * scale)
                cd2_b = max(30.0, self.cd2_budget_s * scale)
            else:
                cd1_b = self.cd1_budget_s
                hung_b = self.hung_budget_s
                cd2_b = self.cd2_budget_s
        else:
            cd1_b = self.cd1_budget_s
            hung_b = self.hung_budget_s
            cd2_b = self.cd2_budget_s

        self._log(f"  Phase 2 (CD1) budget={cd1_b:.0f}s")
        pos = _run_cd_polish(pos, benchmark, plc, cd1_b, self.cd_plateau_threshold)
        cd1_wall = time.time() - t_cd1
        cd1_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self.last_phase_walls["cd1"] = cd1_wall
        self.last_phase_proxies["cd1"] = cd1_proxy
        self._log(f"  Phase 2 (CD1): proxy={cd1_proxy:.5f} wall={cd1_wall:.0f}s")

        # ── Phase 3: Hungarian polish loop ────────────────────────────────
        t_hung = time.time()
        self._log(f"  Phase 3 (Hungarian) budget={hung_b:.0f}s K={self.hung_k} n_slots={self.hung_n_slots}")
        pos_post_hung, hung_stats = _run_hungarian_loop(
            pos, benchmark, plc, budget_s=hung_b,
            k=self.hung_k, n_slots=self.hung_n_slots,
            reject_patience=self.hung_reject_patience,
            seed_base=self.hung_seed_base,
            log_fn=self._log,
        )
        hung_wall = time.time() - t_hung
        post_hung_proxy = float(compute_proxy_cost(pos_post_hung, benchmark, plc)["proxy_cost"])
        post_hung_ovl = compute_overlap_metrics(pos_post_hung, benchmark)["overlap_count"]
        self.last_phase_walls["hungarian"] = hung_wall
        self.last_phase_proxies["hungarian"] = post_hung_proxy
        self.last_hung_stats = hung_stats
        self._log(
            f"  Phase 3 (Hungarian): proxy={post_hung_proxy:.5f} ovl={post_hung_ovl} "
            f"wall={hung_wall:.0f}s iters={hung_stats['iters']} acc={hung_stats['accepts']} "
            f"stop={hung_stats['stop_reason']} delta={hung_stats['delta']:+.6f}"
        )

        # Defensive: if Hungarian produced overlaps (shouldn't — per-move
        # legality guard), revert to CD1 placement.
        if post_hung_ovl > 0:
            self._log(f"  WARN: Hungarian produced {post_hung_ovl} overlaps; reverting to CD1")
            pos_post_hung = pos
            post_hung_proxy = cd1_proxy

        # Hungarian's incremental view can drift from canonical (E1 evaluator
        # accumulates float noise across many move/revert cycles). Even when
        # post_hung_proxy ≈ cd1_proxy in canonical view, the placement is
        # different — CD2 starting from a perturbed-but-equivalent state may
        # find a structurally distinct basin. ALWAYS pass post-Hungarian to
        # CD2; only revert if Hungarian REGRESSED MEANINGFULLY (>0.5%).
        regression_tol = 0.005 * cd1_proxy
        if post_hung_proxy > cd1_proxy + regression_tol:
            self._log(
                f"  WARN: Hungarian cumulative {post_hung_proxy:.5f} > CD1 {cd1_proxy:.5f} "
                f"by >{regression_tol:.5f}; reverting to CD1 placement"
            )
            pos_for_cd2 = pos
        else:
            self._log(
                f"  Passing Hungarian placement to CD2 (post_hung={post_hung_proxy:.5f} "
                f"~= CD1 {cd1_proxy:.5f}; canonical delta {post_hung_proxy-cd1_proxy:+.6f})"
            )
            pos_for_cd2 = pos_post_hung

        # ── Phase 4: CD polish 2 ──────────────────────────────────────────
        t_cd2 = time.time()
        self._log(f"  Phase 4 (CD2) budget={cd2_b:.0f}s")
        pos_final = _run_cd_polish(pos_for_cd2, benchmark, plc, cd2_b, self.cd_plateau_threshold)
        cd2_wall = time.time() - t_cd2
        cd2_proxy = float(compute_proxy_cost(pos_final, benchmark, plc)["proxy_cost"])
        cd2_ovl = compute_overlap_metrics(pos_final, benchmark)["overlap_count"]
        self.last_phase_walls["cd2"] = cd2_wall
        self.last_phase_proxies["cd2"] = cd2_proxy
        self._log(f"  Phase 4 (CD2): proxy={cd2_proxy:.5f} ovl={cd2_ovl} wall={cd2_wall:.0f}s")

        # ── Final safety: pick best of {cd1, post_hung, cd2} ──────────────
        candidates = [
            ("cd1", cd1_proxy, pos),
            ("post_hung", post_hung_proxy, pos_post_hung),
            ("cd2", cd2_proxy, pos_final),
        ]
        # Filter out overlap-positive (cd1 and cd2 are CD outputs so ovl=0
        # already; post_hung was reverted above if ovl>0).
        candidates = [
            (name, p, t) for (name, p, t) in candidates
            if compute_overlap_metrics(t, benchmark)["overlap_count"] == 0
        ]
        best_name, best_proxy, best_pos = min(candidates, key=lambda x: x[1])
        total_wall = time.time() - t0
        self._log(
            f"  Final: pick={best_name} proxy={best_proxy:.5f} "
            f"total_wall={total_wall:.0f}s"
        )

        final_ovl = compute_overlap_metrics(best_pos, benchmark)["overlap_count"]
        if final_ovl > 0:
            raise RuntimeError(f"Final placement has {final_ovl} overlaps")
        return best_pos


# Alias expected by evaluation harness (same convention as v2)
Placer = E158HungarianPolishPlacer
