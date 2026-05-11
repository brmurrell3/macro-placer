"""E70 — E48 hybrid with K=50 Hungarian multi-step polish on the E41 lane.

The E48 champion (`submissions/cd_lns_sa_hybrid/placer.py`) runs E25
and E41 lanes in parallel and returns `min` by proxy. E70 keeps the
hybrid structure but appends a 5th phase to the E41 lane only:

  E25 lane: SDF init  → CD → LNS → SA-v2                              [unchanged]
  E41 lane: DPO init  → CD → LNS → SA-v2 → K-joint K=3 → K-joint Hungarian K=50-multi

The Hungarian polish is the loop verified in
`experiments/E67_kjoint_hungarian/`:
  - cluster_seed = 0..N: random sample of K=50 from top-(K · 3) by
    netlist adjacency (E39 `_adjacency_scores`); diversification per
    step. RNG is per-step seeded for reproducibility.
  - sequential commit (V2): apply moves in priority order with per-move
    legality check against the *committed* state; this avoids the V1
    pairwise-blindness that caused 16-overlap reverts.
  - Early stop on 15 consecutive rejections OR 600 s wall budget.

E25 lane is untouched because the smokes showed it has no plateau slack
on its winners (ibm01 etc.) — top-adjacency macros are already at SA-
optimal positions after the no-K-joint pipeline.

Reference:
  - E67 module: `experiments/E67_kjoint_hungarian/code/kjoint_hungarian.py`
  - E67 manifest + notes: `experiments/E67_kjoint_hungarian/{manifest,notes}.md`
  - E41 parent: `experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`
  - E48 hybrid parent: `submissions/cd_lns_sa_hybrid/placer.py`
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# E25 lane (unchanged — load via importlib so the submission's class name
# survives even if the file is later renamed).
import importlib.util
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer

# E41 inner placer (DPO + CD + LNS + SA + K-joint K=3).
from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import (
    CDLNSSADPOKJointPlacer,
)

# E67 Hungarian module.
_E67_PATH = _ROOT / "experiments" / "E67_kjoint_hungarian" / "code" / "kjoint_hungarian.py"
_E67_SPEC = importlib.util.spec_from_file_location("e67_kjoint_hungarian", str(_E67_PATH))
_E67_MOD = importlib.util.module_from_spec(_E67_SPEC)
_E67_SPEC.loader.exec_module(_E67_MOD)
kjoint_hungarian_step = _E67_MOD.kjoint_hungarian_step


# ── Harness entry point ────────────────────────────────────────────────────
# evaluate.py picks the first class in vars(mod).values() with a .place()
# method. Lazy delegation so the impl can sit lower in the file alongside
# its dependencies.

class CDLNSSAHybridHungarianPlacer:
    """E70 hybrid (entry point).

    Delegates to `_CDLNSSAHybridHungarianImpl` defined below. Per-bench
    winner = min(E25_proxy, E41-with-Hungarian_proxy) on zero-overlap
    outputs. No per-benchmark tuning.
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def place(self, benchmark):
        return _CDLNSSAHybridHungarianImpl(**self._kwargs).place(benchmark)


# ── E41 + Hungarian polish ─────────────────────────────────────────────────


class CDLNSSADPOKJointHungarianPlacer:
    """E41 lane + K=50 Hungarian multi-step polish (post-K-joint K=3).

    Composes by inner-instance: instantiates `CDLNSSADPOKJointPlacer`
    with its production-default kwargs, runs it to completion, then
    builds a fresh evaluator on the post-K-joint placement and runs the
    Hungarian loop until saturation or budget.
    """

    def __init__(
        self,
        # Hungarian phase params (E70's additions — defaults from E67 smokes).
        hungarian_K: int = 50,
        hungarian_n_slots: int = 100,
        hungarian_budget_s: float = 600.0,
        hungarian_max_steps: int = 200,
        hungarian_rejection_streak_max: int = 15,
        hungarian_mode: str = "adjacency",
        hungarian_bbox_pad_frac: float = 0.10,
        verbose: bool = True,
        # E41 inner kwargs forwarded.
        **e41_kwargs,
    ):
        self.hungarian_K = int(hungarian_K)
        self.hungarian_n_slots = int(hungarian_n_slots)
        self.hungarian_budget_s = float(hungarian_budget_s)
        self.hungarian_max_steps = int(hungarian_max_steps)
        self.hungarian_rejection_streak_max = int(hungarian_rejection_streak_max)
        self.hungarian_mode = str(hungarian_mode)
        self.hungarian_bbox_pad_frac = float(hungarian_bbox_pad_frac)
        self.verbose = bool(verbose)

        e41_kwargs.setdefault("verbose", self.verbose)
        self._inner = CDLNSSADPOKJointPlacer(**e41_kwargs)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # 1. Run E41 inner pipeline (returns float32 [N, 2] with fixed restored).
        t_outer = time.perf_counter()
        placement = self._inner.place(benchmark)

        # 2. Build a fresh evaluator on the post-K-joint placement (float64
        # to match the evaluator's internal precision).
        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        placement_f64 = placement.detach().clone().to(torch.float64)
        evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
        baseline_proxy = evaluator.current_cost()["proxy"]

        self._log(
            f"  starting Hungarian K={self.hungarian_K} multi-step phase "
            f"(budget={self.hungarian_budget_s:.0f}s, "
            f"max_steps={self.hungarian_max_steps}, "
            f"rejection_streak_max={self.hungarian_rejection_streak_max})"
        )
        self._log(f"  Hungarian baseline proxy (post-K-joint): {baseline_proxy:.5f}")

        t_hungarian0 = time.perf_counter()
        accept_count = 0
        move_count = 0
        skipped_illegal = 0
        rejection_streak = 0
        steps_run = 0

        for seed in range(self.hungarian_max_steps):
            elapsed = time.perf_counter() - t_hungarian0
            if elapsed > self.hungarian_budget_s:
                self._log(
                    f"  Hungarian budget exhausted at step {seed} "
                    f"(elapsed {elapsed:.1f}s > {self.hungarian_budget_s:.1f}s)"
                )
                break
            new_pl, info = kjoint_hungarian_step(
                benchmark=benchmark,
                placement=placement_f64,
                plc=plc,
                evaluator=evaluator,
                k=self.hungarian_K,
                n_slots=self.hungarian_n_slots,
                mode=self.hungarian_mode,
                bbox_pad_frac=self.hungarian_bbox_pad_frac,
                commit_mode="sequential",
                cluster_seed=seed,
            )
            steps_run = seed + 1
            if info["accepted"]:
                accept_count += 1
                move_count += int(info.get("commit_n_moved", 0))
                placement_f64 = new_pl
                rejection_streak = 0
            else:
                rejection_streak += 1
            skipped_illegal += int(info.get("commit_n_skipped_illegal", 0))

            if rejection_streak >= self.hungarian_rejection_streak_max:
                self._log(
                    f"  Hungarian early-stop at step {steps_run}: "
                    f"{rejection_streak} consecutive rejections"
                )
                break

        hungarian_wall = time.perf_counter() - t_hungarian0
        post_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  Hungarian done: steps={steps_run}, accepts={accept_count}, "
            f"moves={move_count}, skipped_illegal={skipped_illegal}, "
            f"Δ={post_proxy - baseline_proxy:+.5f} "
            f"(baseline={baseline_proxy:.5f} → post={post_proxy:.5f}), "
            f"wall={hungarian_wall:.1f}s"
        )

        # 3. Restore fixed macros and return float32 (mirror E41's contract).
        final_f64 = placement_f64.clone()
        fixed_mask = benchmark.macro_fixed
        if fixed_mask.any():
            original_positions = benchmark.macro_positions.to(torch.float64)
            final_f64[fixed_mask] = original_positions[fixed_mask]
        final_placement = final_f64.to(torch.float32)

        # Defensive validate: any overlap from the Hungarian phase escaping
        # past compute_overlap_metrics's defensive revert would be a bug.
        ov = compute_overlap_metrics(final_placement, benchmark)
        if ov["overlap_count"] > 0:
            raise RuntimeError(
                f"E70 (E41+Hungarian) produced {ov['overlap_count']} overlaps "
                f"on '{benchmark.name}' — Hungarian commit revert path failed?"
            )

        self._log(
            f"  E70 E41+Hungarian total wall: {time.perf_counter() - t_outer:.1f} s"
        )
        return final_placement


# ── Hybrid impl ────────────────────────────────────────────────────────────


class _CDLNSSAHybridHungarianImpl:
    """E70 hybrid impl: E25 (unchanged) + E41-with-Hungarian, return min."""

    def __init__(self, **kwargs):
        # Split kwargs between lanes. Anything starting with `kjoint_` or
        # `hungarian_` is E41-only; the rest goes to both lanes (matches
        # the E48 hybrid pattern).
        e41_only_keys = {
            "kjoint_K", "kjoint_top_N", "kjoint_budget_s", "kjoint_seed",
            "hungarian_K", "hungarian_n_slots", "hungarian_budget_s",
            "hungarian_max_steps", "hungarian_rejection_streak_max",
            "hungarian_mode", "hungarian_bbox_pad_frac",
        }
        e41_kwargs = {k: v for k, v in kwargs.items() if k in e41_only_keys}
        common_kwargs = {k: v for k, v in kwargs.items() if k not in e41_only_keys}
        self._e25 = CDLNSSAPlacer(**common_kwargs)
        self._e41h = CDLNSSADPOKJointHungarianPlacer(**common_kwargs, **e41_kwargs)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True)
        log(
            "=== E70 HYBRID-HUNGARIAN placer: running E25 then E41-with-Hungarian, "
            "returning lower-proxy output ==="
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # E25 lane.
        t0 = time.perf_counter()
        e25_placement = self._e25.place(benchmark)
        e25_wall = time.perf_counter() - t0
        e25_proxy = compute_proxy_cost(e25_placement, benchmark, plc)
        e25_ov = compute_overlap_metrics(e25_placement, benchmark)
        log(
            f"  E25 done: proxy={e25_proxy['proxy_cost']:.5f} "
            f"overlaps={e25_ov['overlap_count']} wall={e25_wall:.1f}s"
        )

        # E41 + Hungarian lane.
        t1 = time.perf_counter()
        e41h_placement = self._e41h.place(benchmark)
        e41h_wall = time.perf_counter() - t1
        e41h_proxy = compute_proxy_cost(e41h_placement, benchmark, plc)
        e41h_ov = compute_overlap_metrics(e41h_placement, benchmark)
        log(
            f"  E41+Hungarian done: proxy={e41h_proxy['proxy_cost']:.5f} "
            f"overlaps={e41h_ov['overlap_count']} wall={e41h_wall:.1f}s"
        )

        # Pick lower-proxy zero-overlap output.
        e25_valid = e25_ov["overlap_count"] == 0
        e41h_valid = e41h_ov["overlap_count"] == 0

        if e25_valid and e41h_valid:
            if e25_proxy["proxy_cost"] <= e41h_proxy["proxy_cost"]:
                winner = "E25"
                final = e25_placement
                final_proxy = e25_proxy["proxy_cost"]
            else:
                winner = "E41+Hungarian"
                final = e41h_placement
                final_proxy = e41h_proxy["proxy_cost"]
        elif e25_valid:
            winner = "E25 (E41+Hungarian had overlaps)"
            final = e25_placement
            final_proxy = e25_proxy["proxy_cost"]
        elif e41h_valid:
            winner = "E41+Hungarian (E25 had overlaps)"
            final = e41h_placement
            final_proxy = e41h_proxy["proxy_cost"]
        else:
            raise RuntimeError(
                f"E70 hybrid: BOTH pipelines produced overlapping placements "
                f"(E25 {e25_ov['overlap_count']}, E41+H {e41h_ov['overlap_count']})"
            )

        total_wall = time.perf_counter() - t0
        log(
            f"  E70 winner: {winner} proxy={final_proxy:.5f} "
            f"(E25={e25_proxy['proxy_cost']:.5f}, "
            f"E41+H={e41h_proxy['proxy_cost']:.5f}) "
            f"total_wall={total_wall:.1f}s"
        )
        return final
