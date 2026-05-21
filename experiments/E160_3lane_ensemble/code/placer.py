"""E158 — 3-lane PARALLEL ensemble with diverse search topologies.

Fixes E155's budget bug (Lane A killed by tight master deadline) by
allocating a generous wall budget (2700s default = 45 min) that fits
the longest lane comfortably.

Architecture:
  Lane A (v2-extCD):    V4+Gauss + legalize + CD polish 700s
  Lane B (E138):        V4+Gauss + legalize + CD polish 700s
                        + bounded saddle 120s + CD2 polish 240s
  Lane C (NEW aggressive): V4+Gauss with overlap_lambda_end=50,
                        overlap_ramp_pct=0.6, num_steps=400
                        + legalize + CD polish 700s

Master spawns 3 subprocesses via multiprocessing.spawn, joins all,
picks canonical-best.
"""
from __future__ import annotations

import multiprocessing as _mp_top
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_EXP_PATHS = (
    str(_ROOT),
    str(_HERE),
    str(_ROOT / "experiments" / "E74_hessian_saddle" / "code"),
    str(_ROOT / "experiments" / "E84_cascading_saddle" / "code"),
    str(_ROOT / "experiments" / "E88_diff_proxy" / "code"),
    str(_ROOT / "experiments" / "E95_diff_proxy_v2" / "code"),
    str(_ROOT / "experiments" / "E110_smooth_global_placer" / "code"),
    str(_ROOT / "experiments" / "E111_per_net_trace_congestion" / "code"),
    str(_ROOT / "experiments" / "E115_triton_kernels" / "code"),
    str(_ROOT / "experiments" / "E117_gaussian_density" / "code"),
    str(_ROOT / "experiments" / "E127_v4_gaussian" / "code"),
    str(_ROOT / "experiments" / "E138_bounded_saddle" / "code"),
)

for _p in _EXP_PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import the worker by NAME so spawn can re-import it the same way.
import e158_lane_worker  # noqa: E402


class E158ThreeLaneEnsemblePlacer:
    """3-lane parallel ensemble. Each lane runs in its own subprocess."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 2700.0,
        # Shared descent params (default = v2-extCD)
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        # CD polish (per lane standalone budget)
        cd_polish_s: float = 700.0,
        cd_plateau_threshold: float = 0.001,
        # Lane B saddle params
        saddle_budget_s: float = 120.0,
        saddle_max_iters: int = 2,
        eigsh_maxiter: int = 50,
        eigsh_tol: float = 1e-2,
        cd2_polish_s: float = 240.0,
        # Lane C aggressive params
        lane_c_overlap_end: float = 50.0,
        lane_c_overlap_ramp_pct: float = 0.6,
        lane_c_num_steps: int = 400,
        # Seeds
        lane_a_seed: int = 42,
        lane_b_seed: int = 142,
        lane_c_seed: int = 242,
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
        self.cd_polish_s = cd_polish_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.saddle_budget_s = saddle_budget_s
        self.saddle_max_iters = saddle_max_iters
        self.eigsh_maxiter = eigsh_maxiter
        self.eigsh_tol = eigsh_tol
        self.cd2_polish_s = cd2_polish_s
        self.lane_c_overlap_end = lane_c_overlap_end
        self.lane_c_overlap_ramp_pct = lane_c_overlap_ramp_pct
        self.lane_c_num_steps = lane_c_num_steps
        self.lane_a_seed = lane_a_seed
        self.lane_b_seed = lane_b_seed
        self.lane_c_seed = lane_c_seed
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _spawn_lane(self, ctx, lane_idx: int, bench_name: str, result_path: str, log_path: str):
        # Per-lane config differs:
        if lane_idx == 0:
            seed = self.lane_a_seed
            num_steps = self.num_steps
            ovl_end = self.overlap_lambda_end
            ovl_ramp = self.overlap_ramp_pct
            saddle_budget = 0.0
            cd2_budget = 0.0
        elif lane_idx == 1:
            seed = self.lane_b_seed
            num_steps = self.num_steps
            ovl_end = self.overlap_lambda_end
            ovl_ramp = self.overlap_ramp_pct
            saddle_budget = self.saddle_budget_s
            cd2_budget = self.cd2_polish_s
        else:  # lane_idx == 2
            seed = self.lane_c_seed
            num_steps = self.lane_c_num_steps
            ovl_end = self.lane_c_overlap_end
            ovl_ramp = self.lane_c_overlap_ramp_pct
            saddle_budget = 0.0
            cd2_budget = 0.0

        args = (
            lane_idx,
            bench_name,
            result_path,
            log_path,
            seed,
            num_steps,
            self.lr_frac,
            self.gamma_start_frac,
            self.gamma_end_frac,
            ovl_end,
            ovl_ramp,
            self.init,
            float(self.cd_polish_s),
            self.cd_plateau_threshold,
            float(saddle_budget),
            self.saddle_max_iters,
            self.eigsh_maxiter,
            self.eigsh_tol,
            float(cd2_budget),
            list(_EXP_PATHS),
        )
        p = ctx.Process(
            target=e158_lane_worker.lane_entry,
            args=args,
            name=f"E158_lane{lane_idx}",
        )
        p.start()
        return p

    def place(self, benchmark) -> torch.Tensor:
        from macro_place.bench_paths import find_benchmark_dir
        from macro_place.cd_core import project_overlaps, sdf_init
        from macro_place.loader import load_benchmark_from_dir
        from macro_place.objective import compute_overlap_metrics

        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E158 3-lane ensemble ({benchmark.name}) ===")

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"e158_{benchmark.name}_"))
        n_lanes = 3
        result_paths = [str(tmp_dir / f"lane{i}.pkl") for i in range(n_lanes)]
        log_paths = [str(tmp_dir / f"lane{i}.log") for i in range(n_lanes)]

        ctx = _mp_top.get_context("spawn")

        self._log(f"  spawning {n_lanes} lanes, tmp={tmp_dir}")
        procs = [
            self._spawn_lane(ctx, i, benchmark.name, result_paths[i], log_paths[i])
            for i in range(n_lanes)
        ]
        self._log(f"  lane PIDs: {[p.pid for p in procs]}")

        # Wait for all lanes uniformly (compute remaining deadline once).
        # Joining sequentially with shared deadline ensures any lane that
        # finishes early doesn't starve later lanes.
        for i, p in enumerate(procs):
            if deadline is not None:
                remaining = deadline - time.time() - 5.0
                timeout = max(0.0, remaining)
            else:
                timeout = None
            self._log(f"  joining lane{i} (timeout={timeout})")
            p.join(timeout=timeout)
            if p.is_alive():
                self._log(f"  lane{i} OVER WALL — terminating")
                p.terminate()
                p.join(timeout=10.0)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=5.0)

        # Read results.
        results = []
        for i in range(n_lanes):
            rp = result_paths[i]
            if not Path(rp).exists():
                self._log(f"  lane{i}: NO RESULT FILE (subprocess died before write)")
                results.append(None)
                continue
            try:
                with open(rp, "rb") as f:
                    r = pickle.load(f)
                results.append(r)
                for line in r.get("log_lines", []):
                    self._log(f"  [lane{i}] {line}")
                if r.get("status") == "ok":
                    self._log(f"  lane{i} OK: proxy={r['proxy']:.5f} wall={r['wall']:.0f}s")
                else:
                    snippet = (r.get("reason") or r.get("traceback", ""))[:200]
                    self._log(f"  lane{i} status={r.get('status')!r} note={snippet!r}")
            except Exception as exc:
                self._log(f"  lane{i}: pickle load failed: {exc!r}")
                results.append(None)

        proxies = []
        for i, r in enumerate(results):
            if r is not None and r.get("status") == "ok":
                proxies.append((r["proxy"], i, r))

        if not proxies:
            self._log("  ALL LANES FAILED — falling back to SDF + project_overlaps")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)
            return pos.to(torch.float32)

        proxies.sort(key=lambda x: x[0])
        picked_proxy, picked_idx, picked_r = proxies[0]
        lane_names = {0: "A", 1: "B", 2: "C"}
        lane_name = lane_names.get(picked_idx, str(picked_idx))

        # Pull individual lane proxies for logging.
        def _get(idx):
            for r in results:
                if r is not None and r.get("lane_idx") == idx and r.get("status") == "ok":
                    return r["proxy"]
            return float("nan")

        def _wall(idx):
            for r in results:
                if r is not None and r.get("lane_idx") == idx:
                    return r.get("wall", float("nan"))
            return float("nan")

        proxy_a, proxy_b, proxy_c = _get(0), _get(1), _get(2)
        wall_a, wall_b, wall_c = _wall(0), _wall(1), _wall(2)
        total_wall = time.time() - t0

        self._log(
            f"=== E158 PICK={lane_name}: proxy_a={proxy_a:.5f} proxy_b={proxy_b:.5f} proxy_c={proxy_c:.5f} "
            f"-> final={picked_proxy:.5f} wall_a={wall_a:.0f}s wall_b={wall_b:.0f}s wall_c={wall_c:.0f}s "
            f"total_wall={total_wall:.0f}s ==="
        )

        pos = torch.from_numpy(picked_r["pos_np"]).to(torch.float32)

        final_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        if final_ovl > 0:
            self._log(f"  picked lane sanity overlap={final_ovl}; trying other lane")
            for proxy, _idx, r in proxies[1:]:
                pos2 = torch.from_numpy(r["pos_np"]).to(torch.float32)
                if compute_overlap_metrics(pos2, benchmark)["overlap_count"] == 0:
                    self._log(f"  fallback to other lane proxy={proxy:.5f}")
                    return pos2
            raise RuntimeError(f"All lanes' final placements have overlaps (picked={final_ovl})")

        return pos


# Standard Placer class for the evaluation harness.
class Placer(E158ThreeLaneEnsemblePlacer):
    """E158 3-lane parallel ensemble. Aliased as Placer for the evaluator."""
    pass
