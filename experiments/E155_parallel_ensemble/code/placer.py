"""E155 — Parallel two-lane ensemble (v2-extCD + E138) via multiprocessing.

E152 in-process sequential ensemble FAILED on EPYC ibm17 (1.18768 vs
v2-extCD 1.18269 standalone): budget split below the plateau-saturation
point destroyed CD convergence in both lanes.

E155 fixes this by spawning two subprocesses that run the lanes IN PARALLEL,
so each gets the full ~900s CD budget. Master joins, deserializes results,
picks canonical-best.

Architecture:
  Lane A (v2-extCD): V4+Gauss + legalize + CD polish 900s (rng_seed=42)
  Lane B (E138):     V4+Gauss + legalize + CD polish 900s
                     + bounded saddle 120s + trailing CD ~360s (rng_seed=142)

Spawning model:
  - `multiprocessing.get_context("spawn").Process(target=lane_entry, ...)`.
  - `spawn` avoids forked CUDA contexts and shared state surprises.
  - Subprocess entry lives in `e155_lane_worker.py` (a normal module that
    spawn can re-import by qualified name).
  - Each subprocess re-loads benchmark+plc via `find_benchmark_dir` +
    `load_benchmark_from_dir` so torch/cuda starts fresh.
  - Results returned via pickle file (positions serialized as numpy).
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
import e155_lane_worker  # noqa: E402


class E155ParallelEnsemblePlacer:
    """Two-lane PARALLEL ensemble using multiprocessing.spawn subprocesses.

    Each lane gets its full standalone budget; total wall is max(lanes) + setup.
    """

    def __init__(
        self,
        budget_seconds: Optional[float] = 3300.0,
        # Shared descent params
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        # Per-lane CD polish budget (each lane gets the full ~900s on its own subprocess)
        cd_polish_s: float = 900.0,
        cd_plateau_threshold: float = 0.001,
        # Lane B saddle params
        saddle_budget_s: float = 120.0,
        saddle_max_iters: int = 2,
        eigsh_maxiter: int = 50,
        eigsh_tol: float = 1e-2,
        # Seeds
        lane_a_seed: int = 42,
        lane_b_seed: int = 142,
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
        self.lane_a_seed = lane_a_seed
        self.lane_b_seed = lane_b_seed
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _spawn_lane(self, ctx, lane_idx: int, bench_name: str, result_path: str, log_path: str):
        if lane_idx == 0:
            seed = self.lane_a_seed
            saddle_budget = 0.0
        else:
            seed = self.lane_b_seed
            saddle_budget = self.saddle_budget_s
        args = (
            lane_idx,
            bench_name,
            result_path,
            log_path,
            seed,
            self.num_steps,
            self.lr_frac,
            self.gamma_start_frac,
            self.gamma_end_frac,
            self.overlap_lambda_end,
            self.overlap_ramp_pct,
            self.init,
            float(self.cd_polish_s),
            self.cd_plateau_threshold,
            float(saddle_budget),
            self.saddle_max_iters,
            self.eigsh_maxiter,
            self.eigsh_tol,
            list(_EXP_PATHS),
        )
        p = ctx.Process(
            target=e155_lane_worker.lane_entry,
            args=args,
            name=f"E155_lane{lane_idx}",
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
        self._log(f"=== E155 parallel ensemble ({benchmark.name}) ===")

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"e155_{benchmark.name}_"))
        result_paths = [str(tmp_dir / f"lane{i}.pkl") for i in range(2)]
        log_paths = [str(tmp_dir / f"lane{i}.log") for i in range(2)]

        ctx = _mp_top.get_context("spawn")

        self._log(f"  spawning 2 lanes, tmp={tmp_dir}")
        procs = [
            self._spawn_lane(ctx, 0, benchmark.name, result_paths[0], log_paths[0]),
            self._spawn_lane(ctx, 1, benchmark.name, result_paths[1], log_paths[1]),
        ]
        self._log(f"  lane PIDs: A={procs[0].pid} B={procs[1].pid}")

        for i, p in enumerate(procs):
            remaining = (deadline - time.time() - 15.0) if deadline is not None else None
            timeout = remaining if (remaining is not None and remaining > 0) else None
            self._log(f"  joining lane{i} (timeout={timeout})")
            p.join(timeout=timeout)
            if p.is_alive():
                self._log(f"  lane{i} OVER WALL — terminating")
                p.terminate()
                p.join(timeout=10.0)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=5.0)

        results = []
        for i in range(2):
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
            self._log("  BOTH LANES FAILED — falling back to SDF + project_overlaps")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)
            return pos.to(torch.float32)

        proxies.sort(key=lambda x: x[0])
        picked_proxy, picked_idx, picked_r = proxies[0]
        lane_name = "A" if picked_idx == 0 else "B"

        proxy_a = next((r["proxy"] for r in results if r is not None and r.get("lane_idx") == 0 and r.get("status") == "ok"), float("nan"))
        proxy_b = next((r["proxy"] for r in results if r is not None and r.get("lane_idx") == 1 and r.get("status") == "ok"), float("nan"))
        wall_a = next((r["wall"] for r in results if r is not None and r.get("lane_idx") == 0), float("nan"))
        wall_b = next((r["wall"] for r in results if r is not None and r.get("lane_idx") == 1), float("nan"))
        total_wall = time.time() - t0
        self._log(
            f"=== E155 PICK={lane_name}: proxy_a={proxy_a:.5f} proxy_b={proxy_b:.5f} "
            f"-> final={picked_proxy:.5f} wall_a={wall_a:.0f}s wall_b={wall_b:.0f}s "
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
