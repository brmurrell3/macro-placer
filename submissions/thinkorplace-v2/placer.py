"""thinkorplace-v2 — 3-lane parallel ensemble (V4+Gaussian + saddle + Hungarian).

Three pipelines per bench, each in its own subprocess:

  A: V4+Gaussian descent → legalize → CD polish.
  B: A + bounded Hessian saddle escape between two CD passes.
  C: A + K=50 Hungarian joint permutation between two CD passes.

Master joins all 3 and returns the canonical-best placement (monotone:
v2 output ≥ Lane A output ≥ prior v2-extCD).
"""
from __future__ import annotations

import multiprocessing as _mp
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_EXP_PATHS: Tuple[str, ...] = (str(_ROOT), str(_HERE), str(_HERE / "lib"))
for _p in _EXP_PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lane_worker  # noqa: E402  — spawn re-imports this by name

_LANE_NAMES: Tuple[str, ...] = ("A", "B", "C")


class Placer:
    """3-lane parallel ensemble (A=v2-extCD, B=+Hessian saddle, C=+Hungarian).

    ``place(benchmark)`` returns a [num_macros, 2] float32 tensor with zero
    hard-macro overlaps. Worst case: SDF + project_overlaps fallback.
    """

    def __init__(
        self,
        budget_seconds: Optional[float] = 2700.0,
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        cd_polish_s: float = 900.0,
        cd_plateau_threshold: float = 0.001,
        saddle_budget_s: float = 120.0,
        saddle_max_iters: int = 2,
        eigsh_maxiter: int = 50,
        eigsh_tol: float = 1e-2,
        hung_budget_s: float = 300.0,
        hung_k: int = 50,
        hung_n_slots: int = 100,
        hung_reject_patience: int = 15,
        hung_seed_base: int = 42,
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
        self.hung_budget_s = hung_budget_s
        self.hung_k = hung_k
        self.hung_n_slots = hung_n_slots
        self.hung_reject_patience = hung_reject_patience
        self.hung_seed_base = hung_seed_base
        self.lane_a_seed = lane_a_seed
        self.lane_b_seed = lane_b_seed
        self.lane_c_seed = lane_c_seed
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def _lane_seed_and_budgets(self, lane_idx: int) -> Tuple[int, float, float]:
        if lane_idx == 0:
            return self.lane_a_seed, 0.0, 0.0
        if lane_idx == 1:
            return self.lane_b_seed, self.saddle_budget_s, 0.0
        return self.lane_c_seed, 0.0, self.hung_budget_s

    def _spawn_lane(self, ctx, lane_idx, bench_name, result_path, log_path):
        seed, saddle_budget, hung_budget = self._lane_seed_and_budgets(lane_idx)
        args = (
            lane_idx, bench_name, result_path, log_path, seed,
            self.num_steps, self.lr_frac,
            self.gamma_start_frac, self.gamma_end_frac,
            self.overlap_lambda_end, self.overlap_ramp_pct, self.init,
            float(self.cd_polish_s), self.cd_plateau_threshold,
            float(saddle_budget), self.saddle_max_iters,
            self.eigsh_maxiter, self.eigsh_tol,
            float(hung_budget), self.hung_k, self.hung_n_slots,
            self.hung_reject_patience, self.hung_seed_base,
            list(_EXP_PATHS),
        )
        p = ctx.Process(target=lane_worker.lane_entry, args=args,
                        name=f"v2_lane{_LANE_NAMES[lane_idx]}")
        p.start()
        return p

    def _collect(self, n_lanes, result_paths):
        out = []
        for i in range(n_lanes):
            name = _LANE_NAMES[i]
            if not Path(result_paths[i]).exists():
                self._log(f"  lane{name}: no result file")
                out.append(None)
                continue
            try:
                with open(result_paths[i], "rb") as f:
                    r = pickle.load(f)
                for line in r.get("log_lines", []):
                    self._log(f"  [lane{name}] {line}")
                if r.get("status") == "ok":
                    self._log(f"  lane{name} OK: proxy={r['proxy']:.5f} wall={r['wall']:.0f}s")
                else:
                    note = (r.get("reason") or r.get("traceback", ""))[:200]
                    self._log(f"  lane{name} status={r.get('status')!r} note={note!r}")
                out.append(r)
            except Exception as exc:
                self._log(f"  lane{name}: pickle load failed: {exc!r}")
                out.append(None)
        return out

    def place(self, benchmark) -> torch.Tensor:
        from macro_place.cd_core import project_overlaps, sdf_init
        from macro_place.objective import compute_overlap_metrics

        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== thinkorplace-v2 ({benchmark.name}) ===")

        n_lanes = 3
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"v2_3lane_{benchmark.name}_"))
        result_paths = [str(tmp_dir / f"lane{i}.pkl") for i in range(n_lanes)]
        log_paths = [str(tmp_dir / f"lane{i}.log") for i in range(n_lanes)]

        ctx = _mp.get_context("spawn")
        self._log(f"  spawning {n_lanes} lanes (A/B/C), tmp={tmp_dir}")
        procs = [
            self._spawn_lane(ctx, i, benchmark.name, result_paths[i], log_paths[i])
            for i in range(n_lanes)
        ]
        self._log("  lane PIDs: " + " ".join(
            f"{_LANE_NAMES[i]}={p.pid}" for i, p in enumerate(procs)))

        for i, p in enumerate(procs):
            name = _LANE_NAMES[i]
            remaining = (deadline - time.time() - 15.0) if deadline is not None else None
            timeout = remaining if (remaining is not None and remaining > 0) else None
            self._log(f"  joining lane{name} (timeout={timeout})")
            p.join(timeout=timeout)
            if p.is_alive():
                self._log(f"  lane{name} OVER WALL — terminating")
                p.terminate()
                p.join(timeout=10.0)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=5.0)

        results = self._collect(n_lanes, result_paths)
        ok = [(r["proxy"], i, r) for i, r in enumerate(results)
              if r is not None and r.get("status") == "ok"]

        if not ok:
            self._log("  ALL LANES FAILED — SDF + project_overlaps fallback")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)
            return pos.to(torch.float32)

        ok.sort(key=lambda x: x[0])
        picked_proxy, picked_idx, picked_r = ok[0]

        parts, walls = [], []
        for li in range(n_lanes):
            r = results[li]
            tag = _LANE_NAMES[li]
            if r is not None and r.get("status") == "ok":
                parts.append(f"{tag}={r['proxy']:.5f}")
                walls.append(f"{tag}={r['wall']:.0f}s")
            else:
                parts.append(f"{tag}=FAIL")
                walls.append(f"{tag}=FAIL")
        self._log(
            f"=== v2 PICK={_LANE_NAMES[picked_idx]}: {' '.join(parts)} "
            f"-> final={picked_proxy:.5f} walls=[{' '.join(walls)}] "
            f"total_wall={time.time() - t0:.0f}s ==="
        )

        pos = torch.from_numpy(picked_r["pos_np"]).to(torch.float32)

        if compute_overlap_metrics(pos, benchmark)["overlap_count"] > 0:
            self._log("  picked lane has overlaps; trying next-best lanes")
            for proxy, _idx, r in ok[1:]:
                pos2 = torch.from_numpy(r["pos_np"]).to(torch.float32)
                if compute_overlap_metrics(pos2, benchmark)["overlap_count"] == 0:
                    self._log(f"  fallback to lane proxy={proxy:.5f}")
                    return pos2
            raise RuntimeError("All ok-status lanes' placements have overlaps")

        return pos
