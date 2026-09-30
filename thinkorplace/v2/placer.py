"""thinkorplace-v2 — 3-lane parallel ensemble (V4+Gaussian + saddle + Hungarian).

Three pipelines per bench, each in its own subprocess:

  A: V4+Gaussian descent → legalize → CD polish.
  B: A + bounded Hessian saddle escape between two CD passes.
  C: A + K=50 Hungarian joint permutation between two CD passes.

Post-competition supervisor selects the canonical-best valid checkpoint produced
in this run. Contention and deadlines can change which lanes finish.
"""
from __future__ import annotations

import multiprocessing as _mp
import os
import math
import shutil
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import torch


from . import worker as lane_worker

_LANE_NAMES: Tuple[str, ...] = ("A", "B", "C")


class Placer:
    """3-lane parallel ensemble (A=v2-extCD, B=+Hessian saddle, C=+Hungarian).

    ``place(benchmark)`` returns a [num_macros, 2] float32 tensor with zero
    hard-macro overlaps. An exhausted run without a valid checkpoint fails clearly.
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
        device: Optional[str] = None,
        num_threads: Optional[int] = None,
        lanes: str = "ABC",
        checkpoint_dir: Optional[str] = None,
    ):
        if budget_seconds is not None and (not math.isfinite(budget_seconds) or budget_seconds < 0):
            raise ValueError("budget_seconds must be finite and nonnegative, or None")
        if lanes not in ("A", "ABC"):
            raise ValueError("lanes must be 'A' or 'ABC'")
        if device not in (None, "cpu", "cuda", "mps"):
            raise ValueError("device must be cpu, cuda, mps, or None")
        if num_threads is not None and num_threads < len(lanes):
            raise ValueError("num_threads must allocate at least one thread per lane")
        self.device = device
        self.num_threads = num_threads
        self.lanes = lanes
        self.checkpoint_dir = checkpoint_dir
        self.last_run_metadata = {}
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

    def _spawn_lane(self, ctx, lane_idx, bench_name, result_path, log_path, deadline):
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
            [], deadline, self.device,
            max(1, (self.num_threads or os.cpu_count() or 8) // len(self.lanes)),
        )
        p = ctx.Process(target=lane_worker.lane_entry, args=args,
                        name=f"v2_lane{_LANE_NAMES[lane_idx]}")
        # Spawn imports NumPy/Torch before lane_entry: set inherited BLAS limits now.
        thread_keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        previous = {key: os.environ.get(key) for key in thread_keys}
        try:
            for key in thread_keys:
                os.environ[key] = str(args[-1])
            p.start()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
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

    @staticmethod
    def _stop_processes(procs):
        # Signal every child before joining, so cleanup does not serialize grace periods.
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=0.5)
            if p.is_alive():
                p.kill()
        for p in procs:
            p.join(timeout=0.5)

    def place(self, benchmark) -> torch.Tensor:
        from ..common.runtime_safety import monotonic_seconds, remaining_seconds, require_time, valid_placement

        t0 = monotonic_seconds()
        deadline = t0 + self.budget_seconds if self.budget_seconds is not None else None
        require_time(deadline)
        n_lanes = len(self.lanes)
        temporary = self.checkpoint_dir is None
        tmp_dir = Path(tempfile.mkdtemp(prefix="v2_lanes_")) if temporary else Path(self.checkpoint_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # Reusing a directory must never admit a prior run's candidate.
        if any(tmp_dir.glob("lane*.pkl")):
            raise ValueError("checkpoint_dir already contains lane checkpoints; use a fresh run directory")
        result_paths = [str(tmp_dir / f"lane{i}.pkl") for i in range(n_lanes)]
        log_paths = [str(tmp_dir / f"lane{i}.log") for i in range(n_lanes)]
        procs = []
        timed_out = []
        try:
            ctx = _mp.get_context("spawn")
            for i in range(n_lanes):
                require_time(deadline)
                procs.append(self._spawn_lane(ctx, i, benchmark.name, result_paths[i], log_paths[i], deadline))
            for i, p in enumerate(procs):
                timeout = None if deadline is None else remaining_seconds(deadline)
                p.join(timeout=timeout)
                if p.is_alive():
                    timed_out.append(_LANE_NAMES[i])
            self._stop_processes(procs)
            results = self._collect(n_lanes, result_paths)
            self.last_run_metadata = dict(
                lanes=self.lanes, timed_out_lanes=timed_out,
                exitcodes={_LANE_NAMES[i]: p.exitcode for i, p in enumerate(procs)},
                lane_results=[{k: v for k, v in r.items() if k not in ("pos_np", "log_lines")} if r else None for r in results],
                wall=monotonic_seconds() - t0,
            )
            completed = {
                _LANE_NAMES[i]: bool(r and r.get("worker_status") == "ok"
                                    and r.get("pipeline_completed", True))
                for i, r in enumerate(results)
            }
            self.last_run_metadata["phase_completion"] = {
                _LANE_NAMES[i]: {"checkpoint_phase": r.get("phase") if r else None,
                                 "pipeline_completed": completed[_LANE_NAMES[i]],
                                 "phase_details": r.get("phase_details", {}) if r else {},
                                 "worker_status": r.get("worker_status", "interrupted" if r.get("status") == "ok" else r.get("status")) if r else "missing"}
                for i, r in enumerate(results)
            }
            self.last_run_metadata["pipeline_completed"] = all(completed.values())
            self.last_run_metadata["degraded_lanes"] = [name for name, done in completed.items() if not done]
            self.last_run_metadata.update(
                solver_outcome="pipeline_completed" if all(completed.values()) else "partial_phase_completion",
                used_fallback=False,
            )
            candidates = []
            for i, r in enumerate(results):
                if r is None or r.get("status") != "ok":
                    continue
                try:
                    pos = torch.from_numpy(r["pos_np"]).to(torch.float32)
                    valid, violations = valid_placement(pos, benchmark)
                    if valid and math.isfinite(float(r["proxy"])):
                        candidates.append((r["proxy"], i, pos))
                    else:
                        self._log(f"lane{_LANE_NAMES[i]} invalid: {violations}")
                except (KeyError, TypeError, ValueError) as exc:
                    self._log(f"lane{_LANE_NAMES[i]} invalid artifact: {exc}")
            if not candidates:
                if timed_out or remaining_seconds(deadline) <= 0:
                    raise TimeoutError("Placement deadline exhausted without a valid lane checkpoint")
                raise RuntimeError("All lanes failed without a valid checkpoint")
            proxy, picked, pos = min(candidates, key=lambda x: x[0])
            used_fallback = results[picked].get("worker_status") != "ok"
            self.last_run_metadata.update(picked_lane=_LANE_NAMES[picked], proxy=float(proxy),
                                          used_fallback=used_fallback)
            if used_fallback:
                self.last_run_metadata["solver_outcome"] = "checkpoint_recovery"
            self._log(f"v2 PICK={_LANE_NAMES[picked]} proxy={proxy:.5f}")
            return pos
        finally:
            self._stop_processes(procs)
            if temporary:
                shutil.rmtree(tmp_dir, ignore_errors=True)
