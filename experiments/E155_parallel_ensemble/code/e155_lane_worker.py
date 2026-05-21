"""E155 lane subprocess worker — top-level module so multiprocessing.spawn
can re-import it by qualified name.

Loading this module via `spec_from_file_location` (as the harness loads
the placer) does NOT register it in `sys.modules` under a name spawn
can re-import. So we place the subprocess entry in a normal module that
the subprocess imports via the sys.path entries that the master passes.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import traceback

import numpy as np


def lane_entry(
    lane_idx: int,
    bench_name: str,
    result_path: str,
    log_path: str,
    rng_seed: int,
    num_steps: int,
    lr_frac: float,
    gamma_start_frac: float,
    gamma_end_frac: float,
    overlap_lambda_end: float,
    overlap_ramp_pct: float,
    init: str,
    cd_polish_s: float,
    cd_plateau_threshold: float,
    saddle_budget_s: float,
    saddle_max_iters: int,
    eigsh_maxiter: int,
    eigsh_tol: float,
    extra_sys_path: list,
) -> None:
    """Subprocess entry: runs ONE lane end-to-end, writes result + log pickle.

    Lane 0 = v2-extCD (descent + legalize + CD polish).
    Lane 1 = E138 (descent + legalize + CD polish + bounded saddle + CD polish).
    """
    # Prime sys.path inside subprocess.
    for p in extra_sys_path:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Reduce per-subprocess thread oversubscription.
    nthreads = max(1, (os.cpu_count() or 8) // 2)
    os.environ.setdefault("OMP_NUM_THREADS", str(nthreads))
    os.environ.setdefault("MKL_NUM_THREADS", str(nthreads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(nthreads))

    log_lines = []

    def _log(s: str) -> None:
        log_lines.append(s)
        print(f"[lane{lane_idx}] {s}", flush=True)

    t0 = time.time()

    def _write(payload: dict) -> None:
        payload["log_lines"] = log_lines
        try:
            with open(result_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            print(f"[lane{lane_idx}] FAILED to write result pickle: {exc!r}", flush=True)
        try:
            with open(log_path, "w") as f:
                f.write("\n".join(log_lines))
        except Exception:
            pass

    try:
        import torch as _torch
        from macro_place.bench_paths import find_benchmark_dir
        from macro_place.cd_core import project_overlaps, run_cd_adaptive, sdf_init
        from macro_place.incremental_evaluator import IncrementalProxyEvaluator
        from macro_place.loader import load_benchmark_from_dir
        from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

        from smooth_global_placer_v4_gaussian import SmoothGlobalPlacerV4Gaussian

        if _torch.cuda.is_available():
            device = "cuda"
        elif getattr(_torch.backends, "mps", None) is not None and _torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        _log(f"start lane{lane_idx} bench={bench_name} device={device} seed={rng_seed} pid={os.getpid()} nthr={nthreads}")

        bench_dir = find_benchmark_dir(bench_name)
        benchmark, plc = load_benchmark_from_dir(str(bench_dir))
        _log(f"loaded bench dir={bench_dir}")

        # ----- Phase 1: V4+Gaussian descent with fallback ladder -----
        pos = None
        for attempt, cfg in enumerate([
            dict(overlap_lambda_end=overlap_lambda_end),
            dict(overlap_lambda_end=50.0),
            dict(overlap_lambda_end=100.0, num_steps=300),
        ]):
            try:
                ns = cfg.get("num_steps", num_steps)
                ovl_end = cfg["overlap_lambda_end"]
                _log(f"  descent attempt {attempt+1}: ovl_end={ovl_end} steps={ns}")
                descender = SmoothGlobalPlacerV4Gaussian(
                    num_steps=ns,
                    lr_frac=lr_frac,
                    gamma_start_frac=gamma_start_frac,
                    gamma_end_frac=gamma_end_frac,
                    overlap_lambda_end=ovl_end,
                    overlap_ramp_pct=overlap_ramp_pct,
                    init=init,
                    device=device,
                    rng_seed=rng_seed + attempt * 100,
                    verbose=False,
                )
                pos_try = descender.place(benchmark)
                ovl_try = compute_overlap_metrics(pos_try, benchmark)["overlap_count"]
                if ovl_try == 0:
                    pos = pos_try
                    _log(f"  descent attempt {attempt+1}: ovl=0")
                    break
                pos_try2, _ = project_overlaps(pos_try, benchmark)
                if compute_overlap_metrics(pos_try2, benchmark)["overlap_count"] == 0:
                    pos = pos_try2
                    _log(f"  descent attempt {attempt+1}: ovl=0 after project_overlaps")
                    break
                _log(f"  descent attempt {attempt+1}: still {ovl_try} overlaps")
            except Exception as exc:
                _log(f"  descent attempt {attempt+1} EXCEPTION: {exc!r}")
                continue

        if pos is None:
            _log("  fallback: SDF + project_overlaps")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)

        descent_wall = time.time() - t0
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        _log(f"  basin: proxy={descent_proxy:.5f} wall={descent_wall:.0f}s")

        if lane_idx == 1:
            # Lane B: CD1 + saddle + CD2.
            cd1_budget = cd_polish_s
            _log(f"  CD1 polish budget={cd1_budget:.0f}s")
            evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
            movable = [i for i in range(benchmark.num_macros) if not bool(benchmark.macro_fixed[i])]
            run_cd_adaptive(
                evaluator, benchmark, plc, movable,
                min_time_s=cd1_budget * 0.5,
                hard_cap_s=cd1_budget,
                patience=5,
                plateau_threshold=cd_plateau_threshold,
                log_fn=None,
            )
            pos = evaluator.placement.detach().clone().to(_torch.float32)
            cd1_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
            cd1_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
            _log(f"  after CD1: proxy={cd1_proxy:.5f} ovl={cd1_ovl} wall={time.time()-t0:.0f}s")

            if cd1_ovl == 0:
                try:
                    from bounded_saddle import bounded_cascading_saddle_escape
                    _log(f"  saddle budget={saddle_budget_s:.0f}s")
                    saddle_pos, _info = bounded_cascading_saddle_escape(
                        pos, benchmark, plc,
                        max_iters=saddle_max_iters,
                        total_budget_s=saddle_budget_s,
                        eigsh_maxiter=eigsh_maxiter,
                        eigsh_tol=eigsh_tol,
                        log=lambda s, _log=_log: _log(f"    [saddle] {s}"),
                    )
                    saddle_ovl = compute_overlap_metrics(saddle_pos, benchmark)["overlap_count"]
                    if saddle_ovl == 0:
                        saddle_proxy = float(compute_proxy_cost(saddle_pos, benchmark, plc)["proxy_cost"])
                        if saddle_proxy <= cd1_proxy:
                            pos = saddle_pos
                            _log(f"  saddle accept: {cd1_proxy:.5f} -> {saddle_proxy:.5f}")
                        else:
                            _log(f"  saddle reject: {saddle_proxy:.5f} > {cd1_proxy:.5f}")
                    else:
                        _log(f"  saddle output has {saddle_ovl} overlaps; rejecting")
                except Exception as exc:
                    _log(f"  saddle EXCEPTION: {exc!r}; skipping")
            else:
                _log("  CD1 produced overlaps -> skipping saddle phase")

            cd2_budget = max(60.0, cd_polish_s * 0.4)
            _log(f"  CD2 polish budget={cd2_budget:.0f}s")
            evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
            movable = [i for i in range(benchmark.num_macros) if not bool(benchmark.macro_fixed[i])]
            run_cd_adaptive(
                evaluator, benchmark, plc, movable,
                min_time_s=cd2_budget * 0.5,
                hard_cap_s=cd2_budget,
                patience=5,
                plateau_threshold=cd_plateau_threshold,
                log_fn=None,
            )
            pos = evaluator.placement.detach().clone().to(_torch.float32)
        else:
            # Lane A: single CD polish (full budget).
            cd_budget = cd_polish_s
            _log(f"  CD polish budget={cd_budget:.0f}s")
            evaluator = IncrementalProxyEvaluator(benchmark, plc, pos)
            movable = [i for i in range(benchmark.num_macros) if not bool(benchmark.macro_fixed[i])]
            run_cd_adaptive(
                evaluator, benchmark, plc, movable,
                min_time_s=cd_budget * 0.5,
                hard_cap_s=cd_budget,
                patience=5,
                plateau_threshold=cd_plateau_threshold,
                log_fn=None,
            )
            pos = evaluator.placement.detach().clone().to(_torch.float32)

        final_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        final_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        wall = time.time() - t0
        _log(f"  FINAL lane{lane_idx}: proxy={final_proxy:.5f} ovl={final_ovl} wall={wall:.0f}s")

        if final_ovl > 0:
            _write({
                "status": "failed",
                "reason": f"final has {final_ovl} overlaps",
                "wall": wall,
                "lane_idx": lane_idx,
            })
            return

        _write({
            "status": "ok",
            "lane_idx": lane_idx,
            "wall": wall,
            "proxy": final_proxy,
            "pos_np": pos.detach().cpu().numpy().astype(np.float32),
        })

    except Exception:
        tb = traceback.format_exc()
        log_lines.append("FATAL TRACEBACK:\n" + tb)
        try:
            _write({
                "status": "exception",
                "lane_idx": lane_idx,
                "traceback": tb,
                "wall": time.time() - t0,
            })
        except Exception:
            pass
