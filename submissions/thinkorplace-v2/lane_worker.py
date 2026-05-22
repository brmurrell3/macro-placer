"""thinkorplace-v3-3lane lane subprocess worker.

Top-level module so multiprocessing.spawn can re-import it by qualified
name. Handles three lane types selected by `lane_idx`:

  lane_idx=0  Lane A — v2-extCD:
              V4+Gaussian descent → legalize → CD polish (full budget).

  lane_idx=1  Lane B — E138:
              V4+Gaussian descent → CD1 → bounded saddle escape → CD2.

  lane_idx=2  Lane C — E159 Hungarian:
              V4+Gaussian descent → CD1 → Hungarian K=50 permutation loop
              → CD2.

OFFLINE per-bench MIN of A/B/C on EPYC = 0.97745 (beats Shoom 0.978).
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
    hung_budget_s: float,
    hung_k: int,
    hung_n_slots: int,
    hung_reject_patience: int,
    hung_seed_base: int,
    extra_sys_path: list,
) -> None:
    """Subprocess entry: runs ONE lane end-to-end, writes result pickle."""
    # Prime sys.path inside subprocess.
    for p in extra_sys_path:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Reduce per-subprocess thread oversubscription.
    nthreads = max(1, (os.cpu_count() or 8) // 3)
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

        if lane_idx == 0:
            # Lane A: single CD polish (full budget).
            cd_budget = cd_polish_s
            _log(f"  [A] CD polish budget={cd_budget:.0f}s")
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

        elif lane_idx == 1:
            # Lane B: CD1 + saddle + CD2.
            cd1_budget = cd_polish_s
            _log(f"  [B] CD1 polish budget={cd1_budget:.0f}s")
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
            _log(f"  [B] after CD1: proxy={cd1_proxy:.5f} ovl={cd1_ovl} wall={time.time()-t0:.0f}s")

            if cd1_ovl == 0:
                try:
                    from bounded_saddle import bounded_cascading_saddle_escape
                    _log(f"  [B] saddle budget={saddle_budget_s:.0f}s")
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
                            _log(f"  [B] saddle accept: {cd1_proxy:.5f} -> {saddle_proxy:.5f}")
                        else:
                            _log(f"  [B] saddle reject: {saddle_proxy:.5f} > {cd1_proxy:.5f}")
                    else:
                        _log(f"  [B] saddle output has {saddle_ovl} overlaps; rejecting")
                except Exception as exc:
                    _log(f"  [B] saddle EXCEPTION: {exc!r}; skipping")
            else:
                _log("  [B] CD1 produced overlaps -> skipping saddle phase")

            cd2_budget = max(60.0, cd_polish_s * 0.4)
            _log(f"  [B] CD2 polish budget={cd2_budget:.0f}s")
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

        elif lane_idx == 2:
            # Lane C: CD1 + Hungarian K=50 permutation loop + CD2.
            from kjoint_hungarian import kjoint_hungarian_step

            cd1_budget = cd_polish_s
            _log(f"  [C] CD1 polish budget={cd1_budget:.0f}s")
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
            _log(f"  [C] after CD1: proxy={cd1_proxy:.5f} ovl={cd1_ovl} wall={time.time()-t0:.0f}s")

            # Hungarian K=50 polish loop (only if CD1 is overlap-clean).
            pos_post_hung = pos
            post_hung_proxy = cd1_proxy
            if cd1_ovl == 0 and hung_budget_s > 1.0:
                try:
                    _log(
                        f"  [C] Hungarian budget={hung_budget_s:.0f}s "
                        f"K={hung_k} n_slots={hung_n_slots} reject_patience={hung_reject_patience}"
                    )
                    n_hard = benchmark.num_hard_macros
                    fixed_np = benchmark.macro_fixed.cpu().numpy()
                    n_movable = sum(1 for i in range(n_hard) if not bool(fixed_np[i]))
                    k_eff = min(hung_k, n_movable)

                    placement_f64 = pos.detach().clone().to(_torch.float64)
                    hung_evaluator = IncrementalProxyEvaluator(
                        benchmark, plc, placement_f64
                    )
                    baseline = hung_evaluator.current_cost()["proxy"]
                    best_proxy = baseline
                    best_placement = hung_evaluator.placement.detach().clone()

                    t_hung_start = time.perf_counter()
                    iters = 0
                    accepts = 0
                    reject_streak = 0
                    stop_reason = "loop_exit"
                    while True:
                        elapsed = time.perf_counter() - t_hung_start
                        if elapsed >= hung_budget_s:
                            stop_reason = "budget"
                            break
                        if reject_streak >= hung_reject_patience:
                            stop_reason = "saturation"
                            break
                        iters += 1
                        cluster_seed = hung_seed_base + iters
                        try:
                            _new_pos, info = kjoint_hungarian_step(
                                benchmark, hung_evaluator.placement, plc,
                                evaluator=hung_evaluator,
                                k=k_eff, n_slots=hung_n_slots,
                                mode="adjacency",
                                commit_mode="sequential",
                                cluster_seed=cluster_seed,
                            )
                        except Exception as h_exc:
                            _log(f"  [C] H iter {iters} EXC: {h_exc!r}; aborting loop")
                            stop_reason = f"exception:{type(h_exc).__name__}"
                            break
                        accepted = bool(info.get("accepted", False))
                        if accepted:
                            accepts += 1
                            reject_streak = 0
                            cur = hung_evaluator.current_cost()["proxy"]
                            if cur < best_proxy - 1e-9:
                                best_proxy = cur
                                best_placement = hung_evaluator.placement.detach().clone()
                        else:
                            reject_streak += 1
                        if iters <= 3 or iters % 10 == 0:
                            _log(
                                f"  [C] H iter {iters}: seed={cluster_seed} acc={accepted} "
                                f"streak={reject_streak} elapsed={elapsed:.0f}s "
                                f"best={best_proxy:.5f}"
                            )

                    hung_wall = time.perf_counter() - t_hung_start
                    pos_post_hung = best_placement.to(_torch.float32)
                    post_hung_ovl = compute_overlap_metrics(pos_post_hung, benchmark)["overlap_count"]
                    if post_hung_ovl > 0:
                        _log(
                            f"  [C] WARN: Hungarian best has {post_hung_ovl} overlaps; "
                            f"reverting to CD1"
                        )
                        pos_post_hung = pos
                        post_hung_proxy = cd1_proxy
                    else:
                        post_hung_proxy = float(
                            compute_proxy_cost(pos_post_hung, benchmark, plc)["proxy_cost"]
                        )
                    _log(
                        f"  [C] Hungarian: iters={iters} acc={accepts} stop={stop_reason} "
                        f"wall={hung_wall:.0f}s best={best_proxy:.5f} canonical={post_hung_proxy:.5f}"
                    )
                except Exception as exc:
                    _log(f"  [C] Hungarian phase EXCEPTION: {exc!r}; using CD1 placement")
                    pos_post_hung = pos
                    post_hung_proxy = cd1_proxy
            else:
                _log("  [C] skipping Hungarian (ovl>0 or zero budget)")

            # Even if Hungarian regressed slightly, pass to CD2 unless it
            # regressed meaningfully (>0.5%).
            regression_tol = 0.005 * cd1_proxy
            if post_hung_proxy > cd1_proxy + regression_tol:
                _log(
                    f"  [C] Hungarian cumulative {post_hung_proxy:.5f} > "
                    f"CD1 {cd1_proxy:.5f}; using CD1 for CD2"
                )
                pos_for_cd2 = pos
            else:
                pos_for_cd2 = pos_post_hung

            cd2_budget = max(60.0, cd_polish_s * 0.4)
            _log(f"  [C] CD2 polish budget={cd2_budget:.0f}s")
            evaluator = IncrementalProxyEvaluator(benchmark, plc, pos_for_cd2)
            movable = [i for i in range(benchmark.num_macros) if not bool(benchmark.macro_fixed[i])]
            run_cd_adaptive(
                evaluator, benchmark, plc, movable,
                min_time_s=cd2_budget * 0.5,
                hard_cap_s=cd2_budget,
                patience=5,
                plateau_threshold=cd_plateau_threshold,
                log_fn=None,
            )
            pos_after_cd2 = evaluator.placement.detach().clone().to(_torch.float32)
            cd2_proxy = float(compute_proxy_cost(pos_after_cd2, benchmark, plc)["proxy_cost"])
            cd2_ovl = compute_overlap_metrics(pos_after_cd2, benchmark)["overlap_count"]
            _log(f"  [C] after CD2: proxy={cd2_proxy:.5f} ovl={cd2_ovl}")

            # Pick best of {cd1, post_hung, cd2} (only ovl=0 candidates).
            candidates = []
            for name, p_proxy, p_pos in (
                ("cd1", cd1_proxy, pos),
                ("post_hung", post_hung_proxy, pos_post_hung),
                ("cd2", cd2_proxy, pos_after_cd2),
            ):
                if compute_overlap_metrics(p_pos, benchmark)["overlap_count"] == 0:
                    candidates.append((name, p_proxy, p_pos))
            if not candidates:
                raise RuntimeError("Lane C: no overlap-clean candidate after CD2")
            best_name, best_p, best_pos = min(candidates, key=lambda x: x[1])
            _log(f"  [C] picked {best_name} proxy={best_p:.5f}")
            pos = best_pos

        else:
            raise ValueError(f"unknown lane_idx={lane_idx}")

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
