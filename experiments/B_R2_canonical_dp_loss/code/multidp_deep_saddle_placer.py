"""CDLNSSACascadeDPLanePlacer — cascade + DP as third basin lane.

Same as `submissions/cd_lns_sa_cascade/placer.py` but adds a third init
lane: stock DREAMPlace via Docker subprocess, given the **same polish
budget** as E25 and E41 (~660s CD + ~200s LNS + ~200s SA each).

The original `submissions/_archive/falsified/cd_lns_sa_hessian_dp/placer.py`
gave DP only 60s brief CD cleanup — that was the falsification's bug,
not "structural mismatch" (see `experiments/E91_dp_full_polish/SUMMARY.md`).

Graceful fallback: if `DREAMPLACE_ROOT` is not set, the placer behaves
identically to `cd_lns_sa_cascade/placer.py`.

Cloud setup (lambda.ai box 129.213.18.245):
  export DREAMPLACE_ROOT=/home/ubuntu/DREAMPlace_cpu/install
  export DP_DOCKER_IMAGE=dreamplace:custom
  export DP_USE_GPU=0   # CUDA build hit nvcc11.0/compute_86 conflict; CPU works.

Status 2026-05-12: NOT YET BENCHMARKED on `--all`. Gated on E91 B-R0'
showing DP+polish competitive across IBM benches. Build out only after
ibm10/12/14/17 final results are in.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# E25 placer (SDF init + polish primitives).
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer
run_lns_gridbin = _E25_MOD.run_lns_gridbin
run_sa_polish_v2 = _E25_MOD.run_sa_polish_v2

# E41 placer (DPO init + K-joint pipeline).
from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import CDLNSSADPOKJointPlacer

# Cascading saddle escape (E84).
_E84_CODE = _ROOT / "experiments" / "E84_cascading_saddle" / "code"
if str(_E84_CODE) not in sys.path:
    sys.path.insert(0, str(_E84_CODE))
from cascading_saddle import cascading_saddle_escape

# DREAMPlace I/O helpers from E76.
_E76 = _ROOT / "experiments" / "E76_dreamplace_integration" / "code"
if str(_E76) not in sys.path:
    sys.path.insert(0, str(_E76))
import tilos_to_bookshelf as _bk_writer
import bookshelf_to_pt as _bk_reader
from macro_legalizer import greedy_macro_legalize

# Stricter legalizer (E71 jitter+project loop) for NG45-class stuck cases.
_E91 = _ROOT / "experiments" / "E91_dp_full_polish" / "code"
if str(_E91) not in sys.path:
    sys.path.insert(0, str(_E91))
from extended_legalize import extended_legalize


def _try_run_dreamplace(benchmark, plc, log) -> Optional[torch.Tensor]:
    """B-R2 variant: run NATIVE DREAMPlace (with B-R2 canonical-loss patch)
    instead of Docker. PlaceObj.py on the install reads BR2_LAMBDA_TOPK and
    BR2_LAMBDA_RUDY from env vars.
    """
    dp_root = os.environ.get("DREAMPLACE_ROOT")
    if not dp_root or not Path(dp_root).exists():
        log(f"  [DP] DREAMPLACE_ROOT not set; skipping DP lane")
        return None
    placer_py = Path(dp_root) / "dreamplace" / "Placer.py"
    if not placer_py.exists():
        log(f"  [DP] {placer_py} missing; skipping")
        return None

    if dp_root not in sys.path:
        sys.path.insert(0, dp_root)
    dp_dreamplace = Path(dp_root) / "dreamplace"
    if str(dp_dreamplace) not in sys.path:
        sys.path.insert(0, str(dp_dreamplace))
    import Params, PlaceDB, NonLinearPlace

    SCALE = float(_bk_writer.SCALE)

    def _run_one(target_density, stop_overflow, dp_iter, dp_lr, label):
        with tempfile.TemporaryDirectory(prefix=f"dp_{benchmark.name}_", dir="/tmp") as tmp:
            tmp = Path(tmp)
            _bk_writer._write_nodes(benchmark, tmp)
            _bk_writer._write_pl(benchmark, tmp)
            _bk_writer._write_nets(benchmark, tmp)
            _bk_writer._write_scl(benchmark, tmp)
            _bk_writer._write_wts(benchmark, tmp)
            _bk_writer._write_aux(benchmark, tmp)
            log(f"  [DP-B-R2 {label}] λ_topk={os.environ.get('BR2_LAMBDA_TOPK','0')} "
                f"λ_rudy={os.environ.get('BR2_LAMBDA_RUDY','0')} "
                f"target_density={target_density:.3f} stop_overflow={stop_overflow} "
                f"iter={dp_iter} lr={dp_lr}")
            cfg = tmp / "dp.json"
            cfg.write_text(json.dumps({
                "aux_input": str(tmp / f"{benchmark.name}.aux"),
                "target_density": float(target_density),
                "density_weight": 8e-5,
                "gpu": 0,
                "num_threads": int(os.environ.get("DP_NUM_THREADS", "4")),
                "global_place_stages": [{
                    "num_bins_x": 1024, "num_bins_y": 1024,
                    "iteration": int(dp_iter), "learning_rate": float(dp_lr),
                    "wirelength": "weighted_average", "optimizer": "nesterov",
                    "Llambda_density_weight_iteration": 1, "Lsub_iteration": 1,
                }],
                "result_dir": str(tmp / "results"),
                "RePlAce_LOWER_PCOF": 0.95, "RePlAce_UPPER_PCOF": 1.05,
                "RePlAce_ref_hpwl": 350000.0,
                "stop_overflow": float(stop_overflow),
                "macro_place_flag": 0, "routability_opt_flag": 0,
                "legalize_flag": 0, "detailed_place_flag": 0,
                "random_seed": 42, "global_place_flag": 1,
                "scale_factor": 1.0, "shift_factor": [0.0, 0.0],
                "ignore_net_degree": 100,
            }))
            params = Params.Params()
            params.load(str(cfg))
            try:
                placedb = PlaceDB.PlaceDB()
                placedb(params)
                t0 = time.time()
                placer = NonLinearPlace.NonLinearPlace(params, placedb, None)
                placer(params, placedb, params.global_place_stages[0]["learning_rate"])
                dp_wall = time.time() - t0
            except Exception as exc:
                log(f"  [DP-B-R2 {label}] failed: {exc}")
                return None
            log(f"  [DP-B-R2 {label}] done in {dp_wall:.0f}s")
            out_dir = Path(params.result_dir) / benchmark.name
            out_dir.mkdir(parents=True, exist_ok=True)
            gp_pl = out_dir / f"{benchmark.name}.gp.pl"
            try:
                placedb.write(params, str(gp_pl))
            except Exception as exc:
                log(f"  [DP-B-R2 {label}] write failed: {exc}")
                return None
            if not gp_pl.exists():
                return None
            pl_map = _bk_reader.parse_pl(gp_pl)
            n = benchmark.num_macros
            sizes = benchmark.macro_sizes.cpu().numpy()
            placement = benchmark.macro_positions.clone().detach()
            for i in range(n):
                if i in pl_map:
                    llx, lly = pl_map[i]
                    placement[i, 0] = llx / SCALE + float(sizes[i, 0]) / 2.0
                    placement[i, 1] = lly / SCALE + float(sizes[i, 1]) / 2.0
            return placement.to(torch.float32)

    # Multi-seed DP basin selection: run DP N times on GPU (fast), pick
    # the one with lowest legalized proxy. Variance probe on ibm10 showed
    # 5 runs produced legal-proxy range 1.376-1.573 (14% spread) — picking
    # best-of-5 gives ~7% better start point than single-run typical.
    # Rule-compliant: single algorithm, deterministic best-pick from variance.
    n_seeds = int(os.environ.get("MULTI_DP_N", "5"))
    log(f"  [multi-DP] running {n_seeds} DP basins (GPU variance source)")
    best_basin = None
    best_legal_proxy = float("inf")
    best_label = ""
    for i in range(n_seeds):
        pl = _run_one(target_density=0.85, stop_overflow=0.07,
                       dp_iter=1000, dp_lr=0.01,
                       label=f"multi-{i+1}/{n_seeds}")
        if pl is None:
            continue
        try:
            legal_i, _ = greedy_macro_legalize(pl, benchmark, step_size_frac=0.01)
            legal_i, _ = project_overlaps(legal_i, benchmark)
            ovl_i = compute_overlap_metrics(legal_i, benchmark)["overlap_count"]
        except Exception as exc:
            log(f"  [multi-DP {i+1}] legalize failed: {exc}")
            continue
        if ovl_i > 0:
            # Skip basins that can't legalize cleanly
            log(f"  [multi-DP {i+1}] ovl={ovl_i} after greedy — skipping")
            continue
        legal_proxy_i = float(compute_proxy_cost(legal_i, benchmark, plc)["proxy_cost"])
        log(f"  [multi-DP {i+1}] legal_proxy={legal_proxy_i:.5f}")
        if legal_proxy_i < best_legal_proxy:
            best_legal_proxy = legal_proxy_i
            best_basin = pl  # return raw (pre-legal) for hybrid's _polish_dp_basin
            best_label = f"multi-{i+1}"

    if best_basin is not None:
        log(f"  [multi-DP] best of {n_seeds}: {best_label} legal_proxy={best_legal_proxy:.5f}")
        return best_basin

    # Fallback: adaptive Stage B if all multi-DP runs failed to legalize
    sizes = benchmark.macro_sizes.cpu().numpy()
    hard_idx = benchmark.num_hard_macros
    macro_area = float((sizes[:hard_idx, 0] * sizes[:hard_idx, 1]).sum())
    canvas_area = float(benchmark.canvas_width * benchmark.canvas_height)
    macro_density = macro_area / max(canvas_area, 1e-9)
    target_b = float(min(0.85, max(0.40, macro_density * 1.5)))
    log(f"  [multi-DP fallback to stageB; macro_density={macro_density:.3f}→target={target_b:.3f}]")
    pl_b = _run_one(target_density=target_b, stop_overflow=0.02, dp_iter=2000,
                    dp_lr=0.005, label="stageB-fallback")
    return pl_b


def _polish_dp_basin(
    raw_placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    cd_hard_cap_s: float,
    lns_budget_s: float,
    sa_budget_s: float,
    log,
) -> Tuple[torch.Tensor, float]:
    """Apply E25 polish pipeline (CD + LNS + SA) to a DP basin.

    Equivalent to E25's polish, but starting from DP init instead of
    SDF init.
    """
    # Greedy legalize first (DP outputs overlap).
    log(f"  [DP polish] greedy_macro_legalize")
    legal, leg_stats = greedy_macro_legalize(
        raw_placement, benchmark, step_size_frac=0.01,
    )
    legal, _ = project_overlaps(legal, benchmark)
    legal_proxy = float(compute_proxy_cost(legal, benchmark, plc)["proxy_cost"])
    ovl = compute_overlap_metrics(legal, benchmark)["overlap_count"]
    log(f"  [DP polish] legal proxy={legal_proxy:.5f} ovl={ovl}")
    if ovl > 0:
        # Fall back to extended_legalize (jitter+project loop) — handles
        # NG45-class commercial designs where greedy can't seat all macros.
        log(f"  [DP polish] {ovl} residuals → extended_legalize fallback")
        try:
            legal, ext_stats = extended_legalize(
                legal, benchmark, max_passes=20, jitter_scale=0.5, seed=42,
            )
            ovl = ext_stats["final_overlap"]
            legal_proxy = float(compute_proxy_cost(legal, benchmark, plc)["proxy_cost"])
            log(f"  [DP polish] extended_legalize: ovl={ovl} "
                f"proxy={legal_proxy:.5f} passes={ext_stats['passes_used']}")
        except Exception as exc:
            log(f"  [DP polish] extended_legalize raised: {exc}; lane will be excluded")
            return None, float('inf')
    if ovl > 0:
        log(f"  [DP polish] still {ovl} overlaps after extended_legalize — "
            f"lane will be excluded from plateau pick")
        return None, float('inf')

    placement_f64 = legal.detach().clone().to(torch.float64)
    ev = IncrementalProxyEvaluator(benchmark, plc, placement_f64)

    n_hard = benchmark.num_hard_macros
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable = [i for i in range(benchmark.num_macros) if not bool(fixed[i])]
    hard_movable = [i for i in range(n_hard) if not bool(fixed[i])]

    log(f"  [DP polish] CD (cap {cd_hard_cap_s:.0f}s)")
    run_cd_adaptive(
        evaluator=ev, benchmark=benchmark, plc=plc, movable=movable,
        min_time_s=min(60.0, cd_hard_cap_s * 0.1),
        hard_cap_s=cd_hard_cap_s,
        patience=3, plateau_threshold=0.001, log_fn=None,
    )

    log(f"  [DP polish] LNS ({lns_budget_s:.0f}s)")
    run_lns_gridbin(
        evaluator=ev, benchmark=benchmark, plc=plc,
        hard_movable=hard_movable, time_budget_s=lns_budget_s,
        destroy_frac=0.05, destroy_cap=30, seed=42, log_fn=None,
    )

    log(f"  [DP polish] SA-v2 ({sa_budget_s:.0f}s)")
    run_sa_polish_v2(
        evaluator=ev, benchmark=benchmark, plc=plc,
        hard_movable=hard_movable, time_budget_s=sa_budget_s,
        T0=5e-4, Tf=1e-6, seed=42, breakpoint_budget=12, log_fn=None,
    )

    polished = ev.placement.detach().clone().to(torch.float32)
    polished, _ = project_overlaps(polished, benchmark)
    final_ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
    proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
    if final_ovl > 0:
        log(f"  [DP polish] FAILED VALIDITY: {final_ovl} residual overlaps "
            f"after polish — DP lane will be excluded from plateau pick")
        return None, float('inf')
    return polished, proxy


class CDLNSSAMultiDPDeepSaddlePlacer:
    """Shoom-style MultiDP + Deep Hessian saddle, no E25/E41 lanes.

    Time budget reorganized to match top-leaderboard approaches:
    - Phase 1: N=10 GPU DP basins (~140s), pick best legalized
    - Phase 2: CD-adaptive refine on best basin (~1000s)
    - Phase 3: LNS + SA polish (~500s)
    - Phase 4: DEEP cascade saddle (k=5 eigvecs, 5 eps values, ~1500s)

    Drops E25/E41 to free time for deep saddle search.
    Matches Shoom v17 (0.978) + vmallela v7 (1.011) techniques combined.
    """

    def __init__(
        self,
        n_dp_basins: int = 10,
        max_saddle_iters: int = 10,
        saddle_n_eigvecs: int = 5,
        saddle_eps: Tuple[float, ...] = (0.1, 0.3, 0.7, 1.5, 3.0),
        saddle_polish_budget: float = 60.0,
        budget_seconds: Optional[float] = 3300.0,
        verbose: bool = True,
    ):
        self.n_dp_basins = n_dp_basins
        self.max_saddle_iters = max_saddle_iters
        self.saddle_n_eigvecs = saddle_n_eigvecs
        self.saddle_eps = saddle_eps
        self.saddle_polish_budget = saddle_polish_budget
        self.budget_seconds = budget_seconds
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSAMultiDPDeepSaddlePlacer ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Import deep cascade saddle (separate from cascading_saddle import)
        import sys as _sys
        _sys.path.insert(0, str(_E84_CODE))
        from cascading_saddle_deep import cascading_saddle_escape_deep

        # Phase 1: Multi-DP basin selection (GPU)
        # Run DP N times, pick best legalized basin
        log(f"  Phase 1: {self.n_dp_basins} GPU DP basins")
        best_raw = None
        best_legal_proxy = float("inf")
        for i in range(self.n_dp_basins):
            if deadline is not None and (deadline - time.time()) < 2000.0:
                log(f"    [budget guard] skipping remaining basins ({deadline-time.time():.0f}s left)")
                break
            pl = _try_run_dreamplace(benchmark, plc, log)
            if pl is None:
                continue
            try:
                legal_i, _ = greedy_macro_legalize(pl, benchmark, step_size_frac=0.01)
                legal_i, _ = project_overlaps(legal_i, benchmark)
                ovl_i = compute_overlap_metrics(legal_i, benchmark)["overlap_count"]
            except Exception:
                continue
            if ovl_i > 0:
                continue
            lp = float(compute_proxy_cost(legal_i, benchmark, plc)["proxy_cost"])
            log(f"    multi-DP {i+1}/{self.n_dp_basins}: legal_proxy={lp:.5f}")
            if lp < best_legal_proxy:
                best_legal_proxy = lp
                best_raw = pl
        if best_raw is None:
            raise RuntimeError(f"All {self.n_dp_basins} DP basins failed to legalize on {benchmark.name}")
        log(f"  Phase 1 done: best basin legal_proxy={best_legal_proxy:.5f} (wall={time.time()-t0:.0f}s)")

        # Phase 2-3: Polish best basin
        remaining = deadline - time.time() if deadline else 3000.0
        # Allocate: 50% polish, 45% deep saddle, 5% buffer
        polish_total = remaining * 0.50
        cd_cap = polish_total * 0.65
        lns_cap = polish_total * 0.20
        sa_cap = polish_total * 0.15
        log(f"  Phase 2-3: polish CD={cd_cap:.0f}s LNS={lns_cap:.0f}s SA={sa_cap:.0f}s")
        dp_polished, dp_polished_proxy = _polish_dp_basin(
            best_raw, benchmark, plc,
            cd_hard_cap_s=cd_cap, lns_budget_s=lns_cap, sa_budget_s=sa_cap,
            log=log,
        )
        if dp_polished is None:
            raise RuntimeError(f"Polish failed on {benchmark.name}")
        log(f"  Polish done: proxy={dp_polished_proxy:.5f} (wall={time.time()-t0:.0f}s)")

        # Phase 4: Deep cascade saddle on polished basin
        remaining = deadline - time.time() if deadline else 1500.0
        saddle_budget = max(60.0, remaining - 60.0)
        log(f"  Phase 4: deep cascade saddle k={self.saddle_n_eigvecs} "
            f"(budget {saddle_budget:.0f}s, max_iters {self.max_saddle_iters})")
        try:
            cascade_state, stats = cascading_saddle_escape_deep(
                dp_polished, benchmark, plc,
                max_iters=self.max_saddle_iters,
                n_eigvecs=self.saddle_n_eigvecs,
                eps_values=self.saddle_eps,
                polish_budget=self.saddle_polish_budget,
                total_budget_s=saddle_budget,
                log=lambda s: None,
            )
            cascade_proxy = float(compute_proxy_cost(cascade_state, benchmark, plc)["proxy_cost"])
            log(f"  Deep saddle done: {cascade_proxy:.5f} iters={stats.get('iters_run','?')}")
        except Exception as exc:
            log(f"  Deep saddle failed: {exc}; falling back to polished")
            cascade_state = dp_polished
            cascade_proxy = dp_polished_proxy

        # Best of polished vs saddle
        if cascade_proxy < dp_polished_proxy:
            best_placement = cascade_state
            best_proxy = cascade_proxy
            best_name = "deep-saddle"
        else:
            best_placement = dp_polished
            best_proxy = dp_polished_proxy
            best_name = "polish"
        log(f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"(polish={dp_polished_proxy:.5f}, saddle={cascade_proxy:.5f}) "
            f"total wall={time.time()-t0:.0f}s")

        ovl = compute_overlap_metrics(best_placement, benchmark)["overlap_count"]
        if ovl > 0:
            raise RuntimeError(
                f"CascadeDPLane winner has {ovl} hard-macro overlaps"
            )
        return best_placement
