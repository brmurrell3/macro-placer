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

    # Two-stage DP: try fast/lenient first (works for IBM out of the box);
    # if greedy_legalize can't fully clean the basin, retry with tighter
    # adaptive config (works for NG45-class commercial designs). Single
    # algorithm — branch is on observed overlap count, not bench name.
    pl_a = _run_one(target_density=0.85, stop_overflow=0.07, dp_iter=1000, dp_lr=0.01,
                    label="stageA")
    if pl_a is None:
        return None
    legal_a, _ = greedy_macro_legalize(pl_a, benchmark, step_size_frac=0.01)
    legal_a, _ = project_overlaps(legal_a, benchmark)
    ovl_a = compute_overlap_metrics(legal_a, benchmark)["overlap_count"]
    log(f"  [DP stageA] greedy_legalize ovl={ovl_a}")
    if ovl_a == 0:
        return pl_a

    # Stage B: tighter config, adaptive target_density.
    sizes = benchmark.macro_sizes.cpu().numpy()
    hard_idx = benchmark.num_hard_macros
    macro_area = float((sizes[:hard_idx, 0] * sizes[:hard_idx, 1]).sum())
    canvas_area = float(benchmark.canvas_width * benchmark.canvas_height)
    macro_density = macro_area / max(canvas_area, 1e-9)
    target_b = float(min(0.85, max(0.40, macro_density * 1.5)))
    log(f"  [DP stageA basin can't legalize ({ovl_a} residuals); stageB "
        f"macro_density={macro_density:.3f}→target={target_b:.3f}]")
    pl_b = _run_one(target_density=target_b, stop_overflow=0.02, dp_iter=2000,
                    dp_lr=0.005, label="stageB")
    return pl_b if pl_b is not None else pl_a


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


class CDLNSSACascadeDPPortfolioPlacer:
    """Portfolio placer with 4 basin lanes: E25 + E41 + stock-DP + B-R2-DP.

    Phase 1: E25 (SDF init + CD + LNS + SA)
    Phase 2: E41 (DPO init + CD + LNS + SA + K-joint)
    Phase 3: Stock DREAMPlace + full polish (λ_topk=0)
    Phase 4: B-R2 DREAMPlace + full polish (λ_topk via BR2_LAMBDA_PORTFOLIO_TOPK env, default 0.05)
    Phase 5: Plateau pick = best of {E25, E41, stock-DP, B-R2-DP}
    Phase 6: Cascading saddle escape on plateau
    Phase 7: Best-of-all across lanes + cascade

    The two DP runs let each bench select its own best DP basin
    (canonical-aware vs canonical-blind) without per-bench tuning —
    it's a single algorithm with deterministic plateau pick.

    Set BR2_LAMBDA_PORTFOLIO_TOPK env to control B-R2 lane's lambda
    (default 0.05 based on Phase A ibm18 -1.3% lift).
    """

    def __init__(
        self,
        max_iters: int = 5,
        eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
        polish_budget: float = 180.0,
        budget_seconds: Optional[float] = 3300.0,
        verbose: bool = True,
    ):
        self.max_iters = max_iters
        self.eps_values = eps_values
        self.polish_budget = polish_budget
        self.budget_seconds = budget_seconds
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSACascadeDPLanePlacer ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Budget allocation: 3 lanes × (CD + LNS + SA) + cascade saddle.
        # Total budget = E25 (0.20) + E41 (0.22) + DP (0.20) + DP raw+legal (~0.03)
        #              + cascade saddle (0.35) = 1.00.
        # Tightened from 0.14 to 0.12 CD-cap per lane (2026-05-13) after
        # ibm17 ran 3655s = 60.9 min (over 60-min hard cap by 55s) because
        # E25+E41 lanes consumed 3113s leaving saddle only 177s. Reclaim
        # 0.06×B for cascade saddle.
        if self.budget_seconds is not None:
            B = self.budget_seconds
            # Portfolio budget: tighter per-lane to fit 2 DP runs.
            # 4 polish lanes × (CD 0.10 + LNS 0.03 + SA 0.03) = 0.16×B per lane.
            # Total: 4 × 0.16×B + 2 × DP raw + cascade saddle 0.18×B = ~0.84×B + raw/overhead ~0.16×B.
            e25_kwargs = dict(
                cd_hard_cap_s=B * 0.10, lns_budget_s=B * 0.03, sa_budget_s=B * 0.03,
            )
            e41_kwargs = dict(
                cd_hard_cap_s=B * 0.10, lns_budget_s=B * 0.025, sa_budget_s=B * 0.025,
                kjoint_budget_s=B * 0.02,
            )
            dp_kwargs = dict(
                cd_hard_cap_s=B * 0.10, lns_budget_s=B * 0.03, sa_budget_s=B * 0.03,
            )
            log(f"  portfolio budget {B:.0f}s; E25={sum(e25_kwargs.values()):.0f}s "
                f"E41={sum(e41_kwargs.values()):.0f}s "
                f"DPx2={sum(dp_kwargs.values()):.0f}s each (+ DP raw ~30s × 2 + legalize ~60s × 2) "
                f"cascade saddle target ~{B * 0.18:.0f}s")
        else:
            e25_kwargs = e41_kwargs = {}
            dp_kwargs = dict(cd_hard_cap_s=600.0, lns_budget_s=180.0, sa_budget_s=180.0)

        # Phase 1: E25 (SDF init + polish)
        log("  Phase 1: E25 (SDF + CD + LNS + SA)")
        e25 = CDLNSSAPlacer(**e25_kwargs).place(benchmark)
        e25_proxy = float(compute_proxy_cost(e25, benchmark, plc)["proxy_cost"])
        log(f"  E25 done: proxy={e25_proxy:.5f} (wall={time.time() - t0:.0f}s)")

        # Phase 2: E41 (DPO init + polish)
        e41 = None
        e41_proxy = float("inf")
        if deadline is not None and (deadline - time.time()) < 700.0:
            log(f"  Phase 2: SKIPPED ({deadline - time.time():.0f}s left)")
        else:
            log("  Phase 2: E41 (DPO + CD + LNS + SA + K-joint)")
            e41 = CDLNSSADPOKJointPlacer(**e41_kwargs).place(benchmark)
            e41_proxy = float(compute_proxy_cost(e41, benchmark, plc)["proxy_cost"])
            log(f"  E41 done: proxy={e41_proxy:.5f} (wall={time.time() - t0:.0f}s)")

        # Phase 3: Stock DP (λ_topk=0) + polish
        # Phase 4: B-R2 DP (λ_topk=BR2_LAMBDA_PORTFOLIO_TOPK, default 0.05) + polish
        # Per-lane skip if remaining < (DP polish budget + saddle floor ~400s).
        dp_stock = None
        dp_stock_proxy = float("inf")
        dp_br2 = None
        dp_br2_proxy = float("inf")
        # Tighter skip threshold for portfolio: each DP lane needs ~520s
        # (polish + raw); saddle floor reduced to 400s.
        if self.budget_seconds is not None:
            dp_lane_budget = self.budget_seconds * 0.16 + 90.0
        else:
            dp_lane_budget = 600.0

        portfolio_topk_lambda = float(os.environ.get("BR2_LAMBDA_PORTFOLIO_TOPK", "0.05"))

        for lane_idx, (lane_name, lambda_topk_val) in enumerate([
            ("stock-DP", 0.0),
            (f"B-R2-DP-λ{portfolio_topk_lambda}", portfolio_topk_lambda),
        ]):
            if deadline is not None and (deadline - time.time()) < (dp_lane_budget + 400.0):
                log(f"  Phase 3+{lane_idx}: {lane_name} SKIPPED ({deadline - time.time():.0f}s left)")
                continue
            log(f"  Phase 3+{lane_idx}: {lane_name} (DP + polish)")
            # Set env vars for this lane (read by patched PlaceObj.py)
            os.environ["BR2_LAMBDA_TOPK"] = str(lambda_topk_val)
            os.environ["BR2_LAMBDA_RUDY"] = "0.0"
            try:
                dp_raw = _try_run_dreamplace(benchmark, plc, log)
            except Exception as exc:
                log(f"  {lane_name} DP raw failed: {exc}; skipping lane")
                dp_raw = None
            if dp_raw is not None:
                pol, pol_proxy = _polish_dp_basin(
                    dp_raw, benchmark, plc,
                    cd_hard_cap_s=dp_kwargs["cd_hard_cap_s"],
                    lns_budget_s=dp_kwargs["lns_budget_s"],
                    sa_budget_s=dp_kwargs["sa_budget_s"],
                    log=log,
                )
                log(f"  {lane_name} polished done: proxy={pol_proxy:.5f} "
                    f"(wall={time.time() - t0:.0f}s)")
                if lambda_topk_val == 0.0:
                    dp_stock, dp_stock_proxy = pol, pol_proxy
                else:
                    dp_br2, dp_br2_proxy = pol, pol_proxy

        # Clear env so other code paths default to 0
        os.environ.pop("BR2_LAMBDA_TOPK", None)
        os.environ.pop("BR2_LAMBDA_RUDY", None)

        # Phase 5: plateau pick = best of {E25, E41, stock-DP, B-R2-DP}
        candidates = [(e25_proxy, e25, "E25")]
        if e41 is not None:
            candidates.append((e41_proxy, e41, "E41"))
        if dp_stock is not None:
            candidates.append((dp_stock_proxy, dp_stock, "stock-DP"))
        if dp_br2 is not None:
            candidates.append((dp_br2_proxy, dp_br2, "B-R2-DP"))
        candidates.sort(key=lambda c: c[0])
        plateau_proxy, plateau, plateau_label = candidates[0]
        log(f"  portfolio plateau pick: {plateau_label} ({plateau_proxy:.5f}) "
            f"[{', '.join(f'{n}={p:.5f}' for p, _, n in candidates)}]")

        # Phase 5: cascading saddle escape
        # Saddle has no mid-iter deadline check; iter once started runs to
        # completion (400-700s on large benches). To guarantee compliance
        # with the 60-min hard cap, skip saddle if remaining < ~400s
        # (one iter likely overruns) and reserve 60s for post-processing
        # (validate, write JSON, eval log).
        cascade_state = plateau
        cascade_proxy = plateau_proxy
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining < 400.0:
                log(f"  Phase 5: SKIPPED ({remaining:.0f}s left; "
                    f"saddle iter > remaining)")
            else:
                cascade_budget = remaining - 60.0
                log(f"  Phase 5: cascading saddle (budget {cascade_budget:.0f}s, "
                    f"max_iters {self.max_iters})")
                try:
                    cascade_state, stats = cascading_saddle_escape(
                        plateau, benchmark, plc,
                        max_iters=self.max_iters,
                        eps_values=self.eps_values,
                        polish_budget=self.polish_budget,
                        total_budget_s=cascade_budget,
                        log=lambda s: None,
                    )
                    cascade_proxy = float(compute_proxy_cost(
                        cascade_state, benchmark, plc)["proxy_cost"])
                    log(f"  cascade done: {cascade_proxy:.5f} "
                        f"iters={stats.get('iters_run', '?')}")
                except Exception as exc:
                    log(f"  cascade failed: {exc}; falling back to plateau")
                    cascade_state = plateau
                    cascade_proxy = plateau_proxy

        # Phase 6: best of all (portfolio + cascade)
        all_candidates = [(e25_proxy, e25, "E25"),
                          (cascade_proxy, cascade_state, "cascade")]
        if e41 is not None:
            all_candidates.append((e41_proxy, e41, "E41"))
        if dp_stock is not None:
            all_candidates.append((dp_stock_proxy, dp_stock, "stock-DP"))
        if dp_br2 is not None:
            all_candidates.append((dp_br2_proxy, dp_br2, "B-R2-DP"))
        all_candidates.sort(key=lambda c: c[0])
        best_proxy, best_placement, best_name = all_candidates[0]
        log(f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"({', '.join(f'{n}={p:.5f}' for p, _, n in all_candidates)}) "
            f"total wall={time.time() - t0:.0f}s")

        ovl = compute_overlap_metrics(best_placement, benchmark)["overlap_count"]
        if ovl > 0:
            raise RuntimeError(
                f"CascadeDPLane winner has {ovl} hard-macro overlaps"
            )
        return best_placement
