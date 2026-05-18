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

_ROOT = Path(__file__).resolve().parents[2]
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
    """Run DREAMPlace; return raw placement or None on failure.

    Discovery order for DREAMPlace install:
      1. ``DREAMPLACE_ROOT`` env var (if set + exists)
      2. ``/submission/dreamplace_install`` (eval_docker bundled extras)
      3. ``submit_deps/dreamplace_install/`` relative to repo root

    Invocation mode:
      - If ``DP_USE_NESTED_DOCKER=1``: legacy nested-Docker mode (requires
        Docker socket; will NOT work in eval_docker's air-gapped runtime).
      - Default: direct ``sys.executable Placer.py config.json`` subprocess —
        works inside eval_docker where the host's PyTorch is already
        DREAMPlace-compatible (2.5.1+cuda12.4).
    """
    dp_root_env = os.environ.get("DREAMPLACE_ROOT")
    candidates = []
    if dp_root_env:
        candidates.append(Path(dp_root_env))
    candidates.extend([
        Path("/submission/dreamplace_install"),
        _ROOT / "submit_deps" / "dreamplace_install",
    ])

    dp_root = None
    for c in candidates:
        if c.exists() and (c / "dreamplace" / "Placer.py").exists():
            dp_root = c
            break
    if dp_root is None:
        log(f"  [DP] no DREAMPlace install found (tried "
            f"{[str(c) for c in candidates]}); skipping DP lane")
        return None
    log(f"  [DP] using install at {dp_root}")

    use_nested_docker = int(os.environ.get("DP_USE_NESTED_DOCKER", "0"))
    dp_image = os.environ.get("DP_DOCKER_IMAGE", "dreamplace:custom")
    use_gpu = int(os.environ.get("DP_USE_GPU", "1"))
    SCALE = float(_bk_writer.SCALE)

    def _run_one(target_density, stop_overflow, dp_iter, dp_lr, label):
        """Single DP invocation. Returns placement tensor or None."""
        with tempfile.TemporaryDirectory(prefix=f"dp_{benchmark.name}_", dir="/tmp") as tmp:
            tmp = Path(tmp)
            _bk_writer._write_nodes(benchmark, tmp)
            _bk_writer._write_pl(benchmark, tmp)
            _bk_writer._write_nets(benchmark, tmp)
            _bk_writer._write_scl(benchmark, tmp)
            _bk_writer._write_wts(benchmark, tmp)
            _bk_writer._write_aux(benchmark, tmp)
            log(f"  [DP {label}] target_density={target_density:.3f} "
                f"stop_overflow={stop_overflow} iter={dp_iter} lr={dp_lr}")
            cfg = tmp / "dp.json"
            # Build DP config. Paths are inside the work tmpdir; the input
            # aux file path differs between direct-Python (host abs) and
            # nested-Docker (mount point) modes.
            aux_path_for_cfg = (
                f"/work/{benchmark.name}.aux" if use_nested_docker
                else str(tmp / f"{benchmark.name}.aux")
            )
            result_dir_for_cfg = "/work" if use_nested_docker else str(tmp)
            cfg.write_text(json.dumps({
                "aux_input": aux_path_for_cfg,
                "target_density": target_density,
                "density_weight": 8e-5,
                "gpu": use_gpu,
                "num_threads": int(os.environ.get("DP_NUM_THREADS", "8")),
                "global_place_stages": [{
                    "num_bins_x": 1024, "num_bins_y": 1024,
                    "iteration": dp_iter, "learning_rate": dp_lr,
                    "wirelength": "weighted_average", "optimizer": "nesterov",
                }],
                "legalize_flag": 0,
                "detailed_place_flag": 0,
                "stop_overflow": stop_overflow,
                "result_dir": result_dir_for_cfg,
            }))

            if use_nested_docker:
                uid, gid = os.getuid(), os.getgid()
                docker_args = ["sudo", "docker", "run", "--rm",
                               "--user", f"{uid}:{gid}",
                               "-v", f"{tmp}:/work",
                               "-v", f"{dp_root}:/dp_install:ro",
                               "-w", "/work"]
                if use_gpu:
                    docker_args.extend(["--gpus", "all"])
                cmd = docker_args + [
                    dp_image,
                    "python", "/dp_install/dreamplace/Placer.py", "/work/dp.json",
                ]
            else:
                # Direct Python invocation (eval_docker default).
                # Placer.py uses sibling-module imports (e.g. `import Params`)
                # so the dreamplace/ subdir must be on PYTHONPATH. The
                # install root also goes on PYTHONPATH for `dreamplace.ops`
                # subpackage discovery.
                env = os.environ.copy()
                env["PYTHONPATH"] = (
                    f"{dp_root / 'dreamplace'}:{dp_root}"
                    f":{env.get('PYTHONPATH', '')}"
                )
                cmd = [sys.executable,
                       str(dp_root / "dreamplace" / "Placer.py"),
                       str(cfg)]
                env_for_subprocess = env
            t0 = time.time()
            try:
                if use_nested_docker:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                else:
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=900,
                        env=env_for_subprocess, cwd=str(tmp),
                    )
            except subprocess.TimeoutExpired:
                log(f"  [DP {label}] timed out 900 s")
                return None
            if proc.returncode != 0:
                log(f"  [DP {label}] failed exit={proc.returncode}: {proc.stderr[-500:]}")
                return None
            log(f"  [DP {label}] done in {time.time() - t0:.0f}s")
            gp_pl = tmp / f"{benchmark.name}.gp.pl"
            if not gp_pl.exists():
                for c in tmp.rglob("*.gp.pl"):
                    gp_pl = c
                    break
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


class CDLNSSACascadeDPLanePlacer:
    """Cascade with DP as a third basin lane (with fair polish budget)."""

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
            e25_kwargs = dict(
                cd_hard_cap_s=B * 0.12, lns_budget_s=B * 0.04, sa_budget_s=B * 0.04,
            )
            e41_kwargs = dict(
                cd_hard_cap_s=B * 0.12, lns_budget_s=B * 0.035, sa_budget_s=B * 0.035,
                kjoint_budget_s=B * 0.03,
            )
            dp_kwargs = dict(
                cd_hard_cap_s=B * 0.12, lns_budget_s=B * 0.04, sa_budget_s=B * 0.04,
            )
            log(f"  budget {B:.0f}s; E25={sum(e25_kwargs.values()):.0f}s "
                f"E41={sum(e41_kwargs.values()):.0f}s "
                f"DP={sum(dp_kwargs.values()):.0f}s (+ DP raw ~30s + legalize ~60s) "
                f"cascade saddle target ~{B * 0.29:.0f}s")
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

        # Phase 3: DP (with fair polish)
        # Skip DP if remaining time can't fit (DP lane ~0.20×B + ~90s raw + ~300s
        # min saddle floor). Stricter than the 700s threshold used pre-2026-05-13
        # to prevent the ibm17 wall-cap violation: DP lane consumes 700-900s on
        # large benches, leaving <100s for saddle when remaining was exactly 700s.
        dp = None
        dp_proxy = float("inf")
        # Threshold = DP lane budget (0.20×B + ~90s raw) + minimum saddle floor (600s).
        # On large benches like ibm17, E25+E41 over-run budgeted shares by 600-900s
        # (init phases unbounded), so DP-skip cutoff must reserve saddle floor.
        if self.budget_seconds is not None:
            dp_skip_threshold = self.budget_seconds * 0.20 + 690.0
        else:
            dp_skip_threshold = 1350.0
        if deadline is not None and (deadline - time.time()) < dp_skip_threshold:
            log(f"  Phase 3: DP SKIPPED ({deadline - time.time():.0f}s left, "
                f"need {dp_skip_threshold:.0f}s)")
        else:
            log("  Phase 3: DREAMPlace + full polish")
            dp_raw = _try_run_dreamplace(benchmark, plc, log)
            if dp_raw is not None:
                dp, dp_proxy = _polish_dp_basin(
                    dp_raw, benchmark, plc,
                    cd_hard_cap_s=dp_kwargs["cd_hard_cap_s"],
                    lns_budget_s=dp_kwargs["lns_budget_s"],
                    sa_budget_s=dp_kwargs["sa_budget_s"],
                    log=log,
                )
                log(f"  DP polished done: proxy={dp_proxy:.5f} "
                    f"(wall={time.time() - t0:.0f}s)")

        # Phase 4: plateau pick = best of {E25, E41, DP}
        candidates = [(e25_proxy, e25, "E25")]
        if e41 is not None:
            candidates.append((e41_proxy, e41, "E41"))
        if dp is not None:
            candidates.append((dp_proxy, dp, "DP"))
        candidates.sort(key=lambda c: c[0])
        plateau_proxy, plateau, plateau_label = candidates[0]
        log(f"  hybrid plateau pick: {plateau_label} ({plateau_proxy:.5f}) "
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

        # Phase 6: best of all
        all_candidates = [(e25_proxy, e25, "E25"),
                          (cascade_proxy, cascade_state, "cascade")]
        if e41 is not None:
            all_candidates.append((e41_proxy, e41, "E41"))
        if dp is not None:
            all_candidates.append((dp_proxy, dp, "DP"))
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
