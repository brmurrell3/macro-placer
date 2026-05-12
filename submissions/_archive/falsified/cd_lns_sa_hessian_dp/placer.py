"""CDLNSSAHessianDP — E74 champion + optional DREAMPlace init lane.

Identical to `submissions/cd_lns_sa_hessian/placer.py` except a 3rd init
lane is added: DREAMPlace global placement, invoked as a subprocess if
the env var `DREAMPLACE_ROOT` points to an install. The DREAMPlace
output is treated like another candidate plateau alongside E25 and E41;
the best-of-{E25, E41, DREAMPlace} plateau is then fed through Hessian
saddle escape (same as E74).

Graceful fallback: if DREAMPlace isn't installed, the placer behaves
identically to the E74 champion. This lets a single submission entry
work on both our M3 Max dev box (no DREAMPlace) and the partcl AMD EPYC
+ RTX 6000 Ada eval box (with DREAMPlace).

To enable on cloud:
  export DREAMPLACE_ROOT=/opt/DREAMPlace
  uv run evaluate submissions/cd_lns_sa_hessian_dp/placer.py --all
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.sparse.linalg as spla
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

# E25 placer.
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
CDLNSSAPlacer = _E25_MOD.CDLNSSAPlacer

# E41 placer.
from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import CDLNSSADPOKJointPlacer

# Hessian saddle escape primitives (reuse the champion's).
from submissions.cd_lns_sa_hessian.placer import (
    _SmoothProxy,
    _saddle_escape,
)

# DREAMPlace I/O helpers from E76.
_E76 = _ROOT / "experiments" / "E76_dreamplace_integration" / "code"
sys.path.insert(0, str(_E76))
import tilos_to_bookshelf as _bk_writer  # noqa: E402
import bookshelf_to_pt as _bk_reader  # noqa: E402


def _try_run_dreamplace(
    benchmark: Benchmark,
    plc,
    log,
) -> Optional[torch.Tensor]:
    """Run DREAMPlace on the benchmark via Bookshelf intermediate, return
    placement [num_macros, 2] or None if DREAMPlace isn't available / fails.
    """
    dp_root = os.environ.get("DREAMPLACE_ROOT")
    if not dp_root or not Path(dp_root).exists():
        log(f"  [DP] DREAMPLACE_ROOT not set or missing; skipping DREAMPlace lane")
        return None
    placer_py = Path(dp_root) / "dreamplace" / "Placer.py"
    if not placer_py.exists():
        log(f"  [DP] {placer_py} missing; skipping")
        return None

    with tempfile.TemporaryDirectory(prefix=f"dp_{benchmark.name}_") as tmp:
        tmp = Path(tmp)
        log(f"  [DP] writing Bookshelf inputs to {tmp}")
        # Write all 6 Bookshelf files; reuse E76's writer helpers.
        _bk_writer._write_nodes(benchmark, tmp)
        _bk_writer._write_pl(benchmark, tmp)
        _bk_writer._write_nets(benchmark, tmp)
        _bk_writer._write_scl(benchmark, tmp)
        _bk_writer._write_wts(benchmark, tmp)
        _bk_writer._write_aux(benchmark, tmp)

        # Write DREAMPlace config JSON.
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        cfg = tmp / "dreamplace.json"
        # Disable legalize_flag: our nodes are macros, not standard cells,
        # and DREAMPlace's abacus_legalize fails on non-row-height nodes.
        # We use only the global placement output (.gp.pl) and let our own
        # project_overlaps + CD polish handle legalization downstream.
        cfg.write_text(f"""{{
  "aux_input": "{tmp / (benchmark.name + '.aux')}",
  "target_density": 0.85,
  "density_weight": 8e-5,
  "gpu": 1,
  "global_place_stages": [
    {{"num_bins_x": 1024, "num_bins_y": 1024,
     "iteration": 1000, "learning_rate": 0.01,
     "wirelength": "weighted_average", "optimizer": "nesterov"}}
  ],
  "legalize_flag": 0,
  "detailed_place_flag": 0,
  "stop_overflow": 0.07,
  "result_dir": "{tmp}"
}}""")

        log(f"  [DP] invoking DREAMPlace ({placer_py})")
        # DREAMPlace's .so files are linked against system torch, so we need
        # the system python interpreter (which also has the deps installed:
        # shapely, matplotlib, etc.). uv's python doesn't see those.
        dp_python = os.environ.get("DREAMPLACE_PYTHON", "")
        if not dp_python:
            for cand in ("/usr/bin/python3", "/usr/bin/python"):
                if Path(cand).exists():
                    dp_python = cand
                    break
        if not dp_python:
            dp_python = sys.executable
        log(f"  [DP] using interpreter: {dp_python}")
        t0 = time.time()
        try:
            proc = subprocess.run(
                [dp_python, str(placer_py), str(cfg)],
                cwd=str(dp_root),
                capture_output=True,
                text=True,
                timeout=900,  # 15 min cap on DREAMPlace
            )
        except subprocess.TimeoutExpired:
            log(f"  [DP] timed out after 15 min; skipping lane")
            return None
        wall = time.time() - t0
        if proc.returncode != 0:
            log(f"  [DP] DREAMPlace exited {proc.returncode}; stderr tail:")
            log(proc.stderr[-500:] if proc.stderr else "<empty>")
            return None
        log(f"  [DP] DREAMPlace done in {wall:.0f}s")

        # Find the output .pl. DREAMPlace writes <name>.gp.pl by default.
        gp_pl = tmp / f"{benchmark.name}.gp.pl"
        if not gp_pl.exists():
            # Some DREAMPlace versions write to result_dir/<name>/<name>.gp.pl
            candidates = list(tmp.rglob("*.gp.pl"))
            if not candidates:
                log(f"  [DP] no .gp.pl output found in {tmp}; skipping")
                return None
            gp_pl = candidates[0]

        log(f"  [DP] parsing {gp_pl}")
        pl_map = _bk_reader.parse_pl(gp_pl)

        # Map back to our placement tensor. DREAMPlace works in the scaled
        # integer units we wrote in tilos_to_bookshelf.py (SCALE = 1000),
        # so divide before mapping to centers.
        SCALE = float(_bk_writer.SCALE)
        n = benchmark.num_macros
        sizes = benchmark.macro_sizes.cpu().numpy()
        placement = benchmark.macro_positions.clone().detach()
        for i in range(n):
            if i in pl_map:
                llx, lly = pl_map[i]
                llx /= SCALE
                lly /= SCALE
                placement[i, 0] = llx + float(sizes[i, 0]) / 2.0
                placement[i, 1] = lly + float(sizes[i, 1]) / 2.0
        placement = placement.to(torch.float32)
        placement, _ = project_overlaps(placement, benchmark)
        ovl = compute_overlap_metrics(placement, benchmark)["overlap_count"]
        if ovl > 0:
            # Brief CD polish (60s) — DREAMPlace global placement isn't fully
            # legal; CD's project + cost-aware sweep cleans it up.
            log(f"  [DP] {ovl} overlaps after project; running brief CD polish")
            ev = IncrementalProxyEvaluator(benchmark, plc, placement.clone())
            n_hard = benchmark.num_hard_macros
            fixed = benchmark.macro_fixed.cpu().numpy()
            movable_idx = [i for i in range(n_hard) if not bool(fixed[i])]
            run_cd_adaptive(
                ev, benchmark, plc, movable_idx,
                min_time_s=30.0, hard_cap_s=60.0,
                patience=2, plateau_threshold=0.001, log_fn=None,
            )
            placement = ev.placement.detach().clone().to(torch.float32)
            ovl = compute_overlap_metrics(placement, benchmark)["overlap_count"]
            log(f"  [DP] after CD polish: {ovl} overlaps")
            if ovl > 0:
                log(f"  [DP] unable to legalize fully; skipping DP lane")
                return None
        return placement


class CDLNSSAHessianDPPlacer:
    """E74 champion + optional DREAMPlace init lane.

    Identical pipeline to CDLNSSAHessianPlacer except for an extra phase
    between E41 and Hessian saddle escape: try DREAMPlace; if available
    and produces a valid placement, include it in the plateau pick.
    """

    def __init__(
        self,
        n_eigvecs: int = 2,
        eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
        polish_budget: float = 240.0,
        budget_seconds: Optional[float] = 3300.0,
        verbose: bool = True,
    ):
        self.n_eigvecs = n_eigvecs
        self.eps_values = eps_values
        self.polish_budget = polish_budget
        self.budget_seconds = budget_seconds
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True) if self.verbose else None
        log(f"=== CDLNSSAHessianDPPlacer ({benchmark.name}) ===")
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        if self.budget_seconds is not None:
            B = self.budget_seconds
            e25_kwargs = dict(
                cd_hard_cap_s=B * 0.20, lns_budget_s=B * 0.06, sa_budget_s=B * 0.06,
            )
            e41_kwargs = dict(
                cd_hard_cap_s=B * 0.20, lns_budget_s=B * 0.05, sa_budget_s=B * 0.05,
                kjoint_budget_s=B * 0.05,
            )
            log(f"  budget enforced: {self.budget_seconds:.0f}s")
        else:
            e25_kwargs, e41_kwargs = {}, {}

        # 1. E25.
        log("  Phase 1: E25 (CDLNSSAPlacer)")
        e25 = CDLNSSAPlacer(**e25_kwargs).place(benchmark)
        e25_proxy = float(compute_proxy_cost(e25, benchmark, plc)["proxy_cost"])
        log(f"  E25 done: proxy={e25_proxy:.5f} (wall={time.time() - t0:.0f}s)")

        # 2. E41.
        e41 = None
        e41_proxy = float("inf")
        if deadline is not None and (deadline - time.time()) < 400.0:
            log(f"  Phase 2: SKIPPED ({deadline - time.time():.0f}s left)")
        else:
            log("  Phase 2: E41 (CDLNSSADPOKJointPlacer)")
            e41 = CDLNSSADPOKJointPlacer(**e41_kwargs).place(benchmark)
            e41_proxy = float(compute_proxy_cost(e41, benchmark, plc)["proxy_cost"])
            log(f"  E41 done: proxy={e41_proxy:.5f} (wall={time.time() - t0:.0f}s)")

        # 3. DREAMPlace (optional, ~30-300s; only if budget left).
        dp = None
        dp_proxy = None
        if deadline is not None and (deadline - time.time()) < 600.0:
            log(f"  Phase 3: DP SKIPPED ({deadline - time.time():.0f}s left)")
        else:
            log("  Phase 3: DREAMPlace (optional)")
            dp = _try_run_dreamplace(benchmark, plc, log)
            if dp is not None:
                dp_proxy = float(compute_proxy_cost(dp, benchmark, plc)["proxy_cost"])
                log(f"  DP done: proxy={dp_proxy:.5f} (wall={time.time() - t0:.0f}s)")

        # 4. Hybrid plateau pick.
        candidates_plateau = [(e25_proxy, e25, "E25")]
        if e41 is not None:
            candidates_plateau.append((e41_proxy, e41, "E41"))
        if dp is not None:
            candidates_plateau.append((dp_proxy, dp, "DP"))
        candidates_plateau.sort(key=lambda c: c[0])
        plateau_proxy, plateau, plateau_label = candidates_plateau[0]
        log(f"  hybrid plateau pick: {plateau_label} ({plateau_proxy:.5f})")

        # 5. Hessian saddle escape with deadline.
        saddle_state = plateau
        saddle_proxy = plateau_proxy
        if deadline is not None and (deadline - time.time()) < 30.0:
            log(f"  Phase 4: SKIPPED ({deadline - time.time():.0f}s left)")
        else:
            log("  Phase 4: Hessian saddle escape")
            try:
                saddle_state, saddle_proxy = _saddle_escape(
                    plateau, benchmark, plc,
                    n_eigvecs=self.n_eigvecs,
                    eps_values=self.eps_values,
                    polish_budget=self.polish_budget,
                    deadline=deadline,
                    log=log if self.verbose else None,
                )
            except Exception as exc:
                log(f"  saddle escape failed: {exc}; falling back to plateau")

        # 6. Best of all.
        all_candidates = [(e25_proxy, e25, "E25"), (saddle_proxy, saddle_state, "saddle")]
        if e41 is not None:
            all_candidates.append((e41_proxy, e41, "E41"))
        if dp is not None:
            all_candidates.append((dp_proxy, dp, "DP"))
        all_candidates.sort(key=lambda c: c[0])
        best_proxy, best_placement, best_name = all_candidates[0]
        log(f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"({', '.join(f'{n}={p:.5f}' for p, _, n in all_candidates)})  "
            f"total wall={time.time() - t0:.0f}s")

        ovl = compute_overlap_metrics(best_placement, benchmark)["overlap_count"]
        if ovl > 0:
            raise RuntimeError(
                f"DP-Hessian winner has {ovl} hard-macro overlaps"
            )
        return best_placement
