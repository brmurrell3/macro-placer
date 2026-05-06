"""E84 v2 — use DREAMPlace as INIT for our pipeline (project + CD + saddle).

Pipeline:
  1. DREAMPlace global+legalize (~30 sec on GPU).
  2. project_overlaps (extended iters — DREAMPlace's macros may overlap heavily).
  3. CD adaptive (legalizes any residual overlaps; polishes to plateau).
  4. Hessian saddle escape on top.

Wall budget on RTX 4070:
  Phase 1 (DREAMPlace):   ~30-60 sec
  Phase 2 (extended project): ~30 sec
  Phase 3 (CD adaptive):  CAP — to be determined; key question is whether
                          DREAMPlace's basin lets CD converge faster than
                          SDF init (which takes ~25 min in E25).
  Phase 4 (Hessian saddle): ~18 min

If CD converges in ~5 min from DREAMPlace init, total wall ~25 min.

This is more like an E25/E83 pipeline with DREAMPlace replacing SDF init.
Hypothesis: DREAMPlace basin is in a different / better region than SDF,
so post-CD plateau is lower → saddle escape lifts it further → final proxy
beats E74's 1.0666.

Reference: experiments/E84_dreamplace_saddle/manifest.md
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Patch loader for Windows path normalization.
import macro_place.loader as _loader_mod
_orig_load_benchmark = _loader_mod.load_benchmark


def _patched_load_benchmark(netlist_file, plc_file=None, name=None):
    netlist_file = str(netlist_file).replace("\\", "/")
    if plc_file is not None:
        plc_file = str(plc_file).replace("\\", "/")
    return _orig_load_benchmark(netlist_file, plc_file, name)


_loader_mod.load_benchmark = _patched_load_benchmark

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# LNS legalizer — uses grid-bin destroy/reinsert to actively resolve overlaps.
# (E25's LNS phase does the same; we re-import it here.)
sys.path.insert(0, str(_REPO_ROOT))
from experiments.E39_kmacro_joint_lns.code.cd_lns_sa_kjoint import run_lns_gridbin

# Reuse Hessian saddle escape from E74.
_E74_PATH = _REPO_ROOT / "submissions" / "cd_lns_sa_hessian" / "placer.py"
_E74_SPEC = importlib.util.spec_from_file_location("e74_for_e84v2", str(_E74_PATH))
_E74_MOD = importlib.util.module_from_spec(_E74_SPEC)
_E74_SPEC.loader.exec_module(_E74_MOD)
_saddle_escape = _E74_MOD._saddle_escape

# DREAMPlace wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_dreamplace import run_dreamplace


def _force_legalize(
    evaluator,
    benchmark: Benchmark,
    plc,
    max_passes: int = 200,
    log_fn=None,
):
    """Force-legalize: move overlapping hard macros to nearest legal grid bin
    regardless of proxy cost.

    Unlike `_gridbin_reinsert` (which only commits proxy-improving moves),
    this function ALWAYS moves a macro if it's currently in an overlapping
    state and a legal alternative exists.  Sacrifices proxy for legality.
    """
    n_hard = benchmark.num_hard_macros
    macro_sizes_np = evaluator.macro_sizes.cpu().numpy()
    fixed = benchmark.macro_fixed.cpu().numpy()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    grid_w = cw / plc.grid_col
    grid_h = ch / plc.grid_row

    # Build sorted list of all legal grid-bin centers (we'll search nearest-first
    # when relocating an overlapping macro).
    bin_centers = []
    for col in range(int(plc.grid_col)):
        cx = (col + 0.5) * grid_w
        for row in range(int(plc.grid_row)):
            cy = (row + 0.5) * grid_h
            bin_centers.append((cx, cy))
    bin_centers_np = np.array(bin_centers, dtype=np.float64)

    moved_total = 0
    for it in range(max_passes):
        # Find overlapping pairs.
        pos = evaluator.placement[:n_hard].cpu().numpy().astype(np.float64)
        half_w = macro_sizes_np[:n_hard, 0] / 2.0
        half_h = macro_sizes_np[:n_hard, 1] / 2.0
        dx = np.abs(pos[:, 0:1] - pos[:, 0:1].T)
        dy = np.abs(pos[:, 1:2] - pos[:, 1:2].T)
        min_dx = half_w[:, None] + half_w[None, :]
        min_dy = half_h[:, None] + half_h[None, :]
        ovl = (dx < min_dx - 1e-9) & (dy < min_dy - 1e-9)
        np.fill_diagonal(ovl, False)
        # Get unique overlapping macros (those involved in any overlap).
        overlapping_set = set()
        for a, b in np.argwhere(np.triu(ovl)):
            overlapping_set.add(int(a))
            overlapping_set.add(int(b))
        if not overlapping_set:
            if log_fn is not None:
                log_fn(f"    force_legalize: clean after {it} passes "
                       f"({moved_total} relocations)")
            return moved_total

        # Sort by overlap-degree (most-overlapping first).
        ovl_count = np.sum(ovl, axis=1)
        candidates = sorted(overlapping_set, key=lambda i: -int(ovl_count[i]))

        moved_this_pass = 0
        for idx in candidates:
            if bool(fixed[idx]):
                continue
            cur_xy = (float(pos[idx, 0]), float(pos[idx, 1]))
            # Sort grid bins by distance to current position.
            dists = (bin_centers_np[:, 0] - cur_xy[0]) ** 2 + (bin_centers_np[:, 1] - cur_xy[1]) ** 2
            sorted_bins = np.argsort(dists)
            placed = False
            # Try ALL grid bins (was 200 nearest — too restrictive when many
            # macros are overlapping; we need to consider the full canvas).
            for bi in sorted_bins:
                bx, by = bin_centers_np[bi]
                if bx < half_w[idx] or bx > cw - half_w[idx]:
                    continue
                if by < half_h[idx] or by > ch - half_h[idx]:
                    continue
                if abs(bx - cur_xy[0]) < 1e-6 and abs(by - cur_xy[1]) < 1e-6:
                    continue
                # Use _is_legal_2d-equivalent check.
                pos_check = evaluator.placement[:n_hard].cpu().numpy().astype(np.float64)
                ddx = np.abs(pos_check[:, 0] - bx)
                ddy = np.abs(pos_check[:, 1] - by)
                mind_x = half_w[idx] + macro_sizes_np[:n_hard, 0] / 2.0
                mind_y = half_h[idx] + macro_sizes_np[:n_hard, 1] / 2.0
                blockers = (ddx < mind_x - 1e-9) & (ddy < mind_y - 1e-9)
                blockers[idx] = False
                if not bool(np.any(blockers)):
                    evaluator.move(idx, (bx, by))
                    moved_this_pass += 1
                    moved_total += 1
                    placed = True
                    break
            # If nothing legal found, leave it (rare).
        if log_fn is not None and (it < 5 or it % 10 == 0):
            log_fn(f"    force_legalize pass {it}: {len(overlapping_set)} "
                   f"overlapping macros, moved {moved_this_pass}")
        if moved_this_pass == 0:
            if log_fn is not None:
                log_fn(f"    force_legalize stuck after {it} passes "
                       f"({moved_total} total relocations, "
                       f"{len(overlapping_set)} macros still overlapping)")
            return moved_total
    return moved_total


def _project_overlaps_extended(
    placement: torch.Tensor,
    benchmark: Benchmark,
    max_iter: int = 500,
    log_fn=None,
) -> Tuple[torch.Tensor, int, int]:
    """Like macro_place.cd_core.project_overlaps but with a higher iteration cap.

    Returns (placement, iters_used, residual_overlap_count).
    """
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.cpu().numpy()
    pos = placement.cpu().numpy().astype(np.float64).copy()

    half_w = sizes[:n_hard, 0] / 2
    half_h = sizes[:n_hard, 1] / 2

    last_count = -1
    for it in range(max_iter):
        dx = np.abs(pos[:n_hard, 0:1] - pos[:n_hard, 0:1].T)
        dy = np.abs(pos[:n_hard, 1:2] - pos[:n_hard, 1:2].T)
        min_dx = half_w[:, None] + half_w[None, :]
        min_dy = half_h[:, None] + half_h[None, :]
        ovl = (dx < min_dx) & (dy < min_dy)
        np.fill_diagonal(ovl, False)
        pairs = np.argwhere(np.triu(ovl))
        n_pairs = len(pairs)
        if n_pairs == 0:
            return torch.tensor(pos, dtype=placement.dtype), it, 0
        if log_fn is not None and (it % 50 == 0 or it == max_iter - 1):
            log_fn(f"    project_overlaps iter {it}: {n_pairs} overlapping pairs")
        for a, b in pairs:
            mov_a = not bool(fixed[a])
            mov_b = not bool(fixed[b])
            if not (mov_a or mov_b):
                continue
            vx = float(min_dx[a, b] - dx[a, b])
            vy = float(min_dy[a, b] - dy[a, b])
            if vx < vy:
                # Push apart along x.
                if pos[a, 0] < pos[b, 0]:
                    if mov_a and mov_b:
                        pos[a, 0] -= vx / 2
                        pos[b, 0] += vx / 2
                    elif mov_a:
                        pos[a, 0] -= vx
                    elif mov_b:
                        pos[b, 0] += vx
                else:
                    if mov_a and mov_b:
                        pos[a, 0] += vx / 2
                        pos[b, 0] -= vx / 2
                    elif mov_a:
                        pos[a, 0] += vx
                    elif mov_b:
                        pos[b, 0] -= vx
            else:
                if pos[a, 1] < pos[b, 1]:
                    if mov_a and mov_b:
                        pos[a, 1] -= vy / 2
                        pos[b, 1] += vy / 2
                    elif mov_a:
                        pos[a, 1] -= vy
                    elif mov_b:
                        pos[b, 1] += vy
                else:
                    if mov_a and mov_b:
                        pos[a, 1] += vy / 2
                        pos[b, 1] -= vy / 2
                    elif mov_a:
                        pos[a, 1] += vy
                    elif mov_b:
                        pos[b, 1] -= vy
        # Clamp to canvas.
        pos[:n_hard, 0] = np.clip(pos[:n_hard, 0], half_w, cw - half_w)
        pos[:n_hard, 1] = np.clip(pos[:n_hard, 1], half_h, ch - half_h)
        last_count = n_pairs

    return torch.tensor(pos, dtype=placement.dtype), max_iter, last_count


class DREAMPlaceInitPlacer:
    """E84 v2 — DREAMPlace init → project → CD → saddle escape.

    Same architecture as E83 but with DREAMPlace replacing SDF init.
    Hypothesis: DREAMPlace's basin is in a different / better region than
    SDF, so post-CD plateau is lower → saddle lift gives sub-1.0666 proxy.
    """

    def __init__(
        self,
        cd_cap_s: float = 1500.0,
        cd_min_time_s: float = 300.0,
        cd_patience: int = 3,
        cd_plateau_threshold: float = 0.001,
        lns_budget_s: float = 600.0,  # LNS legalizes residual overlaps
        n_eigvecs: int = 1,
        eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
        polish_budget: float = 180.0,
        gpu: bool = True,
        verbose: bool = True,
    ):
        self.cd_cap_s = float(cd_cap_s)
        self.cd_min_time_s = float(cd_min_time_s)
        self.cd_patience = int(cd_patience)
        self.cd_plateau_threshold = float(cd_plateau_threshold)
        self.lns_budget_s = float(lns_budget_s)
        self.n_eigvecs = int(n_eigvecs)
        self.eps_values = tuple(eps_values)
        self.polish_budget = float(polish_budget)
        self.gpu = gpu
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = (lambda s: print(s, flush=True)) if self.verbose else (lambda s: None)
        log(f"=== DREAMPlaceInitPlacer ({benchmark.name}) ===")
        t0 = time.time()

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        # Phase 1: DREAMPlace produces an initial placement (with overlaps).
        work_dir = (
            _REPO_ROOT / "experiments" / "E84_dreamplace_saddle" / "dp_runs"
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        log("  Phase 1: DREAMPlace global placement (GPU)")
        dp_placement = run_dreamplace(benchmark, work_dir, gpu=self.gpu, log_fn=log)
        dp_proxy = float(compute_proxy_cost(dp_placement, benchmark, plc)["proxy_cost"])
        dp_overlaps = compute_overlap_metrics(dp_placement, benchmark)["overlap_count"]
        log(f"  DREAMPlace raw: proxy={dp_proxy:.5f}, overlaps={dp_overlaps} "
            f"(wall={time.time() - t0:.0f}s)")

        # Phase 2: extended project_overlaps to fully legalize.
        log("  Phase 2: extended project_overlaps (max_iter=500)")
        legal, proj_iters, residual = _project_overlaps_extended(
            dp_placement, benchmark, max_iter=500, log_fn=log,
        )
        legal_proxy = float(compute_proxy_cost(legal, benchmark, plc)["proxy_cost"])
        legal_overlaps = compute_overlap_metrics(legal, benchmark)["overlap_count"]
        log(f"  Legalized: proxy={legal_proxy:.5f}, overlaps={legal_overlaps}, "
            f"proj iters={proj_iters} (residual_pairs={residual}) "
            f"(wall={time.time() - t0:.0f}s)")

        if legal_overlaps > 0:
            log(f"  WARNING: {legal_overlaps} residual overlaps — CD will absorb them")

        # Phase 3: CD adaptive (legalizes residual + polishes to plateau).
        # Use float64 for the evaluator (matches E25 / E83 patterns).
        log(f"  Phase 3: CD adaptive (cap={self.cd_cap_s:.0f}s)")
        legal_f64 = legal.detach().clone().to(torch.float64)
        evaluator = IncrementalProxyEvaluator(benchmark, plc, legal_f64)
        movable = [
            i for i in range(benchmark.num_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        cd_stats = run_cd_adaptive(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            movable=movable,
            min_time_s=self.cd_min_time_s,
            hard_cap_s=self.cd_cap_s,
            patience=self.cd_patience,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=log if self.verbose else None,
        )
        cd_proxy = evaluator.current_cost()["proxy"]
        log(f"    CD done: exit={cd_stats['exit_reason']}, "
            f"sweeps={cd_stats['sweeps']}, accepts={cd_stats['total_moves']}, "
            f"wall={cd_stats['wall_total_s']:.1f}s, proxy={cd_proxy:.5f}")

        # Phase 3.5: FORCE-LEGALIZE residual overlaps.
        # CD optimized proxy but didn't resolve input overlaps. _gridbin_reinsert
        # only commits proxy-improving moves, leaving overlapping macros stuck.
        # _force_legalize moves overlapping macros to legal positions even at
        # cost of proxy.
        log(f"  Phase 3.5a: force_legalize (resolve residual overlaps)")
        n_moved = _force_legalize(evaluator, benchmark, plc, log_fn=log)
        forced_proxy = evaluator.current_cost()["proxy"]
        log(f"    force_legalize done: {n_moved} relocations, proxy={forced_proxy:.5f}")

        # Phase 3.5a-bis: SECOND CD pass — re-optimize from the legal state.
        # force_legalize moves macros to grid-bin centers regardless of proxy
        # cost (lifts proxy from 0.85 → 1.10 typically).  A second CD pass
        # re-optimizes WITHIN the legal manifold, recovering most of the basin
        # advantage.
        log(f"  Phase 3.5a-bis: second CD adaptive (cap={self.cd_cap_s:.0f}s)")
        cd_stats2 = run_cd_adaptive(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            movable=movable,
            min_time_s=self.cd_min_time_s,
            hard_cap_s=self.cd_cap_s,
            patience=self.cd_patience,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=log if self.verbose else None,
        )
        cd2_proxy = evaluator.current_cost()["proxy"]
        log(f"    CD#2 done: exit={cd_stats2['exit_reason']}, "
            f"sweeps={cd_stats2['sweeps']}, "
            f"wall={cd_stats2['wall_total_s']:.1f}s, proxy={cd2_proxy:.5f}")

        # Phase 3.5b: LNS to polish — final cleanup pass.
        hard_movable = [
            i for i in range(benchmark.num_hard_macros)
            if not bool(benchmark.macro_fixed[i])
        ]
        log(f"  Phase 3.5b: LNS polish (budget={self.lns_budget_s:.0f}s)")
        lns_stats = run_lns_gridbin(
            evaluator=evaluator,
            benchmark=benchmark,
            plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.lns_budget_s,
            destroy_frac=0.05,
            destroy_cap=30,
            seed=42,
            log_fn=log if self.verbose else None,
        )
        lns_proxy = evaluator.current_cost()["proxy"]
        log(f"    LNS done: samples={lns_stats['samples']}, "
            f"Δ={lns_stats['total_improvement']:+.5f}, "
            f"wall={lns_stats['wall_total_s']:.1f}s, proxy={lns_proxy:.5f}")

        # Pull placement back; preserve fixed macros.
        cd_placement = evaluator.placement.detach().clone().cpu()
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            cd_placement[fixed_mask] = original_positions[fixed_mask]
        cd_placement = cd_placement.to(torch.float32)
        plateau_proxy = float(
            compute_proxy_cost(cd_placement, benchmark, plc)["proxy_cost"]
        )
        post_lns_overlaps = compute_overlap_metrics(cd_placement, benchmark)["overlap_count"]
        log(f"  Plateau after CD+LNS: proxy={plateau_proxy:.5f}, "
            f"overlaps={post_lns_overlaps}")

        # Phase 4: Hessian saddle escape.
        log(f"  Phase 4: Hessian saddle escape "
            f"(n_eigvecs={self.n_eigvecs}, eps={self.eps_values}, "
            f"polish={self.polish_budget:.0f}s)")
        try:
            saddle_state, saddle_proxy = _saddle_escape(
                cd_placement, benchmark, plc,
                n_eigvecs=self.n_eigvecs,
                eps_values=self.eps_values,
                polish_budget=self.polish_budget,
                log=log if self.verbose else None,
            )
        except Exception as exc:
            log(f"  Saddle escape failed: {exc}; using CD plateau")
            saddle_state = cd_placement
            saddle_proxy = plateau_proxy

        # Best of (CD plateau, saddle).
        if saddle_proxy < plateau_proxy - 1e-7:
            best_state, best_proxy, best_name = saddle_state, saddle_proxy, "saddle"
        else:
            best_state, best_proxy, best_name = cd_placement, plateau_proxy, "CD"
        total_wall = time.time() - t0
        log(
            f"  WINNER: {best_name} proxy={best_proxy:.5f} "
            f"(CD={plateau_proxy:.5f}, saddle={saddle_proxy:.5f})  "
            f"total wall={total_wall:.0f}s"
        )

        ovl = compute_overlap_metrics(best_state, benchmark)["overlap_count"]
        if ovl > 0:
            raise RuntimeError(
                f"E84 v2 winner has {ovl} hard-macro overlaps"
            )
        return best_state
