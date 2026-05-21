"""E149 — Outlier-targeted perturbation on top of v2 pipeline.

Architecture:

  1. V4 + Gaussian descent (`SmoothGlobalPlacerV4Gaussian.place()`) — fast
     GPU basin. (same as v2 thinkorplace-v2)
  2. Greedy macro legalize + project_overlaps.
  3. CD polish 1 (~400 s) — settle into the local minimum.
  4. Outlier-perturb loop (3 iterations, 100 s each):
     - Compute per-macro cost contribution score:
         wl_contrib + 0.5*density_contrib + 0.5*cong_contrib
       (matches canonical proxy_cost weights).
     - Pick the top K=8 by score.
     - Reset those K to uniformly random valid positions inside a *local*
       neighborhood (±20% canvas radius around current position, clamped
       to the legal canvas bbox).
     - greedy_macro_legalize -> project_overlaps to recover legality.
     - CD polish 100 s.
     - Accept iff canonical proxy_cost strictly improves.
  5. Final CD polish 2 (~300 s).

Total budget 1500 s/bench. Within partcl 60-min/bench cap.

Hypothesis: K-pin random restart (E142) was falsified because random
selection put noise on macros that were already well-placed, so polish
just returns to the prior plateau. Outlier-targeted selection picks the
worst contributors — perturbation has *direction*, like a Newton step on
macro-level coordinates.

DO NOT mutate shipped placer files — this is an experiment.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _ROOT,
    _HERE,
    _ROOT / "experiments" / "E76_dreamplace_integration" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E127_v4_gaussian" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

from smooth_global_placer_v4_gaussian import SmoothGlobalPlacerV4Gaussian  # noqa: E402
from macro_legalizer import greedy_macro_legalize  # noqa: E402


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cd_polish(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    budget_s: float,
    plateau_threshold: float = 0.001,
) -> torch.Tensor:
    """Run CD-adaptive polish under a hard wall budget."""
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement)
    movable = [
        i for i in range(benchmark.num_macros)
        if not bool(benchmark.macro_fixed[i])
    ]
    run_cd_adaptive(
        evaluator, benchmark, plc, movable,
        min_time_s=min(30.0, budget_s * 0.5),
        hard_cap_s=budget_s,
        patience=5,
        plateau_threshold=plateau_threshold,
        log_fn=None,
    )
    return evaluator.placement.detach().clone().to(torch.float32)


def _macro_outlier_scores(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
) -> np.ndarray:
    """Compute per-macro cost-contribution scores.

    Score for macro i =
        wl_contrib[i] + 0.5 * density_contrib[i] + 0.5 * cong_contrib[i]

    where contributions are *proxies* for canonical contribution
    (we don't need exact attribution — we just need relative ordering
    that identifies the worst placed macros).

    WL contribution
        For each net the macro connects to, distance from macro center to
        the bbox center of that net. The larger the distance, the more
        that net is being stretched by this macro. Sum over the macro's
        nets.

    Density contribution
        Density-grid value at the macro's grid cell. (Read from plc's
        density grid via the macro's center.)

    Congestion contribution
        H+V routing-congestion at the macro's grid cell.

    Returns
    -------
    scores : np.ndarray shape (num_macros,)
        Higher = worse-placed.
    """
    pos = placement.detach().cpu().numpy().astype(np.float64)
    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    n = benchmark.num_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    # --- 1) Build per-macro list of net indices using net_nodes.
    macro_nets: List[List[int]] = [[] for _ in range(n)]
    for net_idx in range(benchmark.num_nets):
        nodes = benchmark.net_nodes[net_idx]
        if nodes is None or len(nodes) == 0:
            continue
        seen = set()
        for nd in nodes.tolist():
            # net_nodes uses macro-granularity indices (after dedup); only
            # consider those < num_macros (ports use indices >= num_macros
            # in net_pin_nodes but net_nodes typically has owner-only ids).
            if 0 <= nd < n and nd not in seen:
                seen.add(nd)
                macro_nets[nd].append(net_idx)

    # --- 2) Compute per-net bbox center using current pin positions.
    #     If net_pin_nodes is populated, use pin-level (more accurate).
    #     Else fall back to net_nodes macro-center.
    have_pin_nodes = (
        len(benchmark.net_pin_nodes) > 0
        and benchmark.net_pin_nodes[0] is not None
    )
    port_pos = (
        benchmark.port_positions.detach().cpu().numpy().astype(np.float64)
        if benchmark.port_positions is not None and len(benchmark.port_positions) > 0
        else np.zeros((0, 2), dtype=np.float64)
    )
    n_macros = benchmark.num_macros
    n_ports = port_pos.shape[0]
    net_centers = np.zeros((benchmark.num_nets, 2), dtype=np.float64)
    have_center = np.zeros(benchmark.num_nets, dtype=bool)

    for net_idx in range(benchmark.num_nets):
        x_min, y_min = float("inf"), float("inf")
        x_max, y_max = float("-inf"), float("-inf")
        any_pt = False
        if have_pin_nodes:
            pn = benchmark.net_pin_nodes[net_idx]
            if pn is not None and len(pn) > 0:
                arr = pn.cpu().numpy() if hasattr(pn, "cpu") else np.asarray(pn)
                for row in arr:
                    owner = int(row[0])
                    pin_local = int(row[1]) if arr.shape[1] > 1 else 0
                    if owner < n_macros:
                        # macro center + pin offset (if available)
                        cx, cy = pos[owner, 0], pos[owner, 1]
                        if (
                            owner < benchmark.num_hard_macros
                            and owner < len(benchmark.macro_pin_offsets)
                            and benchmark.macro_pin_offsets[owner] is not None
                            and len(benchmark.macro_pin_offsets[owner]) > 0
                        ):
                            offs = benchmark.macro_pin_offsets[owner]
                            offs_arr = (
                                offs.cpu().numpy() if hasattr(offs, "cpu")
                                else np.asarray(offs)
                            )
                            if pin_local < offs_arr.shape[0]:
                                cx = cx + float(offs_arr[pin_local, 0])
                                cy = cy + float(offs_arr[pin_local, 1])
                        x_min = min(x_min, cx); x_max = max(x_max, cx)
                        y_min = min(y_min, cy); y_max = max(y_max, cy)
                        any_pt = True
                    elif owner - n_macros < n_ports:
                        px = port_pos[owner - n_macros, 0]
                        py = port_pos[owner - n_macros, 1]
                        x_min = min(x_min, px); x_max = max(x_max, px)
                        y_min = min(y_min, py); y_max = max(y_max, py)
                        any_pt = True
        if not any_pt:
            # Fallback: macro centers from net_nodes
            nodes = benchmark.net_nodes[net_idx]
            if nodes is None or len(nodes) == 0:
                continue
            for nd in nodes.tolist():
                if 0 <= nd < n_macros:
                    cx, cy = pos[nd, 0], pos[nd, 1]
                    x_min = min(x_min, cx); x_max = max(x_max, cx)
                    y_min = min(y_min, cy); y_max = max(y_max, cy)
                    any_pt = True
        if any_pt:
            net_centers[net_idx, 0] = 0.5 * (x_min + x_max)
            net_centers[net_idx, 1] = 0.5 * (y_min + y_max)
            have_center[net_idx] = True

    # --- 3) WL contribution: per-macro sum of (macro_center -> net_center)
    wl_contrib = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if not macro_nets[i]:
            continue
        s = 0.0
        for net_idx in macro_nets[i]:
            if not have_center[net_idx]:
                continue
            dx = pos[i, 0] - net_centers[net_idx, 0]
            dy = pos[i, 1] - net_centers[net_idx, 1]
            s += abs(dx) + abs(dy)
        wl_contrib[i] = s

    # Normalize WL by canvas perimeter so it's in the same scale ballpark
    # as canonical wirelength_cost.
    wl_norm = (cw + ch) * max(1, benchmark.num_nets)
    if wl_norm > 0:
        wl_contrib /= wl_norm

    # --- 4) Density and congestion contribution: read plc grids at each
    # macro's grid cell.
    density_contrib = np.zeros(n, dtype=np.float64)
    cong_contrib = np.zeros(n, dtype=np.float64)
    grid_rows = int(plc.grid_row)
    grid_cols = int(plc.grid_col)
    # Refresh plc's grids by calling get_density_cost / get_congestion_cost
    # once with the current placement. compute_proxy_cost already does
    # _set_placement and triggers updates.
    _ = compute_proxy_cost(placement, benchmark, plc)
    # plc.grid_occupied is plc.grid_height x plc.grid_width (density count)
    # We use macro_density per cell if available; else read from
    # internal arrays.
    H_cong = np.array(plc.H_routing_cong, dtype=np.float64)
    V_cong = np.array(plc.V_routing_cong, dtype=np.float64)
    # Density grid: plc.get_grid_cells_density() returns list[rows*cols].
    density_grid = None
    try:
        if hasattr(plc, "get_grid_cells_density"):
            arr = np.array(plc.get_grid_cells_density(), dtype=np.float64)
            if arr.size == grid_rows * grid_cols:
                density_grid = arr.reshape(grid_rows, grid_cols)
    except Exception:
        density_grid = None
    if density_grid is None:
        for cand in ("grid_occupied", "_grid_density", "grid_density"):
            if hasattr(plc, cand):
                obj = getattr(plc, cand)
                try:
                    arr = np.array(obj, dtype=np.float64)
                    if arr.size == grid_rows * grid_cols:
                        density_grid = arr.reshape(grid_rows, grid_cols)
                        break
                except Exception:
                    continue
    grid_w = cw / max(1, grid_cols)
    grid_h = ch / max(1, grid_rows)
    for i in range(n):
        cx = float(np.clip(pos[i, 0], 0.0, cw - 1e-6))
        cy = float(np.clip(pos[i, 1], 0.0, ch - 1e-6))
        col = int(np.clip(np.floor(cx / grid_w), 0, grid_cols - 1))
        row = int(np.clip(np.floor(cy / grid_h), 0, grid_rows - 1))
        flat = row * grid_cols + col
        if 0 <= flat < len(H_cong):
            cong_contrib[i] = H_cong[flat] + V_cong[flat]
        if density_grid is not None:
            density_contrib[i] = density_grid[row, col]
        else:
            # crude proxy: macro footprint area / cell area = density push
            density_contrib[i] = (
                (sizes[i, 0] * sizes[i, 1]) / max(grid_w * grid_h, 1e-9)
            )

    # Normalize density and congestion to comparable scale.
    if density_contrib.max() > 0:
        density_contrib /= density_contrib.max()
    if cong_contrib.max() > 0:
        cong_contrib /= cong_contrib.max()
    if wl_contrib.max() > 0:
        wl_contrib /= wl_contrib.max()

    # Final blended score with canonical-proxy weights.
    score = wl_contrib + 0.5 * density_contrib + 0.5 * cong_contrib
    return score


def _perturb_outliers_local(
    placement: torch.Tensor,
    benchmark: Benchmark,
    chosen_idx: List[int],
    rng: np.random.RandomState,
    radius_frac: float = 0.20,
) -> torch.Tensor:
    """Move each chosen macro to a uniformly random valid position inside
    a local neighborhood (±radius_frac * canvas) clamped to legal bbox.
    """
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    pos = placement.detach().cpu().numpy().astype(np.float64).copy()
    r_x = cw * radius_frac
    r_y = ch * radius_frac
    for i in chosen_idx:
        hw = sizes[i, 0] / 2.0
        hh = sizes[i, 1] / 2.0
        x_lo_legal, x_hi_legal = hw, cw - hw
        y_lo_legal, y_hi_legal = hh, ch - hh
        if x_hi_legal <= x_lo_legal or y_hi_legal <= y_lo_legal:
            continue
        x_lo = max(x_lo_legal, pos[i, 0] - r_x)
        x_hi = min(x_hi_legal, pos[i, 0] + r_x)
        y_lo = max(y_lo_legal, pos[i, 1] - r_y)
        y_hi = min(y_hi_legal, pos[i, 1] + r_y)
        if x_hi <= x_lo or y_hi <= y_lo:
            continue
        pos[i, 0] = rng.uniform(x_lo, x_hi)
        pos[i, 1] = rng.uniform(y_lo, y_hi)
    return torch.tensor(pos, dtype=torch.float32)


def _select_outliers(
    scores: np.ndarray, benchmark: Benchmark, k: int
) -> List[int]:
    """Top-K by score over hard movable macros."""
    fixed = benchmark.macro_fixed.cpu().numpy()
    n_hard = benchmark.num_hard_macros
    candidates = [i for i in range(n_hard) if not bool(fixed[i])]
    if len(candidates) == 0:
        return []
    cand_scores = np.array([scores[i] for i in candidates], dtype=np.float64)
    order = np.argsort(-cand_scores)
    k_eff = min(k, len(candidates))
    return [candidates[int(order[j])] for j in range(k_eff)]


class E149OutlierPerturbPlacer:
    """V4+Gaussian basin -> CD1 -> 3 outlier-targeted perturb iters -> CD2."""

    def __init__(
        self,
        budget_seconds: Optional[float] = 1500.0,
        # Descent params (mirror v2 thinkorplace-v2)
        num_steps: int = 500,
        lr_frac: float = 0.005,
        gamma_start_frac: float = 5e-3,
        gamma_end_frac: float = 5e-5,
        overlap_lambda_end: float = 10.0,
        overlap_ramp_pct: float = 0.7,
        init: str = "sdf",
        # CD params
        cd1_budget_s: float = 400.0,
        cd2_budget_s: float = 300.0,
        cd_plateau_threshold: float = 0.001,
        # Outlier-perturb params
        outlier_k: int = 8,
        outlier_n_iters: int = 3,
        outlier_polish_budget: float = 100.0,
        outlier_radius_frac: float = 0.20,
        rng_seed: int = 42,
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
        self.cd1_budget_s = cd1_budget_s
        self.cd2_budget_s = cd2_budget_s
        self.cd_plateau_threshold = cd_plateau_threshold
        self.outlier_k = outlier_k
        self.outlier_n_iters = outlier_n_iters
        self.outlier_polish_budget = outlier_polish_budget
        self.outlier_radius_frac = outlier_radius_frac
        self.rng_seed = rng_seed
        self.verbose = verbose
        # Phase records.
        self.last_phase_walls: dict = {}
        self.last_phase_proxies: dict = {}
        self.last_outlier_iters: list = []

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        deadline = t0 + self.budget_seconds if self.budget_seconds else None
        self._log(f"=== E149OutlierPerturbPlacer ({benchmark.name}) ===")
        self._log(
            f"  budget={self.budget_seconds}s; descent + CD1={self.cd1_budget_s}s "
            f"+ outlier({self.outlier_n_iters}x{self.outlier_polish_budget}s) "
            f"+ CD2={self.cd2_budget_s}s; K={self.outlier_k}"
        )

        bench_dir = find_benchmark_dir(benchmark.name)
        _, plc = load_benchmark_from_dir(str(bench_dir))
        device = _best_device()
        self._log(f"  device={device}")

        # ------- Phase 1: V4 + Gaussian descent + project_overlaps
        t_descent = time.time()
        pos = None
        for attempt, cfg in enumerate([
            dict(overlap_lambda_end=self.overlap_lambda_end),
            dict(overlap_lambda_end=50.0),
            dict(overlap_lambda_end=100.0, num_steps=300),
        ]):
            try:
                num_steps = cfg.get("num_steps", self.num_steps)
                ovl_end = cfg["overlap_lambda_end"]
                self._log(
                    f"  Phase 1 attempt {attempt+1}: ovl_end={ovl_end} steps={num_steps}"
                )
                descender = SmoothGlobalPlacerV4Gaussian(
                    num_steps=num_steps,
                    lr_frac=self.lr_frac,
                    gamma_start_frac=self.gamma_start_frac,
                    gamma_end_frac=self.gamma_end_frac,
                    overlap_lambda_end=ovl_end,
                    overlap_ramp_pct=self.overlap_ramp_pct,
                    init=self.init,
                    device=device,
                    rng_seed=self.rng_seed + attempt * 100,
                    verbose=False,
                )
                pos_try = descender.place(benchmark)
                ovl_try = compute_overlap_metrics(pos_try, benchmark)["overlap_count"]
                if ovl_try == 0:
                    pos = pos_try
                    self._log(f"    ok: ovl=0")
                    break
                pos_try2, _ = project_overlaps(pos_try, benchmark)
                if compute_overlap_metrics(pos_try2, benchmark)["overlap_count"] == 0:
                    pos = pos_try2
                    self._log(f"    ok: ovl=0 after project_overlaps")
                    break
                self._log(f"    still {ovl_try} overlaps; retrying")
            except Exception as exc:
                self._log(f"    attempt {attempt+1} EXCEPTION: {exc}")
                continue
            if deadline is not None and time.time() > deadline - 600:
                break
        if pos is None:
            self._log("  Phase 1 fallback: SDF + project_overlaps")
            pos = sdf_init(benchmark)
            pos, _ = project_overlaps(pos, benchmark)
        descent_wall = time.time() - t_descent
        descent_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        self._log(
            f"  Phase 1 done: proxy={descent_proxy:.5f} wall={descent_wall:.0f}s"
        )

        # ------- Phase 2: CD polish 1
        if deadline is not None:
            remaining = deadline - time.time()
            reserve = (
                self.outlier_n_iters * self.outlier_polish_budget
                + self.cd2_budget_s
                + 60.0
            )
            cd1_budget = max(30.0, min(self.cd1_budget_s, remaining - reserve))
        else:
            cd1_budget = self.cd1_budget_s
        t_cd1 = time.time()
        self._log(f"  Phase 2: CD1 budget={cd1_budget:.0f}s")
        pos = _cd_polish(pos, benchmark, plc, cd1_budget, self.cd_plateau_threshold)
        cd1_wall = time.time() - t_cd1
        cd1_proxy = float(compute_proxy_cost(pos, benchmark, plc)["proxy_cost"])
        cd1_ovl = compute_overlap_metrics(pos, benchmark)["overlap_count"]
        self._log(
            f"  Phase 2 done: proxy={cd1_proxy:.5f} (Δ={cd1_proxy - descent_proxy:+.5f}) "
            f"ovl={cd1_ovl} wall={cd1_wall:.0f}s"
        )
        if cd1_ovl > 0:
            self._log(f"  WARN: CD1 produced {cd1_ovl} overlaps; project_overlaps")
            pos, _ = project_overlaps(pos, benchmark)

        # ------- Phase 3: Outlier-perturb loop
        best_pos = pos
        best_proxy = cd1_proxy
        rng = np.random.RandomState(self.rng_seed)
        outlier_iters_log: list = []
        t_outlier = time.time()
        for it in range(self.outlier_n_iters):
            t_it = time.time()
            if deadline is not None:
                remaining = deadline - time.time()
                slack = self.cd2_budget_s + 20.0
                iter_budget = max(30.0, min(self.outlier_polish_budget, remaining - slack))
                if iter_budget < 30.0:
                    self._log(
                        f"  Phase 3 iter {it+1} SKIPPED (only {remaining:.0f}s left)"
                    )
                    break
            else:
                iter_budget = self.outlier_polish_budget

            try:
                # 1) Score macros by per-macro cost contribution.
                t_score = time.time()
                scores = _macro_outlier_scores(best_pos, benchmark, plc)
                chosen = _select_outliers(scores, benchmark, self.outlier_k)
                top_score_str = ",".join(f"{scores[i]:.3f}" for i in chosen[:5])
                self._log(
                    f"  Phase 3 iter {it+1}: scored {len(scores)} macros in "
                    f"{time.time()-t_score:.1f}s; top-{len(chosen)} score(0..4)=[{top_score_str}]"
                )
                if not chosen:
                    self._log("    no candidates to perturb; ending loop")
                    break

                # 2) Perturb chosen macros to local random positions.
                perturbed = _perturb_outliers_local(
                    best_pos, benchmark, chosen, rng,
                    radius_frac=self.outlier_radius_frac,
                )
                pert_ovl_pre = compute_overlap_metrics(perturbed, benchmark)["overlap_count"]

                # 3) Legalize.
                try:
                    legal, _ = greedy_macro_legalize(
                        perturbed, benchmark,
                        search_radius_steps=50,
                        step_size_frac=0.05,
                        verbose=False,
                    )
                except Exception as exc:
                    self._log(
                        f"    iter {it+1}: greedy_macro_legalize EXCEPTION: {exc}; fallback project"
                    )
                    legal = perturbed
                legal_ovl = compute_overlap_metrics(legal, benchmark)["overlap_count"]
                if legal_ovl > 0:
                    legal, _ = project_overlaps(legal, benchmark)
                    legal_ovl = compute_overlap_metrics(legal, benchmark)["overlap_count"]
                if legal_ovl > 0:
                    self._log(
                        f"    iter {it+1}: still {legal_ovl} ovl after legalize+project; SKIP"
                    )
                    outlier_iters_log.append(
                        dict(iter=it + 1, perturbed_n=len(chosen),
                             pre_ovl=int(pert_ovl_pre), post_ovl=int(legal_ovl),
                             pre_polish_proxy=None, post_polish_proxy=None,
                             accepted=False, wall=time.time() - t_it)
                    )
                    continue

                # 4) Polish.
                pre_polish_proxy = float(
                    compute_proxy_cost(legal, benchmark, plc)["proxy_cost"]
                )
                polished = _cd_polish(
                    legal, benchmark, plc,
                    iter_budget, self.cd_plateau_threshold,
                )
                polished_ovl = compute_overlap_metrics(polished, benchmark)["overlap_count"]
                if polished_ovl > 0:
                    polished, _ = project_overlaps(polished, benchmark)
                post_proxy = float(
                    compute_proxy_cost(polished, benchmark, plc)["proxy_cost"]
                )
                accepted = post_proxy < best_proxy
                if accepted:
                    self._log(
                        f"  Phase 3 iter {it+1} ACCEPT: K={len(chosen)} "
                        f"pre_polish={pre_polish_proxy:.5f} -> {post_proxy:.5f} "
                        f"(prev best {best_proxy:.5f}) wall={time.time()-t_it:.0f}s"
                    )
                    best_pos = polished
                    best_proxy = post_proxy
                else:
                    self._log(
                        f"  Phase 3 iter {it+1} REJECT: K={len(chosen)} "
                        f"pre_polish={pre_polish_proxy:.5f} -> {post_proxy:.5f} "
                        f"(best {best_proxy:.5f}) wall={time.time()-t_it:.0f}s"
                    )
                outlier_iters_log.append(
                    dict(iter=it + 1, perturbed_n=len(chosen),
                         chosen_indices=list(map(int, chosen)),
                         pre_ovl=int(pert_ovl_pre), post_ovl=int(polished_ovl),
                         pre_polish_proxy=pre_polish_proxy,
                         post_polish_proxy=post_proxy,
                         accepted=bool(accepted),
                         wall=time.time() - t_it)
                )
            except Exception as exc:
                self._log(f"  Phase 3 iter {it+1} EXCEPTION: {exc}; skipping iter")
                outlier_iters_log.append(
                    dict(iter=it + 1, error=str(exc),
                         wall=time.time() - t_it, accepted=False)
                )
                continue
        outlier_wall = time.time() - t_outlier
        self._log(
            f"  Phase 3 done: best_proxy={best_proxy:.5f} "
            f"(Δ vs CD1 {best_proxy - cd1_proxy:+.5f}) wall={outlier_wall:.0f}s"
        )

        pos = best_pos

        # ------- Phase 4: CD polish 2 (final)
        if deadline is not None:
            remaining = deadline - time.time() - 10.0
            cd2_budget = max(30.0, min(self.cd2_budget_s, remaining))
        else:
            cd2_budget = self.cd2_budget_s
        t_cd2 = time.time()
        self._log(f"  Phase 4: CD2 budget={cd2_budget:.0f}s")
        pos = _cd_polish(pos, benchmark, plc, cd2_budget, self.cd_plateau_threshold)
        cd2_wall = time.time() - t_cd2
        final = pos.detach().clone().to(torch.float32)
        final_proxy = float(compute_proxy_cost(final, benchmark, plc)["proxy_cost"])
        final_ovl = compute_overlap_metrics(final, benchmark)["overlap_count"]
        total_wall = time.time() - t0
        self._log(
            f"  Phase 4 done: proxy={final_proxy:.5f} ovl={final_ovl} "
            f"wall={cd2_wall:.0f}s"
        )
        self._log(
            f"  TOTAL: descent={descent_wall:.0f}s CD1={cd1_wall:.0f}s "
            f"outlier={outlier_wall:.0f}s CD2={cd2_wall:.0f}s total={total_wall:.0f}s "
            f"final_proxy={final_proxy:.5f}"
        )

        self.last_phase_walls = {
            "descent": descent_wall,
            "cd1": cd1_wall,
            "outlier": outlier_wall,
            "cd2": cd2_wall,
            "total": total_wall,
        }
        self.last_phase_proxies = {
            "descent": descent_proxy,
            "cd1": cd1_proxy,
            "outlier_best": best_proxy,
            "final": final_proxy,
        }
        self.last_outlier_iters = outlier_iters_log

        if final_ovl > 0:
            raise RuntimeError(f"Final placement has {final_ovl} overlaps")
        return final


# Alias for harness convenience.
Placer = E149OutlierPerturbPlacer
