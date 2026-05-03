"""E63 — Spectral / quadratic init + E41 pipeline (CD + LNS + SA-v2 + K-joint).

Tests netlist-Laplacian eigenvectors as a third basin source orthogonal
to SDF (analytical density spread) and DPO (gradient-descent topology).

Pipeline per benchmark:
  1. Spectral init: top non-trivial eigenvectors of the netlist Laplacian
     (clique-expanded hypergraph) used as (x, y) coordinates.
  2. project_overlaps to clear residuals.
  3. Build IncrementalProxyEvaluator.
  4. CD adaptive (≤ 2400 s).
  5. Grid-bin LNS (≤ 600 s, cost-aware destroy).
  6. SA-v2 (≤ 600 s, T₀=5e-4).
  7. K-joint LNS (≤ 600 s, K=3, top_N=5).
  8. Validate, preserve fixed macros, return.

Reference:
- E41 — pipeline backbone (`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`).
- E18 — net-data extraction helper (`_extract_net_data`).
- Roadmap §4.6.B — motivation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics

from experiments.E18_dpo_init.code.cd_lns_sa_dpo_init import _extract_net_data
from experiments.E39_kmacro_joint_lns.code.cd_lns_sa_kjoint import (
    run_kjoint_lns,
    run_lns_gridbin,
    run_sa_polish_v2,
)
from macro_place.sdf_init import greedy_legalize


# ── Spectral init ──────────────────────────────────────────────────────────


def _build_netlist_laplacian(
    benchmark: Benchmark,
    plc,
) -> scipy.sparse.csr_matrix:
    """Build sparse weighted graph Laplacian L = D − A from the netlist
    hypergraph via clique expansion.

    For each net N with k≥2 macros, contributes weight 1/(k-1) to each
    of the k(k-1)/2 pairs of macros on the net. This is the standard
    'clique expansion' normalization that makes star-shaped (high
    fanout) nets equivalent to clique-shaped nets in graph metrics.
    """
    n = benchmark.num_macros
    net_data = _extract_net_data(benchmark, plc)
    pin_macro_idx = net_data.pin_macro_idx.cpu().numpy()  # [num_nets, max_pins]
    mask = net_data.mask.cpu().numpy()                    # [num_nets, max_pins] bool

    # Net pin index (port_idx) is num_macros (a virtual placeholder pin).
    # We exclude port pins from the Laplacian since they aren't placeable.
    port_idx = n

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    num_nets = pin_macro_idx.shape[0]
    for j in range(num_nets):
        # Collect distinct macro indices on this net (not ports).
        net_macros = []
        for k in range(pin_macro_idx.shape[1]):
            if not mask[j, k]:
                continue
            m = int(pin_macro_idx[j, k])
            if m == port_idx:
                continue
            net_macros.append(m)
        # De-duplicate (same macro could have multiple pins on a net).
        unique_macros = list(set(net_macros))
        if len(unique_macros) < 2:
            continue
        weight = 1.0 / (len(unique_macros) - 1)
        # Symmetric edges between every pair.
        for i_a in range(len(unique_macros)):
            for i_b in range(i_a + 1, len(unique_macros)):
                a, b = unique_macros[i_a], unique_macros[i_b]
                rows.append(a); cols.append(b); data.append(weight)
                rows.append(b); cols.append(a); data.append(weight)

    A = scipy.sparse.csr_matrix(
        (data, (rows, cols)), shape=(n, n), dtype=np.float64
    )
    degrees = np.array(A.sum(axis=1)).flatten()
    D = scipy.sparse.diags(degrees, 0, shape=(n, n), format="csr")
    L = D - A
    # Tiny diagonal regularization to handle disconnected components +
    # numerical stability for shift-invert.
    L = L + 1e-9 * scipy.sparse.eye(n, format="csr")
    return L


def _spectral_init(
    benchmark: Benchmark,
    plc,
    seed: int = 42,
    log_fn: Optional[Callable[[str], None]] = None,
) -> torch.Tensor:
    """Spectral / quadratic init via netlist Laplacian eigenvectors.

    Uses eigenvectors corresponding to the 2nd and 3rd smallest
    eigenvalues (Fiedler-and-next) as (x, y) coordinates. The 1st
    smallest is the constant eigenvector (eigenvalue ≈ 0) and is
    discarded.
    """
    np.random.seed(seed)
    n = benchmark.num_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes_np = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    half_w = sizes_np[:, 0] / 2.0
    half_h = sizes_np[:, 1] / 2.0

    if log_fn is not None:
        log_fn(f"  spectral: building Laplacian (n={n})")
    L = _build_netlist_laplacian(benchmark, plc)

    # Solve for the 4 smallest eigenvalues via shift-invert. We need
    # eigenvalues 2 and 3 (Fiedler = 2nd smallest, plus 3rd).
    if log_fn is not None:
        log_fn(f"  spectral: eigsh k=4 sigma=0 (shift-invert)")
    try:
        vals, vecs = scipy.sparse.linalg.eigsh(
            L, k=4, sigma=0.0, which="LM",  # 'LM' on shift-invert finds smallest
        )
    except Exception as e:
        # Fallback: dense solve for small N (≤ 500), or 'SM' without shift-invert.
        if log_fn is not None:
            log_fn(f"  spectral: shift-invert failed ({type(e).__name__}); fallback")
        if n <= 500:
            L_dense = L.toarray()
            vals_full, vecs_full = np.linalg.eigh(L_dense)
            vals = vals_full[:4]
            vecs = vecs_full[:, :4]
        else:
            vals, vecs = scipy.sparse.linalg.eigsh(L, k=4, which="SM")

    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    # eigvec[:, 0] is ~constant (eigenvalue ~0). Use [:, 1] and [:, 2].
    x_coord = vecs[:, 1].astype(np.float64)
    y_coord = vecs[:, 2].astype(np.float64)

    # **Approach: linearly rescale spectral coords to canvas, then
    # extended-iter push-apart projection.** Project_overlaps from
    # cd_core caps at 50 iters which is too few for spectral inputs;
    # we reimplement here with max_iters=500 and vectorized push.
    fixed_mask_np = benchmark.macro_fixed.cpu().numpy()
    movable_idx = np.where(~fixed_mask_np)[0]

    max_hw = float(half_w.max())
    max_hh = float(half_h.max())
    x_lo, x_hi = float(x_coord.min()), float(x_coord.max())
    y_lo, y_hi = float(y_coord.min()), float(y_coord.max())
    if x_hi - x_lo < 1e-12: x_hi = x_lo + 1.0
    if y_hi - y_lo < 1e-12: y_hi = y_lo + 1.0

    target = np.zeros((n, 2), dtype=np.float64)
    orig_pos = benchmark.macro_positions.cpu().numpy().astype(np.float64)
    for i in range(n):
        if bool(fixed_mask_np[i]):
            target[i, 0] = orig_pos[i, 0]
            target[i, 1] = orig_pos[i, 1]
        else:
            target[i, 0] = max_hw + (cw - 2 * max_hw) * (x_coord[i] - x_lo) / (x_hi - x_lo)
            target[i, 1] = max_hh + (ch - 2 * max_hh) * (y_coord[i] - y_lo) / (y_hi - y_lo)
    target[:, 0] = np.clip(target[:, 0], half_w, cw - half_w)
    target[:, 1] = np.clip(target[:, 1], half_h, ch - half_h)

    # Vectorized iterative push-apart with high iter cap. For each
    # iteration: identify all overlapping pairs, push each by half the
    # overlap on the smaller axis, clamp to canvas. Convergence
    # detected when no overlaps remain among hard macros.
    n_hard = benchmark.num_hard_macros
    pos = target.copy()
    half_w_h = half_w[:n_hard]
    half_h_h = half_h[:n_hard]
    fixed_h = fixed_mask_np[:n_hard]
    if log_fn is not None:
        log_fn(f"  spectral: projecting overlaps (max 500 iters, vectorized)")

    converged = False
    for it in range(500):
        dx = pos[:n_hard, 0:1] - pos[:n_hard, 0:1].T
        dy = pos[:n_hard, 1:2] - pos[:n_hard, 1:2].T
        adx = np.abs(dx)
        ady = np.abs(dy)
        min_dx = half_w_h[:, None] + half_w_h[None, :]
        min_dy = half_h_h[:, None] + half_h_h[None, :]
        ovl = (adx < min_dx - 1e-9) & (ady < min_dy - 1e-9)
        np.fill_diagonal(ovl, False)
        pairs = np.argwhere(np.triu(ovl))
        if len(pairs) == 0:
            converged = True
            break
        # Apply pushes pair-by-pair (sequential to avoid cascading
        # bugs; with vectorized push, two macros pushed in different
        # directions could leave both still overlapping after one
        # iter).
        for a, b in pairs:
            mov_a = not bool(fixed_h[a])
            mov_b = not bool(fixed_h[b])
            if not (mov_a or mov_b):
                continue
            dxv = float(min_dx[a, b] - adx[a, b]) + 1e-3
            dyv = float(min_dy[a, b] - ady[a, b]) + 1e-3
            if dxv < dyv:
                sgn = 1.0 if pos[a, 0] < pos[b, 0] else -1.0
                if mov_a and mov_b:
                    pos[a, 0] -= sgn * dxv / 2
                    pos[b, 0] += sgn * dxv / 2
                elif mov_a:
                    pos[a, 0] -= sgn * dxv
                else:
                    pos[b, 0] += sgn * dxv
            else:
                sgn = 1.0 if pos[a, 1] < pos[b, 1] else -1.0
                if mov_a and mov_b:
                    pos[a, 1] -= sgn * dyv / 2
                    pos[b, 1] += sgn * dyv / 2
                elif mov_a:
                    pos[a, 1] -= sgn * dyv
                else:
                    pos[b, 1] += sgn * dyv
        pos[:n_hard, 0] = np.clip(pos[:n_hard, 0], half_w_h, cw - half_w_h)
        pos[:n_hard, 1] = np.clip(pos[:n_hard, 1], half_h_h, ch - half_h_h)

    if log_fn is not None:
        log_fn(
            f"  spectral: project ended at iter {it+1}, "
            f"converged={converged}, residual_pairs={0 if converged else len(pairs)}"
        )

    placement = torch.tensor(pos, dtype=torch.float32)

    if log_fn is not None:
        log_fn(
            f"  spectral: eigenvalues = {vals[0]:.4e}, {vals[1]:.4e}, "
            f"{vals[2]:.4e}, {vals[3]:.4e} (used eigvecs 2 + 3)"
        )

    return placement


# ── Placer ─────────────────────────────────────────────────────────────────


class CDLNSSASpectralKJointPlacer:
    """E63 — Spectral init + CD + LNS + SA-v2 + K-joint."""

    def __init__(
        self,
        cd_hard_cap_s: float = 2400.0,
        cd_min_time_s: float = 300.0,
        cd_patience: int = 3,
        cd_plateau_threshold: float = 0.001,
        lns_budget_s: float = 600.0,
        lns_destroy_frac: float = 0.05,
        lns_destroy_cap: int = 30,
        lns_seed: int = 42,
        sa_budget_s: float = 600.0,
        sa_T0: float = 5e-4,
        sa_Tf: float = 1e-6,
        sa_seed: int = 42,
        sa_breakpoint_budget: int = 12,
        kjoint_budget_s: float = 600.0,
        kjoint_K: int = 3,
        kjoint_top_N: int = 5,
        kjoint_seed: int = 42,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.cd_hard_cap_s = float(cd_hard_cap_s)
        self.cd_min_time_s = float(cd_min_time_s)
        self.cd_patience = int(cd_patience)
        self.cd_plateau_threshold = float(cd_plateau_threshold)
        self.lns_budget_s = float(lns_budget_s)
        self.lns_destroy_frac = float(lns_destroy_frac)
        self.lns_destroy_cap = int(lns_destroy_cap)
        self.lns_seed = int(lns_seed)
        self.sa_budget_s = float(sa_budget_s)
        self.sa_T0 = float(sa_T0)
        self.sa_Tf = float(sa_Tf)
        self.sa_seed = int(sa_seed)
        self.sa_breakpoint_budget = int(sa_breakpoint_budget)
        self.kjoint_budget_s = float(kjoint_budget_s)
        self.kjoint_K = int(kjoint_K)
        self.kjoint_top_N = int(kjoint_top_N)
        self.kjoint_seed = int(kjoint_seed)
        self.seed = int(seed)
        self.verbose = bool(verbose)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _load_plc_for(self, benchmark: Benchmark):
        bench_dir = find_benchmark_dir(benchmark.name)
        return load_benchmark_from_dir(str(bench_dir))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_total0 = time.perf_counter()
        self._log(
            f"=== CDLNSSASpectralKJointPlacer ({benchmark.name}): "
            f"Spectral -> CD -> LNS -> SA -> KJoint ==="
        )
        self._log(
            f"  num_macros={benchmark.num_macros}, "
            f"n_hard={benchmark.num_hard_macros}, "
            f"canvas={benchmark.canvas_width:.1f}x{benchmark.canvas_height:.1f}"
        )

        # 1. Spectral init.
        _, plc = self._load_plc_for(benchmark)
        t_init0 = time.perf_counter()
        placement = _spectral_init(
            benchmark, plc, seed=self.seed,
            log_fn=self._log if self.verbose else None,
        )
        self._log(f"  spectral init wall = {time.perf_counter() - t_init0:.1f} s")

        # 2. Project overlaps.
        placement, proj_iters = project_overlaps(placement, benchmark)
        init_overlaps = compute_overlap_metrics(placement, benchmark)
        self._log(
            f"  overlap projection: {proj_iters} iters, "
            f"residual count={init_overlaps['overlap_count']}"
        )
        if init_overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"Spectral init projection did not converge — "
                f"{init_overlaps['overlap_count']} residual overlaps. "
                "May need stricter scaling or extra repair iterations."
            )

        # 3. Build evaluator.
        placement_f64 = placement.detach().clone().to(torch.float64)
        evaluator = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
        init_cost = evaluator.current_cost()
        self._log(
            f"  init proxy={init_cost['proxy']:.5f} "
            f"[wl={init_cost['wl']:.4f} d={init_cost['density']:.4f} "
            f"c={init_cost['congestion']:.4f}]"
        )

        movable = [i for i in range(benchmark.num_macros)
                   if not bool(benchmark.macro_fixed[i])]
        hard_movable = [i for i in range(benchmark.num_hard_macros)
                        if not bool(benchmark.macro_fixed[i])]

        # 4. CD phase.
        self._log(f"  starting CD phase (cap={self.cd_hard_cap_s:.0f}s)")
        cd_stats = run_cd_adaptive(
            evaluator=evaluator, benchmark=benchmark, plc=plc,
            movable=movable,
            min_time_s=self.cd_min_time_s,
            hard_cap_s=self.cd_hard_cap_s,
            patience=self.cd_patience,
            plateau_threshold=self.cd_plateau_threshold,
            log_fn=self._log if self.verbose else None,
        )
        cd_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  CD done: exit={cd_stats['exit_reason']}, "
            f"sweeps={cd_stats['sweeps']}, accepts={cd_stats['total_moves']}, "
            f"wall={cd_stats['wall_total_s']:.1f}s, proxy={cd_proxy:.5f}"
        )

        # 5. LNS phase.
        self._log(f"  starting LNS phase (budget={self.lns_budget_s:.0f}s)")
        lns_stats = run_lns_gridbin(
            evaluator=evaluator, benchmark=benchmark, plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.lns_budget_s,
            destroy_frac=self.lns_destroy_frac,
            destroy_cap=self.lns_destroy_cap,
            seed=self.lns_seed,
            log_fn=self._log if self.verbose else None,
        )
        lns_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  LNS done: samples={lns_stats['samples']}, "
            f"Δ={lns_stats['total_improvement']:+.5f}, "
            f"wall={lns_stats['wall_total_s']:.1f}s, proxy={lns_proxy:.5f}"
        )

        # 6. SA-v2 phase.
        self._log(f"  starting SA-v2 phase (budget={self.sa_budget_s:.0f}s)")
        sa_stats = run_sa_polish_v2(
            evaluator=evaluator, benchmark=benchmark, plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.sa_budget_s,
            T0=self.sa_T0, Tf=self.sa_Tf,
            seed=self.sa_seed,
            breakpoint_budget=self.sa_breakpoint_budget,
            log_fn=self._log if self.verbose else None,
        )
        sa_proxy = evaluator.current_cost()["proxy"]
        self._log(
            f"  SA-v2 done: lift_vs_init={sa_stats['improvement_vs_init']:+.5f}, "
            f"best={sa_stats['best_proxy']:.5f}, proxy={sa_proxy:.5f}"
        )

        # 7. K-joint LNS phase.
        self._log(
            f"  starting K-joint phase (budget={self.kjoint_budget_s:.0f}s, "
            f"K={self.kjoint_K}, top_N={self.kjoint_top_N})"
        )
        kj_stats = run_kjoint_lns(
            evaluator=evaluator, benchmark=benchmark, plc=plc,
            hard_movable=hard_movable,
            time_budget_s=self.kjoint_budget_s,
            K=self.kjoint_K,
            top_N=self.kjoint_top_N,
            seed=self.kjoint_seed,
            log_fn=self._log if self.verbose else None,
        )
        final_cost = evaluator.current_cost()
        self._log(
            f"  K-joint done: passes={kj_stats['passes']}, "
            f"tuples_tried={kj_stats['ktuples_tried']}, "
            f"committed={kj_stats['ktuples_committed']}, "
            f"Δ={kj_stats['total_improvement']:+.5f}, "
            f"wall={kj_stats['wall_total_s']:.1f}s, "
            f"final proxy={final_cost['proxy']:.5f}"
        )
        self._log(
            f"  TOTAL lifts: CD={init_cost['proxy'] - cd_proxy:+.5f}, "
            f"LNS={cd_proxy - lns_proxy:+.5f}, "
            f"SA={lns_proxy - sa_proxy:+.5f}, "
            f"KJoint={sa_proxy - final_cost['proxy']:+.5f}"
        )

        # 8. Pull placement back; preserve fixed macros.
        final_placement_f64 = evaluator.placement.detach().clone().cpu()
        fixed_mask = benchmark.macro_fixed
        original_positions = benchmark.macro_positions.to(torch.float64)
        if fixed_mask.any():
            final_placement_f64[fixed_mask] = original_positions[fixed_mask]
        final_placement = final_placement_f64.to(torch.float32)

        overlaps = compute_overlap_metrics(final_placement, benchmark)
        if overlaps["overlap_count"] > 0:
            raise RuntimeError(
                f"CDLNSSASpectralKJointPlacer produced {overlaps['overlap_count']} "
                f"overlaps (area {overlaps['total_overlap_area']:.4f}) on "
                f"'{benchmark.name}'"
            )

        self._log(
            f"  total wall: {time.perf_counter() - t_total0:.1f} s "
            f"(spectral + project + CD + LNS + SA + KJoint + validate)"
        )
        return final_placement
