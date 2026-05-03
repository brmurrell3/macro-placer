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
import scipy.optimize
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


def _row_pack_legalize_spectral(
    spectral_xy: np.ndarray,
    sizes_np: np.ndarray,
    fixed_mask: np.ndarray,
    original_positions: np.ndarray,
    n_hard: int,
    canvas_w: float,
    canvas_h: float,
    log_fn: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """Standard EDA row-packing legalizer that respects spectral 2D order.

    Algorithm:
    1. Sort hard movable macros by (spectral_y_desc, spectral_x_asc).
       This bucketing places macros with similar spectral_y into the
       same row, ordered L→R within the row.
    2. Greedy-pack rows top-down, left-to-right:
       - cur_y starts at canvas_top - max_h/2.
       - cur_x starts at max_half_w.
       - For each macro in sort order:
         - If macro doesn't fit in current row's remaining x: new row
           (cur_y -= max(row macros' heights), cur_x = max_half_w).
         - Place at (cur_x + macro.w/2, cur_y).
         - cur_x += macro.w.
       - Within each row, max_h tracks the tallest macro to set
         next row's cur_y drop.

    Output is no-overlap-by-construction for hard movable macros (their
    row edges abut, but don't overlap). Fixed macros stay at original
    positions; project_overlaps cleanup handles any fixed↔movable
    boundary effects.

    O(N log N) for sort + O(N) for pack — orders of magnitude faster
    than O(N²·R²) greedy_legalize, and handles arbitrary macro size
    distributions (no max-size slot grid required).
    """
    n_macros = len(sizes_np)
    half_w = sizes_np[:, 0] / 2.0
    half_h = sizes_np[:, 1] / 2.0

    # Hard movable indices.
    fixed_h = fixed_mask[:n_hard]
    movable_h_idx = np.where(~fixed_h)[0]
    n_movable = len(movable_h_idx)

    if log_fn is not None:
        log_fn(
            f"  row_pack: {n_movable} hard movables, "
            f"canvas={canvas_w:.1f}×{canvas_h:.1f}"
        )

    # Sort by (spectral_y descending, spectral_x ascending). Tie-breaking
    # by macro size descending (place larger first).
    sort_keys = np.column_stack([
        -spectral_xy[movable_h_idx, 1],          # primary: y desc
        spectral_xy[movable_h_idx, 0],            # secondary: x asc
        -sizes_np[movable_h_idx, 0] * sizes_np[movable_h_idx, 1],  # tertiary: area desc
    ])
    sort_order = np.lexsort(sort_keys.T[::-1])  # lexsort is reverse-priority

    # Greedy row pack. Initialize from original positions — preserves
    # SOFT macros (n_macros > n_hard) at their canonical placement and
    # FIXED hard macros at their pinned positions. Only hard MOVABLE
    # macros get overwritten by the row-pack assignment below.
    pos_np = original_positions.copy()

    # Standard EDA row packing convention: rows fill from top to bottom.
    # Each row has a fixed `row_top_y` (= top edge). Macros sit with their
    # top edges aligned to row_top_y, centers at (cur_x + w/2, row_top_y - h/2).
    # cur_x = next-available left-edge x-coordinate within current row.
    # When row is full (cur_x + w > canvas_w), drop row_top_y by row_max_h.
    row_top_y = canvas_h
    cur_x = 0.0
    row_max_h = 0.0
    row_started = False
    n_rows_used = 0

    for k in sort_order:
        m = int(movable_h_idx[k])
        w = float(sizes_np[m, 0])
        h = float(sizes_np[m, 1])

        if not row_started:
            # First macro of the very first row.
            row_top_y = canvas_h
            cur_x = 0.0
            row_max_h = h
            row_started = True
            n_rows_used += 1
        elif cur_x + w > canvas_w:
            # Doesn't fit in current row → start new row below.
            row_top_y -= row_max_h
            cur_x = 0.0
            row_max_h = h
            n_rows_used += 1
            if row_top_y - h < 0:
                # Out of canvas vertically. Wrap to bottom (rare; signals
                # canvas too small for total macro area).
                if log_fn is not None:
                    log_fn(
                        f"  row_pack WARN: ran out of canvas vertically "
                        f"after {n_rows_used} rows; macros pile up at bottom"
                    )
                row_top_y = h
        else:
            row_max_h = max(row_max_h, h)

        # Place macro: center = (cur_x + w/2, row_top_y - h/2).
        pos_np[m, 0] = cur_x + w / 2.0
        pos_np[m, 1] = row_top_y - h / 2.0

        cur_x += w  # advance left-edge past this macro's right edge

    # Clip to canvas (defensive, should already be in bounds).
    pos_np[:n_hard, 0] = np.clip(
        pos_np[:n_hard, 0], half_w[:n_hard], canvas_w - half_w[:n_hard]
    )
    pos_np[:n_hard, 1] = np.clip(
        pos_np[:n_hard, 1], half_h[:n_hard], canvas_h - half_h[:n_hard]
    )

    if log_fn is not None:
        log_fn(f"  row_pack: packed {n_movable} macros into {n_rows_used} rows")

    return pos_np


def _hungarian_legalize_spectral(
    spectral_xy: np.ndarray,
    sizes_np: np.ndarray,
    fixed_mask: np.ndarray,
    original_positions: np.ndarray,
    n_hard: int,
    canvas_w: float,
    canvas_h: float,
    log_fn: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """Fast Hungarian assignment of spectral coords to a row-aligned grid.

    Replaces greedy_legalize (O(N²·R²) per macro) with O(N³) Hungarian
    on a grid sized by max-macro footprint. By construction:
    - All slots are ≥ max_w apart in x and ≥ max_h apart in y.
    - Each macro's actual size ≤ max size, so adjacent slots fit any
      pair of macros without overlap.
    - Hungarian assignment minimizes sum of squared distances from
      spectral coord to slot center → preserves spectral 2D structure.

    Output is no-overlap by construction (modulo fixed macros, which
    we re-set after assignment). Subsequent project_overlaps handles
    any residual fixed/movable conflicts.
    """
    n_macros = len(sizes_np)
    movable_h_idx = np.where(~fixed_mask[:n_hard])[0]
    n_movable = len(movable_h_idx)

    # Use max sizes for slot grid so any pair of macros fits without overlap.
    half_w = sizes_np[:, 0] / 2.0
    half_h = sizes_np[:, 1] / 2.0
    max_w = float(sizes_np[movable_h_idx, 0].max()) if n_movable > 0 else 1.0
    max_h = float(sizes_np[movable_h_idx, 1].max()) if n_movable > 0 else 1.0

    # Slot grid covering full canvas, slot size = max macro size.
    n_cols = max(1, int(np.floor(canvas_w / max_w)))
    n_rows = max(1, int(np.floor(canvas_h / max_h)))
    n_slots = n_cols * n_rows

    if n_slots < n_movable:
        # Slots too coarse; need finer grid. Use mean instead of max.
        # This relaxes the no-overlap-by-construction guarantee, but
        # project_overlaps cleans up.
        mean_w = float(sizes_np[movable_h_idx, 0].mean())
        mean_h = float(sizes_np[movable_h_idx, 1].mean())
        # 1.1× slack so we have at least n_movable slots.
        target_slots = int(n_movable * 1.15)
        # Scale n_cols × n_rows ≈ target_slots while preserving aspect.
        aspect = canvas_w / canvas_h
        n_rows = max(1, int(np.sqrt(target_slots / aspect)))
        n_cols = max(1, int(target_slots / max(n_rows, 1)) + 1)
        n_slots = n_cols * n_rows
        if log_fn is not None:
            log_fn(
                f"  hungarian: max-size slots ({n_slots}) < n_movable ({n_movable}), "
                f"falling back to finer grid {n_cols}×{n_rows}={n_slots}"
            )

    # Slot centers (uniform on canvas).
    col_centers = np.linspace(canvas_w / (2 * n_cols), canvas_w - canvas_w / (2 * n_cols), n_cols)
    row_centers = np.linspace(canvas_h / (2 * n_rows), canvas_h - canvas_h / (2 * n_rows), n_rows)
    slots = np.zeros((n_slots, 2), dtype=np.float64)
    for r in range(n_rows):
        for c in range(n_cols):
            slots[r * n_cols + c, 0] = col_centers[c]
            slots[r * n_cols + c, 1] = row_centers[r]

    # Normalize spectral coords (movable only) to canvas range.
    s_x = spectral_xy[movable_h_idx, 0]
    s_y = spectral_xy[movable_h_idx, 1]
    sx_lo, sx_hi = float(s_x.min()), float(s_x.max())
    sy_lo, sy_hi = float(s_y.min()), float(s_y.max())
    if sx_hi - sx_lo < 1e-12:
        sx_hi = sx_lo + 1.0
    if sy_hi - sy_lo < 1e-12:
        sy_hi = sy_lo + 1.0
    s_x_norm = (s_x - sx_lo) / (sx_hi - sx_lo) * canvas_w
    s_y_norm = (s_y - sy_lo) / (sy_hi - sy_lo) * canvas_h
    spectral_norm = np.column_stack([s_x_norm, s_y_norm])

    # Cost matrix: squared distances from each movable's spectral pos
    # to each slot. Shape (n_movable, n_slots).
    if log_fn is not None:
        log_fn(
            f"  hungarian: cost matrix {n_movable} × {n_slots} "
            f"(grid {n_cols}×{n_rows}, max_macro={max_w:.2f}×{max_h:.2f})"
        )
    diff = spectral_norm[:, None, :] - slots[None, :, :]
    cost = (diff ** 2).sum(axis=2)

    # Solve assignment. linear_sum_assignment handles non-square (more
    # slots than macros) — returns optimal injection.
    t_h0 = time.perf_counter()
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
    if log_fn is not None:
        log_fn(
            f"  hungarian: assignment solved in {time.perf_counter() - t_h0:.2f}s "
            f"(matched {len(row_ind)} of {n_movable} movables)"
        )

    # Build placement. Initialize from original positions — preserves
    # SOFT macros at their canonical placement and FIXED macros at
    # pinned positions. Hard movables are overwritten via Hungarian below.
    pos_np = original_positions.copy()

    # Place movables according to Hungarian assignment.
    for ii, slot_idx in zip(row_ind, col_ind):
        macro_idx = int(movable_h_idx[ii])
        pos_np[macro_idx, 0] = slots[slot_idx, 0]
        pos_np[macro_idx, 1] = slots[slot_idx, 1]

    # Clip to canvas given each macro's size.
    pos_np[:n_hard, 0] = np.clip(pos_np[:n_hard, 0], half_w[:n_hard], canvas_w - half_w[:n_hard])
    pos_np[:n_hard, 1] = np.clip(pos_np[:n_hard, 1], half_h[:n_hard], canvas_h - half_h[:n_hard])

    return pos_np


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

    # Hungarian assignment to a max-size slot grid. Linearly rescaling +
    # iterative push-apart didn't converge (spectral coords cluster
    # connected macros tightly; push cycles oscillate). Hungarian gives
    # us min-distortion assignment to a NO-OVERLAP-BY-CONSTRUCTION grid
    # in O(N³) instead of O(N²·R²) per macro.
    fixed_mask_np = benchmark.macro_fixed.cpu().numpy()
    spectral_xy = np.column_stack([x_coord, y_coord])
    sizes_full = sizes_np

    orig_pos_np = benchmark.macro_positions.cpu().numpy().astype(np.float64)
    # Row-packing legalizer: standard EDA technique, no-overlap by
    # construction for hard movables, handles arbitrary macro sizes.
    # Hungarian-on-slot-grid only works on benches with low macro
    # density (canvas/max_macro_size² ≥ n_macros).
    pos = _row_pack_legalize_spectral(
        spectral_xy=spectral_xy,
        sizes_np=sizes_full,
        fixed_mask=fixed_mask_np,
        original_positions=orig_pos_np,
        n_hard=benchmark.num_hard_macros,
        canvas_w=cw,
        canvas_h=ch,
        log_fn=log_fn,
    )

    # Multi-pass project_overlaps with jitter-on-stall: cd_core's project
    # caps at 50 iters; after Hungarian on a coarse-grid fallback, residual
    # overlaps can stall in cycles (a pushes b, b pushes a, repeat). When
    # progress stalls (residual unchanged across passes), jitter the
    # overlapping macros to break cycles, then continue.
    placement = torch.tensor(pos, dtype=torch.float32)
    rng = np.random.default_rng(seed=seed)
    half_w_t = sizes_np[:, 0] / 2.0
    half_h_t = sizes_np[:, 1] / 2.0
    fixed_mask_t = benchmark.macro_fixed.cpu().numpy()
    n_hard = benchmark.num_hard_macros
    last_residual = None
    stall_count = 0
    for legal_pass in range(20):
        placement, proj_iters = project_overlaps(placement, benchmark)
        ovl = compute_overlap_metrics(placement, benchmark)["overlap_count"]
        if log_fn is not None:
            log_fn(
                f"  spectral legalize pass {legal_pass + 1}: project_overlaps "
                f"{proj_iters} iters, residual={ovl}"
            )
        if ovl == 0:
            break
        if last_residual is not None and ovl >= last_residual:
            stall_count += 1
        else:
            stall_count = 0
        last_residual = ovl

        if stall_count >= 1:
            # Jitter all hard movable macros by std=0.5 × min macro half-size.
            # Breaks symmetric pushing cycles.
            min_half_w = float(half_w_t[:n_hard].min())
            min_half_h = float(half_h_t[:n_hard].min())
            jitter_std_x = 0.5 * min_half_w
            jitter_std_y = 0.5 * min_half_h
            pos_np_iter = placement.cpu().numpy().astype(np.float64).copy()
            jitter_x = rng.normal(0.0, jitter_std_x, n_hard)
            jitter_y = rng.normal(0.0, jitter_std_y, n_hard)
            for i in range(n_hard):
                if not bool(fixed_mask_t[i]):
                    pos_np_iter[i, 0] += jitter_x[i]
                    pos_np_iter[i, 1] += jitter_y[i]
            pos_np_iter[:n_hard, 0] = np.clip(
                pos_np_iter[:n_hard, 0], half_w_t[:n_hard], cw - half_w_t[:n_hard]
            )
            pos_np_iter[:n_hard, 1] = np.clip(
                pos_np_iter[:n_hard, 1], half_h_t[:n_hard], ch - half_h_t[:n_hard]
            )
            placement = torch.tensor(pos_np_iter, dtype=torch.float32)
            if log_fn is not None:
                log_fn(
                    f"  spectral legalize: stall detected (residual {ovl} "
                    f"unchanged), jittered movables and retrying"
                )
            stall_count = 0

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
