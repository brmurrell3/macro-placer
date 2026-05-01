"""E45 multigrid — Phase 2: V-cycle orchestration.

Active multigrid init for placement:

  1. Cluster hard macros into K super-macros (Phase 1).
  2. SDF init the parent benchmark → fine-scale positions for all macros.
  3. Compute initial super-macro centers as centroids of their constituents'
     SDF positions.
  4. **Coarse-scale relaxation**: coordinate descent on super-macro centers
     to minimize quotient-HPWL with super-macro non-overlap. Operates on K
     entities (small, fast); uses a NEW grid resolution scaled to canvas.
  5. **Uncoarsen**: for each constituent macro, translate it by
     (super_center_post − super_center_pre), preserving its relative
     position within its cluster.
  6. **Project overlaps**: legalize the aggregated placement (this is the
     multigrid init's output).

The output replaces SDF/DPO init in the E41 pipeline. The downstream
phases (CD plateau, grid-bin LNS, SA-v2, K-joint) are unchanged.

Phase 2 design choice: rigid-translate macros with their super-macros
rather than re-running SDF inside each super-macro region. Reasons:
- Avoids needing a `plc` object for sub-benchmarks (SDFPlacer is tied
  to `_load_plc(benchmark.name)`).
- Preserves intra-super-macro structure that SDF already found.
- Simpler V-cycle for first version; can upgrade to per-super SDF later
  if the rigid-translate version doesn't lift enough.

Public API:
- `multigrid_init(benchmark, K=None, log_fn=None) -> torch.Tensor`:
  returns a [num_macros, 2] placement tensor ready to feed into CD or
  the E41 pipeline.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.cd_core import project_overlaps, sdf_init

# Phase 1 helpers in the same package.
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from multigrid_clustering import cluster_macros, super_macro_geometry  # noqa: E402


# ── Quotient netlist ───────────────────────────────────────────────────────


def build_quotient_netlist(
    benchmark: Benchmark, cluster_ids: np.ndarray
) -> List[Tuple[int, ...]]:
    """For each net, the set of super-macros it touches (after deduplication).
    Drops nets with <2 distinct super-macros (intra-super-macro nets).
    """
    n_hard = int(benchmark.num_hard_macros)
    quotient_nets: List[Tuple[int, ...]] = []
    for nodes in benchmark.net_nodes:
        nodes_np = nodes.cpu().numpy()
        super_ids = set()
        for n in nodes_np:
            n_int = int(n)
            if n_int < n_hard:
                super_ids.add(int(cluster_ids[n_int]))
        if len(super_ids) >= 2:
            quotient_nets.append(tuple(sorted(super_ids)))
    return quotient_nets


# ── Coarse-scale CD on super-macros ────────────────────────────────────────


def _quotient_hpwl(
    super_centers: np.ndarray, quotient_nets: List[Tuple[int, ...]]
) -> float:
    """Sum of (max_x - min_x + max_y - min_y) over quotient nets."""
    total = 0.0
    for net in quotient_nets:
        if len(net) < 2:
            continue
        xs = super_centers[list(net), 0]
        ys = super_centers[list(net), 1]
        total += float(xs.max() - xs.min() + ys.max() - ys.min())
    return total


def _supers_overlap(
    centers: np.ndarray, sizes: np.ndarray, k: int, x: float, y: float
) -> bool:
    """Does super-macro k at (x, y) overlap any other non-empty super-macro?"""
    K = centers.shape[0]
    half_w_k = sizes[k, 0] / 2.0
    half_h_k = sizes[k, 1] / 2.0
    if half_w_k <= 0 or half_h_k <= 0:
        return False  # empty super-macro can't overlap
    for j in range(K):
        if j == k:
            continue
        half_w_j = sizes[j, 0] / 2.0
        half_h_j = sizes[j, 1] / 2.0
        if half_w_j <= 0 or half_h_j <= 0:
            continue
        if abs(x - centers[j, 0]) < (half_w_k + half_w_j) and abs(
            y - centers[j, 1]
        ) < (half_h_k + half_h_j):
            return True
    return False


def coarse_cd_supermacros(
    super_centers: np.ndarray,
    super_sizes: np.ndarray,
    quotient_nets: List[Tuple[int, ...]],
    canvas_w: float,
    canvas_h: float,
    grid_n: int = 64,
    max_sweeps: int = 50,
    plateau_threshold: float = 1e-4,
    enforce_nonoverlap: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """Greedy CD on super-macro positions, axis-by-axis. Each super-macro
    independently tries every grid line along each axis; accepts the
    position with lowest quotient-HPWL.

    `enforce_nonoverlap=False` (default) lets super-macros overlap at the
    coarse level — they're approximate aggregates, not real placements.
    The fine-scale uncoarsen step + project_overlaps clean up actual
    constituent collisions. With non-overlap enforcement, on packed
    canvases (e.g., ibm10 with 21 super-macros), most moves get blocked
    and CD stalls (verified empirically: 1/21 moved with constraint).

    Empty super-macros (size 0) are skipped (their positions don't matter).

    Returns updated centers (np.ndarray [K, 2], float64). Operates in-place
    on a copy of super_centers.
    """
    centers = super_centers.copy()
    K = centers.shape[0]
    grid_x = np.linspace(0, canvas_w, grid_n)
    grid_y = np.linspace(0, canvas_h, grid_n)

    cur_cost = _quotient_hpwl(centers, quotient_nets)
    # ⬇ legacy greedy-CD body retained below; new SA primary entry-point
    # is `coarse_sa_supermacros` (see below) which handles dense canvases.
    if log_fn is not None:
        log_fn(
            f"  multigrid coarse-CD start: HPWL={cur_cost:.3f}, K={K}, "
            f"grid={grid_n}, nonoverlap={enforce_nonoverlap}"
        )

    for sweep in range(max_sweeps):
        prev_cost = cur_cost
        for k in range(K):
            half_w = super_sizes[k, 0] / 2.0
            half_h = super_sizes[k, 1] / 2.0
            if half_w <= 0 or half_h <= 0:
                continue
            # X axis
            best_x = centers[k, 0]
            best_cost = cur_cost
            for x_cand in grid_x:
                if x_cand < half_w or x_cand > canvas_w - half_w:
                    continue
                if enforce_nonoverlap and _supers_overlap(
                    centers, super_sizes, k, x_cand, centers[k, 1]
                ):
                    continue
                old_x = centers[k, 0]
                centers[k, 0] = x_cand
                c = _quotient_hpwl(centers, quotient_nets)
                if c < best_cost:
                    best_cost = c
                    best_x = x_cand
                centers[k, 0] = old_x
            centers[k, 0] = best_x
            cur_cost = best_cost
            # Y axis
            best_y = centers[k, 1]
            best_cost = cur_cost
            for y_cand in grid_y:
                if y_cand < half_h or y_cand > canvas_h - half_h:
                    continue
                if enforce_nonoverlap and _supers_overlap(
                    centers, super_sizes, k, centers[k, 0], y_cand
                ):
                    continue
                old_y = centers[k, 1]
                centers[k, 1] = y_cand
                c = _quotient_hpwl(centers, quotient_nets)
                if c < best_cost:
                    best_cost = c
                    best_y = y_cand
                centers[k, 1] = old_y
            centers[k, 1] = best_y
            cur_cost = best_cost
        delta = prev_cost - cur_cost
        if log_fn is not None:
            log_fn(
                f"  multigrid coarse-CD sweep {sweep + 1}: HPWL={cur_cost:.3f}, "
                f"Δ={delta:+.3f}"
            )
        if delta < plateau_threshold:
            if log_fn is not None:
                log_fn(f"  multigrid coarse-CD plateau at sweep {sweep + 1}")
            break

    return centers


# ── Coarse-scale SA with soft overlap penalty ──────────────────────────────


def _total_super_overlap_area(
    centers: np.ndarray, sizes: np.ndarray
) -> float:
    """Sum of bbox-bbox overlap area across all pairs of non-empty
    super-macros."""
    K = centers.shape[0]
    total = 0.0
    for i in range(K):
        if sizes[i, 0] <= 0 or sizes[i, 1] <= 0:
            continue
        for j in range(i + 1, K):
            if sizes[j, 0] <= 0 or sizes[j, 1] <= 0:
                continue
            half_w_i = sizes[i, 0] / 2.0
            half_h_i = sizes[i, 1] / 2.0
            half_w_j = sizes[j, 0] / 2.0
            half_h_j = sizes[j, 1] / 2.0
            ox = max(0.0, (half_w_i + half_w_j) - abs(centers[i, 0] - centers[j, 0]))
            oy = max(0.0, (half_h_i + half_h_j) - abs(centers[i, 1] - centers[j, 1]))
            total += ox * oy
    return total


def coarse_sa_supermacros(
    super_centers: np.ndarray,
    super_sizes: np.ndarray,
    quotient_nets: List[Tuple[int, ...]],
    canvas_w: float,
    canvas_h: float,
    n_moves: int = 50000,
    T0: float = 1.0,
    Tf: float = 1e-3,
    lambda_init: float = 0.01,
    lambda_final: float = 10.0,
    seed: int = 42,
    log_fn: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """SA on super-macro positions with soft overlap penalty.

    Cost = HPWL(quotient netlist) + λ · sum(super-macro bbox overlap area)

    λ starts low (allow exploration where super-macros can overlap to find
    HPWL-better positions) and grows geometrically over n_moves to a high
    value (push super-macros apart by end). T anneals geometrically T0→Tf.

    Empty super-macros (size 0) are not moved.
    """
    rng = np.random.default_rng(seed=seed)
    centers = super_centers.copy()
    K = centers.shape[0]
    movable = [k for k in range(K) if super_sizes[k, 0] > 0 and super_sizes[k, 1] > 0]
    if not movable:
        return centers

    half_w = super_sizes[:, 0] / 2.0
    half_h = super_sizes[:, 1] / 2.0

    def cost_at(c: np.ndarray, lam: float) -> Tuple[float, float, float]:
        hpwl = _quotient_hpwl(c, quotient_nets)
        ovl = _total_super_overlap_area(c, super_sizes)
        return hpwl + lam * ovl, hpwl, ovl

    # Compute T and λ schedule over n_moves.
    if Tf <= 0 or T0 <= 0:
        T_factor = 1.0
    else:
        T_factor = (Tf / T0) ** (1.0 / max(1, n_moves - 1))
    lam_factor = (lambda_final / lambda_init) ** (1.0 / max(1, n_moves - 1))

    cur_lam = lambda_init
    cur_T = T0
    cur_cost, cur_hpwl, cur_ovl = cost_at(centers, cur_lam)
    best_cost = cur_cost
    best_centers = centers.copy()
    best_hpwl = cur_hpwl
    best_ovl = cur_ovl

    accepts = 0
    rejects = 0
    log_step = max(1, n_moves // 10)
    if log_fn is not None:
        log_fn(
            f"  multigrid coarse-SA start: HPWL={cur_hpwl:.3f} "
            f"ovl_area={cur_ovl:.3f} T0={T0:.2g} Tf={Tf:.2g} "
            f"λ_init={lambda_init:.3g} λ_final={lambda_final:.3g}"
        )

    for step in range(n_moves):
        # Pick a random movable super-macro and propose a new position
        # within Gaussian noise of the canvas center scaled to canvas size.
        k = int(rng.choice(movable))
        # Gaussian step scaled by current temperature × canvas dimension.
        sigma = max(canvas_w, canvas_h) * 0.05  # ~5% of canvas per step
        nx = float(np.clip(centers[k, 0] + rng.normal(0, sigma),
                           half_w[k], canvas_w - half_w[k]))
        ny = float(np.clip(centers[k, 1] + rng.normal(0, sigma),
                           half_h[k], canvas_h - half_h[k]))
        # Try the move.
        old_x, old_y = centers[k, 0], centers[k, 1]
        centers[k, 0] = nx
        centers[k, 1] = ny
        new_cost, new_hpwl, new_ovl = cost_at(centers, cur_lam)
        delta = new_cost - cur_cost
        if delta < 0 or rng.random() < np.exp(-delta / max(1e-12, cur_T)):
            cur_cost = new_cost
            cur_hpwl = new_hpwl
            cur_ovl = new_ovl
            accepts += 1
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_centers = centers.copy()
                best_hpwl = cur_hpwl
                best_ovl = cur_ovl
        else:
            centers[k, 0] = old_x
            centers[k, 1] = old_y
            rejects += 1
        cur_T *= T_factor
        cur_lam *= lam_factor
        if log_fn is not None and (step + 1) % log_step == 0:
            log_fn(
                f"  multigrid coarse-SA step {step + 1}/{n_moves}: "
                f"HPWL={cur_hpwl:.3f} ovl={cur_ovl:.3f} "
                f"T={cur_T:.2g} λ={cur_lam:.2g} "
                f"acc={accepts} rej={rejects} "
                f"best_HPWL={best_hpwl:.3f} best_ovl={best_ovl:.3f}"
            )

    if log_fn is not None:
        log_fn(
            f"  multigrid coarse-SA done: best HPWL={best_hpwl:.3f} "
            f"ovl={best_ovl:.3f} accepts={accepts} rejects={rejects}"
        )
    return best_centers


# ── V-cycle orchestration ──────────────────────────────────────────────────


def multigrid_init(
    benchmark: Benchmark,
    K: Optional[int] = None,
    grid_n: int = 64,
    max_sweeps: int = 50,
    log_fn: Optional[Callable[[str], None]] = None,
) -> torch.Tensor:
    """Active multigrid init for `benchmark`.

    Pipeline:
      1. Cluster hard macros into K super-macros via pymetis.
      2. SDF init the parent benchmark.
      3. Compute initial super-macro centers as constituent centroids.
      4. Coarse-scale CD on super-macro centers (minimize quotient-HPWL).
      5. Rigid-translate constituents by (super_center_post − super_center_pre).
      6. project_overlaps to legalize.

    Returns a [num_macros, 2] tensor with the legalized multigrid init.
    Soft macros (if any) are kept at their SDF positions; fixed macros
    are NOT moved.
    """
    n_hard = int(benchmark.num_hard_macros)
    n_macros = int(benchmark.num_macros)
    if K is None:
        K = max(2, int(round(np.sqrt(n_hard))))
    if log_fn is not None:
        log_fn(f"multigrid_init: n_hard={n_hard}, K={K}")

    # Phase 1: cluster.
    cluster_ids = cluster_macros(benchmark, K)
    super_sizes_t, super_areas, super_members = super_macro_geometry(
        benchmark, cluster_ids, K
    )
    super_sizes = super_sizes_t.cpu().numpy().astype(np.float64)

    # Quotient netlist for coarse HPWL.
    quotient_nets = build_quotient_netlist(benchmark, cluster_ids)
    if log_fn is not None:
        log_fn(
            f"multigrid_init: quotient netlist has {len(quotient_nets)} cross-super nets"
        )

    # SDF init the parent benchmark.
    sdf_pos = sdf_init(benchmark)
    sdf_pos_np = sdf_pos.cpu().numpy().astype(np.float64)

    # Compute initial super-macro centers as centroids of their constituents.
    super_centers_init = np.zeros((K, 2), dtype=np.float64)
    for k in range(K):
        members = super_members[k]
        if not members:
            super_centers_init[k] = (
                benchmark.canvas_width / 2.0,
                benchmark.canvas_height / 2.0,
            )
            continue
        super_centers_init[k, 0] = np.mean([sdf_pos_np[m, 0] for m in members])
        super_centers_init[k, 1] = np.mean([sdf_pos_np[m, 1] for m in members])

    # Coarse-scale CD with NO non-overlap constraint (super-macros are
    # approximate aggregates; actual non-overlap is enforced at fine-scale
    # uncoarsen + project_overlaps). The greedy CD finds whatever HPWL
    # improvements are reachable from the SDF centroids; if SDF is already
    # near-optimal at coarse scale (often true on IBM benchmarks), this is
    # a no-op and multigrid_init returns SDF-equivalent placement.
    super_centers_post = coarse_cd_supermacros(
        super_centers_init,
        super_sizes,
        quotient_nets,
        canvas_w=float(benchmark.canvas_width),
        canvas_h=float(benchmark.canvas_height),
        grid_n=grid_n,
        max_sweeps=max_sweeps,
        enforce_nonoverlap=False,
        log_fn=log_fn,
    )

    # Uncoarsen: rigid-translate each constituent by (super_center_post −
    # sdf_centroid). This preserves SDF's intra-cluster relative positions
    # exactly. Super-macro bboxes are sized at SLACK=2.0× constituent area
    # (in clustering layer) so constituents have room without rescaling.
    # project_overlaps cleans up any boundary collisions.
    out = sdf_pos_np.copy()
    fixed = benchmark.macro_fixed.cpu().numpy()
    sizes_np = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    for k in range(K):
        members = super_members[k]
        if not members:
            continue
        member_pos = sdf_pos_np[members, :]
        sdf_centroid = member_pos.mean(axis=0)
        dx = super_centers_post[k, 0] - sdf_centroid[0]
        dy = super_centers_post[k, 1] - sdf_centroid[1]
        for m in members:
            if fixed[m]:
                continue
            out[m, 0] += dx
            out[m, 1] += dy

    # Soft macros stay at SDF positions (already in `out`).
    # Now clip to canvas + legalize.
    sizes_np = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    for i in range(n_macros):
        if fixed[i]:
            continue
        half_w = sizes_np[i, 0] / 2.0
        half_h = sizes_np[i, 1] / 2.0
        out[i, 0] = float(np.clip(out[i, 0], half_w, cw - half_w))
        out[i, 1] = float(np.clip(out[i, 1], half_h, ch - half_h))

    placement = torch.tensor(out, dtype=sdf_pos.dtype)
    proj_result = project_overlaps(placement, benchmark)
    # project_overlaps returns either (tensor, int_residual) or just tensor;
    # handle both for safety.
    if isinstance(proj_result, tuple):
        legalized, residual = proj_result
        if log_fn is not None:
            log_fn(f"multigrid_init: project_overlaps residual={residual}")
    else:
        legalized = proj_result
    return legalized


# ── Smoke test ─────────────────────────────────────────────────────────────


def _smoke_test(bench_name: str) -> None:
    """End-to-end multigrid_init on a benchmark; print diagnostics + proxy."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[3]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    print(f"benchmark={bench_name}  n_hard={benchmark.num_hard_macros}")
    placement = multigrid_init(benchmark, log_fn=lambda s: print(s, flush=True))

    ov = compute_overlap_metrics(placement, benchmark)
    px = compute_proxy_cost(placement, benchmark, plc)
    print(f"\n=== multigrid init result for {bench_name} ===")
    print(
        f"proxy={px['proxy_cost']:.4f}  wl={px['wirelength_cost']:.3f}  "
        f"density={px['density_cost']:.3f}  cong={px['congestion_cost']:.3f}"
    )
    print(f"overlap_count={ov['overlap_count']}  total_overlap_area={ov['total_overlap_area']:.6f}")

    # Compare to plain SDF init.
    sdf_pos = sdf_init(benchmark)
    sdf_px = compute_proxy_cost(sdf_pos, benchmark, plc)
    print(
        f"\n--- plain SDF init for comparison ---\n"
        f"proxy={sdf_px['proxy_cost']:.4f}  wl={sdf_px['wirelength_cost']:.3f}  "
        f"density={sdf_px['density_cost']:.3f}  cong={sdf_px['congestion_cost']:.3f}"
    )
    delta = px['proxy_cost'] - sdf_px['proxy_cost']
    print(
        f"\nmultigrid Δ vs SDF: {delta:+.4f} "
        f"({delta / sdf_px['proxy_cost'] * 100:+.2f} %)"
    )


if __name__ == "__main__":
    import sys
    bench_name = sys.argv[1] if len(sys.argv) > 1 else "ibm10"
    _smoke_test(bench_name)
