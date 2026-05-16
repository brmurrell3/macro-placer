"""12 um-aware legalizer.

ORFS PDN spacing requires >= 12 um edge-to-edge clearance between hard
macros. If our placement has pairs below 12 um, ORFS pushes them at
Tier-2 eval time, which can perturb routing in unpredictable ways.

This script takes a placement and minimally pushes violating pairs apart
to >= 12 um. Preserves:
  - hard-macro non-overlap (no new overlaps introduced)
  - fixed macros (never moved)
  - canvas bounds (macros stay fully inside canvas)

Algorithm: iterative greedy worst-pair push. Each iteration finds the
worst-violating pair, pushes along the axis with the larger current gap
(minimum work to clear), respecting fixed mask + canvas bounds. Stops
when all pairs >= 12 um or no further progress.

Usage:
    uv run python analysis/macro_clearance_diagnostic/legalize_12um.py \\
        --bench ariane133 \\
        --in-pt  /path/to/placement.pt \\
        --out-pt /path/to/placement_12um.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def _pair_clearance(pos: np.ndarray, sz: np.ndarray):
    """Return (gap_x [n,n], gap_y [n,n], clearance [n,n]) edge-to-edge.

    clearance[i,j] = max(gap_x, gap_y); diagonal forced to +inf.
    Non-overlap iff clearance >= 0; ORFS-safe iff clearance >= 12.
    """
    diff = np.abs(pos[:, None, :] - pos[None, :, :])
    half_sum = (sz[:, None, :] + sz[None, :, :]) / 2.0
    gap = diff - half_sum
    gap_x = gap[..., 0]
    gap_y = gap[..., 1]
    clearance = np.maximum(gap_x, gap_y)
    np.fill_diagonal(clearance, np.inf)
    return gap_x, gap_y, clearance


def legalize_12um(
    placement: torch.Tensor,
    benchmark,
    threshold: float = 12.0,
    max_iters: int = 2000,
    safety_margin: float = 0.05,
    step_scale: float = 0.5,
    verbose: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Push hard-macro pairs apart to clearance >= threshold um.

    Force-based batch update: at each iter, sum repulsion forces from all
    violating pairs (per non-fixed macro), apply with damping, clip to
    canvas. Converges in O(log N) iters for most NG45 designs.

    Returns (new_placement_tensor, stats_dict).
    """
    n_hard = benchmark.num_hard_macros
    pos = placement[:n_hard].detach().cpu().numpy().astype(np.float64).copy()
    sz = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed[:n_hard].detach().cpu().numpy().astype(bool)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    half = sz / 2.0
    canvas = np.array([cw, ch])

    # Snapshot stats.
    gx0, gy0, clr0 = _pair_clearance(pos, sz)
    triu = np.triu_indices(n_hard, k=1)
    n_violation_init = int(((clr0[triu] < threshold)).sum())
    n_overlap_init = int(((gx0[triu] < 0) & (gy0[triu] < 0)).sum())
    min_clr_init = float(clr0[triu].min())

    if verbose:
        print(f"[legalize] init: violations={n_violation_init}, "
              f"overlaps={n_overlap_init}, min_clr={min_clr_init:.3f} um, "
              f"target={threshold:.1f} um")

    target = threshold + safety_margin
    last_violations = n_violation_init
    stagnation_count = 0

    for it in range(max_iters):
        gx, gy, clr = _pair_clearance(pos, sz)
        viol_mask = (clr < target)
        np.fill_diagonal(viol_mask, False)
        n_violations = int(np.triu(viol_mask, k=1).sum())
        if n_violations == 0:
            break

        # For each violating pair (i,j), compute repulsion force.
        # Force axis: the axis with the larger current gap (less work to clear).
        # Force magnitude per macro: (target - cur_gap) / 2 (or full if other fixed).
        forces = np.zeros_like(pos)  # [n, 2]

        # Choose axis per pair: 0 if gx >= gy else 1.
        axis_choice = (gx < gy).astype(np.int8)  # 1 if y-axis chosen, 0 if x-axis
        cur_gap = np.where(axis_choice == 0, gx, gy)  # [n, n]
        need = np.maximum(0.0, target - cur_gap)  # [n, n]
        # Direction sign for i (axis k): -1 if pos[i,k] < pos[j,k] else +1.
        dpos_x = pos[:, None, 0] - pos[None, :, 0]
        dpos_y = pos[:, None, 1] - pos[None, :, 1]
        sign_i = np.where(axis_choice == 0,
                          np.sign(dpos_x), np.sign(dpos_y))  # [n, n]
        sign_i = np.where(sign_i == 0, 1.0, sign_i)  # break ties

        # Per-pair push allotment (for macro i in pair (i,j)).
        # If both free: each gets need/2.
        # If i fixed: i gets 0, j gets need.
        # If j fixed: i gets need, j gets 0.
        # If both fixed: 0 (cannot fix).
        fixed_i = fixed[:, None]
        fixed_j = fixed[None, :]
        push_i = np.where(viol_mask,
                          np.where(fixed_i, 0.0,
                                   np.where(fixed_j, need, need / 2.0)),
                          0.0)
        # Accumulate force on macro i from pair (i, j).
        # Force i along chosen axis = sign_i * push_i.
        f_along = sign_i * push_i  # [n, n]
        # Distribute to x or y based on axis_choice.
        f_x = np.where(axis_choice == 0, f_along, 0.0)
        f_y = np.where(axis_choice == 1, f_along, 0.0)
        # Sum over j (axis 1) — but each pair (i,j) contributes to i AND j.
        # The above only computed i's force from pair (i,j). For j's force,
        # we need to sum the (j, i) contribution which is symmetric with
        # opposite sign. Since the matrix is computed for all (i,j) including
        # j<i, the row sum over j gives i's total push and similarly for j.
        forces[:, 0] = f_x.sum(axis=1)
        forces[:, 1] = f_y.sum(axis=1)

        # Apply step (scale * forces). Zero out fixed-macro forces.
        forces[fixed] = 0.0
        # Adaptive step: shrink when oscillating.
        eff_step = step_scale / (1.0 + stagnation_count * 0.1)
        new_pos = pos + eff_step * forces
        # Clip to canvas.
        new_pos = np.maximum(new_pos, half)
        new_pos = np.minimum(new_pos, canvas[None, :] - half)
        # Re-zero fixed.
        new_pos[fixed] = pos[fixed]

        # Update + check progress.
        delta = np.abs(new_pos - pos).max()
        pos = new_pos
        if n_violations >= last_violations:
            stagnation_count += 1
        else:
            stagnation_count = 0
        last_violations = n_violations
        if stagnation_count >= 100 and delta < 0.005:
            if verbose:
                print(f"[legalize] phase-1 stagnated at iter {it}: "
                      f"violations={n_violations}, delta={delta:.4f}")
            break

    # Phase 2: greedy single-pair push for residual violations.
    # After force-based phase converges or stagnates, sweep remaining
    # violations one at a time. Single-pair pushes are surgical and won't
    # cause oscillation as long as we re-evaluate clearance after each.
    n_phase2_pushes = 0
    if verbose:
        print(f"[legalize] entering phase-2 (greedy single-pair on residuals)")
    for it2 in range(3000):
        gx, gy, clr = _pair_clearance(pos, sz)
        mask = np.triu(np.ones_like(clr, dtype=bool), k=1) & (clr < target)
        if not mask.any():
            break
        flat = np.where(mask, clr, np.inf)
        idx = np.argmin(flat)
        i, j = np.unravel_index(idx, flat.shape)
        cur_gx_ij = gx[i, j]
        cur_gy_ij = gy[i, j]
        # Pick axis with larger gap (least work).
        if cur_gx_ij >= cur_gy_ij:
            ax = 0
            cg = cur_gx_ij
            cd = cw
        else:
            ax = 1
            cg = cur_gy_ij
            cd = ch
        needp = target - cg
        if needp <= 0:
            break
        if pos[i, ax] < pos[j, ax]:
            si, sj = -1.0, +1.0
        else:
            si, sj = +1.0, -1.0
        if fixed[i] and fixed[j]:
            break  # cannot resolve
        elif fixed[i]:
            pi, pj = 0.0, needp
        elif fixed[j]:
            pi, pj = needp, 0.0
        else:
            pi, pj = needp / 2.0, needp / 2.0
        hi = sz[i, ax] / 2.0
        hj = sz[j, ax] / 2.0
        ni = max(hi, min(cd - hi, pos[i, ax] + si * pi))
        nj = max(hj, min(cd - hj, pos[j, ax] + sj * pj))
        di = abs(ni - pos[i, ax])
        dj = abs(nj - pos[j, ax])
        if di < 1e-9 and dj < 1e-9:
            # Try other axis.
            ax2 = 1 - ax
            cg2 = cur_gy_ij if ax == 0 else cur_gx_ij
            cd2 = ch if ax == 0 else cw
            needp2 = target - cg2
            if needp2 <= 0:
                break
            if pos[i, ax2] < pos[j, ax2]:
                si2, sj2 = -1.0, +1.0
            else:
                si2, sj2 = +1.0, -1.0
            if fixed[i]:
                pi2, pj2 = 0.0, needp2
            elif fixed[j]:
                pi2, pj2 = needp2, 0.0
            else:
                pi2, pj2 = needp2 / 2.0, needp2 / 2.0
            hi2 = sz[i, ax2] / 2.0
            hj2 = sz[j, ax2] / 2.0
            ni2 = max(hi2, min(cd2 - hi2, pos[i, ax2] + si2 * pi2))
            nj2 = max(hj2, min(cd2 - hj2, pos[j, ax2] + sj2 * pj2))
            if abs(ni2 - pos[i, ax2]) < 1e-9 and abs(nj2 - pos[j, ax2]) < 1e-9:
                break  # truly stuck
            pos[i, ax2] = ni2
            pos[j, ax2] = nj2
        else:
            pos[i, ax] = ni
            pos[j, ax] = nj
        n_phase2_pushes += 1
    if verbose:
        print(f"[legalize] phase-2 pushes={n_phase2_pushes}, iters={it2+1}")

    # Final stats.
    gx1, gy1, clr1 = _pair_clearance(pos, sz)
    n_violation_final = int(((clr1[triu] < threshold)).sum())
    n_overlap_final = int(((gx1[triu] < 0) & (gy1[triu] < 0)).sum())
    min_clr_final = float(clr1[triu].min())

    if verbose:
        print(f"[legalize] iters={it+1}")
        print(f"[legalize] final: violations={n_violation_final}, "
              f"overlaps={n_overlap_final}, min_clr={min_clr_final:.3f} um")

    new_placement = placement.clone()
    new_placement[:n_hard] = torch.from_numpy(pos).to(new_placement.dtype)

    stats = {
        "n_iters": it + 1,
        "n_violation_init": n_violation_init,
        "n_violation_final": n_violation_final,
        "n_overlap_init": n_overlap_init,
        "n_overlap_final": n_overlap_final,
        "min_clr_init_um": min_clr_init,
        "min_clr_final_um": min_clr_final,
    }
    return new_placement, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--in-pt", required=True)
    ap.add_argument("--out-pt", required=True)
    ap.add_argument("--threshold", type=float, default=12.0)
    ap.add_argument("--max-iters", type=int, default=5000)
    args = ap.parse_args()

    bench_dir = find_benchmark_dir(args.bench)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    data = torch.load(args.in_pt, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        for k in ("placement", "polished_placement"):
            if k in data:
                placement = data[k]
                break
        else:
            raise RuntimeError(f"No placement key in {args.in_pt}: {list(data.keys())}")
    else:
        placement = data
    placement = placement.to(torch.float32)

    # Pre stats.
    proxy_in = float(compute_proxy_cost(placement, benchmark, plc)["proxy_cost"])
    ovl_in = int(compute_overlap_metrics(placement, benchmark)["overlap_count"])
    print(f"INPUT  proxy={proxy_in:.5f}  ovl={ovl_in}")

    t0 = time.time()
    new_placement, stats = legalize_12um(
        placement, benchmark, threshold=args.threshold,
        max_iters=args.max_iters, verbose=True,
    )
    wall = time.time() - t0
    print(f"[legalize] wall {wall:.2f}s")

    # Post stats.
    proxy_out = float(compute_proxy_cost(new_placement, benchmark, plc)["proxy_cost"])
    ovl_out = int(compute_overlap_metrics(new_placement, benchmark)["overlap_count"])
    delta_pct = 100.0 * (proxy_out - proxy_in) / max(1e-9, proxy_in)
    print(f"OUTPUT proxy={proxy_out:.5f}  ovl={ovl_out}  "
          f"delta={delta_pct:+.3f}%")

    if ovl_out > 0:
        print(f"WARN: legalizer introduced {ovl_out} overlaps. Not saving.")
        return 1

    torch.save(new_placement, args.out_pt)
    print(f"saved {args.out_pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
