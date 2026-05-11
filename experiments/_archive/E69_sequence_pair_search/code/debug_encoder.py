"""Debug why the encoder finds a cycle on E25 ibm01 output."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sp_inverter import _pair_relation, LEFT, RIGHT, BELOW, ABOVE


def main():
    e25 = torch.load(
        _HERE.parent / "results" / "placements" / "e25_ibm01.pt",
        weights_only=False,
    )
    n_hard = e25["num_hard_macros"]
    pos = e25["placement"][:n_hard].numpy().astype(np.float64)
    siz = e25["macro_sizes"][:n_hard].numpy().astype(np.float64)
    print(f"n_hard = {n_hard}")
    print(f"size range: w=[{siz[:, 0].min():.6f}, {siz[:, 0].max():.6f}], "
          f"h=[{siz[:, 1].min():.6f}, {siz[:, 1].max():.6f}]")
    print(f"pos x range: [{pos[:, 0].min():.6f}, {pos[:, 0].max():.6f}]")
    print(f"pos y range: [{pos[:, 1].min():.6f}, {pos[:, 1].max():.6f}]")

    # Check for degenerate sizes.
    zero_w = np.where(siz[:, 0] < 1e-6)[0]
    zero_h = np.where(siz[:, 1] < 1e-6)[0]
    print(f"macros with width < 1e-6: {len(zero_w)} → {zero_w[:10].tolist()}")
    print(f"macros with height < 1e-6: {len(zero_h)} → {zero_h[:10].tolist()}")

    # Find the bottom-left macro (min x+y).
    sumxy = pos[:, 0] + pos[:, 1]
    bl_idx = int(np.argmin(sumxy))
    print(f"\nBottom-left macro (min x+y): idx={bl_idx} pos=({pos[bl_idx, 0]:.4f}, {pos[bl_idx, 1]:.4f}) size=({siz[bl_idx, 0]:.4f}, {siz[bl_idx, 1]:.4f})")
    print(f"  This macro should have indegree 0 in G+ (no other macro is geometrically left+below it).")

    # Count relations of bl_idx with all others.
    counts = {LEFT: 0, RIGHT: 0, BELOW: 0, ABOVE: 0}
    pred_in_gplus = []  # macros that are predecessors of bl_idx in G+
    succ_in_gplus = []  # successors
    for k in range(n_hard):
        if k == bl_idx:
            continue
        # Compute relation in canonical (i, j) order with i < j.
        if bl_idx < k:
            rel = _pair_relation(pos, siz, bl_idx, k)
            # bl_idx is i; k is j.
            if rel == LEFT:
                # bl_idx LEFT k → bl_idx → k in G+
                succ_in_gplus.append((k, "LEFT"))
            elif rel == RIGHT:
                # bl_idx RIGHT k → k → bl_idx in G+
                pred_in_gplus.append((k, "RIGHT_of_bl"))
            elif rel == BELOW:
                succ_in_gplus.append((k, "BELOW"))
            elif rel == ABOVE:
                pred_in_gplus.append((k, "ABOVE_of_bl"))
        else:
            rel = _pair_relation(pos, siz, k, bl_idx)
            # k is i; bl_idx is j.
            if rel == LEFT:
                # k LEFT bl_idx → k → bl_idx in G+
                pred_in_gplus.append((k, "LEFT_of_bl"))
            elif rel == RIGHT:
                # k RIGHT bl_idx → bl_idx → k in G+
                succ_in_gplus.append((k, "RIGHT_of_bl"))
            elif rel == BELOW:
                # k BELOW bl_idx → k → bl_idx in G+
                pred_in_gplus.append((k, "BELOW_of_bl"))
            elif rel == ABOVE:
                succ_in_gplus.append((k, "ABOVE_of_bl"))
        counts[rel] += 1

    print(f"\nRelations of bl_idx with others:")
    print(f"  LEFT={counts[LEFT]} RIGHT={counts[RIGHT]} BELOW={counts[BELOW]} ABOVE={counts[ABOVE]}")
    print(f"  predecessors in G+: {len(pred_in_gplus)}  (should be 0 for true bottom-left)")
    print(f"  successors in G+: {len(succ_in_gplus)}")

    if pred_in_gplus:
        print(f"\n  Top 5 'predecessors' of bottom-left macro:")
        for k, why in pred_in_gplus[:5]:
            print(f"    macro {k}: pos=({pos[k, 0]:.4f}, {pos[k, 1]:.4f}) size=({siz[k, 0]:.4f}, {siz[k, 1]:.4f}) — {why}")

    # Manual relation check for bl_idx with first few macros.
    print(f"\nDirect inspection of bl_idx ({bl_idx}) with macros 0-5:")
    for k in range(6):
        if k == bl_idx:
            continue
        i, j = (bl_idx, k) if bl_idx < k else (k, bl_idx)
        eps = 1e-3
        hi_w = siz[i, 0] / 2.0; hi_h = siz[i, 1] / 2.0
        hj_w = siz[j, 0] / 2.0; hj_h = siz[j, 1] / 2.0
        i_left = pos[i, 0] - hi_w; i_right = pos[i, 0] + hi_w
        i_bot = pos[i, 1] - hi_h; i_top = pos[i, 1] + hi_h
        j_left = pos[j, 0] - hj_w; j_right = pos[j, 0] + hj_w
        j_bot = pos[j, 1] - hj_h; j_top = pos[j, 1] + hj_h
        h_sep_left = i_right <= j_left + eps
        h_sep_right = i_left + eps >= j_right
        v_sep_below = i_top <= j_bot + eps
        v_sep_above = i_bot + eps >= j_top
        rel = _pair_relation(pos, siz, i, j)
        rel_name = ["LEFT", "RIGHT", "BELOW", "ABOVE"][rel]
        print(
            f"  pair ({i},{j}): pos_i=({pos[i, 0]:.3f},{pos[i, 1]:.3f}) "
            f"pos_j=({pos[j, 0]:.3f},{pos[j, 1]:.3f}) "
            f"sz_i={siz[i, 0]:.3f}x{siz[i, 1]:.3f} sz_j={siz[j, 0]:.3f}x{siz[j, 1]:.3f}\n"
            f"    h_sep_left={h_sep_left} h_sep_right={h_sep_right} v_sep_below={v_sep_below} v_sep_above={v_sep_above}\n"
            f"    relation = {rel_name}"
        )


if __name__ == "__main__":
    main()
