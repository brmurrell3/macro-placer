"""Debug: which hard pairs overlap after V4 row-pack on ibm01."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir

from experiments.E63_spectral_init.code.cd_lns_sa_spectral_kjoint import (
    _build_netlist_laplacian,
    _row_pack_legalize_spectral_fixed_aware,
)
import scipy.sparse.linalg

bench_dir = find_benchmark_dir("ibm01")
benchmark, plc = load_benchmark_from_dir(str(bench_dir))

n_hard = benchmark.num_hard_macros
sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
fixed_mask = benchmark.macro_fixed.cpu().numpy()
orig_pos = benchmark.macro_positions.cpu().numpy().astype(np.float64)

# Spectral coords
L = _build_netlist_laplacian(benchmark, plc)
vals, vecs = scipy.sparse.linalg.eigsh(L, k=4, sigma=0.0, which="LM")
order = np.argsort(vals)
spec_xy = np.column_stack([vecs[:, order[1]], vecs[:, order[2]]])

pos = _row_pack_legalize_spectral_fixed_aware(
    spectral_xy=spec_xy,
    sizes_np=sizes,
    fixed_mask=fixed_mask,
    original_positions=orig_pos,
    n_hard=n_hard,
    canvas_w=float(benchmark.canvas_width),
    canvas_h=float(benchmark.canvas_height),
    log_fn=lambda s: print(s, flush=True),
)

# Manually check hard pairs.
movable_idx = np.where(~fixed_mask[:n_hard])[0]
overlaps = []
for i in range(n_hard):
    for j in range(i + 1, n_hard):
        dx = abs(pos[i, 0] - pos[j, 0])
        dy = abs(pos[i, 1] - pos[j, 1])
        min_sep_x = (sizes[i, 0] + sizes[j, 0]) / 2.0
        min_sep_y = (sizes[i, 1] + sizes[j, 1]) / 2.0
        ox = max(0.0, min_sep_x - dx)
        oy = max(0.0, min_sep_y - dy)
        if ox > 0 and oy > 0:
            overlaps.append((i, j, ox, oy, ox * oy))

print(f"\nTotal hard↔hard overlaps after V4: {len(overlaps)}")
print(f"\nFirst 10 overlapping pairs:")
print(f"{'i':>4} {'j':>4} {'pos_i':>20} {'pos_j':>20} {'size_i':>14} {'size_j':>14} {'ox':>6} {'oy':>6} {'area':>7}")
for i, j, ox, oy, area in overlaps[:10]:
    pi = f"({pos[i,0]:.2f},{pos[i,1]:.2f})"
    pj = f"({pos[j,0]:.2f},{pos[j,1]:.2f})"
    si = f"{sizes[i,0]:.2f}x{sizes[i,1]:.2f}"
    sj = f"{sizes[j,0]:.2f}x{sizes[j,1]:.2f}"
    print(f"{i:>4} {j:>4} {pi:>20} {pj:>20} {si:>14} {sj:>14} {ox:>6.3f} {oy:>6.3f} {area:>7.3f}")
print()

# Are any movables in row-pack overlapping each other? Or only fixed pre-placement?
# Group by sorted spectral_y to figure out row membership.
sort_keys = np.column_stack([
    -spec_xy[movable_idx, 1],
    spec_xy[movable_idx, 0],
    -sizes[movable_idx, 0] * sizes[movable_idx, 1],
])
sort_order = np.lexsort(sort_keys.T[::-1])

# For overlapping pairs, where are they in the sort order?
print(f"Sort positions of first 10 overlap pairs (in movable_idx[sort_order]):")
m_to_sort_pos = {int(movable_idx[sort_order[k]]): k for k in range(len(sort_order))}
for i, j, ox, oy, _ in overlaps[:10]:
    si = m_to_sort_pos.get(i, "fixed?")
    sj = m_to_sort_pos.get(j, "fixed?")
    print(f"  i={i} (sort_pos={si}) j={j} (sort_pos={sj})")
