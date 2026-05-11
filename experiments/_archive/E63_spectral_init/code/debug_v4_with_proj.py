"""Debug: trace overlaps before and after each step of legalization on ibm01."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import scipy.sparse.linalg

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics

from experiments.E63_spectral_init.code.cd_lns_sa_spectral_kjoint import (
    _build_netlist_laplacian,
    _row_pack_legalize_spectral_fixed_aware,
)

bench_dir = find_benchmark_dir("ibm01")
benchmark, plc = load_benchmark_from_dir(str(bench_dir))

n_hard = benchmark.num_hard_macros
sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
fixed_mask = benchmark.macro_fixed.cpu().numpy()
orig_pos = benchmark.macro_positions.cpu().numpy().astype(np.float64)

L = _build_netlist_laplacian(benchmark, plc)
vals, vecs = scipy.sparse.linalg.eigsh(L, k=4, sigma=0.0, which="LM")
order = np.argsort(vals)
spec_xy = np.column_stack([vecs[:, order[1]], vecs[:, order[2]]])

pos_f64 = _row_pack_legalize_spectral_fixed_aware(
    spectral_xy=spec_xy,
    sizes_np=sizes,
    fixed_mask=fixed_mask,
    original_positions=orig_pos,
    n_hard=n_hard,
    canvas_w=float(benchmark.canvas_width),
    canvas_h=float(benchmark.canvas_height),
    log_fn=None,
)

# Step 1: count overlaps in float64.
def count(arr):
    p = torch.tensor(arr, dtype=torch.float32)
    m = compute_overlap_metrics(p, benchmark)
    return m["overlap_count"], m["total_overlap_area"]

# Direct compute with float64 simulating compute_overlap_metrics:
def count_f64(pos):
    cnt = 0
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            dx = abs(pos[i, 0] - pos[j, 0])
            dy = abs(pos[i, 1] - pos[j, 1])
            min_x = (sizes[i, 0] + sizes[j, 0]) / 2.0
            min_y = (sizes[i, 1] + sizes[j, 1]) / 2.0
            if dx < min_x and dy < min_y:
                cnt += 1
    return cnt

print(f"V4 float64 raw overlap count = {count_f64(pos_f64)}")

# Convert to float32 (what _spectral_init does).
pos_f32 = torch.tensor(pos_f64, dtype=torch.float32)
m = compute_overlap_metrics(pos_f32, benchmark)
print(f"V4 → float32 → compute_overlap_metrics: count={m['overlap_count']} area={m['total_overlap_area']:.4f}")

# Now run project_overlaps and recount.
pos_after, proj_iters = project_overlaps(pos_f32, benchmark)
m2 = compute_overlap_metrics(pos_after, benchmark)
print(f"After project_overlaps ({proj_iters} iters): count={m2['overlap_count']} area={m2['total_overlap_area']:.4f}")

# Sample some macro centers before / after project_overlaps for first overlapping pair.
diff = (pos_after.cpu().numpy() - pos_f32.cpu().numpy())
diff_norm = np.linalg.norm(diff, axis=1)
print(f"\nproject_overlaps moved macros: max_dist={diff_norm.max():.3f}, "
      f"non-zero={np.sum(diff_norm > 1e-6)}/{len(diff_norm)}")

# What sort_pos has the largest movement?
top_movers = np.argsort(-diff_norm)[:10]
print(f"\nTop 10 macros moved by project_overlaps:")
for idx in top_movers:
    is_hard = idx < n_hard
    is_fixed = bool(fixed_mask[idx]) if is_hard else False
    sz = sizes[idx]
    print(f"  macro {idx} (hard={is_hard} fixed={is_fixed} size={sz[0]:.2f}x{sz[1]:.2f}): "
          f"({pos_f32[idx,0]:.2f},{pos_f32[idx,1]:.2f}) → "
          f"({pos_after[idx,0]:.2f},{pos_after[idx,1]:.2f}), Δ={diff_norm[idx]:.3f}")
