"""Extended legalization: project_overlaps + jitter on stuck pairs.

The standard project_overlaps caps at 50 iterations. For aggressively-mixed
crossover placements, that's not enough. This module adds "jitter+project"
loops: when project_overlaps returns with residuals, jitter the overlapping
macros by small Gaussian noise (proportional to their size) and re-project.
Repeat until clean or budget exhausted.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.cd_core import project_overlaps
from macro_place.objective import compute_overlap_metrics


def _overlapping_macros(placement: torch.Tensor, benchmark: Benchmark) -> set:
    """Return set of macro indices involved in any hard-macro overlap."""
    n_hard = benchmark.num_hard_macros
    pos = placement[:n_hard].detach().cpu().numpy().astype(np.float64)
    siz = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
    hw = siz[:, 0] / 2.0
    hh = siz[:, 1] / 2.0
    out = set()
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            dx = abs(pos[i, 0] - pos[j, 0])
            dy = abs(pos[i, 1] - pos[j, 1])
            if dx < hw[i] + hw[j] and dy < hh[i] + hh[j]:
                out.add(i)
                out.add(j)
    return out


def extended_legalize(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    max_passes: int = 20,
    jitter_scale: float = 0.5,
    seed: int = 42,
) -> Tuple[torch.Tensor, dict]:
    """Apply project_overlaps repeatedly with jitter on residuals.

    Args:
      placement: candidate placement (may have overlaps).
      max_passes: max number of (jitter, project) cycles.
      jitter_scale: relative to macro size; e.g. 0.5 = jitter ±50% of half-size.
      seed: RNG for jitter.

    Returns:
      (legalized_placement, stats_dict). legalized may have residual overlaps
      if budget exhausted; check overlap count in stats.
    """
    rng = np.random.default_rng(seed)
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes_np = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed[:n_hard].cpu().numpy()
    half_w = sizes_np[:, 0] / 2.0
    half_h = sizes_np[:, 1] / 2.0

    current = placement.detach().clone()
    initial_overlap = compute_overlap_metrics(current, benchmark)["overlap_count"]
    current_overlap = initial_overlap
    history = [initial_overlap]

    for pass_idx in range(max_passes):
        if current_overlap == 0:
            break
        # First, project_overlaps.
        current, _ = project_overlaps(current, benchmark)
        ov = compute_overlap_metrics(current, benchmark)["overlap_count"]
        if ov == 0:
            current_overlap = 0
            history.append(0)
            break
        # Jitter overlapping macros.
        overlapping = _overlapping_macros(current, benchmark)
        if not overlapping:
            current_overlap = 0
            break
        pos = current.detach().cpu().numpy().astype(np.float64).copy()
        for m in overlapping:
            if bool(fixed[m]):
                continue
            # Jitter proportional to half-size, but keep within canvas.
            jx = rng.normal(0, jitter_scale * half_w[m])
            jy = rng.normal(0, jitter_scale * half_h[m])
            pos[m, 0] = float(np.clip(pos[m, 0] + jx, half_w[m], cw - half_w[m]))
            pos[m, 1] = float(np.clip(pos[m, 1] + jy, half_h[m], ch - half_h[m]))
        current = torch.tensor(pos, dtype=current.dtype)
        # Re-project after jitter.
        current, _ = project_overlaps(current, benchmark)
        ov = compute_overlap_metrics(current, benchmark)["overlap_count"]
        current_overlap = ov
        history.append(ov)
        # Decay jitter scale over passes to avoid wandering.
        jitter_scale *= 0.85

    return current, {
        "passes_used": len(history) - 1,
        "initial_overlap": initial_overlap,
        "final_overlap": current_overlap,
        "history": history,
    }
