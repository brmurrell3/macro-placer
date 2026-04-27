"""
E11 — Diverse init strategies for DPO

Four structurally distinct priors that produce a `[num_macros, 2]` tensor of
center positions. Each respects `benchmark.macro_fixed`. Soft macros are
copied from `benchmark.macro_positions`; only hard movable macros are placed
by the strategy, then projected to non-overlap.

Hypothesis (E11): different priors land DPO in different convergence basins,
unlike E5 where Gaussian-perturbed SDF seeds collapsed to the same basin.
"""

import sys
import io
import math
import random
import importlib.util
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# Helper: load a placer module by file path (lets us reuse existing code)
# ---------------------------------------------------------------------------

def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Helper: simple iterative push-apart legalizer (used by random init)
# ---------------------------------------------------------------------------

def _push_apart_legalize(pos_np, sizes_np, movable, n_hard,
                         canvas_w, canvas_h, max_rounds=60, eps=0.002):
    """Iterative pairwise push-apart for hard macros.

    Returns positions; caller should still verify zero overlaps.
    """
    pos = pos_np.copy()
    half_w = sizes_np[:n_hard, 0] / 2
    half_h = sizes_np[:n_hard, 1] / 2

    for _ in range(max_rounds):
        dx_mat = np.abs(pos[:n_hard, 0:1] - pos[:n_hard, 0:1].T)
        dy_mat = np.abs(pos[:n_hard, 1:2] - pos[:n_hard, 1:2].T)
        min_dx = half_w[:, None] + half_w[None, :] + eps
        min_dy = half_h[:, None] + half_h[None, :] + eps
        overlap = (dx_mat < min_dx) & (dy_mat < min_dy)
        np.fill_diagonal(overlap, False)
        pairs = np.argwhere(np.triu(overlap))
        if len(pairs) == 0:
            break

        for a, b in pairs:
            mov_a = bool(movable[a])
            mov_b = bool(movable[b])
            if not (mov_a or mov_b):
                continue
            viol_x = min_dx[a, b] - dx_mat[a, b]
            viol_y = min_dy[a, b] - dy_mat[a, b]
            if viol_x < viol_y:
                sign = 1.0 if pos[a, 0] < pos[b, 0] else -1.0
                if mov_a and mov_b:
                    pos[a, 0] -= sign * viol_x / 2
                    pos[b, 0] += sign * viol_x / 2
                elif mov_a:
                    pos[a, 0] -= sign * viol_x
                else:
                    pos[b, 0] += sign * viol_x
            else:
                sign = 1.0 if pos[a, 1] < pos[b, 1] else -1.0
                if mov_a and mov_b:
                    pos[a, 1] -= sign * viol_y / 2
                    pos[b, 1] += sign * viol_y / 2
                elif mov_a:
                    pos[a, 1] -= sign * viol_y
                else:
                    pos[b, 1] += sign * viol_y

        for i in range(n_hard):
            if movable[i]:
                pos[i, 0] = np.clip(pos[i, 0], half_w[i] + eps,
                                    canvas_w - half_w[i] - eps)
                pos[i, 1] = np.clip(pos[i, 1], half_h[i] + eps,
                                    canvas_h - half_h[i] - eps)
    return pos


# ---------------------------------------------------------------------------
# Strategy 1: SDF (current DPO default)
# ---------------------------------------------------------------------------

def init_sdf(benchmark: Benchmark, seed: int = 42) -> torch.Tensor:
    """Run the SDF placer and return its placement as the DPO init."""
    sdf_path = Path(__file__).parent.parent / "polyhedra" / "init" / "sdf.py"
    mod = _load_module(sdf_path, "sdf")
    placer = mod.SDFPlacer(seed=seed)
    return placer.place(benchmark)


# ---------------------------------------------------------------------------
# Strategy 2: Will's SA seed
# ---------------------------------------------------------------------------

def init_will_seed(benchmark: Benchmark, seed: int = 42,
                   refine_iters: int = 1500) -> torch.Tensor:
    """Run Will's seed placer (legalize + SA refine) for a DPO init."""
    will_path = Path(__file__).parent.parent / "will_seed" / "placer.py"
    mod = _load_module(will_path, "will_seed_placer")
    # Reduce refine_iters from default 3000 — we only need a structurally-
    # distinct starting point, not full convergence
    placer = mod.WillSeedPlacer(seed=seed, refine_iters=refine_iters)
    return placer.place(benchmark)


# ---------------------------------------------------------------------------
# Strategy 3: Greedy row (shelf-packing)
# ---------------------------------------------------------------------------

def init_greedy(benchmark: Benchmark, seed: int = 42) -> torch.Tensor:
    """Run the greedy row placer for a DPO init."""
    greedy_path = (Path(__file__).parent.parent / "examples" /
                   "greedy_row_placer.py")
    mod = _load_module(greedy_path, "greedy_row")
    placer = mod.GreedyRowPlacer()
    return placer.place(benchmark)


# ---------------------------------------------------------------------------
# Strategy 4: Random uniform + push-apart projection
# ---------------------------------------------------------------------------

def init_random_projected(benchmark: Benchmark, seed: int = 42) -> torch.Tensor:
    """Uniform-random hard-macro centers within canvas, projected to
    non-overlap by iterative push-apart.

    Returns full [num_macros, 2] tensor; soft macros copied from
    `benchmark.macro_positions`.
    """
    rng = np.random.default_rng(seed)

    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)
    movable = benchmark.get_movable_mask().numpy()
    init_pos_np = benchmark.macro_positions.numpy().astype(np.float64).copy()

    half_w = sizes_np[:, 0] / 2
    half_h = sizes_np[:, 1] / 2

    pos = init_pos_np.copy()
    for i in range(n_hard):
        if movable[i]:
            lo_x = half_w[i]
            hi_x = max(lo_x + 1e-6, cw - half_w[i])
            lo_y = half_h[i]
            hi_y = max(lo_y + 1e-6, ch - half_h[i])
            pos[i, 0] = rng.uniform(lo_x, hi_x)
            pos[i, 1] = rng.uniform(lo_y, hi_y)

    # Iterative push-apart projection
    pos = _push_apart_legalize(pos, sizes_np, movable, n_hard, cw, ch,
                               max_rounds=80)

    full = benchmark.macro_positions.clone()
    full[:n_hard] = torch.tensor(pos[:n_hard], dtype=torch.float32)
    return full


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

INIT_STRATEGIES = {
    "sdf": init_sdf,
    "will": init_will_seed,
    "greedy": init_greedy,
    "random": init_random_projected,
}


def get_init(name: str):
    if name not in INIT_STRATEGIES:
        raise KeyError(f"Unknown init strategy '{name}'. Choices: "
                       f"{sorted(INIT_STRATEGIES)}")
    return INIT_STRATEGIES[name]
