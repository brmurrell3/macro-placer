"""E45 multigrid — Phase 1 clustering layer.

Hypergraph (macro-net) → graph (clique-expanded) → balanced K-way
partition via pymetis. The clique expansion uses edge weight
`1 / max(1, net_size - 1)` per shared net, consistent with the K-joint
adjacency score in E39.

Public API:
- `build_macro_graph(benchmark)`: returns CSR-style adjacency for hard
  macros only.
- `cluster_macros(benchmark, K)`: returns per-macro cluster assignments
  (length num_hard_macros, values in [0, K)).
- `super_macro_geometry(benchmark, cluster_ids, K)`: returns
  (super_sizes, super_areas, super_member_ids) — bbox dimensions per
  super-macro packed at canvas aspect ratio, total area, member lists.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

import pymetis

from macro_place.benchmark import Benchmark


# ── Macro-macro graph construction ──────────────────────────────────────────


def build_macro_graph(
    benchmark: Benchmark,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build clique-expanded macro-macro graph in METIS CSR format.

    Only hard macros are considered. For each net of size n_i, all C(n_i, 2)
    pairs of HARD macros in the net contribute an edge of weight
    `1 / max(1, n_i - 1)` (the same weighting used by `_adjacency_scores`
    in E39 / E41).

    Returns (xadj, adjncy, eweights) in pymetis CSR format:
    - xadj[i:i+1] points into adjncy for vertex i's neighbors.
    - adjncy is the flat list of all neighbors (each undirected edge
      appears twice).
    - eweights are integer edge weights (scaled by SCALE_INT, see below).
    """
    n_hard = int(benchmark.num_hard_macros)
    edges: Dict[Tuple[int, int], float] = defaultdict(float)

    for net_idx, nodes in enumerate(benchmark.net_nodes):
        node_list = nodes.cpu().numpy().tolist()
        # Filter to hard macros only.
        hard_in_net = [n for n in node_list if n < n_hard]
        if len(hard_in_net) < 2:
            continue
        w_per_pair = 1.0 / max(1, len(hard_in_net) - 1)
        net_w = float(benchmark.net_weights[net_idx]) if net_idx < len(
            benchmark.net_weights
        ) else 1.0
        w = w_per_pair * net_w
        for i in range(len(hard_in_net)):
            for j in range(i + 1, len(hard_in_net)):
                a, b = hard_in_net[i], hard_in_net[j]
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                edges[(a, b)] += w

    # Build per-vertex neighbor lists.
    SCALE_INT = 1000  # so weights down to 0.001 are preserved
    nbrs: List[List[Tuple[int, int]]] = [[] for _ in range(n_hard)]
    for (a, b), w in edges.items():
        iw = max(1, int(round(w * SCALE_INT)))
        nbrs[a].append((b, iw))
        nbrs[b].append((a, iw))

    # Pack into CSR.
    xadj_list: List[int] = [0]
    adjncy_list: List[int] = []
    eweights_list: List[int] = []
    for i in range(n_hard):
        for (j, iw) in nbrs[i]:
            adjncy_list.append(j)
            eweights_list.append(iw)
        xadj_list.append(len(adjncy_list))

    xadj = np.asarray(xadj_list, dtype=np.int32)
    adjncy = np.asarray(adjncy_list, dtype=np.int32)
    eweights = np.asarray(eweights_list, dtype=np.int32)
    return xadj, adjncy, eweights


# ── METIS partitioning ─────────────────────────────────────────────────────


def cluster_macros(
    benchmark: Benchmark, K: int, seed: int = 42
) -> np.ndarray:
    """Partition hard macros into K balanced super-macros via pymetis.

    Returns an int64 array of length num_hard_macros with values in [0, K).

    Edge weights are scaled to ints (pymetis requires integer weights).
    Scale factor: 1000 (so weights down to 0.001 are preserved).
    """
    n_hard = int(benchmark.num_hard_macros)
    if K >= n_hard:
        # Each macro is its own cluster.
        return np.arange(n_hard, dtype=np.int64) % K

    xadj, adjncy, eweights = build_macro_graph(benchmark)

    # Vertex weights = macro area (so balance constraint distributes area,
    # not macro count) — prevents one super-macro from getting all the big
    # macros.
    sizes_np = benchmark.macro_sizes.cpu().numpy()
    vweights = np.asarray(
        [
            max(1, int(round(float(sizes_np[i, 0] * sizes_np[i, 1]))))
            for i in range(n_hard)
        ],
        dtype=np.int32,
    )

    # pymetis is deterministic given the same input; no seed parameter.
    n_cuts, parts = pymetis.part_graph(
        K,
        xadj=xadj,
        adjncy=adjncy,
        vweights=vweights,
        eweights=eweights,
    )
    assignments = np.asarray(parts, dtype=np.int64)
    assert assignments.shape == (n_hard,), (
        f"cluster_macros: pymetis returned {len(assignments)} parts, expected {n_hard}"
    )
    return assignments


# ── Super-macro geometry ───────────────────────────────────────────────────


def super_macro_geometry(
    benchmark: Benchmark,
    cluster_ids: np.ndarray,
    K: int,
) -> Tuple[torch.Tensor, np.ndarray, List[List[int]]]:
    """Compute per-super-macro bounding-box dimensions, total area, and
    member lists.

    Each super-macro's bbox is sized to contain the sum of constituent
    macro areas plus a slack factor (1.1×) for placement room, packed at
    the canvas aspect ratio.

    Returns:
      super_sizes: torch.Tensor [K, 2] — (width, height) per super-macro
      super_areas: np.ndarray [K] — total constituent area per super-macro
      super_members: List[List[int]] — member macro indices per super-macro
    """
    n_hard = int(benchmark.num_hard_macros)
    sizes_np = benchmark.macro_sizes.cpu().numpy()

    super_areas = np.zeros(K, dtype=np.float64)
    super_members: List[List[int]] = [[] for _ in range(K)]
    for i in range(n_hard):
        c = int(cluster_ids[i])
        super_areas[c] += float(sizes_np[i, 0] * sizes_np[i, 1])
        super_members[c].append(i)

    canvas_aspect = float(benchmark.canvas_width) / float(benchmark.canvas_height)
    SLACK = 1.1
    super_sizes = np.zeros((K, 2), dtype=np.float64)
    for k in range(K):
        if super_areas[k] <= 0:
            super_sizes[k] = (0.0, 0.0)
            continue
        a = super_areas[k] * SLACK
        # Pack bbox at canvas aspect ratio: w/h = canvas_aspect, w*h = a
        h = float(np.sqrt(a / canvas_aspect))
        w = float(canvas_aspect * h)
        super_sizes[k] = (w, h)

    return torch.tensor(super_sizes, dtype=torch.float32), super_areas, super_members


# ── Smoke test (run directly: `python multigrid_clustering.py ibm10`) ──────


def _smoke_test(bench_name: str, K: int = 0) -> None:
    """Cluster a benchmark and print summary statistics."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[3]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.loader import load_benchmark_from_dir

    bench_dir = find_benchmark_dir(bench_name)
    benchmark, _plc = load_benchmark_from_dir(str(bench_dir))
    n_hard = int(benchmark.num_hard_macros)
    if K <= 0:
        K = max(2, int(round(np.sqrt(n_hard))))
    print(f"benchmark={bench_name}  n_hard={n_hard}  K={K}")
    print(f"canvas={benchmark.canvas_width:.1f} x {benchmark.canvas_height:.1f}")

    cluster_ids = cluster_macros(benchmark, K)
    super_sizes, super_areas, super_members = super_macro_geometry(
        benchmark, cluster_ids, K
    )

    member_counts = np.array([len(m) for m in super_members])
    print(
        f"super-macro member counts: min={member_counts.min()} "
        f"max={member_counts.max()} mean={member_counts.mean():.1f}"
    )
    print(
        f"super-macro areas (μm²):  min={super_areas.min():.1f} "
        f"max={super_areas.max():.1f} mean={super_areas.mean():.1f}"
    )
    print(
        f"super-macro bbox sizes:   min_w={super_sizes[:,0].min():.1f} "
        f"max_w={super_sizes[:,0].max():.1f} "
        f"min_h={super_sizes[:,1].min():.1f} max_h={super_sizes[:,1].max():.1f}"
    )

    # Sanity: total super-macro area should be ~= sum of macro areas.
    total_macro_area = float(
        (benchmark.macro_sizes[:n_hard, 0] * benchmark.macro_sizes[:n_hard, 1]).sum()
    )
    print(
        f"total area: macros={total_macro_area:.1f}  super-macros (slacked)={super_areas.sum():.1f}  "
        f"canvas={benchmark.canvas_width * benchmark.canvas_height:.1f}"
    )


if __name__ == "__main__":
    import sys
    bench_name = sys.argv[1] if len(sys.argv) > 1 else "ibm10"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    _smoke_test(bench_name, K)
