"""E27 — persistence analysis of CD trajectories from diverse inits.

Reads `trajectories/*.json` (each = one CD trajectory from one init),
clusters their final placements, and emits a verdict on whether the
proxy landscape near the E25 floor has *one big basin* or *several
deep basins*.

Method (H_0 of sublevel-set filtration, simplified):

  1. For each benchmark, gather all trajectories.
  2. Compute pairwise placement distance (L2 over flattened hard-macro
     positions) between every pair of final placements.
  3. Cluster by single-linkage at threshold = 5 % of canvas diagonal
     (= "two trajectories are in the same basin if their final
     placements differ by less than 5 % canvas-diag").
  4. For each cluster: size, mean & std of final proxy.
  5. Verdict per benchmark: "single_basin" / "multi_basin" / "ambiguous".
  6. If `gudhi` is importable, also emit the classic H_0 persistence
     diagram of the Vietoris-Rips complex on final placements; without
     gudhi, the cluster table substitutes (single-linkage with ε
     threshold IS exactly H_0 sublevel sets at filtration value ε).

Output:

  analysis/persistence_summary.md   — per-bench cluster table + verdict
  analysis/<bench>_trajectories.png — trajectories overlay
  analysis/<bench>_clusters.json    — raw cluster assignments

Usage:
  uv run python experiments/E27_basin_persistence/code/analyze_persistence.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Headless plotting for cron / nohup contexts.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir


# E25 per-bench finals on --all (from E25 manifest Outcome table) — the floor
# we're testing whether anything sits below.
E25_PER_BENCH_FLOOR: Dict[str, float] = {
    "ibm11": 0.9136,
    "ibm13": 0.9766,
    "ibm14": 1.2205,
    "ibm15": 1.1797,
}


# ── Loading ─────────────────────────────────────────────────────────────────


def load_trajectories(
    traj_dir: Path, drop_invalid: bool = True
) -> Tuple[Dict[str, List[dict]], Dict[str, int]]:
    """Group `*.json` files by benchmark name.

    Trajectories with ``final_overlap_count > 0`` are dropped by default —
    they're not in the legal feasible region and their final proxy is
    not comparable to valid placements. The number of dropped trajectories
    per benchmark is returned alongside.
    """
    by_bench: Dict[str, List[dict]] = {}
    dropped: Dict[str, int] = {}
    if not traj_dir.exists():
        return by_bench, dropped
    for p in sorted(traj_dir.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        bench = data["benchmark"]
        n_overlap = int(data.get("final_overlap_count", 0))
        if drop_invalid and n_overlap > 0:
            dropped[bench] = dropped.get(bench, 0) + 1
            continue
        by_bench.setdefault(bench, []).append(data)
    return by_bench, dropped


def get_canvas_diag(bench_name: str) -> float:
    """Resolve the canvas diagonal in microns (for normalizing distances)."""
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, _ = load_benchmark_from_dir(str(bench_dir))
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    return math.sqrt(cw ** 2 + ch ** 2)


def get_n_hard(bench_name: str) -> int:
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, _ = load_benchmark_from_dir(str(bench_dir))
    return int(benchmark.num_hard_macros)


# ── Distance + clustering ──────────────────────────────────────────────────


def placement_distance_matrix(
    trajectories: List[dict], n_hard: int
) -> np.ndarray:
    """RMS per-macro distance over hard-macro positions (vectorized).

    Soft macros are excluded — they're not the optimization target and
    they tend to introduce noise in the distance metric.

    Returns the *RMS per-macro* L2 distance, not the total L2. This makes
    the threshold (5% canvas diag) meaningful at the per-macro level
    regardless of how many macros the benchmark has.
    """
    if not trajectories:
        return np.zeros((0, 0))
    positions = []
    for t in trajectories:
        arr = np.asarray(t["final_placement"], dtype=np.float64)
        # Hard macros are the first n_hard rows, shape [n_hard, 2].
        positions.append(arr[:n_hard])
    P = np.stack(positions, axis=0)             # [K, n_hard, 2]
    diff = P[:, None, :, :] - P[None, :, :, :]  # [K, K, n_hard, 2]
    per_macro_sq = np.sum(diff * diff, axis=-1) # [K, K, n_hard]
    rms_per_macro = np.sqrt(np.mean(per_macro_sq, axis=-1))  # [K, K]
    return rms_per_macro


def single_linkage_clusters(D: np.ndarray, threshold: float) -> List[int]:
    """Single-linkage cluster assignments via union-find at threshold ε.

    Returns a list of cluster ids, one per trajectory. Two trajectories
    share a cluster iff there's a chain of pairwise distances all ≤
    threshold connecting them.
    """
    n = D.shape[0]
    if n == 0:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= threshold:
                union(i, j)

    roots = [find(i) for i in range(n)]
    # Compact ids to 0..k-1.
    remap: Dict[int, int] = {}
    for r in roots:
        if r not in remap:
            remap[r] = len(remap)
    return [remap[r] for r in roots]


def cluster_summary(
    trajectories: List[dict], cluster_ids: List[int]
) -> List[dict]:
    """Per-cluster: size, mean & std of final proxy, list of (init, seed)."""
    by_cluster: Dict[int, List[dict]] = {}
    for t, cid in zip(trajectories, cluster_ids):
        by_cluster.setdefault(cid, []).append(t)
    rows: List[dict] = []
    for cid, members in sorted(by_cluster.items(),
                               key=lambda kv: -len(kv[1])):
        finals = np.asarray([m["final_proxy"] for m in members])
        rows.append({
            "cluster_id": int(cid),
            "size": int(finals.size),
            "mean_proxy": float(np.mean(finals)),
            "std_proxy": float(np.std(finals)),
            "min_proxy": float(np.min(finals)),
            "max_proxy": float(np.max(finals)),
            "members": [
                {"init": m["init"], "seed": m["seed"],
                 "final_proxy": float(m["final_proxy"])}
                for m in members
            ],
        })
    return rows


# ── Optional gudhi persistence ─────────────────────────────────────────────


def try_gudhi_persistence(D: np.ndarray) -> List[Tuple[float, float]]:
    """Compute H_0 persistence pairs (birth, death) via Rips on D.

    Returns [] if gudhi is not installed. Each component is born at 0
    and dies when it merges with an older component; we keep the
    finite (birth, death) intervals for H_0.
    """
    try:
        import gudhi
    except ImportError:
        return []
    rips = gudhi.RipsComplex(distance_matrix=D, max_edge_length=float(np.max(D) * 1.1))
    st = rips.create_simplex_tree(max_dimension=1)
    persistence = st.persistence()
    pairs: List[Tuple[float, float]] = []
    for dim, (birth, death) in persistence:
        if dim == 0 and death != float("inf"):
            pairs.append((float(birth), float(death)))
    return pairs


# ── Verdict logic ──────────────────────────────────────────────────────────


def verdict_for_bench(
    clusters: List[dict],
    n_traj: int,
    e25_floor: float,
    near_floor_tol: float = 0.005,
) -> str:
    """Decide single_basin vs multi_basin vs ambiguous for one benchmark.

    - single_basin: largest cluster has ≥ 80 % of trajectories AND its
      mean ≤ e25_floor + near_floor_tol AND no other cluster mean is
      below e25_floor.
    - multi_basin: ≥ 2 clusters each with ≥ 2 members AND each cluster
      mean ≤ e25_floor + near_floor_tol.
    - ambiguous otherwise.
    """
    if not clusters or n_traj == 0:
        return "ambiguous"

    largest = clusters[0]
    largest_frac = largest["size"] / n_traj

    if largest_frac >= 0.80 and largest["mean_proxy"] <= e25_floor + near_floor_tol:
        # Check no other cluster sits below the floor.
        other_below = any(
            c["mean_proxy"] < e25_floor for c in clusters[1:]
        )
        if not other_below:
            return "single_basin"

    near_floor_clusters = [
        c for c in clusters
        if c["size"] >= 2 and c["mean_proxy"] <= e25_floor + near_floor_tol
    ]
    if len(near_floor_clusters) >= 2:
        return "multi_basin"

    return "ambiguous"


# ── Plotting ───────────────────────────────────────────────────────────────


def plot_trajectories(
    trajectories: List[dict],
    cluster_ids: List[int],
    bench_name: str,
    e25_floor: float,
    out_path: Path,
) -> None:
    """Per-bench scatter: every trajectory as a line, colored by cluster."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    n_clusters = max(cluster_ids) + 1 if cluster_ids else 1

    for t, cid in zip(trajectories, cluster_ids):
        sweeps = [pt["sweep"] for pt in t["trajectory"]]
        proxies = [pt["proxy"] for pt in t["trajectory"]]
        color = cmap(cid % 10)
        ax.plot(sweeps, proxies, alpha=0.6, color=color,
                label=f"{t['init']}_{t['seed']}_c{cid}")

    ax.axhline(e25_floor, color="black", linestyle="--", linewidth=1,
               label=f"E25 floor = {e25_floor:.4f}")
    ax.set_xlabel("CD sweep index")
    ax.set_ylabel("proxy")
    ax.set_title(f"{bench_name}: {len(trajectories)} CD trajectories, "
                 f"{n_clusters} cluster(s)")
    # Move legend outside if too many entries.
    if len(trajectories) <= 12:
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Markdown emit ──────────────────────────────────────────────────────────


def emit_summary(
    by_bench: Dict[str, List[dict]],
    bench_results: Dict[str, dict],
    out_md: Path,
) -> None:
    """Write `analysis/persistence_summary.md`."""
    lines: List[str] = []
    lines.append("# E27 — basin persistence summary\n")
    lines.append("THE GATE: do post-E25 plateaus all sit in one big "
                 "basin (kill post-E25 line) or several deep basins "
                 "(justify E28+ multi-day builds)?\n")
    lines.append(
        "Method: K diverse-init CD trajectories per benchmark; "
        "L2 single-linkage clustering of final placements at threshold "
        "= 5 % of canvas diagonal; verdict from cluster sizes + means "
        "vs E25 per-bench floor.\n"
    )

    overall_verdicts = []
    for bench, res in bench_results.items():
        overall_verdicts.append(res["verdict"])
    if overall_verdicts:
        n_single = sum(v == "single_basin" for v in overall_verdicts)
        n_multi = sum(v == "multi_basin" for v in overall_verdicts)
        n_amb = sum(v == "ambiguous" for v in overall_verdicts)
        lines.append(
            f"**Overall verdict across {len(overall_verdicts)} benchmarks:** "
            f"single_basin × {n_single}, multi_basin × {n_multi}, "
            f"ambiguous × {n_amb}.\n"
        )
        if n_single == len(overall_verdicts):
            lines.append("**THE GATE FIRES:** every probed benchmark shows a "
                         "single basin → kill the post-E25 line, ship E25.\n")
        elif n_multi >= 1:
            lines.append("**Post-E25 builds JUSTIFIED:** at least one "
                         "benchmark has multiple deep basins.\n")
        else:
            lines.append("**Verdict ambiguous:** rerun with longer CD budget "
                         "or more diverse inits before deciding.\n")

    for bench in sorted(by_bench.keys()):
        res = bench_results[bench]
        lines.append(f"\n## {bench}\n")
        lines.append(f"- E25 floor: **{res['e25_floor']:.4f}**")
        lines.append(f"- Trajectories: {res['n_trajectories']}")
        lines.append(f"- Canvas diagonal: {res['canvas_diag']:.1f}, "
                     f"clustering threshold: {res['cluster_threshold']:.1f} "
                     f"(= 5 % of diag)")
        lines.append(f"- **Verdict: `{res['verdict']}`**\n")

        clusters = res["clusters"]
        if clusters:
            lines.append("| cluster | size | mean proxy | std | min | max | "
                         "vs E25 floor |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|")
            for c in clusters:
                delta = c["mean_proxy"] - res["e25_floor"]
                marker = ""
                if c["mean_proxy"] <= res["e25_floor"] + 0.001:
                    marker = " (≤ floor)"
                if c["mean_proxy"] < res["e25_floor"] - 0.001:
                    marker = " **(beats floor)**"
                lines.append(
                    f"| {c['cluster_id']} | {c['size']} | "
                    f"{c['mean_proxy']:.4f} | {c['std_proxy']:.4f} | "
                    f"{c['min_proxy']:.4f} | {c['max_proxy']:.4f} | "
                    f"{delta:+.4f}{marker} |"
                )
            lines.append("")
            for c in clusters:
                init_summary = ", ".join(
                    f"{m['init']}/{m['seed']}={m['final_proxy']:.4f}"
                    for m in c["members"]
                )
                lines.append(f"  - cluster {c['cluster_id']}: {init_summary}")

        if res.get("gudhi_pairs"):
            lines.append("\n  H_0 persistence pairs (birth, death):")
            for b, d in res["gudhi_pairs"]:
                lines.append(f"    - ({b:.1f}, {d:.1f})")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Persistence analysis of E27 trajectories."
    )
    parser.add_argument(
        "--traj-dir",
        default=str(_THIS.parent / "trajectories"),
        help="trajectories input directory",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_THIS.parent / "analysis"),
        help="analysis output directory",
    )
    parser.add_argument(
        "--cluster-fraction",
        type=float,
        default=0.05,
        help="cluster threshold as fraction of canvas diagonal",
    )
    args = parser.parse_args()

    traj_dir = Path(args.traj_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    by_bench, dropped = load_trajectories(traj_dir)
    if not by_bench:
        print(f"[E27] no trajectories found in {traj_dir}", flush=True)
        return 1
    if dropped:
        for b, n in sorted(dropped.items()):
            print(f"[E27] {b}: dropped {n} invalid trajectories (final_overlap_count > 0)", flush=True)

    bench_results: Dict[str, dict] = {}
    for bench, trajs in by_bench.items():
        n_hard = get_n_hard(bench)
        canvas_diag = get_canvas_diag(bench)
        thr = args.cluster_fraction * canvas_diag

        D = placement_distance_matrix(trajs, n_hard=n_hard)
        cluster_ids = single_linkage_clusters(D, threshold=thr)
        clusters = cluster_summary(trajs, cluster_ids)
        e25_floor = E25_PER_BENCH_FLOOR.get(bench, float("inf"))
        verdict = verdict_for_bench(clusters, len(trajs), e25_floor)

        gudhi_pairs = try_gudhi_persistence(D)

        bench_results[bench] = {
            "n_trajectories": len(trajs),
            "canvas_diag": canvas_diag,
            "cluster_threshold": thr,
            "e25_floor": e25_floor,
            "clusters": clusters,
            "verdict": verdict,
            "gudhi_pairs": gudhi_pairs,
        }

        plot_trajectories(
            trajs, cluster_ids, bench, e25_floor,
            out_dir / f"{bench}_trajectories.png",
        )

        with open(out_dir / f"{bench}_clusters.json", "w") as f:
            json.dump({
                "benchmark": bench,
                "trajectories": [
                    {"init": t["init"], "seed": t["seed"],
                     "final_proxy": float(t["final_proxy"])}
                    for t in trajs
                ],
                "cluster_ids": cluster_ids,
                "clusters": clusters,
                "verdict": verdict,
                "e25_floor": e25_floor,
                "cluster_threshold": thr,
            }, f, indent=2)

    emit_summary(by_bench, bench_results, out_dir / "persistence_summary.md")
    print(f"[E27] analysis complete → {out_dir}/persistence_summary.md",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
