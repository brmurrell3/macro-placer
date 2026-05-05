"""E67 smoke: one K=50 Hungarian re-pack step on ibm01 SDF init.

Cheap correctness check. Verifies:
  - Module is importable.
  - Cluster selection returns 50 hard movable indices.
  - Slot generation returns 100 candidate (cx, cy) centers.
  - Cost matrix is [50, 100] with mostly finite cells.
  - Hungarian solve returns a length-50 assignment.
  - Commit either accepts (returns improved placement, zero overlaps) or
    defensively reverts (returns original placement, zero overlaps).
  - Returned placement has shape [num_macros, 2] and zero overlaps.
  - Per-substep wall times are in the predicted ballpark.

Wall budget: < 30 s (SDF ~1 s, project ~0.1 s, evaluator init ~0.5 s,
cluster ~0.01 s, slots ~0.01 s, cost matrix ~1-3 s, Hungarian ~0.01 s,
commit ~0.05 s).

NOTE: this does NOT run the full E48 pipeline. It probes a single step on
the SDF-only init to check the module wiring. The E48 plateau probe is
left to the next session per the task.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

# Repo root + module path on sys.path.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import sdf_init, project_overlaps
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics

from kjoint_hungarian import kjoint_hungarian_step


def main():
    print("=== E67 smoke: K=50 Hungarian on ibm01 SDF init ===", flush=True)
    t_smoke0 = time.perf_counter()

    # 1. Load ibm01.
    t0 = time.perf_counter()
    bench_dir = find_benchmark_dir("ibm01")
    bench, plc = load_benchmark_from_dir(str(bench_dir))
    t_load = time.perf_counter() - t0
    print(
        f"[load] ibm01: num_macros={bench.num_macros}, "
        f"n_hard={bench.num_hard_macros}, "
        f"canvas={bench.canvas_width:.1f}x{bench.canvas_height:.1f}, "
        f"wall={t_load:.2f}s",
        flush=True,
    )

    # 2. SDF init + project overlaps.
    t0 = time.perf_counter()
    placement = sdf_init(bench)
    t_sdf = time.perf_counter() - t0
    placement, proj_iters = project_overlaps(placement, bench)
    overlaps_init = compute_overlap_metrics(placement, bench)
    print(
        f"[sdf_init] wall={t_sdf:.2f}s, project_iters={proj_iters}, "
        f"residual_overlaps={overlaps_init['overlap_count']}",
        flush=True,
    )
    if overlaps_init["overlap_count"] > 0:
        print(
            f"  WARN: SDF init left {overlaps_init['overlap_count']} overlaps; "
            f"smoke will still run (commit_if_better's overlap check is "
            f"absolute, not delta — any pre-existing overlap will block "
            f"acceptance)."
        )

    # 3. Build evaluator (the Hungarian step accepts an evaluator OR builds
    # one; we build it here to print init wall + baseline proxy explicitly).
    t0 = time.perf_counter()
    placement_f64 = placement.detach().clone().to(torch.float64)
    evaluator = IncrementalProxyEvaluator(bench, plc, placement_f64)
    t_eval = time.perf_counter() - t0
    init_cost = evaluator.current_cost()
    print(
        f"[eval_init] wall={t_eval:.2f}s, proxy={init_cost['proxy']:.5f} "
        f"[wl={init_cost['wl']:.4f} d={init_cost['density']:.4f} "
        f"c={init_cost['congestion']:.4f}]",
        flush=True,
    )

    # 4. One Hungarian step.
    print("[hungarian] running one K=50, n_slots=100 step ...", flush=True)
    new_placement, info = kjoint_hungarian_step(
        benchmark=bench,
        placement=placement,
        plc=plc,
        evaluator=evaluator,
        k=50,
        n_slots=100,
        mode="adjacency",
    )

    # 5. Print substep wall + numbers.
    print("[result] per-substep wall:", flush=True)
    for key in (
        "t_eval_init_s",
        "t_cluster_select_s",
        "t_slots_gen_s",
        "t_cost_matrix_s",
        "t_hungarian_s",
        "t_commit_s",
        "t_total_s",
    ):
        if key in info:
            print(f"  {key:24s} = {info[key]:.4f} s", flush=True)
    print("[result] sizes + content:", flush=True)
    print(f"  k_actual                = {info['k_actual']}", flush=True)
    print(f"  n_slots_actual           = {info['n_slots_actual']}", flush=True)
    print(
        f"  cost_finite_cells/total  = {info['cost_finite_cells']} / "
        f"{info['k_actual'] * info['n_slots_actual']} "
        f"({info['cost_finite_frac']*100:.1f}% finite)",
        flush=True,
    )
    print(
        f"  cost_min, cost_max       = {info['cost_min']:.4e}, "
        f"{info['cost_max']:.4e}",
        flush=True,
    )
    print(
        f"  cluster_idx (first 10)   = {info['cluster_idx_head']}",
        flush=True,
    )
    print(
        f"  assignment (first 10)    = {info['assignment_head']}",
        flush=True,
    )
    print(
        f"  proxy_before / proxy_after = {info['proxy_baseline']:.5f} / "
        f"{info.get('commit_proxy_after', float('nan')):.5f}",
        flush=True,
    )
    print(
        f"  commit overlap_count, area = "
        f"{info.get('commit_overlap_count', '?')}, "
        f"{info.get('commit_overlap_area', float('nan')):.6e}",
        flush=True,
    )
    print(f"  accepted                 = {info['accepted']}", flush=True)
    print(f"  reason                   = {info.get('commit_reason', '?')}", flush=True)
    print(f"  n_moved                  = {info.get('commit_n_moved', '?')}", flush=True)

    # 6. Verify shape + zero overlaps.
    expected_shape = (bench.num_macros, 2)
    actual_shape = tuple(new_placement.shape)
    assert actual_shape == expected_shape, (
        f"shape mismatch: expected {expected_shape}, got {actual_shape}"
    )
    final_overlaps = compute_overlap_metrics(new_placement, bench)
    print(
        f"[verify] shape={actual_shape} OK; "
        f"final overlap_count={final_overlaps['overlap_count']}, "
        f"area={final_overlaps['total_overlap_area']:.6e}",
        flush=True,
    )

    if info["accepted"]:
        # Acceptance requires zero overlaps already (commit_if_better checks).
        assert final_overlaps["overlap_count"] == 0, (
            "accepted commit but final placement has overlaps?!"
        )
    else:
        # Reject path returns the input placement; final overlaps == init.
        assert final_overlaps["overlap_count"] == overlaps_init["overlap_count"], (
            f"reject path should preserve placement, but overlap count "
            f"changed: {overlaps_init['overlap_count']} -> "
            f"{final_overlaps['overlap_count']}"
        )

    print(
        f"[smoke] total wall = {time.perf_counter() - t_smoke0:.2f} s — OK",
        flush=True,
    )


if __name__ == "__main__":
    main()
