"""Phase 1+2 SP-topology diagnostic.

Reads E25 and E41 .pt outputs (from produce_basin_placement.py), encodes both
to sequence-pair, computes the pair-relation Hamming distance and Kendall-tau
distances, decodes each SP via Murata compaction to verify encoder consistency,
and writes results to JSON.

Optional: also runs the same diagnostic on perturbed-init re-runs (Phase 4 mini)
if those placements are available.

Usage:
  uv run python experiments/E69_sequence_pair_search/code/run_diagnostics.py <bench_name>

Example:
  uv run python experiments/E69_sequence_pair_search/code/run_diagnostics.py ibm01
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sp_inverter import (
    encode, murata_decode, pair_relation_distance, kendall_tau,
)
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def _load_placement(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing placement file: {path}")
    return torch.load(path, weights_only=False)


def _verify_no_overlap(placement: torch.Tensor, sizes: torch.Tensor, n_hard: int) -> int:
    """Count overlapping pairs in [0, n_hard)."""
    pos = placement[:n_hard].detach().cpu().numpy().astype(np.float64)
    siz = sizes[:n_hard].detach().cpu().numpy().astype(np.float64)
    hw = siz[:, 0] / 2.0
    hh = siz[:, 1] / 2.0
    n = pos.shape[0]
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(pos[i, 0] - pos[j, 0])
            dy = abs(pos[i, 1] - pos[j, 1])
            if dx < hw[i] + hw[j] and dy < hh[i] + hh[j]:
                count += 1
    return count


def main():
    if len(sys.argv) != 2:
        print("usage: run_diagnostics.py <bench_name>")
        sys.exit(2)
    bench_name = sys.argv[1]

    placements_dir = _HERE.parent / "results" / "placements"
    e25_path = placements_dir / f"e25_{bench_name}.pt"
    e41_path = placements_dir / f"e41_{bench_name}.pt"

    print(f"[E69 diag] loading {bench_name} placements...")
    e25 = _load_placement(e25_path)
    e41 = _load_placement(e41_path)

    sizes = e25["macro_sizes"]
    n_hard = e25["num_hard_macros"]
    print(f"[E69 diag] n_hard={n_hard}, canvas={e25['canvas_width']:.1f}x{e25['canvas_height']:.1f}")
    print(f"[E69 diag] E25 proxy={e25['proxy']:.5f} overlap={e25['overlap_count']} wall={e25['wall_seconds']:.0f}s")
    print(f"[E69 diag] E41 proxy={e41['proxy']:.5f} overlap={e41['overlap_count']} wall={e41['wall_seconds']:.0f}s")

    # Sanity: confirm placements are non-overlapping (E25/E41 outputs must be).
    e25_overlap = _verify_no_overlap(e25["placement"], sizes, n_hard)
    e41_overlap = _verify_no_overlap(e41["placement"], sizes, n_hard)
    assert e25_overlap == 0, f"E25 has {e25_overlap} hard-macro overlaps; cannot encode"
    assert e41_overlap == 0, f"E41 has {e41_overlap} hard-macro overlaps; cannot encode"

    print("[E69 diag] encoding placements to SP...")
    t0 = time.time()
    sp_25 = encode(e25["placement"], sizes, n_hard)
    sp_41 = encode(e41["placement"], sizes, n_hard)
    t_encode = time.time() - t0
    print(f"[E69 diag] encoded both in {t_encode:.2f}s")

    # Stability check: re-encode and confirm idempotent.
    sp_25_re = encode(e25["placement"], sizes, n_hard)
    re_diff, _ = pair_relation_distance(sp_25, sp_25_re)
    assert re_diff == 0, f"encoder is non-deterministic: {re_diff} diffs on re-encode"

    print("[E69 diag] computing topology distance...")
    t0 = time.time()
    pair_diff, breakdown = pair_relation_distance(sp_25, sp_41)
    kt_plus = kendall_tau(sp_25.gamma_plus, sp_41.gamma_plus)
    kt_minus = kendall_tau(sp_25.gamma_minus, sp_41.gamma_minus)
    t_dist = time.time() - t0

    n_pairs = n_hard * (n_hard - 1) // 2
    print(f"[E69 diag] distance computed in {t_dist:.2f}s")
    print(f"[E69 diag] pair-relation diff: {pair_diff} / {n_pairs} pairs ({100*pair_diff/n_pairs:.2f}%)")
    print(f"[E69 diag] breakdown by relation type:")
    for k, v in sorted(breakdown.items(), key=lambda kv: -kv[1])[:10]:
        print(f"             {k}: {v}")
    print(f"[E69 diag] kendall-tau gamma+: {kt_plus} / {n_pairs}")
    print(f"[E69 diag] kendall-tau gamma-: {kt_minus} / {n_pairs}")

    # Decode SP_25 via Murata compaction; check feasibility.
    print("[E69 diag] decoding SP_E25 via Murata compaction...")
    decoded_25 = murata_decode(sp_25, sizes, n_hard)
    decoded_overlap = _verify_no_overlap(decoded_25, sizes, n_hard)
    print(f"[E69 diag] decoded SP_E25: {decoded_overlap} overlaps (expect 0 by construction)")

    # Compute decoded proxy on the actual benchmark (need plc).
    print("[E69 diag] computing decoded proxy on benchmark...")
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    # Build full placement with soft macros at original positions.
    full_decoded = e25["placement"].clone()
    full_decoded[:n_hard] = decoded_25
    # Compact pack may push hard macros against canvas edges; clamp inside.
    # If any hard macro center is outside canvas, project_overlaps will adjust.
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    hw = sizes[:n_hard, 0] / 2.0
    hh = sizes[:n_hard, 1] / 2.0
    full_decoded[:n_hard, 0] = torch.clamp(full_decoded[:n_hard, 0], hw, cw - hw)
    full_decoded[:n_hard, 1] = torch.clamp(full_decoded[:n_hard, 1], hh, ch - hh)

    # If decoded compact placement extends outside canvas (likely on dense
    # benchmarks), the clamp will introduce overlaps. Run project_overlaps to
    # legalize before proxy computation.
    full_decoded, n_proj_iters = project_overlaps(full_decoded, benchmark)
    decoded_metrics = compute_proxy_cost(full_decoded, benchmark, plc)
    print(
        f"[E69 diag] decoded compact proxy: {decoded_metrics['proxy_cost']:.5f} "
        f"(E25 was {e25['proxy']:.5f}; diff {decoded_metrics['proxy_cost'] - e25['proxy']:+.5f})"
    )
    print(
        f"[E69 diag] decoded overlap_count: {decoded_metrics['overlap_count']} after "
        f"project_overlaps ({n_proj_iters} iters)"
    )

    # Same for E41.
    decoded_41 = murata_decode(sp_41, sizes, n_hard)
    full_decoded_41 = e41["placement"].clone()
    full_decoded_41[:n_hard] = decoded_41
    full_decoded_41[:n_hard, 0] = torch.clamp(full_decoded_41[:n_hard, 0], hw, cw - hw)
    full_decoded_41[:n_hard, 1] = torch.clamp(full_decoded_41[:n_hard, 1], hh, ch - hh)
    full_decoded_41, _ = project_overlaps(full_decoded_41, benchmark)
    decoded_41_metrics = compute_proxy_cost(full_decoded_41, benchmark, plc)
    print(
        f"[E69 diag] decoded SP_E41 proxy: {decoded_41_metrics['proxy_cost']:.5f} "
        f"(E41 was {e41['proxy']:.5f}; diff {decoded_41_metrics['proxy_cost'] - e41['proxy']:+.5f})"
    )

    # Save results.
    # Aggregate disagreement classes into axis-preserving (LEFT↔RIGHT,
    # BELOW↔ABOVE) vs axis-rotating (LEFT↔BELOW, etc).
    axis_preserving = (
        breakdown.get("LEFT->RIGHT", 0) + breakdown.get("RIGHT->LEFT", 0)
        + breakdown.get("BELOW->ABOVE", 0) + breakdown.get("ABOVE->BELOW", 0)
    )
    axis_rotating = pair_diff - axis_preserving

    out = {
        "bench_name": bench_name,
        "n_hard": n_hard,
        "n_pairs": n_pairs,
        "e25_proxy": float(e25["proxy"]),
        "e41_proxy": float(e41["proxy"]),
        "e25_overlap": int(e25["overlap_count"]),
        "e41_overlap": int(e41["overlap_count"]),
        "pair_relation_diff": int(pair_diff),
        "pair_relation_diff_frac": float(pair_diff / n_pairs),
        "axis_preserving_diff": int(axis_preserving),
        "axis_rotating_diff": int(axis_rotating),
        "pair_relation_breakdown": {k: int(v) for k, v in breakdown.items()},
        "kendall_tau_plus": int(kt_plus),
        "kendall_tau_minus": int(kt_minus),
        "kt_plus_frac": float(kt_plus / n_pairs),
        "kt_minus_frac": float(kt_minus / n_pairs),
        "decoded_e25_proxy": float(decoded_metrics["proxy_cost"]),
        "decoded_e25_overlap": int(decoded_metrics["overlap_count"]),
        "decoded_e25_proxy_diff": float(decoded_metrics["proxy_cost"] - e25["proxy"]),
        "decoded_e41_proxy": float(decoded_41_metrics["proxy_cost"]),
        "decoded_e41_overlap": int(decoded_41_metrics["overlap_count"]),
        "decoded_e41_proxy_diff": float(decoded_41_metrics["proxy_cost"] - e41["proxy"]),
        "encode_time_s": float(t_encode),
        "distance_time_s": float(t_dist),
        "sp_25_gamma_plus": sp_25.gamma_plus.tolist(),
        "sp_25_gamma_minus": sp_25.gamma_minus.tolist(),
        "sp_41_gamma_plus": sp_41.gamma_plus.tolist(),
        "sp_41_gamma_minus": sp_41.gamma_minus.tolist(),
    }
    out_path = _HERE.parent / "results" / f"diagnostic_{bench_name}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[E69 diag] saved -> {out_path}")

    # Decision summary against kill gates.
    print()
    print("=" * 60)
    print("PHASE 2 KILL-GATE EVALUATION")
    print("=" * 60)
    print(f"Total pair_relation_diff = {pair_diff} ({100 * pair_diff / n_pairs:.2f}%)")
    print(f"  axis-preserving (LEFT↔RIGHT, BELOW↔ABOVE): {axis_preserving}")
    print(f"  axis-rotating   (H↔V):                     {axis_rotating}")
    print()
    if axis_preserving <= 500:
        print(f"PASS (refined): axis-preserving disagreement ≤ 500 — directed-swap "
              f"search on these pairs is tractable in the swap-budget. Phase 3 GO.")
    elif pair_diff > 2000 and axis_preserving > 500:
        print(f"FAIL: even axis-preserving subset ({axis_preserving}) is too large.")
        print(f"      Block-level SP moves needed (E61 V2-style at finer granularity).")
    elif pair_diff > 200:
        print(f"PARTIAL: total ({pair_diff}) >200 but axis-preserving subset ({axis_preserving}) "
              f"is the more promising target for direct-swap search.")
    else:
        print(f"PASS: full SP-SA tractable.")
    print()


if __name__ == "__main__":
    main()
