"""Iterated E69 SP-search only — no crossover.

Runs E69 SP-guided directed-swap multiple rounds, re-encoding SP after each
to find new disagreements. Each round may compound a small lift.

This is a backup if the crossover-based mechanisms (E71/E72) don't work.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_E69 = _ROOT / "experiments" / "E69_sequence_pair_search" / "code"
if str(_E69) not in sys.path:
    sys.path.insert(0, str(_E69))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sp_search_placer import sp_guided_swap_search
from iter_block_pair_placer import cd_polish

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--n-rounds", type=int, default=4)
    ap.add_argument("--round-attempts", type=int, default=300)
    ap.add_argument("--round-budget", type=float, default=600.0)
    ap.add_argument("--polish-per-round", type=float, default=180.0)
    ap.add_argument("--seed-base", type=int, default=42)
    args = ap.parse_args()

    bench_name = args.bench
    e69 = _ROOT / "experiments" / "E69_sequence_pair_search" / "results" / "placements"
    e25 = torch.load(e69 / f"e25_{bench_name}.pt", weights_only=False)
    e41 = torch.load(e69 / f"e41_{bench_name}.pt", weights_only=False)

    print(f"[E72-e69iter] loading {bench_name}...")
    benchmark, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))

    if e25["proxy"] <= e41["proxy"]:
        state = e25["placement"].detach().clone(); state_label = "E25"
        other = e41["placement"]; other_label = "E41"
    else:
        state = e41["placement"].detach().clone(); state_label = "E41"
        other = e25["placement"]; other_label = "E25"

    starting_proxy = float(compute_proxy_cost(state, benchmark, plc)["proxy_cost"])
    print(f"[E72-e69iter] start={state_label} ({starting_proxy:.5f}), target={other_label}")

    best_proxy = starting_proxy
    best_state = state.detach().clone()
    rounds_log = []
    t_total = time.time()

    for r in range(args.n_rounds):
        print(f"\n--- round {r+1}/{args.n_rounds} ---")
        t_round = time.time()
        rotated, sp_stats = sp_guided_swap_search(
            state, other, benchmark, plc,
            max_attempts=args.round_attempts,
            budget_seconds=args.round_budget,
            seed=args.seed_base + r,
            axis_preserving_only=True,
            log=lambda s: print(f"  {s}"),
        )
        post_swap_proxy = float(sp_stats["final_proxy"])
        print(f"  round {r+1} swap: tried={sp_stats['swaps_tried']} accepted={sp_stats['swaps_accepted']} "
              f"proxy={post_swap_proxy:.5f}")

        polished = cd_polish(rotated, benchmark, plc,
                             budget_seconds=args.polish_per_round,
                             log=lambda s: print(f"  {s}"))
        post_polish_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
        post_polish_overlap = compute_overlap_metrics(polished, benchmark)["overlap_count"]
        print(f"  round {r+1} polish: proxy={post_polish_proxy:.5f} overlap={post_polish_overlap}")

        round_wall = time.time() - t_round
        rounds_log.append({
            "round": r,
            "swaps_tried": sp_stats["swaps_tried"],
            "swaps_accepted": sp_stats["swaps_accepted"],
            "post_swap_proxy": post_swap_proxy,
            "post_polish_proxy": post_polish_proxy,
            "post_polish_overlap": post_polish_overlap,
            "wall_seconds": round_wall,
        })

        # Track best_seen across {post_swap, post_polish}, not just post_polish.
        # On some benches, polish regresses the swap state due to evaluator
        # float drift (gotchas.md).
        if post_polish_overlap == 0 and post_polish_proxy < best_proxy - 1e-7:
            best_proxy = post_polish_proxy
            best_state = polished.detach().clone()
            print(f"  round {r+1} NEW BEST (polished): {best_proxy:.5f} "
                  f"(Δ vs start = {best_proxy - starting_proxy:+.5f})")
        rotated_overlap = compute_overlap_metrics(rotated, benchmark)["overlap_count"]
        if rotated_overlap == 0 and post_swap_proxy < best_proxy - 1e-7:
            best_proxy = post_swap_proxy
            best_state = rotated.detach().clone()
            print(f"  round {r+1} NEW BEST (swap, polish regressed): {best_proxy:.5f} "
                  f"(Δ vs start = {best_proxy - starting_proxy:+.5f})")

        # Use BEST_SEEN as next round's start (not polished, which may have
        # regressed). This prevents compounding regression across rounds.
        state = best_state.detach().clone()

    total_wall = time.time() - t_total

    out = {
        "bench_name": bench_name,
        "start_proxy": starting_proxy,
        "best_proxy": best_proxy,
        "lift_vs_start": best_proxy - starting_proxy,
        "lift_frac": (best_proxy - starting_proxy) / starting_proxy,
        "n_rounds": args.n_rounds,
        "rounds_log": rounds_log,
        "wall_seconds": total_wall,
    }
    out_path = _HERE.parent / "results" / f"iter_e69_{bench_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    out_pt = _HERE.parent / "results" / f"iter_e69_{bench_name}.pt"
    torch.save({"placement": best_state.cpu(), "rounds_log": rounds_log,
                "bench_name": bench_name}, out_pt)
    print(f"\n[E72-e69iter] saved -> {out_path}, {out_pt}")
    print(f"  start: {starting_proxy:.5f}")
    print(f"  best:  {best_proxy:.5f}")
    print(f"  lift:  {best_proxy - starting_proxy:+.5f} ({100 * (best_proxy - starting_proxy) / starting_proxy:+.3f}%)")


if __name__ == "__main__":
    main()
