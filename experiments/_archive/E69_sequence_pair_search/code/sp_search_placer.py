"""SP-guided directed-swap placer.

Mechanism: given E25 (SDF basin) and E41 (DPO basin) placements, encode both
to sequence-pair, identify pairs (i, j) where the relation differs between
SPs, and try DIRECTED Cartesian swaps on those pairs only. Each swap is
feasibility-checked via project_overlaps; accepted iff proxy strictly
improves.

This is E15 pair-swap (which was falsified random) restricted to the
SP-disagreement subset — the pairs where E25 and E41 disagree on
left/right/below/above. Hypothesis: targeted basin-transition swaps escape
the multi-mechanism plateau where random swaps don't.

After targeted swaps, optional short CD polish.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sp_inverter import encode

from macro_place.benchmark import Benchmark
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def _spatial_dist(pos: np.ndarray, i: int, j: int) -> float:
    return float(np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]))


def _relation_code(p, m):
    """Encode (p_in_gamma_plus, m_in_gamma_minus) booleans as relation."""
    if p and m:
        return 0  # LEFT
    if (not p) and (not m):
        return 1  # RIGHT
    if p and (not m):
        return 2  # BELOW
    return 3  # ABOVE


def _build_disagree_set(sp_25, sp_41, n_hard, pos_np, axis_preserving_only: bool = False):
    """Return list of (i, j, spatial_dist) tuples where sp_25 and sp_41 disagree.

    If axis_preserving_only: only pairs where both relations are H (LEFT/RIGHT)
    or both are V (BELOW/ABOVE) — i.e., a position swap directly flips the
    topology. Excludes axis-rotating disagreements (LEFT↔BELOW etc.) which
    a simple position swap cannot achieve.
    """
    rp25 = sp_25.rank_plus(); rm25 = sp_25.rank_minus()
    rp41 = sp_41.rank_plus(); rm41 = sp_41.rank_minus()
    disagree = []
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            p25 = rp25[i] < rp25[j]; m25 = rm25[i] < rm25[j]
            p41 = rp41[i] < rp41[j]; m41 = rm41[i] < rm41[j]
            if (p25 != p41) or (m25 != m41):
                if axis_preserving_only:
                    rel25 = _relation_code(p25, m25)
                    rel41 = _relation_code(p41, m41)
                    h_pair = (rel25 < 2 and rel41 < 2)  # both H (LEFT or RIGHT)
                    v_pair = (rel25 >= 2 and rel41 >= 2)  # both V (BELOW or ABOVE)
                    if not (h_pair or v_pair):
                        continue
                disagree.append((i, j, _spatial_dist(pos_np, i, j)))
    return disagree


def sp_guided_swap_search(
    e25_placement: torch.Tensor,
    e41_placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    max_attempts: int = 1000,
    budget_seconds: float = 1800.0,
    log: Optional[Callable[[str], None]] = None,
    seed: int = 42,
    swap_order: str = "spatial_close_first",
    axis_preserving_only: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """SP-guided directed-swap search starting from E25 placement.

    Each swap candidate: take the position swap of (i, j) where SP_25 and
    SP_41 disagree, project_overlaps, accept iff proxy strictly improves.
    By default, restricts to axis-preserving disagreements where a position
    swap directly achieves the topology transition (LEFT↔RIGHT, BELOW↔ABOVE).

    Returns final placement (zero hard-overlaps) and stats.
    """
    if log is None:
        log = lambda s: print(s, flush=True)

    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes

    # Build current placement (we mutate this).
    current = e25_placement.detach().clone()
    pos_np = current[:n_hard].detach().cpu().numpy().astype(np.float64).copy()

    # Encode SP for both basins.
    log("[SP-search] encoding SP for E25 and E41 basins...")
    sp_25 = encode(e25_placement, sizes, n_hard)
    sp_41 = encode(e41_placement, sizes, n_hard)
    disagree = _build_disagree_set(
        sp_25, sp_41, n_hard, pos_np, axis_preserving_only=axis_preserving_only
    )
    n_pairs = n_hard * (n_hard - 1) // 2
    log(f"[SP-search] disagreement set "
        f"({'axis-preserving only' if axis_preserving_only else 'all'}): "
        f"{len(disagree)} / {n_pairs} pairs "
        f"({100 * len(disagree) / n_pairs:.1f}%)")
    if not disagree:
        log("[SP-search] basins agree everywhere — no swap targets")
        return current, {
            "disagree_count": 0, "swaps_tried": 0, "swaps_accepted": 0,
            "starting_proxy": float(compute_proxy_cost(current, benchmark, plc)["proxy_cost"]),
            "final_proxy": None,
        }

    # Order swap attempts.
    rng = np.random.default_rng(seed)
    if swap_order == "spatial_close_first":
        disagree.sort(key=lambda t: t[2])
    elif swap_order == "random_seeded":
        rng.shuffle(disagree)
    elif swap_order == "spatial_far_first":
        disagree.sort(key=lambda t: -t[2])
    else:
        raise ValueError(f"unknown swap_order: {swap_order}")

    # Starting proxy.
    starting_proxy = float(compute_proxy_cost(current, benchmark, plc)["proxy_cost"])
    log(f"[SP-search] starting proxy: {starting_proxy:.5f}")

    current_proxy = starting_proxy
    accepted = 0
    feas_failed = 0
    proxy_failed = 0
    proxy_history = [starting_proxy]
    t0 = time.time()

    for attempt_idx, (i, j, d) in enumerate(disagree):
        if attempt_idx >= max_attempts:
            log(f"[SP-search] reached max_attempts={max_attempts}")
            break
        if (time.time() - t0) > budget_seconds:
            log(f"[SP-search] reached budget {budget_seconds:.0f}s")
            break

        # Try the position swap.
        candidate = current.clone()
        candidate[i], candidate[j] = current[j].clone(), current[i].clone()

        # Legalize via project_overlaps (50-iter cap).
        candidate, n_iters = project_overlaps(candidate, benchmark)
        ovl_metrics = compute_overlap_metrics(candidate, benchmark)

        if ovl_metrics["overlap_count"] > 0:
            feas_failed += 1
            continue

        # Compute proxy of the legalized candidate.
        cand_proxy = float(compute_proxy_cost(candidate, benchmark, plc)["proxy_cost"])

        if cand_proxy < current_proxy - 1e-7:
            # Accept.
            current = candidate
            current_proxy = cand_proxy
            accepted += 1
            proxy_history.append(current_proxy)
            if accepted <= 5 or accepted % 10 == 0:
                log(f"[SP-search] [{attempt_idx + 1}/{len(disagree)}] swap ({i},{j}) "
                    f"d={d:.2f} ACCEPT → proxy {current_proxy:.5f} "
                    f"(Δ={cand_proxy - proxy_history[-2]:.5f}, accepted={accepted})")
        else:
            proxy_failed += 1

    wall = time.time() - t0
    attempts = min(len(disagree), max_attempts, attempt_idx + 1 if disagree else 0)
    log(
        f"[SP-search] done: attempts={attempts} accepted={accepted} "
        f"feas_failed={feas_failed} proxy_failed={proxy_failed} wall={wall:.0f}s"
    )
    log(f"[SP-search] final proxy: {current_proxy:.5f} (vs starting {starting_proxy:.5f}, "
        f"Δ={current_proxy - starting_proxy:+.5f})")

    return current, {
        "disagree_count": len(disagree),
        "swaps_tried": attempts,
        "swaps_accepted": accepted,
        "feas_failed": feas_failed,
        "proxy_failed": proxy_failed,
        "starting_proxy": starting_proxy,
        "final_proxy": current_proxy,
        "improvement": starting_proxy - current_proxy,
        "improvement_frac": (starting_proxy - current_proxy) / starting_proxy if starting_proxy > 0 else 0.0,
        "proxy_history": proxy_history,
        "wall_seconds": wall,
    }


def cd_polish(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    budget_seconds: float = 600.0,
    log: Optional[Callable[[str], None]] = None,
) -> torch.Tensor:
    """Short CD-adaptive polish on a placement; returns mutated placement."""
    if log is None:
        log = lambda s: print(s, flush=True)
    log(f"[cd-polish] starting (budget {budget_seconds:.0f}s)...")
    placement = placement.detach().clone()
    evaluator = IncrementalProxyEvaluator(benchmark, plc, placement)
    n_hard = benchmark.num_hard_macros
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable = [i for i in range(n_hard) if not bool(fixed[i])]
    info = run_cd_adaptive(
        evaluator, benchmark, plc, movable,
        min_time_s=60.0,
        hard_cap_s=budget_seconds,
        patience=3,
        plateau_threshold=0.001,
        log_fn=log,
    )
    log(f"[cd-polish] done: {info.get('exit_reason')} after {info.get('total_moves', '?')} moves")
    return evaluator.placement.detach().clone().to(torch.float32)


class SPGuidedSwapPlacer:
    """Standalone E69 placer used for --fast / --all evaluation if Phase 2 OK.

    Pipeline: load cached E25 + E41, SP-guided swap search, optional CD polish,
    return min-proxy among {E25, E41, swap-result, polished}.
    """

    def __init__(
        self,
        max_swap_attempts: int = 1000,
        swap_budget_seconds: float = 1800.0,
        cd_polish_budget_seconds: float = 600.0,
        seed: int = 42,
    ):
        self.max_swap_attempts = max_swap_attempts
        self.swap_budget_seconds = swap_budget_seconds
        self.cd_polish_budget_seconds = cd_polish_budget_seconds
        self.seed = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = lambda s: print(s, flush=True)
        bench_name = benchmark.name
        log(f"=== SPGuidedSwapPlacer ({bench_name}) ===")

        placements_dir = _HERE.parent / "results" / "placements"
        e25_path = placements_dir / f"e25_{bench_name}.pt"
        e41_path = placements_dir / f"e41_{bench_name}.pt"
        if not e25_path.exists() or not e41_path.exists():
            raise FileNotFoundError(
                f"Need precomputed E25 and E41 placements at {placements_dir}. "
                f"Run produce_basin_placement.py first for {bench_name}."
            )
        e25 = torch.load(e25_path, weights_only=False)
        e41 = torch.load(e41_path, weights_only=False)

        from macro_place.bench_paths import find_benchmark_dir
        from macro_place.loader import load_benchmark_from_dir
        bench_dir = find_benchmark_dir(bench_name)
        _, plc = load_benchmark_from_dir(str(bench_dir))

        log(f"  E25 cached proxy {e25['proxy']:.5f} | E41 cached proxy {e41['proxy']:.5f}")

        # SP-guided directed-swap search starting from E25.
        swapped, stats = sp_guided_swap_search(
            e25["placement"], e41["placement"], benchmark, plc,
            max_attempts=self.max_swap_attempts,
            budget_seconds=self.swap_budget_seconds,
            log=log, seed=self.seed,
        )
        log(f"  SP-search stats: tried={stats['swaps_tried']} accepted={stats['swaps_accepted']} "
            f"final={stats['final_proxy']:.5f}")

        # CD polish.
        polished = cd_polish(
            swapped, benchmark, plc,
            budget_seconds=self.cd_polish_budget_seconds, log=log,
        )
        polished_proxy = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
        log(f"  post-polish proxy: {polished_proxy:.5f}")

        # Return min over {E25, E41, swap, polished}.
        candidates = [
            (float(e25["proxy"]), e25["placement"], "E25"),
            (float(e41["proxy"]), e41["placement"], "E41"),
            (stats.get("final_proxy", 1e9) or 1e9, swapped, "SP-swap"),
            (polished_proxy, polished, "polished"),
        ]
        candidates.sort(key=lambda c: c[0])
        best_proxy, best_placement, best_name = candidates[0]
        log(f"  best of {[c[2] for c in candidates]}: {best_name} at {best_proxy:.5f}")
        return best_placement
