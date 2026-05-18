"""LP-dual-guided LNS destroy ranking.

Drop-in replacement for `submissions/cd_lns_sa/placer.py::_cost_aware_destroy`.
The monkey-patch in `submissions/cd_lns_sa_cascade_v2/placer.py` (set by env
`MPC_V2_H1=1`) routes LNS through this implementation. On LP solver failure
or timeout it falls back to the original cost-aware probe transparently —
LNS NEVER breaks because of an LP issue.

Caching policy: the LP is solved once per *sample* (per call), not per
destroy candidate. The cache key is `id(evaluator)` — when LNS rebuilds an
evaluator (rare), the cache is invalidated. The cost of one LP solve is
~0.5–5 s on IBM benches (smaller benches faster); destroying K macros per
sample amortizes that cost.

Variant policy: by default the function mixes LP-dual + cost-aware halves,
which calibration suggested may outperform pure replacement (Spearman 0.32
on ibm01 = signal but not full agreement). Toggleable via env
`MPC_V2_H1_MIX={pure_lp|mix|cost_aware}`.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Local-folder import (mcf_lp lives next to this file).
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from mcf_lp import LPDuals, build_and_solve_subset_mcf, score_macros_by_duals


# Cache: id(evaluator) -> (duals, generation_counter).
# Generation counter tracks how many destroy calls have hit this cache;
# we refresh after a configured number of hits to keep duals current as
# LNS moves the placement.
_DUAL_CACHE: dict = {}
_CACHE_REFRESH_EVERY = int(os.environ.get("MPC_V2_H1_CACHE_REFRESH", "1"))

# Configuration (env-overridable for May 19 hyperparameter sweep).
_TOP_EDGE_FRAC = float(os.environ.get("MPC_V2_H1_EDGE_FRAC", "0.20"))
_TOP_K_NETS = int(os.environ.get("MPC_V2_H1_K_NETS", "2000"))
_LP_WALL_S = float(os.environ.get("MPC_V2_H1_LP_WALL_S", "5.0"))
_MIX_MODE = os.environ.get("MPC_V2_H1_MIX", "mix")  # "pure_lp" | "mix" | "cost_aware"


def _fallback_cost_aware(evaluator, hard_movable: List[int], K: int) -> List[int]:
    """Inline copy of the original cost-aware destroy (move-to-center probe)."""
    cw = evaluator.width / 2.0
    ch = evaluator.height / 2.0
    baseline_p = evaluator.current_cost()["proxy"]
    scores = []
    for idx in hard_movable:
        try:
            p = evaluator.delta_cost(idx, (cw, ch))["proxy"]
            scores.append((idx, p - baseline_p))
        except Exception:
            scores.append((idx, 0.0))
    scores.sort(key=lambda s: s[1])
    return [s[0] for s in scores[:K]]


def lp_dual_destroy(
    evaluator,
    hard_movable: List[int],
    K: int,
) -> List[int]:
    """Pick K hard movables to destroy using LP-dual congestion gradient.

    Signature-compatible with the original `_cost_aware_destroy`.
    """
    eid = id(evaluator)
    cache_entry = _DUAL_CACHE.get(eid)
    refresh = (cache_entry is None) or (cache_entry["gen"] >= _CACHE_REFRESH_EVERY)

    if refresh:
        try:
            duals = build_and_solve_subset_mcf(
                evaluator,
                top_edge_frac=_TOP_EDGE_FRAC,
                top_k_nets=_TOP_K_NETS,
                wall_budget_s=_LP_WALL_S,
            )
        except Exception:
            return _fallback_cost_aware(evaluator, hard_movable, K)
        if duals.status != "ok":
            # Graceful degradation: do NOT poison the cache; future calls
            # may try again on a different placement.
            return _fallback_cost_aware(evaluator, hard_movable, K)
        _DUAL_CACHE[eid] = {"duals": duals, "gen": 0}
    else:
        _DUAL_CACHE[eid]["gen"] += 1
        duals = cache_entry["duals"]

    # Score by LP-dual.
    scored = score_macros_by_duals(evaluator, hard_movable, duals)
    lp_top: List[int] = [m for (m, _s) in scored[:K]]

    if _MIX_MODE == "pure_lp":
        return lp_top

    if _MIX_MODE == "cost_aware":
        return _fallback_cost_aware(evaluator, hard_movable, K)

    # Mix mode: union of top K/2 from each method, padded with extra cost-aware
    # candidates to reach exactly K. Preserves diversity in destroy set.
    lp_pick = lp_top[: max(1, K // 2)]
    cost_pick = _fallback_cost_aware(evaluator, hard_movable, K)
    out: List[int] = list(lp_pick)
    seen = set(out)
    for m in cost_pick:
        if m not in seen:
            out.append(m)
            seen.add(m)
        if len(out) >= K:
            break
    # Pad with remaining LP candidates if still short.
    if len(out) < K:
        for m in lp_top[K // 2:]:
            if m not in seen:
                out.append(m)
                seen.add(m)
            if len(out) >= K:
                break
    return out[:K]
