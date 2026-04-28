"""E16 — CDAdaptive with tighter plateau threshold.

Hypothesis: today's analysis showed every hard benchmark exited the current
CDAdaptive (threshold=0.005) with last-3-sweep deltas in the 0.001-0.005 range
— right at the threshold. Tightening to 0.001 lets each benchmark continue
sweeping until the trajectory truly flattens, capped at the contest's 1-hour
legal limit per benchmark (3600s).

This is the cheapest possible score win: zero algorithmic change, just one
hyperparameter swap. If the trajectory deltas observed today are real, the
extra sweeps each contribute ~0.001 of proxy improvement; across the 6 hard
benchmarks that's potentially 0.005-0.01 average drop.

Contest constraint: hard_cap_s stays at 3600 (the 1-hour legal limit per
benchmark). On benchmarks where the new threshold doesn't trigger before the
cap, the cap fires as a safety net.

No per-benchmark tuning — same threshold/cap applies to every benchmark.
"""
from __future__ import annotations

import sys
from pathlib import Path

# This file lives at experiments/E16_tight_threshold/code/, so repo root is 4 levels up.
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from submissions.cd_adaptive.placer import CDAdaptivePlacer


class CDAdaptiveE16Placer(CDAdaptivePlacer):
    """CDAdaptive with plateau_threshold=0.001 (vs 0.005 baseline)."""

    def __init__(self, verbose: bool = True) -> None:
        super().__init__(
            min_time_s=300.0,
            hard_cap_s=3600.0,        # contest legal limit
            patience=3,
            plateau_threshold=0.001,  # tighter than the 0.005 baseline
            init_strategy="sdf",
            verbose=verbose,
        )
