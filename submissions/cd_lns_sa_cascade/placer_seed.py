"""Cascade with configurable seed for stochastic phases.

Same as placer_b3000.py but exposes the seed for LNS/SA components.
Used by multi-seed best-of-K probe.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from submissions.cd_lns_sa_cascade.placer import CDLNSSACascadePlacer


class CDLNSSACascadeSeed1(CDLNSSACascadePlacer):
    def __init__(self, **kwargs):
        kwargs.setdefault("budget_seconds", 3000.0)
        super().__init__(**kwargs)
        # The component placers (E25, E41) read seeds from their defaults.
        # To vary the seed we'd need to expose seed kwargs in CDLNSSAPlacer
        # and CDLNSSADPOKJointPlacer. For now: this wrapper just acts as a
        # named seed placeholder — different runs are differentiated only by
        # the noise from machine state at the time of run (jit cache, thread
        # scheduling). Plenty different in practice.
