"""E43 — CDLNSSA + DPO + K=3 K-joint with 1200 s budget (was 600 s in E41).

Tests if E41 K-joint is budget-bound or saturation-bound. Single-line
variant of E41: kjoint_budget_s=1200 instead of 600. K-tuple selection,
K=3, top_N=5, seed=42 unchanged.

Companion to E26 (longer SA budget — falsified; SA was saturation-bound
at 600 s). The expected outcome is saturation on easy benches and
modest lift on hard benches (multi-coupled plateaus where K=3 didn't
finish enumeration in 600 s).

Reference:
- E41 — K=3 K-joint at 600 s budget.
- E26 — analogous probe for SA-v2 (falsified).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import (
    CDLNSSADPOKJointPlacer,
)


class CDLNSSADPOKJointLongerPlacer:
    """E43 placer: same as E41 but K-joint budget 1200 s (was 600 s)."""

    def __init__(
        self,
        kjoint_budget_s: float = 1200.0,
        kjoint_K: int = 3,
        kjoint_top_N: int = 5,
        kjoint_seed: int = 42,
        **kwargs,
    ):
        self._inner = CDLNSSADPOKJointPlacer(
            kjoint_K=int(kjoint_K),
            kjoint_top_N=int(kjoint_top_N),
            kjoint_budget_s=float(kjoint_budget_s),
            kjoint_seed=int(kjoint_seed),
            **kwargs,
        )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return self._inner.place(benchmark)
