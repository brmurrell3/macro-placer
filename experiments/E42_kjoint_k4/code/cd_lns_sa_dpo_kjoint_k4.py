"""E42 — CDLNSSA + DPO best_of_v2 init + K=4 K-macro joint LNS.

Single-line variant of E41: K-joint phase uses K=4 instead of K=3 in the
brute-force enumeration. The inner cartesian product is N^K = 5^4 = 625
combos per K-tuple (vs E41's 125). Per-K-tuple wall scales ~5×; total
K-tuples covered in the 600 s budget drops ~5×, but each K-tuple covers
a larger reachable set.

Tests whether the hard-plateau IBM benches (ibm14, 15, 17 — where E25
ties with E12 under five different mechanisms, and E41 K=3 only partially
escaped) are *4-coupled* multi-basin floors.

Pipeline identical to E41 (DPO init → project_overlaps → CD → LNS →
SA-v2 → K-joint K=4 → validate). All hyperparameters match E41 except
`kjoint_K=4`. K-joint legality / pairwise check uses the post-fix
strict-eps version (committed in E39's module 2026-04-30 04:35).

Reference:
- E41 — K=3 K-joint parent.
  `experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`.
- E39 — K-joint primitive. eps fix 2026-04-30 04:35.
- E18 — DPO init parent.
- E25 — shared CD+LNS+SA-v2 backbone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Repo root on sys.path for cross-experiment imports.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark

# E41's placer accepts kjoint_K as a constructor kwarg. E42 = E41 with K=4.
from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import (
    CDLNSSADPOKJointPlacer,
)


class CDLNSSADPOKJointK4Placer:
    """E42 placer: same as E41 but K-joint phase uses K=4."""

    def __init__(
        self,
        kjoint_K: int = 4,
        kjoint_top_N: int = 5,
        kjoint_budget_s: float = 600.0,
        kjoint_seed: int = 42,
        **kwargs,
    ):
        # Force K=4 even if a caller passed K=3 by accident.
        self._inner = CDLNSSADPOKJointPlacer(
            kjoint_K=int(kjoint_K),
            kjoint_top_N=int(kjoint_top_N),
            kjoint_budget_s=float(kjoint_budget_s),
            kjoint_seed=int(kjoint_seed),
            **kwargs,
        )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return self._inner.place(benchmark)
