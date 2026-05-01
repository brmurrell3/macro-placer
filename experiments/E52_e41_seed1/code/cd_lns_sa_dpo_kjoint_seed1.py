"""E52 — E41 with DPO seed=1 (was seed=42). Directly measures DPO seed-noise."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import CDLNSSADPOKJointPlacer


class CDLNSSADPOKJointSeed1Placer:
    """E52: E41 with seed=1 (instead of seed=42). Probes DPO non-determinism."""
    def __init__(self, **kwargs):
        kwargs.setdefault("seed", 1)
        kwargs.setdefault("sa_seed", 1)
        kwargs.setdefault("kjoint_seed", 1)
        kwargs.setdefault("lns_seed", 1)
        self._inner = CDLNSSADPOKJointPlacer(**kwargs)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return self._inner.place(benchmark)
