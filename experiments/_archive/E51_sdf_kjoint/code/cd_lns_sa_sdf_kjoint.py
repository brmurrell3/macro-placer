"""E51 placer: SDF init + CD + LNS + SA-v2 + K-joint (DPO REPLACED by SDF).

Basin attribution probe. E41's lift over E18 (-0.46% on --all) and over
E25 (-0.97%) came from the COMPOSITION of "DPO basin" + "K=3 K-joint".
This experiment isolates the K-joint contribution by running the same
post-fix K-joint phase on the SDF basin (instead of DPO basin).

If E51 ≈ E25: K-joint requires DPO basin to extract its lift.
If E51 < E25 by ≥ 0.3 %: K-joint helps SDF basin too — additively
                          composable with E41's DPO basin gain.
If E51 > E25: K-joint regresses SDF basin (DPO basin is load-bearing
              for K-joint's commits to be productive).

Implementation: monkey-patch `_best_of_v2_init` in E41's module to
return SDF init, then run CDLNSSADPOKJointPlacer normally. The patch
is applied per-place() call so other E41-using placers are unaffected.

Reference:
- E41 — DPO + K-joint parent.
- E25 — SDF + CD + LNS + SA-v2 (no K-joint).
- E39 — original "SDF + K-joint" attempt; --all crashed at ibm07
  (overlap-validation bug, fixed 2026-04-30 04:35).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.cd_core import sdf_init

import experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint as _e41_mod
from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import (
    CDLNSSADPOKJointPlacer,
)


def _sdf_only_init(benchmark, plc, seed=42, log_fn=None):
    """Drop-in replacement for `_best_of_v2_init` that returns SDF init."""
    if log_fn is not None:
        log_fn(f"  SDF-only init (E51 override; seed={seed})")
    return sdf_init(benchmark)


class CDLNSSASDFKJointPlacer:
    """E51: same as E41 but uses SDF init instead of DPO best_of_v2."""

    def __init__(self, **kwargs):
        self._original_init = _e41_mod._best_of_v2_init
        self._inner = CDLNSSADPOKJointPlacer(**kwargs)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # Monkey-patch _best_of_v2_init in E41's module namespace.
        original = _e41_mod._best_of_v2_init
        _e41_mod._best_of_v2_init = _sdf_only_init
        try:
            return self._inner.place(benchmark)
        finally:
            _e41_mod._best_of_v2_init = original
