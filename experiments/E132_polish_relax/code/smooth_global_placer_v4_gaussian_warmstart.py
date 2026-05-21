"""E132 — V4 Gaussian descender with warm-start init mode.

Subclasses SmoothGlobalPlacerV4Gaussian to add an ``init="warmstart"``
branch that uses positions supplied via the ``warmstart_pos`` ctor
keyword. Useful for re-running Adam descent from a CD-polished basin
to attempt group moves CD's single-axis sweep cannot make.

No modification of the shipped V4 / V4Gaussian files.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E115_triton_kernels" / "code",
    _ROOT / "experiments" / "E117_gaussian_density" / "code",
    _ROOT / "experiments" / "E127_v4_gaussian" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.benchmark import Benchmark  # noqa: E402

from smooth_global_placer_v4_gaussian import SmoothGlobalPlacerV4Gaussian  # noqa: E402


class SmoothGlobalPlacerV4GaussianWarmstart(SmoothGlobalPlacerV4Gaussian):
    """V4 Gaussian descender with init="warmstart" support.

    Pass a precomputed ``warmstart_pos`` tensor; when ``init="warmstart"``
    the descent loop will start from those positions (no SDF / DPO init).
    All other behavior identical to the parent.
    """

    def __init__(self, *args, warmstart_pos: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmstart_pos = warmstart_pos

    def _init_positions(self, benchmark: Benchmark, plc=None) -> torch.Tensor:
        if self.init == "warmstart":
            if self.warmstart_pos is None:
                raise ValueError(
                    "init='warmstart' requires warmstart_pos= ctor kwarg"
                )
            pos = self.warmstart_pos.detach().clone().to(torch.float32)
            self._log(
                f"  init=warmstart: using supplied pos shape={tuple(pos.shape)}"
            )
            return pos
        return super()._init_positions(benchmark, plc=plc)
