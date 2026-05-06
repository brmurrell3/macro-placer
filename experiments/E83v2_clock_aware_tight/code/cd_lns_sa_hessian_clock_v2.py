"""E83 v2 — tighter wall budgets, otherwise identical to E83 v1.

E83 v1 measured walls of 49-60 min on Windows --jobs 4; 5/17 within 1 min
of the 60-min cap. EPYC slowdown (1.2-1.5×) would push those over the cap.

v2 tightens budgets to leave a real safety margin:
  Phase 1+2 cap:  1500 → 1200 s (CD), 360 → 240 s (LNS), 360 → 240 s (SA)
                  per-lane sum 28 min (was 37 min)
  Hessian full:   3 ε values, 180 s polish → 2 ε values, 120 s polish
                  ~8 min (was 18 min)
  Total max wall: ~36 min on M3-equivalent (vs ~55 min for v1)

Trade: ~1-2 % proxy regression on hard benches in exchange for guaranteed
cap fit on EPYC.

Reference: experiments/E83v2_clock_aware_tight/manifest.md
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Patch loader for Windows path normalization (consistent with E83 v1).
import macro_place.loader as _loader_mod
_orig_load_benchmark = _loader_mod.load_benchmark


def _patched_load_benchmark(netlist_file, plc_file=None, name=None):
    netlist_file = str(netlist_file).replace("\\", "/")
    if plc_file is not None:
        plc_file = str(plc_file).replace("\\", "/")
    return _orig_load_benchmark(netlist_file, plc_file, name)


_loader_mod.load_benchmark = _patched_load_benchmark

# Load E83 v1 placer (we subclass it and just override the budgets).
_E83_PATH = (
    _REPO_ROOT / "experiments" / "E83_clock_aware" / "code"
    / "cd_lns_sa_hessian_clock.py"
)
_E83_SPEC = importlib.util.spec_from_file_location("e83_for_v2", str(_E83_PATH))
_E83_MOD = importlib.util.module_from_spec(_E83_SPEC)
_E83_SPEC.loader.exec_module(_E83_MOD)

CDLNSSAHessianClockPlacer = _E83_MOD.CDLNSSAHessianClockPlacer

# E83 v2 worker is the v1 worker (same arg signature; budgets passed at runtime).
# But we need to point the v1 placer at the right worker — its module-level
# constant `_WORKER_SCRIPT` resolves to E83 v1's _worker.py, which is exactly
# what we want (same code path; only the call-site budgets change).


class CDLNSSAHessianClockV2Placer(CDLNSSAHessianClockPlacer):
    """E83 v2 — same algorithm as v1 with tighter wall budgets.

    Hard wall cap unchanged (55 min); per-phase budgets squeezed to leave
    real safety margin under EPYC slowdown (1.2-1.5×):

      M3-equivalent total wall worst case: 28 (phase 1+2) + 8 (Hessian) = 36 min
      EPYC total wall worst case:           ~54 min  ✓ within 60-min cap
    """

    # Phase 1+2 (per-lane).  Sum ≤ 1680 s = 28 min.
    CD_CAP_S: float = 1200.0     # 20 min (vs v1 25 min)
    LNS_BUDGET_S: float = 240.0  #  4 min (vs v1 6 min)
    SA_BUDGET_S: float = 240.0   #  4 min (vs v1 6 min)
    # K-joint stays disabled (Mitigation #4).
    KJOINT_BUDGET_S: float = 0.0

    # Phase 3 (Hessian) — keep same thresholds, but full mode is cheaper.
    HESSIAN_FULL_THRESHOLD_S: float = 12 * 60  # 12 min remaining → full
    HESSIAN_MIN_THRESHOLD_S: float = 4 * 60    # 4 min remaining → minimal
    HESSIAN_FULL_POLISH_S: float = 120.0
    HESSIAN_FULL_EPS: Tuple[float, ...] = (0.3, 1.0)  # 2 ε values (was 3)
    HESSIAN_MIN_POLISH_S: float = 60.0
    HESSIAN_MIN_EPS: Tuple[float, ...] = (1.0,)
