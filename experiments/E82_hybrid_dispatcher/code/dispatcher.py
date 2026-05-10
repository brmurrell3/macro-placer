"""E82 HybridDispatcherPlacer — per-bench configuration based on bench size.

Dispatches to E79 (full parallel E25⊥E41 + saddle) for small/medium benches
and E81 (CD-only + saddle) for large benches that wouldn't fit the 60-min cap.

Threshold: `num_hard_macros`. Default 350 (calibrated from E79 v2 measurements:
ibm01 with 246 hard macros = 65 min, ibm13 with 575 hard macros = 89 min on
Windows).  The threshold is the dispatcher's ONLY per-bench logic; everything
else is shared between the two paths.

Proxy expectation:
  * small benches: E74-class (proxy from E79 path)
  * large benches: gated on E81 smoke result; floor is E48 plateau (1.10ish)

Reference: experiments/E82_hybrid_dispatcher/manifest.md
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Tuple

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Patch loader for Windows path normalization (avoid modifying loader.py).
import macro_place.loader as _loader_mod
_orig_load_benchmark = _loader_mod.load_benchmark


def _patched_load_benchmark(netlist_file, plc_file=None, name=None):
    netlist_file = str(netlist_file).replace("\\", "/")
    if plc_file is not None:
        plc_file = str(plc_file).replace("\\", "/")
    return _orig_load_benchmark(netlist_file, plc_file, name)


_loader_mod.load_benchmark = _patched_load_benchmark

from macro_place.benchmark import Benchmark

# Load both placer modules (each via spec_from_file_location to avoid
# polluting sys.path with worker-script-only paths).
_E79_PATH = (
    _ROOT / "experiments" / "E79_hardware_portability" / "code"
    / "cd_lns_sa_hessian_fast.py"
)
_E79_SPEC = importlib.util.spec_from_file_location("e79_for_e82", str(_E79_PATH))
_E79_MOD = importlib.util.module_from_spec(_E79_SPEC)
_E79_SPEC.loader.exec_module(_E79_MOD)

# E48 hybrid (verified champion fallback at 1.08151 --all, fits 1-hr cap).
_E48_PATH = _ROOT / "submissions" / "cd_lns_sa_hybrid" / "placer.py"
_E48_SPEC = importlib.util.spec_from_file_location("e48_for_e82", str(_E48_PATH))
_E48_MOD = importlib.util.module_from_spec(_E48_SPEC)
_E48_SPEC.loader.exec_module(_E48_MOD)

CDLNSSAHessianFastPlacer = _E79_MOD.CDLNSSAHessianFastPlacer
# E48 placer class: introspect the module to find it (typically CDLNSSAHybridPlacer).
_e48_classes = [
    getattr(_E48_MOD, n) for n in dir(_E48_MOD)
    if isinstance(getattr(_E48_MOD, n, None), type)
    and getattr(_E48_MOD, n).__module__ == "e48_for_e82"
    and "Placer" in n
]
if not _e48_classes:
    raise ImportError("Could not find a *Placer class in submissions/cd_lns_sa_hybrid/placer.py")
CDLNSSAHybridPlacer = _e48_classes[0]


class HybridDispatcherPlacer:
    """E82 v2 — best-of dispatcher for hardware-portable submission.

    Routes per-bench based on `num_hard_macros`:
      * ≤ threshold → E79 (parallel E25⊥E41 + compressed Hessian saddle).
        Delivers E74-class proxy (-1.4 % vs E48); fits 60-min cap on small
        benches.  Risk: still over cap on borderline-medium benches under
        EPYC slowdown.
      * > threshold → E48 hybrid (verified 1.08151 --all, fits cap, no surprises).

    Why not E81 for large?  Smoke on ibm01 (E81 = 0.911 ≈ E48) showed the
    cheap CD-only plateau forfeits the saddle lift entirely; using E48 is
    no worse and is already verified.

    Default threshold: 350 hard macros (calibrated from E79 v2 measurements
    on Windows: ibm01 with 246 hard = 65 min, ibm13 with 575 hard = 89 min).
    Lower the threshold to be more conservative on EPYC.
    """

    def __init__(
        self,
        hard_macro_threshold: int = 350,
        verbose: bool = True,
    ):
        self.hard_macro_threshold = int(hard_macro_threshold)
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log = (lambda s: print(s, flush=True)) if self.verbose else (lambda s: None)
        n_hard = int(benchmark.num_hard_macros)
        log(f"=== HybridDispatcherPlacer ({benchmark.name}) ===")
        log(f"  num_hard_macros={n_hard}, threshold={self.hard_macro_threshold}")

        if n_hard <= self.hard_macro_threshold:
            log(f"  → small/medium bench: dispatching to E79 (parallel + saddle)")
            placer = CDLNSSAHessianFastPlacer(verbose=self.verbose)
        else:
            log(f"  → large bench: dispatching to E48 hybrid (verified 1.08151)")
            placer = CDLNSSAHybridPlacer(verbose=self.verbose)

        return placer.place(benchmark)
