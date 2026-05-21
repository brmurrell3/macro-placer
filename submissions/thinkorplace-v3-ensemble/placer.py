"""thinkorplace-v3-ensemble — 2-lane parallel ensemble (v2-extCD + E138 saddle).

Builds on v2-extCD by running TWO independent lanes in parallel
subprocesses and picking the canonical-better. Lane A is v2-extCD
(V4+Gaussian descent + extended CD polish). Lane B adds Hessian-based
saddle escape on the CD-polished local optimum (E138).

The two lanes converge to STRUCTURALLY DIFFERENT basins:
 - Lane A reaches the CD local minimum from the descent basin.
 - Lane B perturbs that minimum along the softest Hessian mode and
   re-polishes; on benches where the perturbed basin is canonically
   better, we win.

Verified on EPYC ibm17: Lane A 1.18269 vs Lane B 1.17653 (-0.52%).
Offline projection across 17 IBM: 0.98064 vs v2-extCD 0.98387
(-0.33% lift) — see `experiments/E157_adaptive_lane/attrs.json`.

Each lane has its full standalone budget (~25-50 min). Master joins
both, picks canonical-better. Total wall ~ max(lane A, lane B) ~ 50
min/bench on hardest benches; well under 60-min cap.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    """Locate the repo root by searching for `macro_place/` + `experiments/`."""
    candidates = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parents[1],
        Path(__file__).resolve().parents[2] if Path(__file__).resolve().parents and len(Path(__file__).resolve().parents) > 1 else None,
        Path("/submission/repo"),
        Path("/submission/macro-place-challenge-2026"),
    ]
    for c in candidates:
        if c is None:
            continue
        if (c / "macro_place").is_dir() and (c / "experiments").is_dir():
            return c
    # Walk up from /submission looking for repo
    submission_root = Path("/submission")
    if submission_root.is_dir():
        for child in submission_root.iterdir():
            if child.is_dir() and (child / "macro_place").is_dir() and (child / "experiments").is_dir():
                return child
    # Last resort: assume repo is parent of this file's grandparent
    return Path(__file__).resolve().parents[1]


_ROOT = _find_repo_root()
_E155 = _ROOT / "experiments" / "E155_parallel_ensemble" / "code"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_E155) not in sys.path:
    sys.path.insert(0, str(_E155))

# Import the E155 placer module explicitly (not via "from placer import")
# to avoid colliding with this file's own name.
_E155_PLACER = _E155 / "placer.py"
_spec = importlib.util.spec_from_file_location("_e155_placer_mod", str(_E155_PLACER))
_e155_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e155_mod)
_E155ParallelEnsemblePlacer = _e155_mod.E155ParallelEnsemblePlacer


class Placer(_E155ParallelEnsemblePlacer):
    """thinkorplace-v3-ensemble. Aliased from E155ParallelEnsemblePlacer.

    Default budget_seconds = 3300 (55 min) gives each lane its full
    standalone budget while staying under the 60-min cap.
    """
    pass
