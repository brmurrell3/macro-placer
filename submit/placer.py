"""Launcher placer for partcl eval_docker.

The eval_docker base image mounts the placer's parent dir at /submission/.
Our real placer lives in submissions/cd_lns_sa_cascade_stacked_periphery/
and depends on experiments/, submissions/cd_lns_sa/, etc.

Usage in eval_docker:
  ./eval_docker/run_eval.sh thinkorplace path/to/submit/placer.py path/to/repo

The full repo is mounted as /submission/repo. This launcher adjusts sys.path
to make all our cross-dir imports work.

If /submission/repo isn't present, falls back to repo discovery via the
launcher's own location.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_repo_root():
    """Locate the macro-place-challenge-2026 repo root."""
    candidates = [
        Path("/submission/repo"),
        Path("/submission/macro-place-challenge-2026"),
        Path(__file__).resolve().parents[1],  # if launcher is in repo/submit/
        Path(__file__).resolve().parents[2],
    ]
    for c in candidates:
        if (c / "macro_place").is_dir() and (c / "submissions").is_dir():
            return c
    raise RuntimeError(
        f"Could not locate repo root. Tried: {[str(c) for c in candidates]}"
    )


_REPO = _find_repo_root()
sys.path.insert(0, str(_REPO))

# Now import the real placer (this works because _REPO is on sys.path)
import importlib.util

_PLACER_PATH = _REPO / "submissions" / "cd_lns_sa_cascade_stacked_periphery" / "placer.py"
_spec = importlib.util.spec_from_file_location("_thinkorplace_inner", str(_PLACER_PATH))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class Placer(_mod.CDLNSSACascadeStackedPeripheryPlacer):
    """Thinkorplace champion: cascade saddle (canonical) → portfolio saddle
    (3 non-canonical Hessian weights) → periphery wrap. 1.0575 IBM avg
    on M3 --all 2026-05-17. Zero external dependencies."""
    pass
