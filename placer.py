"""thinkorplace entry placer for partcl eval_docker.

This launcher sits at the repo root so that when `eval_docker/run_eval.sh`
mounts the placer's parent dir at `/submission/`, the WHOLE REPO is
mounted — no extras arg needed for our internal cross-dir imports.

Standard invocation:

  ./eval_docker/run_eval.sh thinkorplace placer.py

Optional DREAMPlace mount (enables the 3rd init lane):

  ./eval_docker/run_eval.sh thinkorplace placer.py submit_deps/dreamplace_install

This launcher adds the repo root to `sys.path` and loads
`submissions/cd_lns_sa_cascade_dp_lane/placer.py` (Option B).
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root():
    """Locate the macro-place-challenge-2026 repo root.

    When the eval_docker mount works as designed, the launcher sits at
    `/submission/placer.py` and the rest of the repo (`macro_place/`,
    `submissions/`, `experiments/`, etc.) is in the same `/submission/`
    directory. So `Path(__file__).parents[0]` is the repo root.

    Fallbacks handle running the launcher locally or with the repo passed
    as a separate extras mount.
    """
    fallback_candidates = [
        Path(__file__).resolve().parent,          # launcher next to repo files
        Path(__file__).resolve().parents[1],      # launcher in repo/<subdir>/
        Path("/submission/repo"),
        Path("/submission/macro-place-challenge-2026"),
    ]
    for c in fallback_candidates:
        if (c / "macro_place").is_dir() and (c / "submissions").is_dir():
            return c
    submission_root = Path("/submission")
    if submission_root.is_dir():
        for child in submission_root.iterdir():
            if child.is_dir() and (child / "macro_place").is_dir() and (child / "submissions").is_dir():
                return child
    raise RuntimeError(
        f"Could not locate repo root. Tried fallbacks "
        f"{[str(c) for c in fallback_candidates]} and a scan of /submission/. "
        f"Make sure you cloned the repo and run from its root: "
        f"`./eval_docker/run_eval.sh thinkorplace placer.py`"
    )


_REPO = _find_repo_root()
sys.path.insert(0, str(_REPO))

# Load the real placer (cd_lns_sa_cascade_dp_lane).
import importlib.util  # noqa: E402

_PLACER_PATH = _REPO / "submissions" / "cd_lns_sa_cascade_dp_lane" / "placer.py"
_spec = importlib.util.spec_from_file_location("_thinkorplace_inner", str(_PLACER_PATH))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class Placer(_mod.CDLNSSACascadeDPLanePlacer):
    """thinkorplace submission: cascade pipeline + (optional) DREAMPlace
    3rd init lane. Mount `submit_deps/dreamplace_install/` to enable the
    DP lane. Without DREAMPlace: 1.0782 IBM / 0.6810 NG45 (2-lane fallback).
    With DREAMPlace: 1.0665 IBM / 0.6809 NG45 (3-lane verified 2026-05-14).
    """
    pass
