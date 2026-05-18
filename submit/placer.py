"""Launcher placer for partcl eval_docker.

The eval_docker base image mounts the placer's parent dir at /submission/.
Our real placer (`cd_lns_sa_cascade_dp_lane/placer.py`) depends on
`experiments/`, `submissions/cd_lns_sa/`, etc. — paths that aren't in the
placer's own parent dir. This launcher discovers the repo (mounted as an
`eval_docker` extras arg) and prepends it to `sys.path`, then loads the
real placer.

Standard invocation (from a clone of our repo):

  ./eval_docker/run_eval.sh thinkorplace submit/placer.py .

Optional DREAMPlace mount (lets the placer enable its 3rd init lane):

  ./eval_docker/run_eval.sh thinkorplace submit/placer.py . submit_deps/dreamplace_install
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root():
    """Locate the macro-place-challenge-2026 repo root.

    Tries (in order):
      1. Launcher's own ../ (works when launcher is at <repo>/submit/placer.py
         and the WHOLE repo is mounted)
      2. /submission/<basename> auto-scan — handles `run_eval.sh ... .` where
         the repo gets mounted at `/submission/macro-place-challenge-2026/`
      3. /submission/repo (canonical name if user passes a renamed dir)
    """
    fallback_candidates = [
        Path(__file__).resolve().parents[1],
        Path("/submission/repo"),
        Path("/submission/macro-place-challenge-2026"),
    ]
    for c in fallback_candidates:
        if (c / "macro_place").is_dir() and (c / "submissions").is_dir():
            return c
    submission_root = Path("/submission")
    if submission_root.is_dir():
        for child in submission_root.iterdir():
            if (child / "macro_place").is_dir() and (child / "submissions").is_dir():
                return child
    raise RuntimeError(
        f"Could not locate repo root. Tried fallbacks "
        f"{[str(c) for c in fallback_candidates]} and a scan of /submission/. "
        f"Mount the whole repo as an extra arg: "
        f"`./eval_docker/run_eval.sh team submit/placer.py .`"
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
