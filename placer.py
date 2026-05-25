"""thinkorplace entry placer for partcl eval_docker.

Auto-updated by update_launcher.py to point at Placer.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root():
    fallback_candidates = [
        Path(__file__).resolve().parent,
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
            if child.is_dir() and (child / "macro_place").is_dir() and (child / "submissions").is_dir():
                return child
    raise RuntimeError("Could not locate repo root")


_REPO = _find_repo_root()
sys.path.insert(0, str(_REPO))

import importlib.util  # noqa: E402

_PLACER_PATH = _REPO / "submissions/thinkorplace-v2" / "placer.py"
_spec = importlib.util.spec_from_file_location("_thinkorplace_inner", str(_PLACER_PATH))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class Placer(_mod.Placer):
    """thinkorplace submission, auto-routed."""
    pass
