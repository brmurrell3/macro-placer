"""Update the root placer.py launcher to point to a specific submission.

Usage:
  uv run python experiments/E110_smooth_global_placer/code/update_launcher.py \
    submissions/cd_lns_sa_cascade_stacked_periphery_e110_ovl10/placer.py \
    CDLNSSACascadeStackedPeripheryE110Ovl10Placer

This script:
  1. Updates root placer.py to import from the given submission path
  2. Updates the Placer class to inherit from the given inner class
  3. Backs up the original launcher to placer.py.bak
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


def update_launcher(submission_path: str, class_name: str) -> None:
    placer_py = _ROOT / "placer.py"
    backup = _ROOT / "placer.py.bak"
    if not backup.exists():
        shutil.copy(placer_py, backup)
        print(f"  backup: placer.py.bak created")

    submission_path = submission_path.lstrip("./").rstrip("/")
    rel = Path(submission_path).relative_to("submissions") if "submissions/" in submission_path else Path(submission_path)
    if str(rel).endswith("/placer.py"):
        sub_dir = str(rel.parent)
    else:
        sub_dir = str(rel)
    if not sub_dir.startswith("submissions/"):
        sub_dir = f"submissions/{sub_dir}"

    parts = sub_dir.split("/")
    full_path = _ROOT / Path(*parts) / "placer.py"
    if not full_path.exists():
        raise FileNotFoundError(f"Submission not found: {full_path}")

    new = f'''"""thinkorplace entry placer for partcl eval_docker.

Auto-updated by update_launcher.py to point at {class_name}.
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

_PLACER_PATH = _REPO / "{sub_dir}" / "placer.py"
_spec = importlib.util.spec_from_file_location("_thinkorplace_inner", str(_PLACER_PATH))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class Placer(_mod.{class_name}):
    """thinkorplace submission, auto-routed."""
    pass
'''
    placer_py.write_text(new)
    print(f"  launcher → {sub_dir}/placer.py (class {class_name})")
    print(f"  test: uv run evaluate placer.py -b ibm01")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission_path", help="e.g. submissions/cd_lns_sa_cascade_stacked_periphery_e110_ovl10")
    ap.add_argument("class_name", help="e.g. CDLNSSACascadeStackedPeripheryE110Ovl10Placer")
    args = ap.parse_args()
    update_launcher(args.submission_path, args.class_name)


if __name__ == "__main__":
    main()
