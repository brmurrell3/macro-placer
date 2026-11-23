"""Compatibility entry for the corrected post-competition v1 runtime.

Original competition code is preserved at Git 2ab92a3; install/mount the
thinkorplace package together with canonical macro_place infrastructure.
"""
from pathlib import Path
import sys

# A checkout adapter also works when the package has not been installed yet.
for _parent in Path(__file__).resolve().parents:
    if (_parent / "thinkorplace" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from thinkorplace.v1.placer import (
    Placer, CDLNSSACascadeStackedPeripheryPlacer, periphery_push_and_polish,
)
