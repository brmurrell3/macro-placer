"""Compatibility entry for the corrected post-competition v2 runtime.

Original competition code is preserved at Git 2ab92a3. Runtime dependencies
are packaged under thinkorplace rather than added to the canonical namespace.
"""
from pathlib import Path
import sys

# A checkout adapter also works when the package has not been installed yet.
for _parent in Path(__file__).resolve().parents:
    if (_parent / "thinkorplace" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from thinkorplace.v2.placer import Placer
from thinkorplace.v2 import worker as lane_worker
