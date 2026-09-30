"""Compatibility import; new subprocesses use thinkorplace.v2.worker."""
from pathlib import Path
import sys

# A checkout adapter also works when the package has not been installed yet.
for _parent in Path(__file__).resolve().parents:
    if (_parent / "thinkorplace" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from thinkorplace.v2.worker import lane_entry
