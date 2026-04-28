"""B=1 sanity wrapper for batched_seeds_placer (E5).

Used solely to compare against B=64 wall clock and confirm the batched code
path matches the serial DPO baseline within seed noise on --fast.
"""

import sys
from pathlib import Path
import importlib.util


_name = "batched_seeds_placer"
_spec = importlib.util.spec_from_file_location(
    _name,
    str(Path(__file__).parent / "batched_seeds_placer.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_name] = _mod  # required for dataclasses inside the module
_spec.loader.exec_module(_mod)


class BatchedSeedsB1Placer(_mod.BatchedSeedsPlacer):
    def __init__(self):
        super().__init__(batch_size=1, seed=42)
