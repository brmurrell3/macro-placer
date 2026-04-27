"""E11 ablation — SDF prior only.

Should match `best_of_v2_placer.py` fast result (1.1749) since both run a
single SDF-initialized DPO; the only difference is the wrapper. If E11
matches, we know the diverse-priors infrastructure adds no overhead.
"""

import importlib.util
from pathlib import Path

_p = (Path(__file__).parent / "diverse_priors_placer.py").resolve()
_spec = importlib.util.spec_from_file_location("diverse_priors_placer", str(_p))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class E11SdfOnlyPlacer:
    """DPO-from-SDF only (1-prior ablation)."""

    def __init__(self):
        self._inner = _mod.DiversePriorsPlacer(seed=42, priors=["sdf"])

    def place(self, benchmark):
        return self._inner.place(benchmark)
