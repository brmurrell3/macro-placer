"""E11 ablation — SDF + random-projected priors only.

Tests whether one structurally distinct prior (random) on top of the SDF
default is enough to capture most of the diverse-priors gain, or whether
the will/greedy priors materially contribute.
"""

import importlib.util
from pathlib import Path

_p = (Path(__file__).parent / "diverse_priors_placer.py").resolve()
_spec = importlib.util.spec_from_file_location("diverse_priors_placer", str(_p))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class E11SdfRandomPlacer:
    """DPO from SDF + random-projected (2-prior ablation)."""

    def __init__(self):
        self._inner = _mod.DiversePriorsPlacer(
            seed=42, priors=["sdf", "random"]
        )

    def place(self, benchmark):
        return self._inner.place(benchmark)
