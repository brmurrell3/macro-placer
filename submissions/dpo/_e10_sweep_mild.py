"""E10 sweep: mild variant (cong_mult=2, n_refine=500). Sweep helper."""

import importlib.util
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark


_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "crp", str(_HERE / "congestion_refine_placer.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CongestionRefinePlacer = _mod.CongestionRefinePlacer


class E10SweepMildPlacer:
    """Mild but longer: n_refine=500, cong_mult=2."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._inner = CongestionRefinePlacer(
            seed=seed, n_refine=500, wl_mult=0.5, dens_mult=0.8,
            cong_mult=2.0, verbose=True,
        )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return self._inner.place(benchmark)
