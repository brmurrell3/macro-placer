"""BestOfV2Placer with seed=43 for multi-seed stability analysis."""
import torch
from pathlib import Path
import importlib.util
from macro_place.benchmark import Benchmark


class BestOfV2Seed43Placer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        spec = importlib.util.spec_from_file_location(
            "bov2", str(Path(__file__).parent / "best_of_v2_placer.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.BestOfV2Placer(seed=43).place(benchmark)
