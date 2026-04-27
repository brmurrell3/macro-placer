# STATUS: SUPERSEDED 2026-04-27. Earlier DPO best-of variant (1.4145 --all).
# Replaced by best_of_v2 (1.3834) and ultimately by CDAdaptive (1.1055).
"""
Best-of-Two Placer: SDF init vs DPO

Runs both SDFPlacer and DPOPlacer, evaluates each with the real proxy cost,
and returns whichever placement scores lower. DPO sometimes worsens results
vs SDF on certain benchmarks (ibm01, ibm02, ibm06, ibm12), so this wrapper
captures the best of both.
"""

import torch
from pathlib import Path

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


def _load_plc(name):
    from macro_place.loader import load_benchmark_from_dir
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    return None


class BestOfPlacer:
    """Runs SDF and DPO, returns the placement with lower proxy cost."""

    def __init__(self, seed=42):
        self.seed = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        import importlib.util

        # Load SDFPlacer
        sdf_path = Path(__file__).parent.parent / "polyhedra" / "init" / "sdf.py"
        spec = importlib.util.spec_from_file_location("sdf", str(sdf_path))
        sdf_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sdf_mod)
        SDFPlacer = sdf_mod.SDFPlacer

        # Load DPOPlacer
        dpo_path = Path(__file__).parent / "placer.py"
        spec = importlib.util.spec_from_file_location("dpo_placer", str(dpo_path))
        dpo_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dpo_mod)
        DPOPlacer = dpo_mod.DPOPlacer

        # Run SDF
        sdf_placer = SDFPlacer(seed=self.seed)
        sdf_positions = sdf_placer.place(benchmark)

        # Run DPO (uses SDF init internally)
        dpo_placer = DPOPlacer(seed=self.seed, use_sdf_init=True)
        dpo_positions = dpo_placer.place(benchmark)

        # Load PlacementCost for proxy evaluation
        plc = _load_plc(benchmark.name)
        if plc is None:
            # Can't evaluate -- default to DPO
            return dpo_positions

        # Evaluate both
        sdf_cost = compute_proxy_cost(sdf_positions, benchmark, plc)["proxy_cost"]
        dpo_cost = compute_proxy_cost(dpo_positions, benchmark, plc)["proxy_cost"]

        winner = "SDF" if sdf_cost < dpo_cost else "DPO"
        print(f"  >> Best-of-two: SDF={sdf_cost:.4f}  DPO={dpo_cost:.4f}  -> {winner}")

        if sdf_cost < dpo_cost:
            return sdf_positions
        return dpo_positions
