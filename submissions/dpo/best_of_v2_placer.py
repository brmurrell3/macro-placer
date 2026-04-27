"""
Best-of-Two Placer: SDF init vs DPO v2-steps

Runs both SDFPlacer and DPOv2StepsPlacer, evaluates each with the real proxy
cost, and returns whichever placement scores lower. Combines:
  - v2-steps DPO (step_scale floor of 0.6) for better optimization quality
  - Best-of selection to capture cases where SDF alone beats DPO
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


class BestOfV2Placer:
    """Runs SDF and DPO v2-steps, returns the placement with lower proxy cost."""

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

        # Load DPOv2StepsPlacer
        v2_path = Path(__file__).parent / "ablation_v2_steps.py"
        spec = importlib.util.spec_from_file_location("dpo_v2_steps", str(v2_path))
        v2_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v2_mod)
        DPOv2StepsPlacer = v2_mod.DPOv2StepsPlacer

        # Run SDF
        sdf_placer = SDFPlacer(seed=self.seed)
        sdf_positions = sdf_placer.place(benchmark)

        # Run DPO v2-steps (uses SDF init internally)
        dpo_placer = DPOv2StepsPlacer(seed=self.seed, use_sdf_init=True)
        dpo_positions = dpo_placer.place(benchmark)

        # Load PlacementCost for proxy evaluation
        plc = _load_plc(benchmark.name)
        if plc is None:
            # Can't evaluate -- default to DPO v2
            return dpo_positions

        # Evaluate both
        sdf_cost = compute_proxy_cost(sdf_positions, benchmark, plc)["proxy_cost"]
        dpo_cost = compute_proxy_cost(dpo_positions, benchmark, plc)["proxy_cost"]

        winner = "SDF" if sdf_cost < dpo_cost else "DPO-v2"
        print(f"  >> Best-of-two: SDF={sdf_cost:.4f}  DPO-v2={dpo_cost:.4f}  -> {winner}")

        if sdf_cost < dpo_cost:
            return sdf_positions
        return dpo_positions
