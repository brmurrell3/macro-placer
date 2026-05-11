"""E83 subprocess worker — runs E25 or E41 with reduced hard wall budgets.

Usage:
    python _worker.py (e25|e41) <bench_name> <root_str> <out_npy_path> \
        <cd_cap_s> <lns_budget_s> <sa_budget_s> <kjoint_budget_s>

Budgets are explicitly passed so the same worker handles E83's
clock-aware reduced budgets without baking them in.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _patch_loader_for_windows(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import macro_place.loader as _loader_mod
    _orig = _loader_mod.load_benchmark

    def _patched(netlist_file, plc_file=None, name=None):
        netlist_file = str(netlist_file).replace("\\", "/")
        if plc_file is not None:
            plc_file = str(plc_file).replace("\\", "/")
        return _orig(netlist_file, plc_file, name)

    _loader_mod.load_benchmark = _patched


if __name__ == "__main__":
    mode = sys.argv[1]
    bench_name = sys.argv[2]
    root_str = sys.argv[3]
    out_path = sys.argv[4]
    cd_cap_s = float(sys.argv[5])
    lns_budget_s = float(sys.argv[6])
    sa_budget_s = float(sys.argv[7])
    kjoint_budget_s = float(sys.argv[8])

    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    root = Path(root_str)

    _patch_loader_for_windows(root)

    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.loader import load_benchmark_from_dir

    bench_dir = find_benchmark_dir(bench_name)
    benchmark, _ = load_benchmark_from_dir(str(bench_dir))

    if mode == "e25":
        e25_path = root / "submissions" / "cd_lns_sa" / "placer.py"
        spec = importlib.util.spec_from_file_location("e25_placer_for_e83", str(e25_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        placer = mod.CDLNSSAPlacer(
            cd_hard_cap_s=cd_cap_s,
            lns_budget_s=lns_budget_s,
            sa_budget_s=sa_budget_s,
        )
        result = placer.place(benchmark)

    elif mode == "e41":
        from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import (
            CDLNSSADPOKJointPlacer,
        )
        placer = CDLNSSADPOKJointPlacer(
            cd_hard_cap_s=cd_cap_s,
            lns_budget_s=lns_budget_s,
            sa_budget_s=sa_budget_s,
            kjoint_budget_s=kjoint_budget_s,
        )
        result = placer.place(benchmark)

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    np.save(out_path, result.detach().cpu().numpy())
    sys.exit(0)
