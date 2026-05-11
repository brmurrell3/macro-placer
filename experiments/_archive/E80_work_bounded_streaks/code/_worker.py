"""E80 subprocess worker — monkey-patches E25/E41 with streaked phases, then runs.

Usage:
    python _worker.py (e25|e41) <bench_name> <root_str> <out_npy_path>

Patching strategy:
  - Before importing E25/E41 placers, import the streaked function module.
  - Monkey-patch the function attributes on the source modules:
      experiments.E39_kmacro_joint_lns.code.cd_lns_sa_kjoint.{run_lns_gridbin,
        run_sa_polish_v2}
      submissions.cd_lns_sa.placer.{run_lns_gridbin, run_sa_polish_v2}
  - For E41: also force kjoint_budget_s=0 (skip K-joint per Mitigation #4).
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


def _patch_streaked_phases(root: Path) -> None:
    """Monkey-patch run_lns_gridbin and run_sa_polish_v2 in both source modules."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from experiments.E80_work_bounded_streaks.code.streaked_phases import (
        run_lns_gridbin_streaked,
        run_sa_polish_v2_streaked,
    )

    # Patch in cd_lns_sa_kjoint (used by E41).
    import experiments.E39_kmacro_joint_lns.code.cd_lns_sa_kjoint as _kj
    _kj.run_lns_gridbin = run_lns_gridbin_streaked
    _kj.run_sa_polish_v2 = run_sa_polish_v2_streaked

    # Mitigation #4: replace K-joint with a no-op (Hessian saddle escape that
    # follows in the main placer finds deeper minima than K-joint K=3 can).
    def _noop_kjoint(*args, **kwargs):
        log_fn = kwargs.get("log_fn")
        if log_fn is not None:
            log_fn("  K-joint SKIPPED (E80 Mitigation #4: Hessian replaces it)")
        return {
            "ktuples_tried": 0, "ktuples_committed": 0, "passes": 0,
            "total_improvement": 0.0, "wall_total_s": 0.0,
        }

    _kj.run_kjoint_lns = _noop_kjoint

    # E25 has its own run_lns_gridbin / run_sa_polish_v2 defined inline in
    # submissions/cd_lns_sa/placer.py.  Load that module and patch its
    # module-level function attributes — Python looks up `run_lns_gridbin`
    # in the module's globals at call time, so the patch takes effect.
    e25_path = root / "submissions" / "cd_lns_sa" / "placer.py"
    spec = importlib.util.spec_from_file_location("e25_placer_for_patch", str(e25_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e25_placer_for_patch"] = mod
    spec.loader.exec_module(mod)
    mod.run_lns_gridbin = run_lns_gridbin_streaked
    mod.run_sa_polish_v2 = run_sa_polish_v2_streaked
    return mod  # caller can use this same module instance for E25


if __name__ == "__main__":
    mode, bench_name, root_str, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    root = Path(root_str)

    _patch_loader_for_windows(root)
    e25_mod = _patch_streaked_phases(root)

    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.loader import load_benchmark_from_dir
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, _ = load_benchmark_from_dir(str(bench_dir))

    if mode == "e25":
        result = e25_mod.CDLNSSAPlacer().place(benchmark)

    elif mode == "e41":
        from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import CDLNSSADPOKJointPlacer
        # Mitigation #4: skip K-joint (Hessian saddle does the same job better
        # downstream, and dropping K-joint saves ~10 min/bench on hard benches).
        result = CDLNSSADPOKJointPlacer(kjoint_budget_s=0.0).place(benchmark)

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    np.save(out_path, result.detach().cpu().numpy())
    sys.exit(0)
