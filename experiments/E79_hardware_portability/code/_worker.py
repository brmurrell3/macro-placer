"""E79 subprocess worker — runs E25 or E41 and saves result as .npy.

Usage:
    python _worker.py (e25|e41) <bench_name> <root_str> <out_npy_path>

The worker normalizes Windows backslash paths to forward slashes before
calling load_benchmark directly — avoids modifying macro_place/loader.py
while still working around the upstream plc_client_os.py rsplit('/', -1) bug.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np


def _patch_loader_for_windows(root: Path) -> None:
    """Monkey-patch macro_place.loader.load_benchmark to normalize Windows paths.

    Upstream plc_client_os.py uses rsplit('/', -1)[-2] and crashes on backslash
    paths. We patch at the loader entry point so all downstream call chains
    (sdf_init, load_benchmark_from_dir, etc.) receive posix-style paths
    without modifying macro_place/loader.py.
    """
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


def _load_benchmark_posix(bench_name: str, root: Path):
    """Load a benchmark by name (loader is already patched)."""
    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.loader import load_benchmark_from_dir

    bench_dir = find_benchmark_dir(bench_name)
    return load_benchmark_from_dir(str(bench_dir))


if __name__ == "__main__":
    mode, bench_name, root_str, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    root = Path(root_str)

    _patch_loader_for_windows(root)
    benchmark, _ = _load_benchmark_posix(bench_name, root)

    if mode == "e25":
        e25_path = root / "submissions" / "cd_lns_sa" / "placer.py"
        spec = importlib.util.spec_from_file_location("e25_placer_worker", str(e25_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        result = mod.CDLNSSAPlacer().place(benchmark)

    elif mode == "e41":
        from experiments.E41_dpo_kjoint.code.cd_lns_sa_dpo_kjoint import CDLNSSADPOKJointPlacer
        result = CDLNSSADPOKJointPlacer().place(benchmark)

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    np.save(out_path, result.detach().cpu().numpy())
    sys.exit(0)
