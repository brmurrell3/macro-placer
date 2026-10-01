#!/usr/bin/env python3
"""Check whether the post-competition input repairs change any IBM input.

Compares the 20 May 2026 single-pipeline extraction code (commit 5485310) with
the corrected runtime on all 17 IBM designs: hard-macro pin ownership, smooth
net tensors, SDF hard-macro edges, and the initial canvas clamp. No optimization
or scoring is run. Writes writeup/data/publication/ibm_repair_equivalence.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from thinkorplace.common import sdf_init as new_sdf  # noqa: E402
from thinkorplace.common import smooth_primitives as new_prim  # noqa: E402
from thinkorplace.common.runtime_safety import clamp_to_canvas  # noqa: E402

MAY20_COMMIT = "5485310535b052193d3b3a908ee03f53d3a0a64b"
MAY20_FILES = {
    "prim": "writeup/archive/submissions/dpo/ablation_v2_steps.py",
    "sdf": "macro_place/sdf_init.py",
}
IBM = ["ibm01", "ibm02", "ibm03", "ibm04"] + ["ibm%02d" % i for i in range(6, 19)]
OUT = ROOT / "writeup/data/publication/ibm_repair_equivalence.json"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def may20_modules(tmp):
    modules, hashes = {}, {}
    for key, rel in MAY20_FILES.items():
        text = subprocess.run(["git", "show", f"{MAY20_COMMIT}:{rel}"], cwd=ROOT,
                              check=True, capture_output=True).stdout
        path = Path(tmp) / f"may20_{key}.py"
        path.write_bytes(text)
        hashes[rel] = hashlib.sha256(text).hexdigest()
        modules[key] = load_module(f"may20_{key}", path)
    return modules, hashes


def main():
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        old, may20_hashes = may20_modules(tmp)
        for name in IBM:
            src = ROOT / "external/MacroPlacement/Testcases/ICCAD04" / name
            bench, plc = load_benchmark_from_dir(str(src))
            n_hard = int(bench.num_hard_macros)
            fixed = bench.macro_fixed.bool()

            owner_mismatch = macro_pins = 0
            for pin in plc.modules_w_pins:
                if pin.get_type() in ("MACRO_PIN", "SOFT_MACRO_PIN"):
                    macro_pins += 1
                    if pin.get_name().split("/")[0] != pin.get_macro_name():
                        owner_mismatch += 1

            a = old["prim"]._extract_net_data(bench, plc)
            b = new_prim._extract_net_data(bench, plc)
            net_equal = all(
                torch.equal(getattr(a, f), getattr(b, f))
                for f in ("pin_macro_idx", "pin_offsets", "mask", "weights")
            ) and a.num_macros == b.num_macros and a.total_net_count == b.total_net_count

            ea, wa = old["sdf"]._extract_edges(bench, plc)
            eb, wb = new_sdf._extract_edges(bench, plc)
            edges_equal = torch.equal(ea, eb) and torch.equal(wa, wb)

            init = bench.macro_positions.clone().to(torch.float32)
            clamped = clamp_to_canvas(init, bench)
            moved = (clamped != init).any(dim=1)
            rows.append(dict(
                benchmark=name,
                netlist_sha256=sha256_file(src / "netlist.pb.txt"),
                hard_macros=n_hard,
                soft_macros=int(bench.num_macros) - n_hard,
                fixed_hard=int(fixed[:n_hard].sum()),
                fixed_soft=int(fixed[n_hard:].sum()),
                macro_pins=macro_pins,
                owner_split_vs_plc_mismatch=owner_mismatch,
                net_data_identical=bool(net_equal),
                sdf_edges=int(ea.shape[0]),
                sdf_edges_identical=bool(edges_equal),
                initial_soft_clusters_moved_by_clamp=int(moved[n_hard:].sum()),
                initial_hard_moved_by_clamp=int(moved[:n_hard].sum()),
            ))
            print(json.dumps(rows[-1]), flush=True)

    record = dict(
        description=__doc__.strip(),
        may20_commit=MAY20_COMMIT,
        may20_source_sha256=may20_hashes,
        current_source_sha256={
            rel: sha256_file(ROOT / rel) for rel in (
                "thinkorplace/common/smooth_primitives.py",
                "thinkorplace/common/sdf_init.py",
                "thinkorplace/common/runtime_safety.py",
            )
        },
        generator_sha256=sha256_file(__file__),
        summary=dict(
            designs=len(rows),
            total_macro_pins=sum(r["macro_pins"] for r in rows),
            owner_mismatches=sum(r["owner_split_vs_plc_mismatch"] for r in rows),
            net_data_identical=sum(r["net_data_identical"] for r in rows),
            sdf_edges_identical=sum(r["sdf_edges_identical"] for r in rows),
            designs_with_fixed_macros=sum(r["fixed_hard"] + r["fixed_soft"] > 0 for r in rows),
            designs_with_clamped_initial_soft=sum(
                r["initial_soft_clusters_moved_by_clamp"] > 0 for r in rows),
        ),
        rows=rows,
    )
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print("wrote", OUT.relative_to(ROOT))
    print(json.dumps(record["summary"]))


if __name__ == "__main__":
    main()
