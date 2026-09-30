#!/usr/bin/env python3
"""Compare the current and proposed SDF hard-macro graph extractors only."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from macro_place.loader import load_benchmark_from_dir
from thinkorplace.common.bench_paths import find_benchmark_dir
from thinkorplace.common import sdf_init as namespaced_sdf
from macro_place import sdf_init as local_sdf


OLD_BLOCK = '''        for pin in [driver] + sinks:
            parent = pin.split("/")[0]
            if parent in name_to_bidx:
                macros.add(name_to_bidx[parent])'''

FIXED_SETUP = '''    pin_to_bidx = {}
    for pin_idx in plc.hard_macro_pin_indices:
        pin = plc.modules_w_pins[pin_idx]
        parent_bidx = name_to_bidx.get(pin.get_macro_name())
        if parent_bidx is not None:
            pin_to_bidx[pin.get_name()] = parent_bidx'''

FIXED_BLOCK = '''        for pin_name in [driver] + sinks:
            bidx = pin_to_bidx.get(pin_name)
            if bidx is not None:
                macros.add(bidx)'''


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def proposed_source(before_source: str) -> str:
    if before_source.count(OLD_BLOCK) != 1:
        raise RuntimeError("Current SDF extractor no longer matches the frozen control")
    setup_anchor = "    edge_dict = {}"
    if before_source.count(setup_anchor) != 1:
        raise RuntimeError("Cannot locate SDF pin-map insertion point")
    with_setup = before_source.replace(setup_anchor, FIXED_SETUP + "\n" + setup_anchor)
    return with_setup.replace(OLD_BLOCK, FIXED_BLOCK)


def compile_extractor(source: str, module) -> Callable:
    namespace = dict(module.__dict__)
    exec(compile(source, "<proposed-sdf-owner-extraction>", "exec"), namespace)
    return namespace["_extract_edges"]


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(header + b"\0" + array.tobytes(order="C"))


def describe(extractor: Callable, benchmark, plc) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
    edges, weights = extractor(benchmark, plc)
    order = torch.argsort(edges[:, 0] * max(1, benchmark.num_hard_macros) + edges[:, 1]) if len(edges) else None
    if order is not None:
        edges = edges[order]
        weights = weights[order]
    record = {
        "unique_edge_count": int(edges.shape[0]),
        "edge_weight_sum_float32": float(weights.sum()) if len(weights) else 0.0,
        "edge_weight_sum_float64": float(weights.to(torch.float64).sum()) if len(weights) else 0.0,
        "edges_sha256": tensor_sha256(edges),
        "weights_sha256": tensor_sha256(weights),
    }
    return record, edges, weights


def hard_pin_counts(benchmark, plc) -> Dict[str, int]:
    hard_ids = set(benchmark.hard_macro_indices)

    def is_hard_pin(node) -> bool:
        return (
            node.get_type() == "MACRO_PIN"
            and plc.mod_name_to_indices.get(node.get_macro_name()) in hard_ids
        )

    raw = [plc.modules_w_pins[index] for index in plc.hard_macro_pin_indices]
    used = [
        plc.modules_w_pins[plc.mod_name_to_indices[pin_name]]
        for driver, sinks in plc.nets.items()
        for pin_name in [driver] + sinks
        if pin_name in plc.mod_name_to_indices
        and is_hard_pin(plc.modules_w_pins[plc.mod_name_to_indices[pin_name]])
    ]
    return {
        "raw_hard_pin_objects": len(raw),
        "hard_pin_occurrences_in_net_traversal": len(used),
        "unique_used_hard_pin_names": len({node.get_name() for node in used}),
    }


def verify_record(output: Path) -> None:
    record = json.loads(output.read_text(encoding="utf-8"))
    if record.get("classification") != "controlled_sdf_topology_extraction_only":
        raise RuntimeError("Unexpected SDF topology evidence classification")

    before_source = record["before_function_source"]
    fixed_source = record["fixed_function_source"]
    for key, source in (("before_function_sha256", before_source),
                        ("fixed_function_sha256", fixed_source)):
        actual = sha256_bytes(source.encode("utf-8"))
        if actual != record[key]:
            raise RuntimeError(f"Recorded {key} does not match its embedded function source")

    raw_script = ROOT / "writeup/data/publication/raw/measure_publication_sdf_topology.py"
    if sha256_file(raw_script) != record["diagnostic_script_sha256"]:
        raise RuntimeError("Archived evidence-producing script does not match the record")

    expected_fixed_hash = record["fixed_function_sha256"]
    active_function_hashes = {}
    for path, module in (
        ("thinkorplace/common/sdf_init.py", namespaced_sdf),
        ("macro_place/sdf_init.py", local_sdf),
    ):
        active_source = inspect.getsource(module._extract_edges)
        active_hash = sha256_bytes(active_source.encode("utf-8"))
        active_function_hashes[path] = active_hash
        if active_hash != expected_fixed_hash:
            raise RuntimeError(f"Active fixed extractor does not match recorded proposal: {path}")

    before_extractor = compile_extractor(before_source, namespaced_sdf)
    fixed_extractor = compile_extractor(fixed_source, namespaced_sdf)
    replayed = []
    for expected in record["benchmarks"]:
        input_paths = {relative: ROOT / relative for relative in expected["inputs"]}
        for relative, path in input_paths.items():
            actual_hash = sha256_file(path)
            if actual_hash != expected["inputs"][relative]:
                raise RuntimeError(f"Benchmark input hash changed: {relative}")
        netlist_paths = [path for path in input_paths.values() if path.name == "netlist.pb.txt"]
        if len(netlist_paths) != 1:
            raise RuntimeError(f"Expected one netlist path for {expected['benchmark']}")
        directory = netlist_paths[0].parent
        benchmark, plc = load_benchmark_from_dir(str(directory))

        actual_before, before_edges, before_weights = describe(
            before_extractor, benchmark, plc
        )
        actual_fixed, fixed_edges, fixed_weights = describe(
            fixed_extractor, benchmark, plc
        )
        for label, actual in (
            ("before_split_owner", actual_before),
            ("fixed_plc_owner", actual_fixed),
        ):
            if actual != expected[label]:
                raise RuntimeError(
                    f"Replayed {expected['benchmark']} {label} differs from the record: "
                    f"expected={expected[label]!r}, actual={actual!r}"
                )

        expected_metadata = {
            "hard_macros": int(benchmark.num_hard_macros),
            "plc_nets": len(plc.nets),
            **hard_pin_counts(benchmark, plc),
        }
        for key, actual in expected_metadata.items():
            if actual != expected[key]:
                raise RuntimeError(
                    f"Replayed {expected['benchmark']} metadata differs for {key}: "
                    f"expected={expected[key]!r}, actual={actual!r}"
                )
        exact_match = bool(
            torch.equal(before_edges, fixed_edges)
            and torch.equal(before_weights, fixed_weights)
        )
        if exact_match != expected["fixed_matches_before_exactly"]:
            raise RuntimeError(
                f"Replayed exact-match status differs for {expected['benchmark']}"
            )
        replayed.append(
            {
                "benchmark": expected["benchmark"],
                "before_split_owner": actual_before,
                "fixed_plc_owner": actual_fixed,
                "fixed_matches_before_exactly": exact_match,
            }
        )

    print(json.dumps({
        "verified": True,
        "record": str(output.relative_to(ROOT)),
        "record_sha256": sha256_file(output),
        "archived_script": str(raw_script.relative_to(ROOT)),
        "archived_script_sha256": sha256_file(raw_script),
        "active_fixed_function_sha256": active_function_hashes,
        "benchmarks": replayed,
    }, indent=2))


def generate_record(output: Path) -> None:
    if output.exists():
        raise SystemExit("Evidence exists; preserve it and choose a new path for a changed study.")

    source_paths = {
        "thinkorplace/common/sdf_init.py": ROOT / "thinkorplace/common/sdf_init.py",
        "macro_place/sdf_init.py": ROOT / "macro_place/sdf_init.py",
    }
    before_sources = {
        "thinkorplace/common/sdf_init.py": inspect.getsource(namespaced_sdf._extract_edges),
        "macro_place/sdf_init.py": inspect.getsource(local_sdf._extract_edges),
    }
    if len(set(before_sources.values())) != 1:
        raise RuntimeError("The two SDF extractor copies have diverged")
    before_source = next(iter(before_sources.values()))
    fixed_source = proposed_source(before_source)
    fixed_extractor = compile_extractor(fixed_source, namespaced_sdf)

    records = []
    for name in ("ibm01", "ariane133"):
        directory = find_benchmark_dir(name)
        benchmark, plc = load_benchmark_from_dir(str(directory))
        before, before_edges, before_weights = describe(
            namespaced_sdf._extract_edges, benchmark, plc
        )
        fixed, fixed_edges, fixed_weights = describe(fixed_extractor, benchmark, plc)
        input_paths = [directory / "netlist.pb.txt", directory / "initial.plc"]
        records.append(
            {
                "benchmark": name,
                "hard_macros": int(benchmark.num_hard_macros),
                "plc_nets": len(plc.nets),
                **hard_pin_counts(benchmark, plc),
                "before_split_owner": before,
                "fixed_plc_owner": fixed,
                "fixed_matches_before_exactly": bool(
                    torch.equal(before_edges, fixed_edges)
                    and torch.equal(before_weights, fixed_weights)
                ),
                "inputs": {
                    str(path.relative_to(ROOT)): sha256_file(path) for path in input_paths
                },
            }
        )

    ibm = next(record for record in records if record["benchmark"] == "ibm01")
    ariane = next(record for record in records if record["benchmark"] == "ariane133")
    if not ibm["fixed_matches_before_exactly"]:
        raise RuntimeError("Proposed extraction changed the IBM01 graph")
    if ariane["before_split_owner"]["unique_edge_count"] != 0:
        raise RuntimeError("Expected the pre-fix ariane133 SDF graph to be empty")
    if ariane["fixed_plc_owner"]["unique_edge_count"] == 0:
        raise RuntimeError("Proposed extraction did not restore ariane133 edges")

    report = {
        "schema_version": 1,
        "classification": "controlled_sdf_topology_extraction_only",
        "control": (
            "The before and proposed functions differ only in resolving hard-macro pin "
            "ownership through the PlacementCost API instead of splitting pin names."
        ),
        "limits": "No placement optimization, scoring, transfer, or downstream PnR claim.",
        "diagnostic_script_sha256": sha256_file(Path(__file__)),
        "runtime_sources": {
            path: {
                "file_sha256": sha256_file(source_paths[path]),
                "before_function_sha256": sha256_bytes(before_sources[path].encode("utf-8")),
            }
            for path in source_paths
        },
        "before_function_source": before_source,
        "before_function_sha256": sha256_bytes(before_source.encode("utf-8")),
        "fixed_function_source": fixed_source,
        "fixed_function_sha256": sha256_bytes(fixed_source.encode("utf-8")),
        "benchmarks": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-record",
        action="store_true",
        help="Replay and verify the immutable SDF topology evidence without writing it.",
    )
    args = parser.parse_args()
    output = ROOT / "writeup/data/publication/sdf_topology_diagnostic.json"
    if args.verify_record:
        verify_record(output)
    else:
        generate_record(output)


if __name__ == "__main__":
    main()
