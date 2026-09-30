#!/usr/bin/env python3
"""Cross-check frozen E74/E84 geometry against canonical validation utilities."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, value: object) -> None:
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rescore",
        type=Path,
        default=ROOT / "writeup/data/publication/e74_e84_canonical_rescore_20260907.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "writeup/data/publication/e74_e84_geometry_crosscheck_20260907.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new path to preserve evidence")

    import torch
    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.utils import validate_placement

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rescore = json.loads(args.rescore.read_text())
    rows = []
    for benchmark_record in rescore["benchmarks"]:
        name = benchmark_record["name"]
        benchmark, _ = load_benchmark_from_dir(find_benchmark_dir(name))
        for placement_record in benchmark_record["placements"]:
            stage = placement_record["stage"]
            source = ROOT / placement_record["source_artifacts"]["tensor_path"]
            placement = torch.load(source, map_location="cpu", weights_only=False)["placement"]
            valid, violations = validate_placement(placement, benchmark, check_overlaps=True)
            half = benchmark.macro_sizes.to(placement.dtype) / 2
            excess = torch.stack((
                torch.clamp(-(placement[:, 0] - half[:, 0]), min=0),
                torch.clamp(placement[:, 0] + half[:, 0] - benchmark.canvas_width, min=0),
                torch.clamp(-(placement[:, 1] - half[:, 1]), min=0),
                torch.clamp(placement[:, 1] + half[:, 1] - benchmark.canvas_height, min=0),
            ), dim=1)
            offenders = []
            for index in (excess > 0).any(dim=1).nonzero(as_tuple=True)[0].tolist():
                components = {
                    label: float(excess[index, axis])
                    for axis, label in enumerate(("x_below", "x_above", "y_below", "y_above"))
                    if excess[index, axis] > 0
                }
                offenders.append({
                    "row": index,
                    "plc_module_index": int(
                        (list(benchmark.hard_macro_indices) + list(benchmark.soft_macro_indices))[index]
                    ),
                    "macro_name": benchmark.macro_names[index],
                    "kind": "hard" if index < benchmark.num_hard_macros else "soft",
                    "fixed": bool(benchmark.macro_fixed[index]),
                    "center": [float(value) for value in placement[index]],
                    "size": [float(value) for value in benchmark.macro_sizes[index]],
                    "boundary_excess": components,
                })
            recorded_full_legal = placement_record["geometry_validation"]["full_legal"]
            if bool(valid) != bool(recorded_full_legal):
                raise RuntimeError("Validation disagreement for %s %s" % (name, stage))
            sidecar_overlap = placement_record["recorded"]["sidecar_overlap_count"]
            rows.append({
                "benchmark": name,
                "stage": stage,
                "tensor_path": str(source.relative_to(ROOT)),
                "tensor_sha256": digest(source),
                "canonical_validate_placement": {
                    "full_legal": bool(valid),
                    "violations": violations,
                },
                "agrees_with_rescore_geometry": True,
                "out_of_bounds_macros": offenders,
                "out_of_bounds_hard_count": sum(row["kind"] == "hard" for row in offenders),
                "out_of_bounds_soft_count": sum(row["kind"] == "soft" for row in offenders),
                "historical_sidecar_overlap_count": sidecar_overlap,
                "metadata_scope_mismatch": (
                    "sidecar reports zero hard-macro overlaps but does not encode bounds legality"
                    if sidecar_overlap == 0 and not valid else None
                ),
            })
    failures = [row for row in rows if not row["canonical_validate_placement"]["full_legal"]]
    record = {
        "schema_version": 1,
        "kind": "canonical_validate_placement_crosscheck_and_boundary_offender_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": digest(Path(__file__))},
        "source_rescore": {"path": str(args.rescore.relative_to(ROOT)), "sha256": digest(args.rescore)},
        "canonical_validation_source": {
            "path": "macro_place/utils.py",
            "sha256": digest(ROOT / "macro_place/utils.py"),
        },
        "summary": {
            "placements_checked": len(rows),
            "canonical_agreement_count": sum(row["agrees_with_rescore_geometry"] for row in rows),
            "fully_legal_count": len(rows) - len(failures),
            "strict_bounds_failure_count": len(failures),
            "failure_ids": [row["benchmark"] + "_" + row["stage"] for row in failures],
            "all_failures_have_zero_recorded_overlap": all(
                row["historical_sidecar_overlap_count"] == 0 for row in failures
            ),
            "maximum_boundary_excess": max(
                (value for row in failures for macro in row["out_of_bounds_macros"]
                 for value in macro["boundary_excess"].values()), default=0.0
            ),
            "offending_hard_macro_instances": sum(row["out_of_bounds_hard_count"] for row in failures),
            "offending_soft_macro_instances": sum(row["out_of_bounds_soft_count"] for row in failures),
        },
        "placements": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output, record)
    print(json.dumps(record["summary"], indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
