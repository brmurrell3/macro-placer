#!/usr/bin/env python3
"""Canonically rescore and validate the 34 frozen E74/E84 placements.

The script is intentionally separate from the historical freezing and study
runner code.  It reads, but never modifies, the frozen tensors, sidecars,
benchmark inputs, or evaluator sources.  Results are checkpointed atomically
after each benchmark, so rerunning the same command resumes missing records.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "writeup" / "data" / "publication"
RAW = DATA / "raw"
PLACEMENTS = RAW / "placements"
SIDECARS = RAW / "cascade"
DEFAULT_OUTPUT = DATA / "e74_e84_canonical_rescore_20260907.json"
BENCHMARKS = (
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08",
    "ibm09", "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15",
    "ibm16", "ibm17", "ibm18",
)
STAGES = ("hessian", "cascade")
PINNED_REPOSITORY_COMMIT = "2ab92a33d352f0dbc4ba97d7c2bfad41ccf0ecfd"
PINNED_TILOS_COMMIT = "45a721d01dfe56fc4800b95da52e9b4193b76d04"
CANONICAL_SOURCES = (
    "macro_place/_plc.py",
    "macro_place/loader.py",
    "macro_place/objective.py",
    "external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py",
)
HISTORICAL_FULL_SUITE = (
    ("v1_stacked_periphery", "CDLNSSACascadeStackedPeripheryPlacer_20260517_062436.json"),
    ("earlier_v2_epyc", "EPYC_thinkorplace_v2_validation_20260520.json"),
    ("gaussian_ancestor", "E111MinimalGaussianPlacer_20260520_135149.json"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def git_output(arguments: Sequence[str], cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *arguments], cwd=cwd, text=True).strip()


def nested_coordinate_keys(value: Any, prefix: str = "") -> List[str]:
    hits: List[str] = []
    coordinate_names = {
        "placement", "placements", "position", "positions", "coordinates",
        "macro_positions", "pos", "pos_np", "x", "y",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = "%s.%s" % (prefix, key) if prefix else str(key)
            if str(key).lower() in coordinate_names:
                hits.append(child_prefix)
            hits.extend(nested_coordinate_keys(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(nested_coordinate_keys(child, "%s[%d]" % (prefix, index)))
    return hits


def historical_coordinate_audit() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for label, filename in HISTORICAL_FULL_SUITE:
        path = RAW / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        sibling_candidates = sorted(
            str(candidate.relative_to(ROOT))
            for candidate in ROOT.rglob(path.stem + ".*")
            if candidate.resolve() != path.resolve()
        )
        record: Dict[str, Any] = {
            "label": label,
            "json_path": str(path.relative_to(ROOT)),
            "json_sha256": sha256_file(path),
            "benchmark_names": [row["name"] for row in payload["benchmarks"]],
            "benchmark_count": len(payload["benchmarks"]),
            "embedded_coordinate_keys": nested_coordinate_keys(payload),
            "same_basename_artifacts_elsewhere": sibling_candidates,
        }
        if label == "v1_stacked_periphery":
            record["repository_coordinate_candidates"] = {
                "paths": [
                    "writeup/data/publication/raw/placements/hessian_ibm*.pt",
                    "writeup/data/publication/raw/placements/cascade_ibm*.pt",
                ],
                "classification": (
                    "E74/E84 stage endpoints from a different, earlier experiment; "
                    "their provenance and recorded proxies do not identify them as "
                    "the 17 final stacked-periphery outputs in this JSON"
                ),
            }
        elif label == "earlier_v2_epyc":
            record["repository_coordinate_candidates"] = {
                "paths": [],
                "classification": (
                    "No same-run or same-basename coordinate artifact was located "
                    "in the repository; unrelated later pilot outputs are not recoveries"
                ),
            }
        else:
            candidate = ROOT / "experiments/E117_gaussian_density/results/ibm17_descent_gaussian_s1_0.pt"
            record["repository_coordinate_candidates"] = {
                "paths": [str(candidate.relative_to(ROOT))] if candidate.exists() else [],
                "sha256": sha256_file(candidate) if candidate.exists() else None,
                "classification": (
                    "One IBM17 descent-only diagnostic tensor exists. Its producing "
                    "script labels it an intermediate with no CD polish, so it is not "
                    "provenance-linked to the full-suite JSON's final IBM17 output; "
                    "no candidates were located for the other 16 designs"
                ),
            }
        record["scoped_conclusion"] = (
            "This JSON embeds component values but no coordinates. The repository "
            "audit above did not locate provenance-linked final coordinates for its "
            "17 rows; this is a scoped finding, not a claim about unavailable external artifacts."
        )
        records.append(record)
    return records


def source_identity() -> Dict[str, Any]:
    repository_head = git_output(["rev-parse", "HEAD"])
    tilos_head = git_output(["rev-parse", "HEAD"], ROOT / "external/MacroPlacement")
    if repository_head != PINNED_REPOSITORY_COMMIT:
        raise RuntimeError("Repository HEAD is not the pinned competition revision")
    if tilos_head != PINNED_TILOS_COMMIT:
        raise RuntimeError("TILOS submodule is not at the pinned evaluator revision")
    files = {}
    for relative in CANONICAL_SOURCES:
        path = ROOT / relative
        files[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    # The three local adapter sources are tracked at the pinned repository commit.
    for relative in CANONICAL_SOURCES[:3]:
        pinned = subprocess.check_output(
            ["git", "show", "%s:%s" % (PINNED_REPOSITORY_COMMIT, relative)], cwd=ROOT
        )
        pinned_sha = hashlib.sha256(pinned).hexdigest()
        files[relative]["pinned_blob_sha256"] = pinned_sha
        files[relative]["matches_pinned_blob"] = pinned_sha == files[relative]["sha256"]
        if not files[relative]["matches_pinned_blob"]:
            raise RuntimeError("Canonical adapter differs from pinned blob: %s" % relative)
    tilos_relative = CANONICAL_SOURCES[3].split("external/MacroPlacement/", 1)[1]
    pinned = subprocess.check_output(
        ["git", "show", "%s:%s" % (PINNED_TILOS_COMMIT, tilos_relative)],
        cwd=ROOT / "external/MacroPlacement",
    )
    pinned_sha = hashlib.sha256(pinned).hexdigest()
    files[CANONICAL_SOURCES[3]]["pinned_blob_sha256"] = pinned_sha
    files[CANONICAL_SOURCES[3]]["matches_pinned_blob"] = (
        pinned_sha == files[CANONICAL_SOURCES[3]]["sha256"]
    )
    if not files[CANONICAL_SOURCES[3]]["matches_pinned_blob"]:
        raise RuntimeError("Canonical PLC implementation differs from pinned submodule blob")
    return {
        "repository_commit": repository_head,
        "tilos_submodule_commit": tilos_head,
        "source_files": files,
        "interface": (
            "macro_place.loader.load_benchmark_from_dir; "
            "macro_place.objective._set_placement; PlacementCost.get_cost, "
            "get_density_cost, get_congestion_cost"
        ),
        "component_meaning": "Evaluator-normalized wirelength, density, and congestion",
        "proxy_formula": "wirelength + 0.5*density + 0.5*congestion",
    }


def geometry_record(placement: Any, benchmark: Any) -> Dict[str, Any]:
    import torch

    expected = [int(benchmark.num_macros), 2]
    actual = list(placement.shape)
    shape_ok = actual == expected
    result: Dict[str, Any] = {
        "expected_shape": expected,
        "actual_shape": actual,
        "shape_ok": shape_ok,
        "dtype": str(placement.dtype),
    }
    if not shape_ok:
        result.update({
            "finite": {"all": False, "nonfinite_coordinate_count": None},
            "bounds": {"all_within_or_on_canvas": False},
            "fixed": {"all_within_atol_1e_3": False},
            "hard_macro_overlap": None,
            "full_legal": False,
        })
        return result
    finite_mask = torch.isfinite(placement)
    result["finite"] = {
        "all": bool(finite_mask.all()),
        "nonfinite_coordinate_count": int((~finite_mask).sum()),
        "nan_coordinate_count": int(torch.isnan(placement).sum()),
        "inf_coordinate_count": int(torch.isinf(placement).sum()),
    }
    half = benchmark.macro_sizes.to(placement.dtype) / 2
    lower = placement - half
    upper = placement + half
    x_low = torch.clamp(-lower[:, 0], min=0)
    y_low = torch.clamp(-lower[:, 1], min=0)
    x_high = torch.clamp(upper[:, 0] - benchmark.canvas_width, min=0)
    y_high = torch.clamp(upper[:, 1] - benchmark.canvas_height, min=0)
    bound_excess = torch.stack((x_low, x_high, y_low, y_high), dim=1)
    out_mask = (bound_excess > 0).any(dim=1)
    result["bounds"] = {
        "definition": "center +/- half macro size must lie in [0, canvas dimension]",
        "all_within_or_on_canvas": bool(not out_mask.any()),
        "out_of_bounds_macro_count": int(out_mask.sum()),
        "x_below_count": int((x_low > 0).sum()),
        "x_above_count": int((x_high > 0).sum()),
        "y_below_count": int((y_low > 0).sum()),
        "y_above_count": int((y_high > 0).sum()),
        "maximum_boundary_excess": float(bound_excess.max()),
    }
    fixed_mask = benchmark.macro_fixed
    fixed_delta = (placement[fixed_mask] - benchmark.macro_positions[fixed_mask].to(placement.dtype)).abs()
    fixed_bad = (fixed_delta > 1e-3).any(dim=1) if fixed_delta.numel() else torch.zeros(0, dtype=torch.bool)
    result["fixed"] = {
        "definition": "saved coordinate compared with initial.plc coordinate; atol=1e-3, rtol=0",
        "fixed_macro_count": int(fixed_mask.sum()),
        "moved_beyond_tolerance_count": int(fixed_bad.sum()),
        "maximum_absolute_coordinate_difference": float(fixed_delta.max()) if fixed_delta.numel() else 0.0,
        "all_within_atol_1e_3": bool(not fixed_bad.any()),
    }
    # This is the same hard-macro rectangle check used by the publication runner.
    from macro_place.objective import compute_overlap_metrics
    overlap = compute_overlap_metrics(placement, benchmark)
    result["hard_macro_overlap"] = {
        key: int(value) if key in ("overlap_count", "num_macros_with_overlaps") else float(value)
        for key, value in overlap.items()
    }
    result["full_legal"] = bool(
        result["finite"]["all"]
        and result["bounds"]["all_within_or_on_canvas"]
        and result["fixed"]["all_within_atol_1e_3"]
        and overlap["overlap_count"] == 0
    )
    return result


def recorded_values(stage: str, sidecar: Mapping[str, Any], saved: Mapping[str, Any]) -> Dict[str, Any]:
    proxy_key = "final_proxy" if stage == "hessian" else "best_proxy"
    stats = saved.get("stats", {})
    stats_proxy_key = "best_proxy"
    return {
        "sidecar_proxy": float(sidecar[proxy_key]),
        "sidecar_proxy_field": proxy_key,
        "sidecar_overlap_count": int(sidecar["final_overlap"]),
        "tensor_stats_proxy": float(stats[stats_proxy_key]) if stats_proxy_key in stats else None,
        "tensor_stats_proxy_field": stats_proxy_key if stats_proxy_key in stats else None,
    }


def stage_record(stage: str, name: str, benchmark: Any, plc: Any) -> Dict[str, Any]:
    import torch
    from macro_place.objective import _set_placement

    total_started = time.perf_counter()
    tensor_path = PLACEMENTS / ("%s_%s.pt" % (stage, name))
    sidecar_path = SIDECARS / ("%s_%s.json" % (stage, name))
    load_started = time.perf_counter()
    saved = torch.load(tensor_path, map_location="cpu", weights_only=False)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    artifact_load_seconds = time.perf_counter() - load_started
    if not isinstance(saved, Mapping) or "placement" not in saved:
        raise TypeError("Unexpected saved placement container: %s" % tensor_path)
    if saved.get("bench_name") != name or sidecar.get("bench_name") != name:
        raise ValueError("Benchmark identity mismatch for %s %s" % (stage, name))
    placement = saved["placement"].detach().cpu().contiguous()
    geometry_started = time.perf_counter()
    geometry = geometry_record(placement, benchmark)
    geometry_seconds = time.perf_counter() - geometry_started
    if not geometry["shape_ok"] or not geometry["finite"]["all"]:
        raise ValueError("Cannot canonically score malformed placement: %s" % tensor_path)
    score_started = time.perf_counter()
    _set_placement(plc, placement, benchmark)
    wirelength = float(plc.get_cost())
    density = float(plc.get_density_cost())
    congestion = float(plc.get_congestion_cost())
    proxy = wirelength + 0.5 * density + 0.5 * congestion
    canonical_seconds = time.perf_counter() - score_started
    recorded = recorded_values(stage, sidecar, saved)
    fresh = {
        "normalized_components": {
            "wirelength": wirelength,
            "density": density,
            "congestion": congestion,
        },
        "proxy_cost": proxy,
    }
    difference = {
        "fresh_minus_recorded_proxy": proxy - recorded["sidecar_proxy"],
        "absolute_proxy_difference": abs(proxy - recorded["sidecar_proxy"]),
        "relative_proxy_difference": (
            (proxy - recorded["sidecar_proxy"]) / recorded["sidecar_proxy"]
            if recorded["sidecar_proxy"] != 0 else None
        ),
        "fresh_minus_recorded_overlap_count": (
            geometry["hard_macro_overlap"]["overlap_count"] - recorded["sidecar_overlap_count"]
        ),
    }
    return {
        "stage": "E74" if stage == "hessian" else "E84",
        "source_artifacts": {
            "tensor_path": str(tensor_path.relative_to(ROOT)),
            "tensor_file_sha256": sha256_file(tensor_path),
            "coordinate_tensor_sha256": tensor_sha256(placement),
            "sidecar_path": str(sidecar_path.relative_to(ROOT)),
            "sidecar_sha256": sha256_file(sidecar_path),
            "container_benchmark_name": saved.get("bench_name"),
            "coordinate_key": "placement",
        },
        "recorded": recorded,
        "fresh_canonical": fresh,
        "recorded_vs_fresh": difference,
        "geometry_validation": geometry,
        "timing_seconds": {
            "artifact_load": artifact_load_seconds,
            "geometry_validation": geometry_seconds,
            "canonical_components": canonical_seconds,
            "stage_total": time.perf_counter() - total_started,
        },
    }


def benchmark_record(name: str, threads: int) -> Dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    sys.path.insert(0, str(ROOT))
    import torch
    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.loader import load_benchmark_from_dir

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    total_started = time.perf_counter()
    benchmark_dir = Path(find_benchmark_dir(name)).resolve()
    load_started = time.perf_counter()
    benchmark, plc = load_benchmark_from_dir(str(benchmark_dir))
    benchmark_load_seconds = time.perf_counter() - load_started
    canvas_area = float(benchmark.canvas_width * benchmark.canvas_height)
    areas = benchmark.macro_sizes[:, 0].double() * benchmark.macro_sizes[:, 1].double()
    hard_area = float(areas[:benchmark.num_hard_macros].sum())
    total_area = float(areas.sum())
    row_mapping = []
    for row, (plc_index, macro_name) in enumerate(zip(
        list(benchmark.hard_macro_indices) + list(benchmark.soft_macro_indices),
        benchmark.macro_names,
    )):
        row_mapping.append([
            row,
            int(plc_index),
            str(macro_name),
            "hard" if row < benchmark.num_hard_macros else "soft",
            bool(benchmark.macro_fixed[row]),
        ])
    input_files = {}
    for filename in ("netlist.pb.txt", "initial.plc"):
        path = benchmark_dir / filename
        input_files[str(path.relative_to(ROOT))] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "role": "parsed netlist" if filename.startswith("netlist") else "initial placement and evaluator parameters",
        }
    stages = [stage_record(stage, name, benchmark, plc) for stage in STAGES]
    return {
        "name": name,
        "status": "completed",
        "worker_pid": os.getpid(),
        "input_files": input_files,
        "coordinate_row_mapping": {
            "contract": (
                "Saved placement row i is Benchmark row i. Rows are PLC hard_macro_indices "
                "in parser order, followed by PLC soft_macro_indices in parser order."
            ),
            "fields": ["row", "plc_module_index", "macro_name", "kind", "fixed"],
            "mapping_sha256": sha256_json(row_mapping),
            "rows": row_mapping,
        },
        "parsed_benchmark_statistics": {
            "num_macros": int(benchmark.num_macros),
            "num_hard_macros": int(benchmark.num_hard_macros),
            "num_soft_macros": int(benchmark.num_soft_macros),
            "num_fixed_macros": int(benchmark.macro_fixed.sum()),
            "num_fixed_hard_macros": int(benchmark.macro_fixed[:benchmark.num_hard_macros].sum()),
            "num_fixed_soft_macros": int(benchmark.macro_fixed[benchmark.num_hard_macros:].sum()),
            "plc_net_count": int(plc.net_cnt),
            "benchmark_filtered_net_count": int(benchmark.num_nets),
            "net_count_definition": (
                "plc_net_count is the parser's full count; benchmark_filtered_net_count "
                "counts loader nets with at least one resolved benchmark node"
            ),
            "canvas_width": float(benchmark.canvas_width),
            "canvas_height": float(benchmark.canvas_height),
            "canvas_area": canvas_area,
            "grid_rows": int(benchmark.grid_rows),
            "grid_cols": int(benchmark.grid_cols),
            "hard_macro_area": hard_area,
            "total_macro_area": total_area,
            "hard_macro_area_over_canvas": hard_area / canvas_area,
            "total_macro_area_over_canvas": total_area / canvas_area,
            "utilization_definition": (
                "sum of rectangular macro widths*heights divided by full canvas width*height; "
                "hard ratio includes fixed and movable hard macros; total adds all soft macros"
            ),
        },
        "placements": stages,
        "pair_comparison": {
            "recorded_e84_minus_e74_proxy": (
                stages[1]["recorded"]["sidecar_proxy"] - stages[0]["recorded"]["sidecar_proxy"]
            ),
            "fresh_e84_minus_e74_proxy": (
                stages[1]["fresh_canonical"]["proxy_cost"] - stages[0]["fresh_canonical"]["proxy_cost"]
            ),
        },
        "timing_seconds": {
            "benchmark_load": benchmark_load_seconds,
            "benchmark_total": time.perf_counter() - total_started,
        },
    }


def error_record(name: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "name": name,
        "status": "error",
        "error": "%s: %s" % (type(exc).__name__, exc),
        "traceback": traceback.format_exc(),
    }


def atomic_write(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def aggregate(record: Dict[str, Any]) -> None:
    complete = [row for row in record["benchmarks"] if row.get("status") == "completed"]
    record["progress"] = {
        "completed_benchmarks": len(complete),
        "target_benchmarks": len(BENCHMARKS),
        "completed_placements": 2 * len(complete),
        "target_placements": 2 * len(BENCHMARKS),
    }
    if not complete:
        record["aggregate"] = None
        return
    by_stage: Dict[str, List[Mapping[str, Any]]] = {"E74": [], "E84": []}
    for benchmark in complete:
        for placement in benchmark["placements"]:
            by_stage[placement["stage"]].append(placement)
    stage_aggregates = {}
    for stage, rows in by_stage.items():
        fresh = [row["fresh_canonical"]["proxy_cost"] for row in rows]
        recorded = [row["recorded"]["sidecar_proxy"] for row in rows]
        stage_aggregates[stage] = {
            "coverage": len(rows),
            "fresh_mean_proxy": statistics.mean(fresh),
            "recorded_mean_proxy": statistics.mean(recorded),
            "fresh_mean_minus_recorded_mean": statistics.mean(fresh) - statistics.mean(recorded),
            "maximum_absolute_per_placement_proxy_difference": max(
                row["recorded_vs_fresh"]["absolute_proxy_difference"] for row in rows
            ),
            "full_legal_count": sum(row["geometry_validation"]["full_legal"] for row in rows),
            "zero_overlap_count": sum(
                row["geometry_validation"]["hard_macro_overlap"]["overlap_count"] == 0
                for row in rows
            ),
        }
    record["aggregate"] = {
        "by_stage": stage_aggregates,
        "fresh_mean_e84_minus_e74": (
            stage_aggregates["E84"]["fresh_mean_proxy"]
            - stage_aggregates["E74"]["fresh_mean_proxy"]
        ),
        "benchmark_total_seconds_sum": sum(row["timing_seconds"]["benchmark_total"] for row in complete),
    }


def initial_record(threads: int, workers: int) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "canonical_rescore_and_geometry_validation_of_frozen_e74_e84_endpoints",
        "measurement_status": "in_progress",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "script": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "device": "cpu",
            "torch_threads_per_worker": threads,
            "maximum_worker_processes": workers,
        },
        "canonical_evaluator": source_identity(),
        "protocol": {
            "read_only_inputs": [
                "writeup/data/publication/raw/placements/*.pt",
                "writeup/data/publication/raw/cascade/*.json",
                "external/MacroPlacement/Testcases/ICCAD04/*/{netlist.pb.txt,initial.plc}",
                *CANONICAL_SOURCES,
            ],
            "checkpointing": "Atomic rewrite after every completed benchmark; completed records are skipped on resume",
            "parallelism": "At most two spawned workers; each worker uses one PyTorch/inter-op thread",
            "geometry": "Strict finite/bounds/shape, fixed-position atol=1e-3, and hard-macro rectangle overlap checks",
        },
        "target_benchmark_order": list(BENCHMARKS),
        "historical_full_suite_coordinate_audit": historical_coordinate_audit(),
        "benchmarks": [],
        "progress": {},
        "aggregate": None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads != 1:
        parser.error("This evidence protocol requires exactly one torch thread per worker")
    output = args.output.resolve()
    if output.exists():
        record = json.loads(output.read_text(encoding="utf-8"))
        if record.get("kind") != "canonical_rescore_and_geometry_validation_of_frozen_e74_e84_endpoints":
            parser.error("Existing output is not a compatible checkpoint")
        if record["script"]["sha256"] != sha256_file(Path(__file__).resolve()):
            parser.error("Script changed since checkpoint creation; use a new output path")
        if record["canonical_evaluator"] != source_identity():
            parser.error("Canonical evaluator identity changed since checkpoint creation")
    else:
        record = initial_record(args.threads, args.workers)
        atomic_write(output, record)
    existing = {row["name"]: row for row in record["benchmarks"]}
    pending = [name for name in args.benchmarks if existing.get(name, {}).get("status") != "completed"]
    if pending:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context
        ) as pool:
            futures = {pool.submit(benchmark_record, name, args.threads): name for name in pending}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                except BaseException as exc:
                    result = error_record(name, exc)
                existing[name] = result
                record["benchmarks"] = [existing[name] for name in BENCHMARKS if name in existing]
                record["updated_utc"] = utc_now()
                aggregate(record)
                atomic_write(output, record)
                seconds = result.get("timing_seconds", {}).get("benchmark_total")
                suffix = " %.2fs" % seconds if seconds is not None else ""
                print("%s: %s%s (%d/17 benchmarks)" % (
                    name, result["status"], suffix, record["progress"]["completed_benchmarks"]
                ), flush=True)
    aggregate(record)
    errors = [row for row in record["benchmarks"] if row.get("status") == "error"]
    if record["progress"]["completed_benchmarks"] == len(BENCHMARKS) and not errors:
        record["measurement_status"] = "completed"
        record["completed_utc"] = utc_now()
    else:
        record["measurement_status"] = "in_progress" if not errors else "completed_with_errors"
    record["updated_utc"] = utc_now()
    atomic_write(output, record)
    print(output, flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
