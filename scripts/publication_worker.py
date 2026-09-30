#!/usr/bin/env python3
"""Isolated worker for one publication-study benchmark run."""
from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import random
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Type


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # A script launched by absolute path otherwise sees only scripts/ on sys.path.
    sys.path.insert(0, str(REPO_ROOT))


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256(relative_paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        path = (REPO_ROOT / relative).resolve()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def verify_frozen_inputs(request: Mapping[str, Any]) -> None:
    source = request["source"]
    actual_source = source_tree_sha256(source["source_files"])
    if actual_source != source["variant_source_sha256"]:
        raise RuntimeError("source tree changed after the run identity was created")
    benchmark = request["benchmark"]
    for path_key, hash_key in (
        ("benchmark_path", "benchmark_sha256"),
        ("netlist_path", "netlist_sha256"),
        ("initial_plc_path", "initial_plc_sha256"),
    ):
        path_value, expected = benchmark.get(path_key), benchmark.get(hash_key)
        if path_value and expected and sha256_file(Path(str(path_value))) != expected:
            raise RuntimeError("benchmark input changed after identity creation: %s" % path_value)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else repr(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return repr(value)


def load_placer_class(path: Path, class_name: str) -> Type[Any]:
    module_name = "publication_placer_%s" % hashlib.sha256(
        str(path.resolve()).encode("utf-8")
    ).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("cannot load placer module %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    placer_class = getattr(module, class_name)
    if not inspect.isclass(placer_class) or not hasattr(placer_class, "place"):
        raise TypeError("%s.%s is not a placer class" % (path, class_name))
    return placer_class


def validate_constructor_parameters(cls: Type[Any], parameters: Mapping[str, Any]) -> None:
    signature = inspect.signature(cls.__init__)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    unknown = sorted(
        name for name in parameters
        if name not in signature.parameters and not accepts_kwargs
    )
    if unknown:
        raise TypeError(
            "%s does not accept constructor parameter(s): %s"
            % (cls.__name__, ", ".join(unknown))
        )


def tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    """Hash coordinates independently of their pickle/container metadata."""
    import numpy as np
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def checkpoint_inventory(directory: Path) -> Sequence[Dict[str, Any]]:
    """Record every saved checkpoint plus its phase and coordinate identity."""
    records = []
    for path in sorted(directory.rglob("*.pkl")):
        record: Dict[str, Any] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "artifact_sha256": sha256_file(path),
        }
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, Mapping):
                record.update({
                    "phase": payload.get("phase"),
                    "lane_index": payload.get("lane_idx"),
                    "rng_seed": payload.get("rng_seed"),
                    "effective_rng_seed": payload.get("effective_rng_seed"),
                    "initializer_attempt": payload.get("initializer_attempt"),
                    "initializer_method": payload.get("initializer_method"),
                    "coordinates_sha256": (
                        array_sha256(payload["pos_np"])
                        if payload.get("pos_np") is not None else None
                    ),
                })
        except Exception as exc:
            record["inventory_error"] = "%s: %s" % (type(exc).__name__, exc)
        records.append(record)
    return records


def installed_packages() -> Dict[str, str]:
    packages: Dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def torch_environment(torch: Any, requested_device: str, threads: int) -> Dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    device_detail: Optional[str] = None
    if requested_device == "cuda" and cuda_available:
        device_detail = torch.cuda.get_device_name(torch.cuda.current_device())
    elif requested_device == "mps" and mps_available:
        device_detail = "Apple Metal Performance Shaders"
    elif requested_device == "cpu":
        device_detail = platform.processor() or None
    packages = installed_packages()
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "requested_thread_limit": threads,
        "requested_device": requested_device,
        "device_detail": device_detail,
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch.version, "cuda", None),
        "mps_available": mps_available,
        "thread_environment": {
            name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            )
        },
        "installed_distributions_sha256": hashlib.sha256(
            json.dumps(
                packages, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        ).hexdigest(),
        "installed_packages": packages,
    }


def placer_definition(variant: str) -> Dict[str, str]:
    if variant == "v1":
        return {
            "module": "thinkorplace.v1.placer",
            "class": "Placer",
        }
    return {
        "module": "thinkorplace.v2.placer",
        "class": "Placer",
    }


def seed_provenance(
    variant: str, nominal_seed: int, params: Mapping[str, Any], metadata: Any,
) -> Dict[str, Any]:
    if variant == "v1":
        nominal_search = {"rng_seed": params.get("rng_seed")}
        initializer_seeds = {"pipeline": 42} if params.get("init", "sdf") == "sdf" else {}
    else:
        nominal_search = {
            "lane_a_seed": params.get("lane_a_seed"),
            "lane_b_seed": params.get("lane_b_seed"),
            "lane_c_seed": params.get("lane_c_seed"),
            "hung_seed_base": params.get("hung_seed_base", 42),
        }
        initializer_seeds = {}
        if isinstance(metadata, Mapping):
            lanes = list(str(metadata.get("lanes", "")))
            results = metadata.get("lane_results", [])
            for index, lane in enumerate(lanes):
                record = results[index] if index < len(results) else None
                if isinstance(record, Mapping):
                    initializer_seeds[lane] = record.get("effective_rng_seed")
        if params.get("init", "sdf") == "sdf":
            for lane in str(params.get("lanes", "A")):
                initializer_seeds.setdefault(lane, 42)
    return {
        "nominal_seed": nominal_seed,
        "nominal_search_seeds": nominal_search,
        "effective_initializer_seeds": initializer_seeds,
        "constructor_seeds": nominal_search,
        "initializer_seed": 42 if params.get("init", "sdf") == "sdf" else None,
        "initializer_seed_note": (
            "The current SDF initializer uses seed 42 independently of lane/rng seeds."
            if params.get("init", "sdf") == "sdf" else None
        ),
        "runtime_seed_metadata": json_safe(
            metadata.get("seed_provenance")
            if isinstance(metadata, Mapping) else None
        ),
    }


def solver_outcome(variant: str, metadata: Any) -> Dict[str, Any]:
    """Separate normal pipeline completion from valid checkpoint recovery."""
    if not isinstance(metadata, Mapping):
        return {
            "solver_outcome": "completion_metadata_unavailable",
            "used_fallback": None,
            "phase_completion": None,
            "degraded_lanes": [],
            "pipeline_completed": False,
            "lane_worker_errors": {},
            "phase_outcomes": None,
            "incomplete_phases": [],
            "phase_errors": {},
        }
    explicit = metadata.get("solver_outcome")
    if explicit is not None:
        phase_completion = json_safe(metadata.get("phase_completion"))
        degraded = json_safe(metadata.get("degraded_lanes", []))
        if variant != "v1" and not degraded and isinstance(phase_completion, Mapping):
            degraded = [
                str(name) for name, phase in phase_completion.items()
                if isinstance(phase, Mapping) and not phase.get("pipeline_completed", False)
            ]
        declared_completion = metadata.get("pipeline_completed")
        pipeline_completed = (
            bool(declared_completion)
            if isinstance(declared_completion, bool)
            else str(explicit) == "pipeline_completed" and not degraded
        )
        picked = metadata.get("picked_lane")
        lane_errors = {}
        lane_results = metadata.get("lane_results", [])
        lanes = list(str(metadata.get("lanes", "")))
        selected_lane = None
        for index, name in enumerate(lanes):
            lane = lane_results[index] if index < len(lane_results) else None
            if name == picked and isinstance(lane, Mapping):
                selected_lane = lane
            lane_worker_status = None
            if isinstance(lane, Mapping):
                lane_worker_status = lane.get("worker_status")
                if lane_worker_status is None:
                    lane_worker_status = (
                        "interrupted" if lane.get("status") == "ok"
                        else lane.get("status", "missing")
                    )
            if isinstance(lane, Mapping) and str(lane_worker_status) != "ok":
                lane_errors[name] = {
                    key: json_safe(lane.get(key))
                    for key in ("worker_status", "reason", "traceback")
                    if lane.get(key) is not None
                }
                lane_errors[name]["worker_status"] = str(lane_worker_status)
        if variant != "v1" and selected_lane is not None:
            used_fallback = selected_lane.get("worker_status") != "ok"
        else:
            used_fallback = bool(metadata.get("used_fallback", False))

        if variant == "v1":
            phase_outcomes = json_safe(metadata.get("phase_details", {}))
        else:
            phase_outcomes = {}
            for index, name in enumerate(lanes):
                lane_phase = None
                if isinstance(phase_completion, Mapping):
                    completion = phase_completion.get(name)
                    if isinstance(completion, Mapping):
                        lane_phase = completion.get("phase_details")
                if lane_phase is None:
                    lane = lane_results[index] if index < len(lane_results) else None
                    if isinstance(lane, Mapping):
                        lane_phase = lane.get("phase_details")
                phase_outcomes[name] = json_safe(lane_phase or {})

        incomplete_phases = []
        phase_errors = {}
        groups = {"": phase_outcomes} if variant == "v1" else phase_outcomes
        if isinstance(groups, Mapping):
            for group, details in groups.items():
                if not isinstance(details, Mapping):
                    continue
                for phase, detail in details.items():
                    if not isinstance(detail, Mapping):
                        continue
                    label = "%s.%s" % (group, phase) if group else str(phase)
                    disabled = bool(detail.get("skipped")) and detail.get("reason") == "disabled"
                    if not bool(detail.get("completed")) and not disabled:
                        incomplete_phases.append(label)
                    if bool(detail.get("failed")):
                        phase_errors[label] = detail.get("error") or detail.get("reason")
        if pipeline_completed:
            normalized = "pipeline_completed"
        elif used_fallback:
            normalized = "checkpoint_recovery"
        else:
            normalized = "partial_phase_completion"
        return {
            "solver_outcome": normalized,
            "used_fallback": used_fallback,
            "phase_completion": phase_completion,
            "degraded_lanes": degraded,
            "pipeline_completed": pipeline_completed,
            "lane_worker_errors": lane_errors,
            "phase_outcomes": phase_outcomes,
            "incomplete_phases": sorted(incomplete_phases),
            "phase_errors": phase_errors,
        }
    if variant == "v1":
        return {
            "solver_outcome": "pipeline_completed",
            "used_fallback": False,
            "phase_completion": {"selected_phase": metadata.get("phase")},
            "degraded_lanes": [],
            "pipeline_completed": True,
            "lane_worker_errors": {},
            "phase_outcomes": None,
            "incomplete_phases": [],
            "phase_errors": {},
        }

    lane_names = list(str(metadata.get("lanes", "")))
    lane_results = metadata.get("lane_results", [])
    picked = metadata.get("picked_lane")
    phases: Dict[str, Any] = {}
    degraded = []
    selected_used_fallback = False
    for index, lane_name in enumerate(lane_names):
        lane = lane_results[index] if index < len(lane_results) else None
        if not isinstance(lane, Mapping):
            phases[lane_name] = {"worker_status": "missing", "phase": None}
            degraded.append(lane_name)
            if lane_name == picked:
                selected_used_fallback = True
            continue
        worker_status = str(lane.get("worker_status", "unknown"))
        phases[lane_name] = {
            "worker_status": worker_status,
            "checkpoint_status": lane.get("status"),
            "selected_checkpoint_phase": lane.get("phase"),
            "worker_wall_seconds": lane.get("worker_wall", lane.get("wall")),
        }
        if worker_status != "ok":
            degraded.append(lane_name)
            if lane_name == picked:
                selected_used_fallback = True
    if not degraded and lane_names:
        outcome = "pipeline_completed"
    elif selected_used_fallback:
        outcome = "checkpoint_recovery"
    else:
        outcome = "partial_lane_completion"
    return {
        "solver_outcome": outcome,
        "used_fallback": selected_used_fallback,
        "phase_completion": phases,
        "degraded_lanes": degraded,
        "pipeline_completed": not degraded and bool(lane_names),
        "lane_worker_errors": {
            name: {
                key: json_safe(phases[name].get(key))
                for key in ("worker_status", "reason", "traceback")
                if phases[name].get(key) is not None
            }
            for name in degraded
        },
        "phase_outcomes": None,
        "incomplete_phases": [],
        "phase_errors": {},
    }


def run_request(request: Mapping[str, Any]) -> Dict[str, Any]:
    worker_started_mono = time.monotonic()
    started_wall = time.time()
    # Thread limits are set in the parent environment before this module imports torch.
    verify_started = time.monotonic()
    verify_frozen_inputs(request)
    input_verification_seconds = time.monotonic() - verify_started
    imports_started = time.monotonic()
    import numpy as np
    import torch
    from macro_place.benchmark import Benchmark
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_overlap_metrics, compute_proxy_cost
    from macro_place.utils import validate_placement
    runtime_import_seconds = time.monotonic() - imports_started

    setup_started = time.monotonic()
    resources = request["resource_allocation"]
    threads = int(resources["threads_per_benchmark_run"])
    requested_device = str(resources["device"])
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(threads, 4)))
    except RuntimeError:
        pass

    seed = int(request["nominal_seed"])
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    benchmark_info = request["benchmark"]
    benchmark = Benchmark.load(str(benchmark_info["benchmark_path"]))
    initial_hash = tensor_sha256(benchmark.macro_positions)
    _, plc = load_benchmark_from_dir(str(benchmark_info["source_dir"]))

    definition = placer_definition(str(request["variant"]))
    placer_class = getattr(importlib.import_module(definition["module"]), definition["class"])
    params = dict(request["constructor_parameters"])
    params["checkpoint_dir"] = str(request["checkpoint_dir"])
    validate_constructor_parameters(placer_class, params)
    placer = placer_class(**params)
    setup_seconds = time.monotonic() - setup_started
    place_started = time.monotonic()
    try:
        placement = placer.place(benchmark)
    except BaseException as exc:
        exc.solver_metadata = json_safe(getattr(placer, "last_run_metadata", None))
        raise
    elapsed_solver = time.monotonic() - place_started
    verify_frozen_inputs(request)

    evaluation_started = time.monotonic()
    full_legal, full_violations = validate_placement(
        placement, benchmark, check_overlaps=True
    )
    structural_legal, structural_violations = validate_placement(
        placement, benchmark, check_overlaps=False
    )
    overlaps = compute_overlap_metrics(placement, benchmark)
    costs = compute_proxy_cost(placement, benchmark, plc)
    canonical = {
        key: float(costs[key]) for key in (
            "proxy_cost", "wirelength_cost", "density_cost", "congestion_cost"
        )
    }
    overlap_record = {
        "overlap_count": int(overlaps["overlap_count"]),
        "total_overlap_area": float(overlaps["total_overlap_area"]),
        "max_overlap_area": float(overlaps["max_overlap_area"]),
        "num_macros_with_overlaps": int(overlaps["num_macros_with_overlaps"]),
        "overlap_ratio": float(overlaps["overlap_ratio"]),
    }
    final_path = Path(str(request["final_placement_path"]))
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(".%s.%d.tmp" % (final_path.name, os.getpid()))
    torch.save(placement.detach().cpu(), str(temporary))
    os.replace(str(temporary), str(final_path))
    evaluation_and_export_seconds = time.monotonic() - evaluation_started
    metadata = json_safe(getattr(placer, "last_run_metadata", None))
    outcome = solver_outcome(str(request["variant"]), metadata)
    checkpoints = checkpoint_inventory(Path(str(request["checkpoint_dir"])))
    environment = torch_environment(torch, requested_device, threads)
    environment_matches_request = (
        environment["installed_distributions_sha256"]
        == request["host"]["installed_distributions_sha256"]
    )
    source_consistent = True
    usable_operating_result = bool(full_legal) and source_consistent
    study_eligible = usable_operating_result and environment_matches_request

    result = {
        "schema_version": request["schema_version"],
        "run_id": request["run_id"],
        "identity_sha256": request["identity_sha256"],
        "status": "completed",
        "valid": bool(full_legal),
        "started_at": _datetime.datetime.fromtimestamp(
            started_wall, _datetime.timezone.utc
        ).isoformat(),
        "finished_at": utc_now(),
        "timing": {
            "input_verification_seconds": input_verification_seconds,
            "runtime_import_seconds": runtime_import_seconds,
            "benchmark_and_placer_setup_seconds": setup_seconds,
            "solver_place_seconds": elapsed_solver,
            "canonical_validation_and_export_seconds": evaluation_and_export_seconds,
            "worker_through_environment_capture_seconds": (
                time.monotonic() - worker_started_mono
            ),
            "solver_budget_seconds": request["constructor_parameters"].get(
                "budget_seconds"
            ),
            "external_timeout_seconds": resources.get("external_timeout_seconds"),
        },
        "variant": request["variant"],
        "benchmark": benchmark_info,
        "repository": request["repository"],
        "source": request["source"],
        "resource_allocation": resources,
        "constructor_parameters": request["constructor_parameters"],
        "runtime_artifact_parameters": {"checkpoint_dir": str(request["checkpoint_dir"])},
        "nominal_seed": seed,
        "seeds": seed_provenance(str(request["variant"]), seed, params, metadata),
        "initial_placement_sha256": initial_hash,
        "final_placement_tensor_sha256": tensor_sha256(placement),
        "final_placement_file_sha256": sha256_file(final_path),
        "final_placement_path": str(final_path),
        "checkpoint_dir": str(request["checkpoint_dir"]),
        "checkpoint_inventory": checkpoints,
        "canonical_cost": canonical,
        "legality": {
            "full_legal": bool(full_legal),
            "full_violations": full_violations,
            "structural_legal_without_overlap": bool(structural_legal),
            "structural_violations_without_overlap": structural_violations,
            "hard_macro_overlap": overlap_record,
        },
        "solver_metadata": metadata,
        "environment": environment,
        "environment_matches_manifest": environment_matches_request,
        "source_consistent": source_consistent,
        "usable_operating_result": usable_operating_result,
        "study_eligible": study_eligible,
        "pipeline_completed": outcome["pipeline_completed"],
        "worker_error": None,
    }
    result.update(outcome)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    request_path = Path(args.request).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result_path = Path(str(request["worker_result_path"]))
    try:
        result = run_request(request)
        atomic_write_json(result_path, result)
        return 0
    except BaseException as exc:
        result = {
            "schema_version": request.get("schema_version"),
            "run_id": request.get("run_id"),
            "identity_sha256": request.get("identity_sha256"),
            "status": "error",
            "valid": False,
            "finished_at": utc_now(),
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
            "environment": {
                "python_version": platform.python_version(),
                "python_executable": sys.executable,
                "worker_runtime": "worker failed before complete environment capture",
            },
        }
        metadata = getattr(exc, "solver_metadata", None)
        if isinstance(metadata, Mapping):
            result["solver_metadata"] = metadata
            result.update(solver_outcome(str(request.get("variant", "")), metadata))
            result["solver_outcome"] = "worker_did_not_complete"
            result["pipeline_completed"] = False
        atomic_write_json(result_path, result)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
