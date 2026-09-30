#!/usr/bin/env python3
"""Reproducible, externally supervised Thinkorplace benchmark studies.

This module intentionally does not import torch.  Each benchmark/variant run is
started in a fresh Python process after its thread environment has been set.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_SCRIPT = REPO_ROOT / "scripts" / "publication_worker.py"
VARIANTS = ("v1", "v2-a", "v2-abc")


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


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_paths(paths: Iterable[Path], root: Path = REPO_ROOT) -> str:
    digest = hashlib.sha256()
    unique = sorted({path.resolve() for path in paths if path.is_file()}, key=str)
    for path in unique:
        try:
            label = str(path.relative_to(root.resolve()))
        except ValueError:
            label = str(path)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _git(args: Sequence[str], cwd: Path = REPO_ROOT) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git"] + list(args), cwd=str(cwd), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def repository_provenance() -> Dict[str, Any]:
    status = _git(["status", "--porcelain=v1", "--untracked-files=all"])
    diff = _git(["diff", "--binary", "HEAD", "--"])
    entries = status.splitlines() if status else []
    generated_prefix = "?? writeup/runs/"
    generated_entries = [entry for entry in entries if entry.startswith(generated_prefix)]
    meaningful_entries = [entry for entry in entries if not entry.startswith(generated_prefix)]
    return {
        "head_commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["branch", "--show-current"]),
        "dirty": bool(meaningful_entries),
        "dirty_entries": meaningful_entries,
        "excluded_generated_run_artifact_entries": len(generated_entries),
        "tracked_diff_sha256": (
            hashlib.sha256(diff.encode("utf-8")).hexdigest()
            if diff is not None else None
        ),
        "tilos_submodule_commit": _git(
            ["-C", "external/MacroPlacement", "rev-parse", "HEAD"]
        ),
    }


def source_provenance(variant: str) -> Dict[str, Any]:
    roots = [REPO_ROOT / "macro_place", REPO_ROOT / "thinkorplace" / "common"]
    roots.append(REPO_ROOT / "thinkorplace" / ("v1" if variant == "v1" else "v2"))
    files: List[Path] = [
        Path(__file__).resolve(), WORKER_SCRIPT, REPO_ROOT / "thinkorplace" / "__init__.py",
    ]
    for root in roots:
        if root.exists():
            files.extend(root.rglob("*.py"))
    evaluator_files = [
        REPO_ROOT / "macro_place" / name
        for name in ("objective.py", "loader.py", "benchmark.py", "_plc.py", "utils.py")
    ]
    return {
        "variant_source_sha256": sha256_paths(files),
        "evaluator_source_sha256": sha256_paths(evaluator_files),
        "worker_sha256": sha256_file(WORKER_SCRIPT),
        "source_file_count": len({p.resolve() for p in files if p.is_file()}),
        "source_files": sorted(
            str(p.resolve().relative_to(REPO_ROOT.resolve()))
            for p in {item.resolve() for item in files if item.is_file()}
        ),
    }


def host_provenance() -> Dict[str, Any]:
    cpu_model: Optional[str] = None
    total_memory: Optional[int] = None
    if sys.platform == "darwin":
        cpu_model = _command_text(["sysctl", "-n", "machdep.cpu.brand_string"])
        memory = _command_text(["sysctl", "-n", "hw.memsize"])
        if memory and memory.isdigit():
            total_memory = int(memory)
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    total_memory = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            pass
    distributions = {
        str(distribution.metadata.get("Name")): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    distributions = dict(sorted(distributions.items(), key=lambda item: item[0].lower()))
    return {
        "hostname": platform.node(),
        "os": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "total_memory_bytes": total_memory,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "installed_distributions_sha256": hashlib.sha256(
            canonical_json(distributions)
        ).hexdigest(),
        "installed_distributions": distributions,
    }


def _command_text(command: Sequence[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            list(command), text=True, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _strip_benchmark_suffix(name: str) -> Tuple[str, Optional[str]]:
    base = re.sub(r"_random$", "", name)
    for suffix, technology in (("_ng45", "NanGate45"), ("_asap7", "ASAP7")):
        if base.endswith(suffix):
            return base[:-len(suffix)], technology
    return base, None


def resolve_benchmark(
    value: str, source_overrides: Mapping[str, str]
) -> Dict[str, Any]:
    supplied = Path(value)
    benchmark_path = supplied if supplied.is_file() else (
        REPO_ROOT / "benchmarks" / "processed" / "public" / (value + ".pt")
    )
    benchmark_path = benchmark_path.resolve()
    if not benchmark_path.is_file():
        raise FileNotFoundError("benchmark tensor not found: %s" % benchmark_path)
    public_name = supplied.stem if supplied.is_file() else value
    design_name, technology = _strip_benchmark_suffix(public_name)

    override = source_overrides.get(public_name) or source_overrides.get(design_name)
    candidates: List[Tuple[Path, Path]] = []
    if override:
        source = Path(override).resolve()
        candidates.append((source, source.parent))
    if design_name.startswith("ibm"):
        root = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"
        candidates.append((root / design_name, root))
    technologies = [technology] if technology else ["NanGate45", "ASAP7"]
    for tech in technologies:
        root = REPO_ROOT / "external" / "MacroPlacement" / "Flows" / str(tech)
        candidates.append((root / design_name / "netlist" / "output_CT_Grouping", root))
    candidates.append((benchmark_path.parent / design_name, benchmark_path.parent))

    source_path: Optional[Path] = None
    bench_root: Optional[Path] = None
    tried: List[str] = []
    for candidate, candidate_root in candidates:
        tried.append(str(candidate))
        if (candidate / "netlist.pb.txt").is_file():
            source_path, bench_root = candidate.resolve(), candidate_root.resolve()
            break
    if source_path is None or bench_root is None:
        raise FileNotFoundError(
            "source netlist for %s not found; tried %s; use --benchmark-source NAME=DIR"
            % (public_name, tried)
        )
    netlist = source_path / "netlist.pb.txt"
    config = source_path / "initial.plc"
    return {
        "requested_name": public_name,
        "design_name": design_name,
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": sha256_file(benchmark_path),
        "source_dir": str(source_path),
        "bench_root": str(bench_root),
        "netlist_path": str(netlist),
        "netlist_sha256": sha256_file(netlist),
        "initial_plc_path": str(config) if config.is_file() else None,
        "initial_plc_sha256": sha256_file(config) if config.is_file() else None,
    }


def parse_json_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_key_values(values: Sequence[str], variants: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    parsed: Dict[str, Dict[str, Any]] = {variant: {} for variant in variants}
    for item in values:
        if "=" not in item:
            raise ValueError("placer parameter must be [VARIANT:]NAME=JSON: %s" % item)
        left, raw = item.split("=", 1)
        if ":" in left:
            scope, name = left.split(":", 1)
            if scope not in variants:
                raise ValueError("parameter scope %r is not a selected variant" % scope)
            targets = [scope]
        else:
            name, targets = left, list(variants)
        if not name or name.startswith("_"):
            raise ValueError("invalid constructor parameter name: %r" % name)
        for target in targets:
            parsed[target][name] = parse_json_value(raw)
    return parsed


def build_constructor_parameters(
    variant: str,
    seed: int,
    budget_seconds: float,
    num_steps: Optional[int],
    cd_seconds: Optional[float],
    device: str,
    threads: int,
    extra: Mapping[str, Any],
) -> Dict[str, Any]:
    if variant == "v1":
        params: Dict[str, Any] = {
            "budget_seconds": budget_seconds,
            "rng_seed": seed,
            "device": device,
            "num_threads": threads,
        }
    else:
        params = {
            "budget_seconds": budget_seconds,
            "lane_a_seed": seed,
            "lane_b_seed": seed + 100,
            "lane_c_seed": seed + 200,
            "lanes": "A" if variant == "v2-a" else "ABC",
            "device": device,
            "num_threads": threads,
        }
        if num_steps is not None:
            params["num_steps"] = num_steps
        if cd_seconds is not None:
            params["cd_polish_s"] = cd_seconds
    params.update(extra)
    return params


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "run"


def make_run_spec(
    benchmark: Mapping[str, Any], variant: str, seed: int,
    constructor: Mapping[str, Any], device: str, threads: int, jobs: int,
    timeout_seconds: float, source: Mapping[str, Any], repository: Mapping[str, Any],
    host: Mapping[str, Any],
) -> Dict[str, Any]:
    identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": {
            key: benchmark[key] for key in (
                "requested_name", "benchmark_sha256", "netlist_sha256",
                "initial_plc_sha256",
            )
        },
        "variant": variant,
        "seed": seed,
        "constructor_parameters": constructor,
        "device": device,
        "threads": threads,
        "jobs": jobs,
        "external_timeout_seconds": timeout_seconds,
        "source": source,
        "repository_head": repository.get("head_commit"),
        "host": host,
    }
    identity = hashlib.sha256(canonical_json(identity_payload)).hexdigest()
    run_id = "%s__%s__seed%d__%s" % (
        _slug(str(benchmark["requested_name"])), variant, seed, identity[:12]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "identity_sha256": identity,
        "identity_payload": identity_payload,
        "benchmark": dict(benchmark),
        "variant": variant,
        "nominal_seed": seed,
        "constructor_parameters": dict(constructor),
        "resource_allocation": {
            "device": device,
            "threads_per_benchmark_run": threads,
            "concurrent_benchmark_jobs": jobs,
            "external_timeout_seconds": timeout_seconds,
        },
        "source": dict(source),
        "repository": dict(repository),
        "host": dict(host),
    }


def thread_environment(threads: int, device: str) -> Dict[str, str]:
    env = os.environ.copy()
    value = str(threads)
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = value
    env["THINKORPLACE_NUM_THREADS"] = value
    env["THINKORPLACE_DEVICE"] = device
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _process_group_rss_bytes(pgid: int) -> Optional[int]:
    """Observed RSS of current process-group members, not a whole-tree claim."""
    try:
        result = subprocess.run(
            ["ps", "-axo", "pgid=,rss="], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
            timeout=1.0,
        )
        total_kib = 0
        found = False
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) >= 2 and int(fields[0]) == pgid:
                total_kib += int(fields[1])
                found = True
        return total_kib * 1024 if found else None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def supervise_process(
    command: Sequence[str], cwd: Path, env: Mapping[str, str], log_path: Path,
    timeout_seconds: float, terminate_grace_seconds: float = 2.0,
    poll_seconds: float = 0.1,
) -> Dict[str, Any]:
    """Run command in a new session and enforce a wall deadline on its group."""
    started = time.monotonic()
    deadline = started + timeout_seconds
    peak_rss: Optional[int] = None
    memory_samples = 0
    timed_out = False
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            list(command), cwd=str(cwd), env=dict(env), stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        while process.poll() is None:
            sample = _process_group_rss_bytes(process.pid)
            if sample is not None:
                peak_rss = sample if peak_rss is None else max(peak_rss, sample)
                memory_samples += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=terminate_grace_seconds)
                except subprocess.TimeoutExpired:
                    pass
                # The group leader may exit promptly while a descendant ignores
                # SIGTERM. Always remove any remaining group members.
                if _process_group_exists(process.pid):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()
                break
            time.sleep(min(poll_seconds, max(0.0, remaining)))
        return_code = process.wait()
        # A failed worker can exit while leaving a grandchild behind. A run is
        # complete only after its isolated process group has been cleaned up.
        if _process_group_exists(process.pid):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return {
        "status": "timeout" if timed_out else "exited",
        "return_code": return_code,
        "timed_out": timed_out,
        "wall_seconds": time.monotonic() - started,
        "timeout_seconds": timeout_seconds,
        "terminate_grace_seconds": terminate_grace_seconds,
        "peak_rss_bytes": peak_rss,
        "peak_memory_scope": (
            "observed_process_group_member_rss_sum" if peak_rss is not None
            else "unavailable"
        ),
        "memory_sample_count": memory_samples,
        "process_group_note": (
            "RSS is summed over processes observed in the worker process group; "
            "it is not asserted to cover descendants that changed process group."
        ),
    }


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def classify_operating_result(
    worker: Optional[Mapping[str, Any]], status: str, valid: bool, timed_out: bool,
) -> Dict[str, bool]:
    """Classify usability independently of whether every solver lane completed."""
    source_consistent = bool(worker and worker.get("source_consistent", False))
    environment_matches = bool(
        worker and worker.get("environment_matches_manifest", False)
    )
    completed_within_external_timeout = status == "completed" and not timed_out
    usable_operating_result = bool(
        valid and completed_within_external_timeout and source_consistent
    )
    return {
        "source_consistent": source_consistent,
        "environment_matches_manifest": environment_matches,
        "completed_within_external_timeout": completed_within_external_timeout,
        "usable_operating_result": usable_operating_result,
        "study_eligible": usable_operating_result and environment_matches,
    }


def execute_run(
    spec: Mapping[str, Any], output: Path, timeout_seconds: float,
    terminate_grace_seconds: float, retry_failures: bool,
) -> Dict[str, Any]:
    run_dir = output / "runs" / str(spec["run_id"])
    result_path = run_dir / "result.json"
    existing = _load_json(result_path)
    if existing and existing.get("identity_sha256") != spec["identity_sha256"]:
        raise RuntimeError("refusing to overwrite result with a different identity: %s" % result_path)
    if existing and existing.get("identity_sha256") == spec["identity_sha256"]:
        completed_valid = (
            existing.get("status") == "completed" and bool(existing.get("valid", False))
        )
        if completed_valid or not retry_failures:
            return {
                "run_id": spec["run_id"],
                "status": existing.get("status", "error"),
                "valid": bool(existing.get("valid", False)),
                "study_eligible": bool(existing.get("study_eligible", False)),
                "usable_operating_result": bool(
                    existing.get("usable_operating_result", False)
                ),
                "solver_outcome": existing.get("solver_outcome", "unavailable"),
                "pipeline_completed": bool(existing.get("pipeline_completed", False)),
                "used_fallback": existing.get("used_fallback"),
                "degraded_lanes": existing.get("degraded_lanes", []),
                "worker_error": existing.get("worker_error"),
                "phase_errors": existing.get("phase_errors", {}),
                "incomplete_phases": existing.get("incomplete_phases", []),
                "action": "resumed" if existing.get("status") == "completed" else "retained",
                "result_path": str(result_path),
            }

    run_dir.mkdir(parents=True, exist_ok=True)
    attempts = int(existing.get("attempt_count", 0)) + 1 if existing else 1
    checkpoints = run_dir / "checkpoints" / ("attempt-%03d" % attempts)
    checkpoints.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / ("stdout.log" if attempts == 1 else "stdout.attempt-%03d.log" % attempts)
    worker_result = run_dir / ("worker-result.attempt-%03d.json" % attempts)
    request_path = run_dir / "request.json"
    request = dict(spec)
    request.update({
        "attempt": attempts,
        "requested_at": utc_now(),
        "checkpoint_dir": str(checkpoints),
        "final_placement_path": str(run_dir / "final.pt"),
        "worker_result_path": str(worker_result),
    })
    atomic_write_json(request_path, request)

    env = thread_environment(
        int(spec["resource_allocation"]["threads_per_benchmark_run"]),
        str(spec["resource_allocation"]["device"]),
    )
    env["BENCH_ROOT"] = str(spec["benchmark"]["bench_root"])
    command = [sys.executable, str(WORKER_SCRIPT), "--request", str(request_path)]
    try:
        supervision = supervise_process(
            command, REPO_ROOT, env, log_path, timeout_seconds,
            terminate_grace_seconds=terminate_grace_seconds,
        )
    except Exception as exc:
        supervision = {
            "status": "supervisor_error", "return_code": None,
            "timed_out": False, "wall_seconds": None,
            "timeout_seconds": timeout_seconds,
            "terminate_grace_seconds": terminate_grace_seconds,
            "peak_rss_bytes": None, "peak_memory_scope": "unavailable",
            "memory_sample_count": 0, "error": repr(exc),
        }
    worker = _load_json(worker_result)
    if supervision["timed_out"]:
        status = "timeout"
        valid = False
        error = "external wall-clock timeout"
    elif supervision["status"] == "supervisor_error":
        status = "error"
        valid = False
        error = "failed to start or supervise worker: %s" % supervision.get("error")
    elif supervision["return_code"] != 0:
        status = "error"
        valid = False
        error = "worker exited with code %s" % supervision["return_code"]
    elif worker is None:
        status = "error"
        valid = False
        error = "worker did not produce readable atomic JSON"
    elif worker.get("identity_sha256") != spec["identity_sha256"]:
        status = "error"
        valid = False
        error = "worker result identity mismatch"
    else:
        status = str(worker.get("status", "error"))
        valid = status == "completed" and bool(worker.get("valid", False))
        error = worker.get("error")

    eligibility = classify_operating_result(
        worker, status, valid, bool(supervision["timed_out"])
    )
    completed_within_external_timeout = eligibility["completed_within_external_timeout"]
    source_consistent = eligibility["source_consistent"]
    usable_operating_result = eligibility["usable_operating_result"]
    study_eligible = eligibility["study_eligible"]
    worker_error = None
    if worker is not None and worker.get("status") == "error":
        worker_error = {
            key: worker.get(key) for key in ("error", "traceback")
            if worker.get(key) is not None
        }

    prior_attempts = list(existing.get("attempts", [])) if existing else []
    attempt_summary = {
        "attempt": attempts,
        "status": status,
        "valid": valid,
        "started_at": request["requested_at"],
        "finished_at": utc_now(),
        "log_path": str(log_path),
        "worker_result_path": str(worker_result),
        "checkpoint_dir": str(checkpoints),
        "supervisor": supervision,
        "error": error,
    }
    result: Dict[str, Any] = {
        "variant": spec["variant"],
        "benchmark": spec["benchmark"],
        "repository": spec["repository"],
        "source": spec["source"],
        "host": spec["host"],
        "resource_allocation": spec["resource_allocation"],
        "constructor_parameters": spec["constructor_parameters"],
        "nominal_seed": spec["nominal_seed"],
        "solver_outcome": "worker_did_not_complete",
        "pipeline_completed": False,
        "used_fallback": None,
        "phase_completion": None,
        "degraded_lanes": [],
        "lane_worker_errors": {},
        "phase_outcomes": None,
        "incomplete_phases": [],
        "phase_errors": {},
        "study_eligible": False,
        "usable_operating_result": False,
        "source_consistent": False,
        "completed_within_external_timeout": False,
        "worker_error": worker_error,
        "environment": {
            "worker_runtime": "unavailable because the worker did not complete",
            "supervisor_python_version": platform.python_version(),
            "supervisor_python_executable": sys.executable,
        },
    }
    result.update(worker or {})
    result.update({
        "schema_version": SCHEMA_VERSION,
        "run_id": spec["run_id"],
        "identity_sha256": spec["identity_sha256"],
        "status": status,
        "valid": valid,
        "error": error,
        "worker_error": worker_error,
        "source_consistent": source_consistent,
        "completed_within_external_timeout": completed_within_external_timeout,
        "usable_operating_result": usable_operating_result,
        "study_eligible": study_eligible,
        "attempt_count": attempts,
        "attempts": prior_attempts + [attempt_summary],
        "supervisor": supervision,
        "request_path": str(request_path),
        "stdout_log": str(log_path),
        "checkpoint_dir": str(checkpoints),
        "checkpoint_artifacts": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(checkpoints.rglob("*")) if path.is_file()
        ],
        "final_placement_path": (
            worker.get("final_placement_path") if worker is not None else None
        ),
    })
    atomic_write_json(result_path, result)
    return {
        "run_id": spec["run_id"], "status": status, "valid": valid,
        "study_eligible": bool(result.get("study_eligible", False)),
        "usable_operating_result": bool(
            result.get("usable_operating_result", False)
        ),
        "solver_outcome": result.get("solver_outcome", "unavailable"),
        "pipeline_completed": bool(result.get("pipeline_completed", False)),
        "used_fallback": result.get("used_fallback"),
        "degraded_lanes": result.get("degraded_lanes", []),
        "worker_error": result.get("worker_error"),
        "phase_errors": result.get("phase_errors", {}),
        "incomplete_phases": result.get("incomplete_phases", []),
        "action": "executed", "result_path": str(result_path),
    }


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("study config must be a JSON object")
    return value


def _pick(cli_value: Any, config: Mapping[str, Any], name: str, default: Any) -> Any:
    return cli_value if cli_value is not None else config.get(name, default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON study configuration")
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    parser.add_argument("--seeds", "--seed", nargs="+", type=int, default=None)
    parser.add_argument("--budget-seconds", type=float, default=None,
                        help="solver constructor budget per benchmark/variant run")
    parser.add_argument("--timeout-seconds", type=float, default=None,
                        help="hard external wall cap; defaults to budget-seconds")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="v2 smooth-placement steps")
    parser.add_argument("--cd-seconds", type=float, default=None,
                        help="v2 cd_polish_s constructor value")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=None)
    parser.add_argument("--threads", type=int, default=None,
                        help="total threads allocated to one benchmark run")
    parser.add_argument("--jobs", type=int, default=None,
                        help="concurrent benchmark workers; default 1")
    parser.add_argument("--output", default=None, help="study output directory")
    parser.add_argument("--placer-param", action="append", default=[],
                        metavar="[VARIANT:]NAME=JSON")
    parser.add_argument("--benchmark-source", action="append", default=[],
                        metavar="NAME=DIR")
    parser.add_argument("--terminate-grace-seconds", type=float, default=None)
    parser.add_argument("--retry-failures", action="store_true", default=None)
    parser.add_argument(
        "--fail-on-degraded", action="store_true", default=None,
        help="return nonzero when a valid result used checkpoint recovery",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _load_config(args.config)
        benchmarks = list(_pick(args.benchmarks, config, "benchmarks", []))
        variants = list(_pick(args.variants, config, "variants", list(VARIANTS)))
        seeds = [int(seed) for seed in _pick(args.seeds, config, "seeds", [42])]
        budget_seconds = float(_pick(args.budget_seconds, config, "budget_seconds", 3600.0))
        timeout_raw = _pick(args.timeout_seconds, config, "timeout_seconds", budget_seconds)
        timeout_seconds = float(timeout_raw)
        num_steps_raw = _pick(args.num_steps, config, "num_steps", None)
        num_steps = int(num_steps_raw) if num_steps_raw is not None else None
        cd_raw = _pick(args.cd_seconds, config, "cd_seconds", None)
        cd_seconds = float(cd_raw) if cd_raw is not None else None
        device = str(_pick(args.device, config, "device", "cpu"))
        threads = int(_pick(args.threads, config, "threads", 1))
        jobs = int(_pick(args.jobs, config, "jobs", 1))
        output_value = _pick(args.output, config, "output", None)
        grace = float(_pick(
            args.terminate_grace_seconds, config, "terminate_grace_seconds", 2.0
        ))
        retry_failures = bool(_pick(args.retry_failures, config, "retry_failures", False))
        fail_on_degraded = bool(_pick(
            args.fail_on_degraded, config, "fail_on_degraded", False
        ))
        if not benchmarks:
            raise ValueError("at least one --benchmarks value is required")
        if not output_value:
            raise ValueError("--output is required")
        if importlib.util.find_spec("torch") is None:
            raise ValueError(
                "the selected Python interpreter has no torch; run with .venv/bin/python"
            )
        if budget_seconds <= 0 or timeout_seconds <= 0 or threads < 1 or jobs < 1 or grace < 0:
            raise ValueError("budget/timeout must be >0; threads/jobs >=1; grace >=0")
        if any(variant not in VARIANTS for variant in variants):
            raise ValueError("unknown variant in %r" % variants)
        if "v1" in variants and device != "cpu":
            raise ValueError("v1 supports --device cpu only")
        if "v2-abc" in variants and threads < 3:
            raise ValueError("v2-abc requires --threads 3 or greater (one per lane)")

        source_pairs = list(config.get("benchmark_sources", {}).items())
        for item in args.benchmark_source:
            if "=" not in item:
                raise ValueError("benchmark source must be NAME=DIR: %s" % item)
            source_pairs.append(tuple(item.split("=", 1)))
        source_overrides = {str(k): str(v) for k, v in source_pairs}

        file_params = config.get("variant_parameters", {})
        if not isinstance(file_params, dict):
            raise ValueError("variant_parameters must be an object")
        extra = {variant: dict(file_params.get(variant, {})) for variant in variants}
        cli_params = parse_key_values(args.placer_param, variants)
        for variant in variants:
            extra[variant].update(cli_params[variant])
            reserved = sorted(
                set(extra[variant]).intersection(
                    {"checkpoint_dir", "device", "num_threads", "lanes"}
                )
            )
            if reserved:
                raise ValueError(
                    "%s parameter(s) are controlled by runner options: %s"
                    % (variant, ", ".join(reserved))
                )

        output = Path(str(output_value)).resolve()
        output.mkdir(parents=True, exist_ok=True)
        repository = repository_provenance()
        host = host_provenance()
        benchmark_records = [resolve_benchmark(name, source_overrides) for name in benchmarks]
        source_records = {variant: source_provenance(variant) for variant in variants}
        specs: List[Dict[str, Any]] = []
        for benchmark in benchmark_records:
            for variant in variants:
                for seed in seeds:
                    constructor = build_constructor_parameters(
                        variant, seed, budget_seconds, num_steps, cd_seconds,
                        device, threads, extra[variant],
                    )
                    constructor["checkpoint_dir"] = "<per-run-checkpoint-dir>"
                    specs.append(make_run_spec(
                        benchmark, variant, seed, constructor, device, threads,
                        jobs, timeout_seconds, source_records[variant], repository, host,
                    ))

        invocation = {
            "config_path": str(Path(args.config).resolve()) if args.config else None,
            "argv": [sys.executable, str(Path(__file__).resolve())] + list(argv or sys.argv[1:]),
            "benchmarks": benchmarks, "variants": variants, "seeds": seeds,
            "budget_seconds": budget_seconds, "num_steps": num_steps,
            "timeout_seconds": timeout_seconds,
            "cd_seconds": cd_seconds, "device": device, "threads": threads,
            "jobs": jobs, "output": str(output),
            "retry_failures": retry_failures,
            "fail_on_degraded": fail_on_degraded,
            "study_kind": config.get("study_kind"),
            "study_notes": config.get("study_notes"),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "study_id": output.name,
            "generated_at": utc_now(),
            "repository": repository,
            "host": host,
            "invocation": invocation,
            "run_count": len(specs),
            "runs": [
                {"run_id": spec["run_id"], "identity_sha256": spec["identity_sha256"],
                 "benchmark": spec["benchmark"]["requested_name"],
                 "variant": spec["variant"], "seed": spec["nominal_seed"]}
                for spec in specs
            ],
        }
        atomic_write_json(output / "manifest.json", manifest)
        if args.dry_run:
            print("planned %d runs in %s" % (len(specs), output))
            return 0

        results: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            future_map = {
                pool.submit(
                    execute_run, spec, output, timeout_seconds, grace,
                    retry_failures,
                ): spec for spec in specs
            }
            for future in concurrent.futures.as_completed(future_map):
                spec = future_map[future]
                try:
                    item = future.result()
                except Exception as exc:  # retain supervisor failures in the summary
                    item = {
                        "run_id": spec["run_id"], "status": "error", "valid": False,
                        "action": "supervisor_error", "error": repr(exc),
                        "result_path": str(output / "runs" / spec["run_id"] / "result.json"),
                    }
                results.append(item)
                print("%-9s valid=%-5s eligible=%-5s outcome=%s %s (%s)" % (
                    item["status"], str(item["valid"]),
                    str(item.get("study_eligible", False)),
                    item.get("solver_outcome", "unavailable"),
                    item["run_id"], item["action"]
                ), flush=True)
                atomic_write_json(output / "summary.json", {
                    "schema_version": SCHEMA_VERSION, "updated_at": utc_now(),
                    "manifest_path": str(output / "manifest.json"),
                    "planned_run_count": len(specs), "finished_run_count": len(results),
                    "completed_run_count": sum(
                        row["status"] == "completed" for row in results
                    ),
                    "valid_run_count": sum(bool(row["valid"]) for row in results),
                    "study_eligible_run_count": sum(
                        bool(row.get("study_eligible", False)) for row in results
                    ),
                    "usable_operating_result_count": sum(
                        bool(row.get("usable_operating_result", False))
                        for row in results
                    ),
                    "pipeline_completed_run_count": sum(
                        bool(row.get("pipeline_completed", False)) for row in results
                    ),
                    "degraded_run_count": sum(
                        not bool(row.get("pipeline_completed", False)) for row in results
                    ),
                    "runs": sorted(results, key=lambda row: row["run_id"]),
                })

        failed = sum(1 for result in results if result["status"] != "completed")
        invalid = sum(1 for result in results if not result["valid"])
        ineligible = sum(1 for result in results if not result.get("study_eligible", False))
        print("finished %d runs: %d non-completed, %d invalid, %d study-ineligible" % (
            len(results), failed, invalid, ineligible
        ))
        degraded = sum(
            1 for result in results if not result.get("pipeline_completed", False)
        )
        return 0 if (
            failed == 0 and invalid == 0 and ineligible == 0
            and (not fail_on_degraded or degraded == 0)
        ) else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("publication study configuration error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
