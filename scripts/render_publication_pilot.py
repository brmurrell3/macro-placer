#!/usr/bin/env python3
"""Render a fixed 12-run publication validation profile without running a solver."""
from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_STUDY = ROOT / "writeup/runs/pilot-20260906-final"
PRIMARY_JSON = ROOT / "writeup/data/publication/pilot_results.json"
PRIMARY_TEX = ROOT / "writeup/latex/generated/pilot_results.tex"
PROFILES = {
    "primary": {
        "study_dir": PRIMARY_STUDY,
        "json_output": PRIMARY_JSON,
        "tex_output": PRIMARY_TEX,
        "study_kind": "limited-budget-pilot",
        "study_notes": (
            "Twelve-run readiness pilot; the 180-second solver budget is not the "
            "publication study budget."
        ),
        "jobs": 1,
        "fail_on_degraded": True,
        "classification": "limited_budget_engineering_pilot",
        "title": "Limited-budget pilot",
        "label_prefix": "pilot",
        "warning": None,
    },
    "repair": {
        "study_dir": ROOT / "writeup/runs/repair-20260906",
        "json_output": ROOT / "writeup/data/publication/repair_results.json",
        "tex_output": ROOT / "writeup/latex/generated/repair_results.tex",
        "study_kind": "post-pilot-repair-validation",
        "study_notes": (
            "engineering only, concurrent jobs; not pooled with primary or causal "
            "quality comparison"
        ),
        "jobs": 3,
        "fail_on_degraded": False,
        "classification": "post_pilot_repair_validation_engineering_evidence",
        "title": "Post-pilot repair validation",
        "label_prefix": "repair-validation",
        "warning": (
            "Separate concurrent engineering validation; not pooled with the primary "
            "pilot and not a causal quality comparison."
        ),
    },
}
EXPECTED_RUNS = 12
EXPECTED_BENCHMARKS = ("ibm01", "ibm10", "ibm17", "ariane133_ng45")
EXPECTED_VARIANTS = ("v1", "v2-a", "v2-abc")
EXPECTED_SEEDS = (42,)
PREFIX_PHASES = ("initializer", "descent_legalized")
ABC_LANES = ((0, "A"), (1, "B"), (2, "C"))
# Later phases rank higher when several saved checkpoints share the returned coordinates.
V2_PHASE_ORDER = (
    "initializer", "descent_legalized", "cd1", "post_saddle", "post_hungarian", "cd2", "final",
)
V1_PHASE_ORDER = ("e25", "e41", "cascade", "portfolio", "periphery")
PHASE_LABELS = {
    "initializer": "init.", "descent_legalized": "descent", "cd1": "CD1",
    "post_saddle": "curvature", "post_hungarian": "assignment", "cd2": "CD2",
    "final": "final", "e25": "E25", "e41": "E41", "cascade": "cascade",
    "portfolio": "portfolio", "periphery": "periphery",
}


def returned_state(result: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Identify the saved lane/phase whose coordinates equal the returned placement.

    Uses only hashes already recorded in the finalized result: the returned
    tensor hash and the checkpoint inventory. Returns None when no match exists.
    """
    final_hash = result.get("final_placement_tensor_sha256")
    inventory = result.get("checkpoint_inventory")
    if not isinstance(final_hash, str) or not isinstance(inventory, list):
        return None
    matches = [
        (item.get("lane_index"), item.get("phase")) for item in inventory
        if isinstance(item, Mapping) and item.get("coordinates_sha256") == final_hash
    ]
    if result.get("variant") == "v1":
        phases = [phase for _, phase in matches if phase in V1_PHASE_ORDER]
        if not phases:
            return None
        return {"lanes": [], "phase": max(phases, key=V1_PHASE_ORDER.index)}
    latest: Dict[str, str] = {}
    for lane_index, phase in matches:
        if phase not in V2_PHASE_ORDER or not isinstance(lane_index, int):
            continue
        lane = dict(ABC_LANES).get(lane_index)
        if lane is None:
            continue
        if lane not in latest or V2_PHASE_ORDER.index(phase) > V2_PHASE_ORDER.index(latest[lane]):
            latest[lane] = phase
    if not latest:
        return None
    best = max(V2_PHASE_ORDER.index(phase) for phase in latest.values())
    lanes = sorted(lane for lane, phase in latest.items() if V2_PHASE_ORDER.index(phase) == best)
    return {"lanes": lanes, "phase": V2_PHASE_ORDER[best]}


def threads_per_lane(result: Mapping[str, Any]) -> Optional[int]:
    """Recorded CPU threads per solver process: lane records for v2, constructor for v1."""
    metadata = result.get("solver_metadata")
    lanes = metadata.get("lane_results") if isinstance(metadata, Mapping) else None
    if isinstance(lanes, list) and lanes:
        values = {lane.get("num_threads") for lane in lanes if isinstance(lane, Mapping)}
        if len(values) == 1 and isinstance(next(iter(values)), int):
            return int(next(iter(values)))
        return None
    parameters = result.get("constructor_parameters")
    if result.get("variant") == "v1" and isinstance(parameters, Mapping):
        value = parameters.get("num_threads")
        return int(value) if isinstance(value, int) else None
    return None


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON in %s: %s" % (path, exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object in %s" % path)
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def planned_rows(
    manifest: Mapping[str, Any], profile: Optional[Mapping[str, Any]] = None
) -> List[Dict[str, Any]]:
    profile = PROFILES["primary"] if profile is None else profile
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise ValueError("manifest.runs must be a list")
    if int(manifest.get("run_count", -1)) != EXPECTED_RUNS or len(runs) != EXPECTED_RUNS:
        raise ValueError(
            "this renderer requires the fixed %d-run profile manifest; found run_count=%r and %d entries"
            % (EXPECTED_RUNS, manifest.get("run_count"), len(runs))
        )
    normalized = []
    seen_ids = set()
    seen_cells = set()
    for item in runs:
        if not isinstance(item, Mapping):
            raise ValueError("every manifest run must be an object")
        row = {
            "run_id": str(item.get("run_id", "")),
            "identity_sha256": str(item.get("identity_sha256", "")),
            "benchmark": str(item.get("benchmark", "")),
            "variant": str(item.get("variant", "")),
            "nominal_seed": int(item.get("seed")),
        }
        cell = (row["benchmark"], row["variant"], row["nominal_seed"])
        if not row["run_id"] or row["run_id"] in seen_ids:
            raise ValueError("manifest contains a missing or duplicate run_id")
        if cell in seen_cells:
            raise ValueError("manifest contains a duplicate benchmark/variant/seed cell: %r" % (cell,))
        if len(row["identity_sha256"]) != 64:
            raise ValueError("manifest identity is not a SHA-256 for %s" % row["run_id"])
        seen_ids.add(row["run_id"])
        seen_cells.add(cell)
        normalized.append(row)
    expected_cells = {
        (benchmark, variant, seed)
        for benchmark in EXPECTED_BENCHMARKS
        for variant in EXPECTED_VARIANTS
        for seed in EXPECTED_SEEDS
    }
    if seen_cells != expected_cells:
        raise ValueError("manifest does not contain the fixed 12-run benchmark matrix")
    invocation = manifest.get("invocation")
    if not isinstance(invocation, Mapping):
        raise ValueError("manifest.invocation must be an object")
    expected_settings = {
        "budget_seconds": 180.0, "timeout_seconds": 240.0,
        "num_steps": 500, "cd_seconds": 30.0,
        "device": "cpu", "threads": 3, "jobs": profile["jobs"],
        "study_kind": profile["study_kind"],
        "study_notes": profile["study_notes"],
        "fail_on_degraded": profile["fail_on_degraded"],
        "benchmarks": list(EXPECTED_BENCHMARKS),
        "variants": list(EXPECTED_VARIANTS),
        "seeds": list(EXPECTED_SEEDS),
    }
    mismatches = {
        name: {"expected": expected, "actual": invocation.get(name)}
        for name, expected in expected_settings.items()
        if invocation.get(name) != expected
    }
    if mismatches:
        raise ValueError("manifest profile settings mismatch: %r" % mismatches)
    return normalized


def result_row(plan: Mapping[str, Any], result_path: Path) -> Dict[str, Any]:
    result = load_object(result_path)
    if result.get("run_id") != plan["run_id"]:
        raise ValueError("result run_id mismatch in %s" % result_path)
    if result.get("identity_sha256") != plan["identity_sha256"]:
        raise ValueError("result identity mismatch in %s" % result_path)
    benchmark = result.get("benchmark", {})
    if not isinstance(benchmark, Mapping) or benchmark.get("requested_name") != plan["benchmark"]:
        raise ValueError("result benchmark mismatch in %s" % result_path)
    if result.get("variant") != plan["variant"]:
        raise ValueError("result variant mismatch in %s" % result_path)
    if int(result.get("nominal_seed")) != plan["nominal_seed"]:
        raise ValueError("result seed mismatch in %s" % result_path)

    cost = result.get("canonical_cost")
    cost = cost if isinstance(cost, Mapping) else {}
    timing = result.get("timing")
    timing = timing if isinstance(timing, Mapping) else {}
    supervisor = result.get("supervisor")
    supervisor = supervisor if isinstance(supervisor, Mapping) else {}
    legality = result.get("legality")
    legality = legality if isinstance(legality, Mapping) else {}
    overlap = legality.get("hard_macro_overlap")
    overlap = overlap if isinstance(overlap, Mapping) else {}
    resources = result.get("resource_allocation")
    resources = resources if isinstance(resources, Mapping) else {}

    row = dict(plan)
    row.update({
        "result_record": relative(result_path),
        "result_record_sha256": sha256_file(result_path),
        "status": str(result.get("status", "unknown")),
        "valid": bool(result.get("valid", False)),
        "usable_operating_result": bool(result.get("usable_operating_result", False)),
        "study_eligible": bool(result.get("study_eligible", False)),
        "pipeline_completed": bool(result.get("pipeline_completed", False)),
        "solver_outcome": str(result.get("solver_outcome", "unknown")),
        "used_fallback": result.get("used_fallback"),
        "degraded_lanes": result.get("degraded_lanes", []),
        "incomplete_phases": result.get("incomplete_phases", []),
        "phase_outcomes": result.get("phase_outcomes"),
        "phase_errors": result.get("phase_errors", {}),
        "lane_worker_errors": result.get("lane_worker_errors", {}),
        "worker_error": result.get("worker_error"),
        "error": result.get("error"),
        "canonical_cost": {
            name: finite_number(cost.get(name)) for name in (
                "proxy_cost", "wirelength_cost", "density_cost", "congestion_cost"
            )
        },
        "timing": {
            "solver_place_seconds": finite_number(timing.get("solver_place_seconds")),
            "worker_total_seconds": finite_number(
                timing.get("worker_through_environment_capture_seconds")
            ),
            "supervisor_wall_seconds": finite_number(supervisor.get("wall_seconds")),
            "solver_budget_seconds": finite_number(timing.get("solver_budget_seconds")),
            "external_timeout_seconds": finite_number(
                resources.get("external_timeout_seconds")
            ),
        },
        "legality": {
            "full_legal": bool(legality.get("full_legal", False)),
            "overlap_count": overlap.get("overlap_count"),
            "full_violations": legality.get("full_violations", []),
        },
        "peak_rss_bytes": supervisor.get("peak_rss_bytes"),
        "checkpoint_inventory": result.get("checkpoint_inventory", []),
        "returned_state": returned_state(result) if result.get("valid") else None,
        "threads_per_lane": threads_per_lane(result),
        "seed_provenance": result.get("seeds"),
        "constructor_parameters": result.get("constructor_parameters"),
        "source_sha256": (
            result.get("source", {}).get("variant_source_sha256")
            if isinstance(result.get("source"), Mapping) else None
        ),
    })
    # Pipeline status is a separate engineering outcome. Eligibility may remain
    # true for a valid partial or recovered operating result.
    if row["pipeline_completed"] and row["solver_outcome"] != "pipeline_completed":
        raise ValueError("contradictory pipeline outcome in %s" % result_path)
    return row


def missing_row(plan: Mapping[str, Any], result_path: Path) -> Dict[str, Any]:
    row = dict(plan)
    row.update({
        "result_record": relative(result_path), "result_record_sha256": None,
        "status": "missing", "valid": False, "usable_operating_result": False,
        "study_eligible": False, "pipeline_completed": False,
        "solver_outcome": "not_yet_recorded", "used_fallback": None,
        "degraded_lanes": [], "incomplete_phases": [], "phase_outcomes": None,
        "phase_errors": {},
        "lane_worker_errors": {}, "worker_error": None, "error": "result.json is missing",
        "canonical_cost": {name: None for name in (
            "proxy_cost", "wirelength_cost", "density_cost", "congestion_cost"
        )},
        "timing": {name: None for name in (
            "solver_place_seconds", "worker_total_seconds", "supervisor_wall_seconds",
            "solver_budget_seconds", "external_timeout_seconds"
        )},
        "legality": {"full_legal": False, "overlap_count": None, "full_violations": []},
        "peak_rss_bytes": None, "checkpoint_inventory": [],
        "returned_state": None, "threads_per_lane": None,
        "seed_provenance": None, "constructor_parameters": None,
        "source_sha256": None,
    })
    return row


def aggregate(rows: Sequence[Mapping[str, Any]], variants: Sequence[str]) -> Dict[str, Any]:
    result = {}
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        proxies = [row["canonical_cost"]["proxy_cost"] for row in subset]
        full_proxy_coverage = bool(subset) and all(
            row["study_eligible"] and proxy is not None
            for row, proxy in zip(subset, proxies)
        )
        all_walls = [row["timing"]["supervisor_wall_seconds"] for row in subset]
        full_wall_coverage = bool(subset) and all(value is not None for value in all_walls)
        result[variant] = {
            "planned_run_count": len(subset),
            "recorded_run_count": sum(row["status"] != "missing" for row in subset),
            "completed_run_count": sum(row["status"] == "completed" for row in subset),
            "valid_run_count": sum(bool(row["valid"]) for row in subset),
            "usable_operating_result_count": sum(
                bool(row["usable_operating_result"]) for row in subset
            ),
            "study_eligible_run_count": sum(bool(row["study_eligible"]) for row in subset),
            "eligible_noncompleted_operating_result_count": sum(
                bool(row["study_eligible"]) and not bool(row["pipeline_completed"])
                for row in subset
            ),
            "pipeline_completed_run_count": sum(bool(row["pipeline_completed"]) for row in subset),
            "partial_phase_completion_count": sum(
                row["solver_outcome"] == "partial_phase_completion" for row in subset
            ),
            "checkpoint_recovery_count": sum(
                row["solver_outcome"] == "checkpoint_recovery" for row in subset
            ),
            "error_count": sum(row["status"] == "error" for row in subset),
            "timeout_count": sum(row["status"] == "timeout" for row in subset),
            "missing_count": sum(row["status"] == "missing" for row in subset),
            "matched_pilot_mean_proxy": (
                sum(proxies) / len(proxies) if full_proxy_coverage else None
            ),
            "matched_pilot_mean_proxy_available": full_proxy_coverage,
            "matched_pilot_mean_proxy_unavailable_reason": (
                None if full_proxy_coverage
                else "requires an eligible finite proxy for every planned benchmark"
            ),
            "all_run_mean_supervisor_wall_seconds": (
                sum(all_walls) / len(all_walls) if full_wall_coverage else None
            ),
        }
    return result


def checkpoint_phase_hashes(row: Mapping[str, Any]) -> Dict[tuple, str]:
    """Return one coordinate hash per lane/phase, validating checkpoint aliases."""
    inventory = row.get("checkpoint_inventory", [])
    if not isinstance(inventory, list):
        raise ValueError("checkpoint_inventory must be a list for %s" % row["run_id"])
    candidates: Dict[tuple, set] = {}
    for item in inventory:
        if not isinstance(item, Mapping):
            raise ValueError("checkpoint inventory entry must be an object for %s" % row["run_id"])
        phase = item.get("phase")
        lane_index = item.get("lane_index")
        coordinate_hash = item.get("coordinates_sha256")
        if phase not in PREFIX_PHASES or not isinstance(lane_index, int):
            continue
        if not isinstance(coordinate_hash, str) or len(coordinate_hash) != 64:
            continue
        candidates.setdefault((lane_index, phase), set()).add(coordinate_hash)
    hashes = {}
    for key, values in candidates.items():
        if len(values) != 1:
            raise ValueError(
                "conflicting coordinate hashes for lane %d phase %s in %s"
                % (key[0], key[1], row["run_id"])
            )
        hashes[key] = next(iter(values))
    return hashes


def _configured_lane_seeds(row: Optional[Mapping[str, Any]]) -> Dict[str, Optional[int]]:
    if row is None:
        return {lane: None for _, lane in ABC_LANES}
    parameters = row.get("constructor_parameters")
    parameters = parameters if isinstance(parameters, Mapping) else {}
    return {
        lane: (
            int(parameters["lane_%s_seed" % lane.lower()])
            if isinstance(parameters.get("lane_%s_seed" % lane.lower()), int)
            else None
        )
        for _, lane in ABC_LANES
    }


def _effective_initializer_seeds(row: Optional[Mapping[str, Any]]) -> Dict[str, Optional[int]]:
    values = {lane: None for _, lane in ABC_LANES}
    if row is None:
        return values
    inventory = row.get("checkpoint_inventory", [])
    if not isinstance(inventory, list):
        return values
    for item in inventory:
        if not isinstance(item, Mapping) or item.get("phase") != "initializer":
            continue
        lane_index = item.get("lane_index")
        effective_seed = item.get("effective_rng_seed")
        for expected_index, lane in ABC_LANES:
            if lane_index == expected_index and isinstance(effective_seed, int):
                if values[lane] not in (None, effective_seed):
                    raise ValueError(
                        "conflicting effective initializer seeds for lane %s in %s"
                        % (lane, row["run_id"])
                    )
                values[lane] = effective_seed
    return values


def prefix_diversity(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize measured v2 prefix identity without inferring search diversity."""
    by_cell = {(row["benchmark"], row["variant"]): row for row in rows}
    benchmark_rows = []
    all_three_cells = 0
    standalone_comparison_cells = 0
    missing_cells = 0
    for benchmark in EXPECTED_BENCHMARKS:
        abc = by_cell.get((benchmark, "v2-abc"))
        standalone = by_cell.get((benchmark, "v2-a"))
        abc_hashes = checkpoint_phase_hashes(abc) if abc is not None else {}
        standalone_hashes = checkpoint_phase_hashes(standalone) if standalone is not None else {}
        phases = {}
        for phase in PREFIX_PHASES:
            lane_hashes = {
                lane: abc_hashes.get((lane_index, phase))
                for lane_index, lane in ABC_LANES
            }
            available = [lane for _, lane in ABC_LANES if lane_hashes[lane] is not None]
            missing = [lane for _, lane in ABC_LANES if lane_hashes[lane] is None]
            all_three = len(available) == len(ABC_LANES)
            standalone_hash = standalone_hashes.get((0, phase))
            comparisons = {
                lane: (
                    lane_hashes[lane] == standalone_hash
                    if lane_hashes[lane] is not None and standalone_hash is not None
                    else None
                )
                for _, lane in ABC_LANES
            }
            comparison_available = standalone_hash is not None and bool(available)
            if all_three:
                all_three_cells += 1
            if comparison_available:
                standalone_comparison_cells += 1
            if not all_three or standalone_hash is None:
                missing_cells += 1
            phases[phase] = {
                "abc_lane_coordinate_sha256": lane_hashes,
                "abc_available_lanes": available,
                "abc_missing_lanes": missing,
                "abc_all_three_available": all_three,
                "abc_distinct_coordinate_hash_count": (
                    len(set(lane_hashes.values())) if all_three else None
                ),
                "abc_all_three_equal": (
                    len(set(lane_hashes.values())) == 1 if all_three else None
                ),
                "standalone_a_coordinate_sha256": standalone_hash,
                "standalone_a_available": standalone_hash is not None,
                "abc_lane_equal_to_standalone_a": comparisons,
                "comparison_available": comparison_available,
                "all_three_equal_to_standalone_a": (
                    all(comparisons.values())
                    if all_three and standalone_hash is not None else None
                ),
            }
        benchmark_rows.append({
            "benchmark": benchmark,
            "abc_run_id": abc.get("run_id") if abc is not None else None,
            "standalone_a_run_id": standalone.get("run_id") if standalone is not None else None,
            "configured_nominal_lane_seeds": _configured_lane_seeds(abc),
            "measured_effective_initializer_seeds": _effective_initializer_seeds(abc),
            "phases": phases,
        })
    total_cells = len(EXPECTED_BENCHMARKS) * len(PREFIX_PHASES)
    return {
        "classification": "measured_checkpoint_coordinate_hash_equality",
        "checkpoint_phases": list(PREFIX_PHASES),
        "alias_handling": (
            "Checkpoint aliases such as laneN.pkl and laneN_PHASE.pkl are collapsed "
            "by lane, phase, and coordinate hash; conflicting hashes fail rendering."
        ),
        "interpretation": (
            "Exact coordinate-hash equality is reported only from checkpoint inventories "
            "embedded in finalized result metadata. Missing metadata does not prove that "
            "no physical checkpoint file was written; a timed-out worker may leave files "
            "that were not inventoried because result finalization did not finish. "
            "Different nominal lane seeds do not establish independent basin starts; "
            "effective initializer seeds are reported where recorded. Equality can "
            "motivate a future shared-prefix implementation, but no speedup is measured "
            "or claimed here."
        ),
        "coverage": {
            "benchmark_phase_cell_count": total_cells,
            "abc_all_three_available_cell_count": all_three_cells,
            "standalone_comparison_available_cell_count": standalone_comparison_cells,
            "cells_with_missing_abc_or_standalone_coverage": missing_cells,
        },
        "benchmarks": benchmark_rows,
    }


def latex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def format_number(value: Any, digits: int) -> str:
    number = finite_number(value)
    return "---" if number is None else ("%.*f" % (digits, number))


def display_design(value: Any) -> str:
    name = str(value).replace("_ng45", "")
    return name.upper() if name.startswith("ibm") else name


def display_variant(value: Any) -> str:
    return {"v1": "v1", "v2-a": "A", "v2-abc": "A+B+C"}.get(str(value), str(value))


def outcome_code(row: Mapping[str, Any]) -> str:
    status = row["status"]
    if status == "missing":
        return "---"
    if status == "timeout":
        return "T"
    if status == "error":
        return "E"
    if not row["valid"]:
        return "I"
    if row["pipeline_completed"]:
        code = "C"
    elif row["solver_outcome"] == "checkpoint_recovery":
        code = "R"
    else:
        code = "P"
    return code if row["study_eligible"] else code + r"$^{\dagger}$"


def returned_label(row: Mapping[str, Any]) -> str:
    state = row.get("returned_state")
    if not isinstance(state, Mapping):
        return "---"
    phase = PHASE_LABELS.get(str(state.get("phase")), latex_escape(state.get("phase")))
    lanes = "".join(state.get("lanes") or [])
    return phase if not lanes else "%s/%s" % (lanes, phase)


def threads_label(row: Mapping[str, Any]) -> str:
    value = row.get("threads_per_lane")
    return "---" if not isinstance(value, int) else str(value)


def coverage_sentence(summary: Mapping[str, Any]) -> str:
    parts = []
    for variant in summary["variants"]:
        item = summary["aggregate_by_variant"][variant]
        parts.append(
            "%s %d/%d eligible, %d/%d completed"
            % (display_variant(variant), item["study_eligible_run_count"],
               item["planned_run_count"], item["pipeline_completed_run_count"],
               item["planned_run_count"])
        )
    available = [
        variant for variant in summary["variants"]
        if summary["aggregate_by_variant"][variant]["matched_pilot_mean_proxy_available"]
    ]
    tail = (
        " No variant has an eligible result on every planned design, so no mean is reported."
        if not available else
        " Means are reported only for variants with an eligible result on every planned design."
    )
    return "Coverage by variant: " + "; ".join(parts) + "." + tail


def prefix_sentence(summary: Mapping[str, Any]) -> str:
    """One sentence replacing the former coordinate-hash table."""
    equal, differing = [], []
    for benchmark in summary["prefix_diversity"]["benchmarks"]:
        design = benchmark["benchmark"].replace("_ng45", "").upper().replace("ARIANE", "ariane")
        for phase, label in (("initializer", "initializer"), ("descent_legalized", "legalized descent")):
            item = benchmark["phases"][phase]
            if not item["abc_all_three_available"] or item["standalone_a_coordinate_sha256"] is None:
                continue
            target = equal if item["all_three_equal_to_standalone_a"] else differing
            target.append("%s %s" % (design, label))
    if not equal and not differing:
        return ""
    sentence = (
        "Where initializer or legalized-descent checkpoints survive for all three A+B+C lanes "
        "and for standalone A (%s), " % ", ".join(equal + differing)
    )
    if not differing:
        sentence += (
            "the four coordinate tensors are bit-identical; the nominal lane seeds therefore "
            "did not diversify the shared prefix. Full SHA-256 values are in the JSON summary."
        )
    else:
        sentence += "coordinates differ for %s; full SHA-256 values are in the JSON summary." % (
            ", ".join(differing))
    return sentence


def render_latex(summary: Mapping[str, Any]) -> str:
    rows = summary["runs"]
    title = summary["profile_title"]
    label_prefix = summary["latex_label_prefix"]
    context = summary["resource_context"]
    observed_codes = {outcome_code(row)[0] for row in rows}
    descriptions = {"C": "completed pipeline", "P": "valid result after partial phases",
                    "R": "valid checkpoint recovery", "E": "error", "T": "external timeout",
                    "I": "invalid completed result"}
    legend = "; ".join(code + ": " + meaning for code, meaning in descriptions.items()
                       if code in observed_codes) + "."
    if any(row["valid"] and not row["study_eligible"] for row in rows):
        legend += " A dagger marks a valid result excluded by eligibility checks."
    scope = (
        "%.0f\\,s solver budget, %.0f\\,s external limit, %d concurrent job%s, %d CPU threads per run"
        % (context["budget_seconds"], context["timeout_seconds"], context["jobs"],
           "" if context["jobs"] == 1 else "s", context["threads"])
    )
    lines = [
        "% Generated by scripts/render_publication_pilot.py; do not edit.",
        "% Limited-budget engineering evidence; no official-budget extrapolation.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{%s on four designs (%s). %s Returned: lane and phase whose saved "
        r"coordinates equal the returned placement. Threads: CPU threads per solver process "
        r"as recorded. Every planned run is shown.}" % (latex_escape(title), scope, legend),
        r"\label{tab:%s-results}" % label_prefix,
        r"\begin{tabular}{@{}llrrrclr@{}}",
        r"\toprule",
        r"Design & Variant & Proxy & Solver (s) & Wall (s) & Outcome & Returned & Threads \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join([
            latex_escape(display_design(row["benchmark"])),
            latex_escape(display_variant(row["variant"])),
            format_number(row["canonical_cost"]["proxy_cost"], 5),
            format_number(row["timing"]["solver_place_seconds"], 1),
            format_number(row["timing"]["supervisor_wall_seconds"], 1),
            outcome_code(row),
            returned_label(row),
            threads_label(row),
        ]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    lines.append(coverage_sentence(summary) + " " + prefix_sentence(summary))
    lines.append("")
    if summary["artifact_status"] != "final_complete":
        lines.insert(2, r"\textbf{Interim artifact: the 12-run profile is incomplete.}")
    return "\n".join(lines)


def build_summary(
    study_dir: Path, allow_incomplete: bool, profile_name: str = "primary"
) -> Dict[str, Any]:
    profile = PROFILES[profile_name]
    manifest_path = study_dir / "manifest.json"
    manifest = load_object(manifest_path)
    plans = planned_rows(manifest, profile)
    expected_paths = {
        (study_dir / "runs" / plan["run_id"] / "result.json").resolve()
        for plan in plans
    }
    actual_paths = {path.resolve() for path in (study_dir / "runs").glob("*/result.json")}
    unexpected = sorted(actual_paths - expected_paths, key=str)
    if unexpected:
        raise ValueError("unplanned result records found: %s" % ", ".join(map(str, unexpected)))
    missing = [path for path in sorted(expected_paths, key=str) if not path.is_file()]
    if missing and not allow_incomplete:
        raise ValueError(
            "%s profile is incomplete: %d/%d result.json records exist; missing %d. "
            "Use --allow-incomplete only for a clearly labelled interim artifact."
            % (profile_name, len(actual_paths), EXPECTED_RUNS, len(missing))
        )
    rows = []
    for plan in plans:
        path = study_dir / "runs" / plan["run_id"] / "result.json"
        rows.append(result_row(plan, path) if path.is_file() else missing_row(plan, path))
    variants = list(dict.fromkeys(str(row["variant"]) for row in rows))
    artifact_status = "final_complete" if not missing else "interim_incomplete"
    coverage = {
        "planned_run_count": EXPECTED_RUNS,
        "recorded_run_count": sum(row["status"] != "missing" for row in rows),
        "completed_run_count": sum(row["status"] == "completed" for row in rows),
        "valid_run_count": sum(bool(row["valid"]) for row in rows),
        "usable_operating_result_count": sum(
            bool(row["usable_operating_result"]) for row in rows
        ),
        "study_eligible_run_count": sum(bool(row["study_eligible"]) for row in rows),
        "eligible_noncompleted_operating_result_count": sum(
            bool(row["study_eligible"]) and not bool(row["pipeline_completed"])
            for row in rows
        ),
        "pipeline_completed_run_count": sum(bool(row["pipeline_completed"]) for row in rows),
        "partial_phase_completion_count": sum(
            row["solver_outcome"] == "partial_phase_completion" for row in rows
        ),
        "checkpoint_recovery_count": sum(
            row["solver_outcome"] == "checkpoint_recovery" for row in rows
        ),
        "error_count": sum(row["status"] == "error" for row in rows),
        "timeout_count": sum(row["status"] == "timeout" for row in rows),
        "missing_count": sum(row["status"] == "missing" for row in rows),
    }
    return {
        "schema_version": 2,
        "artifact_status": artifact_status,
        "profile": profile_name,
        "profile_title": profile["title"],
        "latex_label_prefix": profile["label_prefix"],
        "evidence_classification": profile["classification"],
        "separation_warning": profile["warning"],
        "pooled_with_primary_pilot": False if profile_name == "repair" else None,
        "causal_quality_comparison": False,
        "official_budget_extrapolation": False,
        "generated_at": utc_now(),
        "study_directory": relative(study_dir),
        "manifest": relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "planned_run_count": EXPECTED_RUNS,
        "recorded_run_count": EXPECTED_RUNS - len(missing),
        "missing_run_count": len(missing),
        "benchmarks": list(dict.fromkeys(str(row["benchmark"]) for row in rows)),
        "variants": variants,
        "nominal_seeds": sorted({int(row["nominal_seed"]) for row in rows}),
        "resource_context": manifest.get("invocation", {}),
        "interpretation": (
            "Finite 180-second solver-budget evidence only. Results include every planned "
            "run, including errors, timeouts, partial phases, and checkpoint recoveries. "
            "No extrapolation to the predeclared publication budget is made."
            + (
                " This post-fix validation used three concurrent jobs and remains separate "
                "from the primary pilot; it is not a causal quality comparison."
                if profile_name == "repair" else ""
            )
        ),
        "runs": rows,
        "coverage": coverage,
        "aggregate_by_variant": aggregate(rows, variants),
        "prefix_diversity": prefix_diversity(rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="primary")
    parser.add_argument("--study-dir")
    parser.add_argument("--json-output")
    parser.add_argument("--tex-output")
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="write an explicitly interim artifact when fewer than 12 results exist",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = PROFILES[args.profile]
        study_dir = Path(args.study_dir or profile["study_dir"]).resolve()
        json_output = Path(args.json_output or profile["json_output"]).resolve()
        tex_output = Path(args.tex_output or profile["tex_output"]).resolve()
        if args.profile == "repair" and (
            json_output == PRIMARY_JSON.resolve() or tex_output == PRIMARY_TEX.resolve()
        ):
            raise ValueError("the repair profile may not overwrite primary pilot artifacts")
        summary = build_summary(study_dir, args.allow_incomplete, args.profile)
        json_text = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        latex_text = render_latex(summary)
        atomic_write_text(json_output, json_text)
        atomic_write_text(tex_output, latex_text)
        print(
            "%s: recorded %d/%d runs; wrote %s and %s"
            % (
                summary["artifact_status"], summary["recorded_run_count"],
                summary["planned_run_count"], json_output, tex_output,
            )
        )
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print("publication pilot rendering error: %s" % exc, file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
