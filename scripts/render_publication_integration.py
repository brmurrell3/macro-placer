#!/usr/bin/env python3
"""Summarize the final short NG45 integration, retaining exact run identities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "writeup/runs/final-ng45-integration-20260906"
OUT = ROOT / "writeup/data/publication"


def display_variant(value):
    return {"v1": "v1", "v2-a": "A", "v2-abc": "A+B+C"}.get(value, value)


def main():
    manifest_path = STUDY / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for key, expected in {
        "budget_seconds": 60, "timeout_seconds": 120, "num_steps": 5,
        "cd_seconds": 0, "jobs": 3, "threads": 3, "device": "cpu",
        "study_kind": "final-sdf-topology-integration",
    }.items():
        assert manifest["invocation"][key] == expected, (key, manifest["invocation"][key])
    planned = manifest["runs"]
    assert {(r["benchmark"], r["variant"], r["seed"]) for r in planned} == {
        ("ariane133_ng45", variant, 42) for variant in ("v1", "v2-a", "v2-abc")}
    assert len(planned) == 3
    rows = []
    for plan in planned:
        path = STUDY / "runs" / plan["run_id"] / "result.json"
        record = json.loads(path.read_text())
        assert record["identity_sha256"] == plan["identity_sha256"]
        assert record["variant"] == plan["variant"]
        rows.append({
            "result_record": str(path.relative_to(ROOT)),
            "result_record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **{key: record.get(key) for key in (
                "run_id", "identity_sha256", "variant", "status", "valid",
                "study_eligible", "pipeline_completed", "solver_outcome",
                "canonical_cost", "legality", "timing", "supervisor", "source",
                "source_consistent", "environment_matches_manifest",
                "constructor_parameters", "phase_outcomes")},
        })
    summary = {
        "classification": "final_sdf_graph_integration_only",
        "scope": "Three entry configurations on ariane133 after SDF graph repair; not a quality comparison.",
        "configuration_note": "V2 uses five analytical steps and zero-second CD phase budgets. These options do not disable v1's distinct budget-derived CD stages.",
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "run_count": 3,
        "valid_count": sum(r["valid"] for r in rows),
        "eligible_count": sum(r["study_eligible"] for r in rows),
        "pipeline_completed_count": sum(r["pipeline_completed"] for r in rows),
        "runs": rows,
    }
    (OUT / "final_ng45_integration.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        r"\subsection{Final NG45 integration}",
        "A separate check of runtime archive " + r"\code{5aa4c719cf18}" +
        " ran all three entry configurations on ariane133 with a 60-second solver "
        "budget, a 120-second external limit, and three concurrent jobs. A and A+B+C used "
        "five analytical steps and zero-second CD phase budgets; v1 retained its "
        "distinct budget-derived refinement stages. All three outputs were valid "
        "and study-eligible. Only A completed its configured pipeline; v1 and "
        "A+B+C returned valid partial-phase outputs. These are integration checks, "
        "not comparable quality measurements.",
        "",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Final NG45 integration after the SDF graph repair; every row returned a valid placement within its external limit.}",
        r"\label{tab:final-ng45-integration}",
        r"\begin{tabular}{lrrl}\toprule",
        r"Variant & Solver time (s) & Wall time (s) & Outcome \\ \midrule",
    ]
    for row in rows:
        outcome = "complete" if row["pipeline_completed"] else "partial phases"
        lines.append(f"{display_variant(row['variant'])} & {row['timing']['solver_place_seconds']:.1f} & "
                     f"{row['supervisor']['wall_seconds']:.1f} & {outcome}" + r" \\")
    lines += [r"\bottomrule\end{tabular}\end{table}", ""]
    # This prose asserts full validity; reject rather than silently emit it for a different result set.
    assert all(r["valid"] and r["study_eligible"] and r["source_consistent"] for r in rows)
    assert [r["variant"] for r in rows if r["pipeline_completed"]] == ["v2-a"]
    (ROOT / "writeup/latex/generated/final_ng45_integration.tex").write_text("\n".join(lines))
    print(f"{summary['valid_count']}/3 valid; {summary['pipeline_completed_count']}/3 pipelines completed.")


if __name__ == "__main__":
    main()
