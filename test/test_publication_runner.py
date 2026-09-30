"""Fast tests for the external publication-study runner."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module("publication_runner_test", ROOT / "scripts" / "run_publication_study.py")
worker = load_module("publication_worker_test", ROOT / "scripts" / "publication_worker.py")


def test_scoped_arbitrary_placer_parameters_are_explicit():
    parsed = runner.parse_key_values(
        ["verbose=false", "v2-a:lr_frac=0.004", "v1:periphery_alpha=0.02"],
        ["v1", "v2-a"],
    )
    assert parsed == {
        "v1": {"verbose": False, "periphery_alpha": 0.02},
        "v2-a": {"verbose": False, "lr_frac": 0.004},
    }
    with pytest.raises(ValueError, match="not a selected variant"):
        runner.parse_key_values(["v2-abc:lr_frac=1"], ["v2-a"])


def test_worker_loads_and_validates_tiny_dummy_placer(tmp_path):
    placer_file = tmp_path / "tiny_placer.py"
    placer_file.write_text(
        "class TinyPlacer:\n"
        "    def __init__(self, scale=1): self.scale = scale\n"
        "    def place(self, benchmark): return benchmark * self.scale\n",
        encoding="utf-8",
    )
    cls = worker.load_placer_class(placer_file, "TinyPlacer")
    worker.validate_constructor_parameters(cls, {"scale": 3})
    assert cls(scale=3).place(4) == 12
    with pytest.raises(TypeError, match="unknown"):
        worker.validate_constructor_parameters(cls, {"unknown": 3})


def test_checkpoint_recovery_is_valid_but_not_normal_completion():
    outcome = worker.solver_outcome("v2-a", {
        "lanes": "A",
        "picked_lane": "A",
        "lane_results": [{
            "status": "ok", "worker_status": "exception",
            "phase": "initializer", "proxy": 1.2,
        }],
    })
    assert outcome["solver_outcome"] == "checkpoint_recovery"
    assert outcome["used_fallback"] is True
    assert outcome["degraded_lanes"] == ["A"]
    assert outcome["phase_completion"]["A"]["selected_checkpoint_phase"] == "initializer"

    normal = worker.solver_outcome("v2-abc", {
        "lanes": "ABC", "picked_lane": "B",
        "lane_results": [
            {"status": "ok", "worker_status": "ok", "phase": "final"},
            {"status": "ok", "worker_status": "ok", "phase": "cd1"},
            {"status": "ok", "worker_status": "ok", "phase": "final"},
        ],
    })
    assert normal["solver_outcome"] == "pipeline_completed"
    assert normal["degraded_lanes"] == []

    explicit = worker.solver_outcome("v2-abc", {
        "solver_outcome": "checkpoint_recovery", "used_fallback": True,
        "phase_completion": {
            "A": {"pipeline_completed": True},
            "B": {"pipeline_completed": False},
            "C": {"pipeline_completed": False},
        },
    })
    assert explicit["degraded_lanes"] == ["B", "C"]


def test_valid_checkpoint_recovery_remains_in_study_aggregate():
    fields = runner.classify_operating_result({
        "source_consistent": True,
        "environment_matches_manifest": True,
        "pipeline_completed": False,
        "solver_outcome": "checkpoint_recovery",
        "used_fallback": True,
    }, status="completed", valid=True, timed_out=False)
    assert fields["usable_operating_result"] is True
    assert fields["study_eligible"] is True

    timed_out = runner.classify_operating_result({
        "source_consistent": True,
        "environment_matches_manifest": True,
    }, status="timeout", valid=False, timed_out=True)
    assert timed_out["usable_operating_result"] is False
    assert timed_out["study_eligible"] is False


def test_partial_phase_completion_is_not_checkpoint_fallback_or_worker_error():
    outcome = worker.solver_outcome("v2-a", {
        "solver_outcome": "partial_phase_completion",
        "pipeline_completed": False,
        "lanes": "A",
        "picked_lane": "A",
        "degraded_lanes": ["A"],
        "lane_results": [{
            "status": "ok", "worker_status": "ok", "phase": "final",
            "phase_details": {
                "initializer": {
                    "attempted": True, "completed": True, "skipped": False,
                    "failed": False, "accepted": True, "reason": None, "error": None,
                },
                "saddle": {
                    "attempted": False, "completed": False, "skipped": True,
                    "failed": False, "accepted": None,
                    "reason": "disabled", "error": None,
                },
                "cd": {
                    "attempted": False, "completed": False, "skipped": True,
                    "failed": False, "accepted": None,
                    "reason": "budget_exhausted", "error": None,
                },
                "descent": {
                    "attempted": True, "completed": False, "skipped": False,
                    "failed": True, "accepted": None,
                    "reason": None, "error": "RuntimeError: descent failed",
                },
            },
        }],
        "phase_completion": {
            "A": {"pipeline_completed": False, "phase_details": {
                "initializer": {"completed": True},
                "saddle": {"completed": False, "skipped": True, "reason": "disabled"},
                "cd": {"completed": False, "skipped": True, "reason": "budget_exhausted"},
                "descent": {"completed": False, "failed": True,
                            "error": "RuntimeError: descent failed"},
            }},
        },
        "used_fallback": False,
    })
    assert outcome["solver_outcome"] == "partial_phase_completion"
    assert outcome["pipeline_completed"] is False
    assert outcome["used_fallback"] is False
    assert outcome["lane_worker_errors"] == {}
    assert outcome["phase_errors"] == {"A.descent": "RuntimeError: descent failed"}
    assert outcome["incomplete_phases"] == ["A.cd", "A.descent"]


def test_selected_lane_worker_failure_is_checkpoint_recovery():
    outcome = worker.solver_outcome("v2-a", {
        "solver_outcome": "checkpoint_recovery", "pipeline_completed": False,
        "lanes": "A", "picked_lane": "A", "degraded_lanes": ["A"],
        "lane_results": [{
            "status": "ok", "worker_status": "exception", "phase": "initializer",
            "error": "RuntimeError: boom", "phase_details": {},
        }],
        "phase_completion": {"A": {"pipeline_completed": False, "phase_details": {}}},
    })
    assert outcome["solver_outcome"] == "checkpoint_recovery"
    assert outcome["used_fallback"] is True
    assert outcome["lane_worker_errors"]["A"]["worker_status"] == "exception"


def test_selected_checkpoint_without_worker_completion_is_recovery():
    outcome = worker.solver_outcome("v2-a", {
        "solver_outcome": "checkpoint_recovery", "pipeline_completed": False,
        "lanes": "A", "picked_lane": "A", "degraded_lanes": ["A"],
        "lane_results": [{"status": "ok", "phase": "initializer", "phase_details": {}}],
        "phase_completion": {"A": {
            "worker_status": "interrupted", "pipeline_completed": False,
            "phase_details": {},
        }},
    })
    assert outcome["used_fallback"] is True
    assert outcome["solver_outcome"] == "checkpoint_recovery"
    assert outcome["lane_worker_errors"]["A"]["worker_status"] == "interrupted"


def test_external_timeout_kills_descendant_process_group(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    script = tmp_path / "spawn_descendant.py"
    script.write_text(
        "import pathlib, subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid))\n"
        "print('spawned', p.pid, flush=True)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    result = runner.supervise_process(
        [sys.executable, str(script), str(child_pid_file)],
        tmp_path, os.environ.copy(), tmp_path / "stdout.log",
        timeout_seconds=0.4, terminate_grace_seconds=0.2, poll_seconds=0.02,
    )
    assert result["status"] == "timeout"
    assert result["timed_out"] is True
    assert result["return_code"] != 0
    assert "spawned" in (tmp_path / "stdout.log").read_text(encoding="utf-8")
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    deadline = time.time() + 2
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        # A zombie has already exited and cannot do work; it may remain briefly
        # until init reaps it. Accept only that state.
        import subprocess
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(child_pid)], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout.strip()
        assert state.startswith("Z"), "descendant survived process-group timeout"


def test_resume_requires_matching_identity(tmp_path, monkeypatch):
    spec = {
        "run_id": "ibm01__v2-a__seed42__abc",
        "identity_sha256": "abc",
        "resource_allocation": {"threads_per_benchmark_run": 1, "device": "cpu"},
        "benchmark": {"bench_root": str(tmp_path)},
    }
    result_path = tmp_path / "runs" / spec["run_id"] / "result.json"
    runner.atomic_write_json(result_path, {
        "identity_sha256": "abc", "status": "completed", "valid": True,
    })

    def should_not_run(*args, **kwargs):
        raise AssertionError("identity-matched completed run was not resumed")

    monkeypatch.setattr(runner, "supervise_process", should_not_run)
    resumed = runner.execute_run(spec, tmp_path, 1, 0.1, retry_failures=False)
    assert resumed["action"] == "resumed"
    assert resumed["valid"] is True

    mismatched = dict(spec)
    mismatched["identity_sha256"] = "different"
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        runner.execute_run(mismatched, tmp_path, 1, 0.1, retry_failures=False)


def test_dry_run_writes_manifest_without_importing_torch(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "netlist.pb.txt").write_text("netlist", encoding="utf-8")
    (source / "initial.plc").write_text("config", encoding="utf-8")
    benchmark = tmp_path / "toy.pt"
    benchmark.write_bytes(b"benchmark")
    output = tmp_path / "study"
    code = runner.main([
        "--benchmarks", str(benchmark), "--benchmark-source", "toy=%s" % source,
        "--variants", "v2-a", "--budget-seconds", "1", "--device", "cpu",
        "--threads", "1", "--output", str(output), "--dry-run",
    ])
    assert code == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_count"] == 1
    assert manifest["runs"][0]["variant"] == "v2-a"
    assert "torch" not in runner.__dict__


def test_checkpoint_inventory_hashes_artifact_phase_and_coordinates(tmp_path):
    np = pytest.importorskip("numpy")
    payload = {
        "status": "ok", "phase": "initializer", "lane_idx": 0,
        "rng_seed": 142, "effective_rng_seed": 42,
        "initializer_attempt": 1, "initializer_method": "sdf",
        "pos_np": np.array([[1.0, 2.0]], dtype=np.float32),
    }
    import pickle
    path = tmp_path / "lane0_initializer.pkl"
    path.write_bytes(pickle.dumps(payload))
    inventory = worker.checkpoint_inventory(tmp_path)
    assert len(inventory) == 1
    assert inventory[0]["phase"] == "initializer"
    assert inventory[0]["rng_seed"] == 142
    assert inventory[0]["effective_rng_seed"] == 42
    assert len(inventory[0]["artifact_sha256"]) == 64
    assert len(inventory[0]["coordinates_sha256"]) == 64
