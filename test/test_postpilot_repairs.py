"""Small regressions for observed frozen-pilot engineering failures."""
import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from macro_place.utils import validate_placement
from thinkorplace.common.runtime_safety import finalize_placement, monotonic_seconds


def benchmark():
    bench = SimpleNamespace(
        name="synthetic", num_macros=4, num_hard_macros=2,
        macro_sizes=torch.tensor([[1., 1.], [1., 1.], [2., 2.], [2., 2.]]),
        macro_positions=torch.tensor([[2., 2.], [7., 7.], [-.5, 5.], [5., 11.]]),
        macro_fixed=torch.tensor([True, False, False, False]),
        canvas_width=10., canvas_height=10.)
    bench.get_movable_mask = lambda: ~bench.macro_fixed
    return bench


def test_sdf_repairs_soft_bounds_without_relying_on_cd(monkeypatch):
    import thinkorplace.common.sdf_init as sdf
    bench = benchmark()
    monkeypatch.setattr(sdf, "_load_plc", lambda name: None)
    result = sdf.SDFPlacer(n_iters=0, grid_resolution=4).place(bench)
    assert validate_placement(result, bench) == (True, [])
    assert torch.equal(result[0], bench.macro_positions[0])
    assert torch.equal(result[2:], torch.tensor([[1., 5.], [5., 9.]]))


def test_sdf_legalizer_places_fixed_obstacles_before_larger_movables():
    from thinkorplace.common.sdf_init import greedy_legalize
    result = greedy_legalize(np.array([[5., 5.], [5., 5.]]),
                             np.array([[2., 2.], [1., 1.]]),
                             np.array([True, False]), 10., 10.)
    assert np.array_equal(result[1], [5., 5.])
    assert np.max(np.abs(result[0] - result[1])) >= 1.5


def test_float64_export_restores_fixed_and_repairs_soft_bounds():
    bench = benchmark()
    candidate = bench.macro_positions.double()
    candidate[0] = torch.tensor([7., 7.])  # Must restore before overlap repair.
    result = finalize_placement(candidate, bench, "v1 export")
    assert result.dtype == torch.float32
    assert validate_placement(result, bench) == (True, [])
    assert torch.equal(result[0], bench.macro_positions[0])


def test_float32_upper_endpoint_steps_inward_without_tolerance():
    width = 98 / 173
    bench = SimpleNamespace(num_macros=1, num_hard_macros=0,
                            macro_sizes=torch.tensor([[width, 1.]]),
                            macro_positions=torch.tensor([[1., 1.]]),
                            macro_fixed=torch.tensor([False]),
                            canvas_width=10.03, canvas_height=10.)
    half = bench.macro_sizes[0, 0] / 2
    upper = torch.tensor(bench.canvas_width) - half
    candidate = torch.tensor([[upper, 1.]])
    assert not validate_placement(candidate, bench)[0]
    result = finalize_placement(candidate, bench)
    assert validate_placement(result, bench) == (True, [])
    assert result[0, 0] == torch.nextafter(upper, torch.tensor(-float("inf")))


def test_checkpoint_steps_rounding_overshoot_inward_but_rejects_real_violation(monkeypatch):
    import macro_place.objective as objective
    from thinkorplace.common.runtime_safety import checkpoint
    monkeypatch.setattr(objective, "compute_proxy_cost", lambda pos, bench, plc: {"proxy_cost": 1.0})
    bench = benchmark()
    upper = torch.tensor(bench.canvas_width) - bench.macro_sizes[1, 0] / 2
    candidate = torch.tensor([[2., 2.], [7., 7.], [1., 5.], [5., 9.]])
    candidate[1, 0] = torch.nextafter(upper, torch.tensor(float("inf")))
    assert not validate_placement(candidate, bench)[0]
    record = checkpoint(None, candidate, bench, None, "cd", monotonic_seconds())
    assert validate_placement(torch.from_numpy(record["pos_np"]), bench) == (True, [])
    assert 0 < record["bounds_clamp_max_shift"] <= 1e-6
    candidate[1, 0] = upper + 0.5
    with pytest.raises(ValueError, match="Invalid cd placement"):
        checkpoint(None, candidate, bench, None, "cd", monotonic_seconds())


@pytest.mark.parametrize("overlap", [False, True])
def test_infeasible_fixed_reference_is_rejected(overlap):
    bench = benchmark()
    bench.macro_fixed[:2] = True
    bench.macro_positions[1] = bench.macro_positions[0] if overlap else torch.tensor([-1., 5.])
    with pytest.raises(ValueError, match="Invalid fixed reference geometry"):
        finalize_placement(bench.macro_positions, bench)


def test_unresolved_hard_geometry_is_rejected(monkeypatch):
    import thinkorplace.common.cd_core as cd
    bench = benchmark()
    candidate = bench.macro_positions.clone()
    candidate[1] = candidate[0]
    monkeypatch.setattr(cd, "project_overlaps", lambda pos, benchmark: (pos, 50))
    with pytest.raises(ValueError, match="Hard macro overlaps"):
        finalize_placement(candidate, bench)


@pytest.mark.parametrize("layout", ["", "netlist/output_CT_Grouping"])
def test_ng45_processed_alias_resolves_source_directory(tmp_path, monkeypatch, layout):
    import thinkorplace.common.bench_paths as paths
    target = tmp_path / "ariane133" / layout
    target.mkdir(parents=True)
    (target / "netlist.pb.txt").write_text("fixture")
    monkeypatch.setattr(paths, "_candidate_roots", lambda: [tmp_path])
    assert paths.find_benchmark_dir("ariane133_ng45") == target


def test_exact_benchmark_name_precedes_alias_and_no_recursive_fallback(tmp_path, monkeypatch):
    import thinkorplace.common.bench_paths as paths
    for name in ("ariane133", "ariane133_ng45", "fixtures/missing"):
        target = tmp_path / name
        target.mkdir(parents=True)
        (target / "netlist.pb.txt").write_text("fixture")
    monkeypatch.setattr(paths, "_candidate_roots", lambda: [tmp_path])
    assert paths.find_benchmark_dir("ariane133_ng45") == tmp_path / "ariane133_ng45"
    with pytest.raises(FileNotFoundError):
        paths.find_benchmark_dir("missing_ng45")


def test_rejected_initializer_is_persisted_as_diagnostic_not_candidate(tmp_path, monkeypatch):
    import thinkorplace.common.bench_paths as paths
    import thinkorplace.v2.worker as worker
    import thinkorplace.v2.smooth_global_placer_v4_gaussian as gaussian
    import macro_place.loader
    bench = benchmark()
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(paths, "find_benchmark_dir", lambda name: tmp_path)
    monkeypatch.setattr(macro_place.loader, "load_benchmark_from_dir", lambda path: (bench, None))
    result = tmp_path / "lane0.pkl"
    class Descender:
        def __init__(self, **kwargs): self.callback = kwargs["checkpoint_callback"]
        def place(self, benchmark):
            with pytest.raises(ValueError):
                self.callback(benchmark.macro_positions)
            # This file must exist before the long descent operation, even if
            # the process is terminated before its final error handler runs.
            diagnostic = pickle.loads(result.read_bytes())
            assert diagnostic["status"] == "initializer_rejected"
            assert "pos_np" not in diagnostic
            assert diagnostic["phase_details"]["initializer"]["failed"]
            raise TimeoutError("simulated descent deadline")
    monkeypatch.setattr(gaussian, "SmoothGlobalPlacerV4Gaussian", Descender)
    worker.lane_entry(0, "synthetic", str(result), str(tmp_path / "lane.log"), 42,
                      1, .005, .005, .00005, 10., .7, "sdf", 1., .001,
                      0., 2, 50, .01, 0., 50, 100, 15, 42, [],
                      monotonic_seconds() + 60, "cpu", 1)
    diagnostic = pickle.loads(result.read_bytes())
    assert diagnostic["status"] == "timeout"
    assert "pos_np" not in diagnostic
    assert "canvas bounds" in diagnostic["phase_details"]["initializer"]["error"]


def test_fatal_worker_result_preserves_phase_errors(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("postpilot_publication_worker", root / "scripts/publication_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    result = tmp_path / "result.json"
    request = tmp_path / "request.json"
    request.write_text(json.dumps(dict(worker_result_path=str(result), variant="v1")))
    from thinkorplace.common.runtime_safety import phase_detail
    def fail(request):
        exc = ValueError("no valid plateau")
        exc.solver_metadata = dict(solver_outcome="worker_did_not_complete", pipeline_completed=False,
            phase_details={"e25": dict(phase_detail(), attempted=True, failed=True, error="invalid bounds")})
        raise exc
    monkeypatch.setattr(worker, "run_request", fail)
    assert worker.main(["--request", str(request)]) == 1
    payload = json.loads(result.read_text())
    assert payload["status"] == "error"
    assert payload["phase_errors"]["e25"] == "invalid bounds"
