"""Focused post-competition safety regressions; no solver benchmarks."""
import importlib.util
import multiprocessing
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from thinkorplace.common.runtime_safety import monotonic_seconds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(path, name):
    if path.startswith("thinkorplace/"):
        return importlib.import_module(path[:-3].replace("/", "."))
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def v2():
    return load("submissions/thinkorplace-v2/placer.py", "safety_v2")


@pytest.fixture
def bench():
    return SimpleNamespace(name="synthetic", num_macros=2, num_hard_macros=2,
                           canvas_width=10., canvas_height=10.,
                           macro_sizes=torch.ones(2, 2),
                           macro_positions=torch.tensor([[2., 2.], [7., 7.]]),
                           macro_fixed=torch.tensor([True, False]))


class FakeProcess:
    def __init__(self, alive=True, join_error=False):
        self.alive = alive
        self.join_error = join_error
        self.timeouts = []
        self.exitcode = None
        self.terminated = False
    def join(self, timeout=None):
        self.timeouts.append(timeout)
        if self.join_error:
            self.join_error = False
            raise RuntimeError("join failed")
    def is_alive(self):
        return self.alive
    def terminate(self):
        self.terminated = True
        self.alive = False
        self.exitcode = -15
    def kill(self):
        self.alive = False
        self.exitcode = -9


def test_expired_budget_never_spawns(v2, bench, monkeypatch):
    monkeypatch.setattr(v2.Placer, "_spawn_lane", lambda *a: pytest.fail("spawn after expiry"))
    with pytest.raises(TimeoutError):
        v2.Placer(budget_seconds=0, verbose=False).place(bench)


def test_expired_join_is_zero_and_all_children_cleaned(v2, bench, tmp_path, monkeypatch):
    processes = [FakeProcess() for _ in range(3)]
    placer = v2.Placer(budget_seconds=1, checkpoint_dir=str(tmp_path), verbose=False)
    # Spawn is instantaneous; simulate expiration immediately after the third spawn.
    clock = [0.]
    monkeypatch.setattr(time, "clock_gettime", lambda clock_id: clock[0])
    def spawn(ctx, i, *args):
        if i == 2:
            clock[0] = 2.
        return processes[i]
    monkeypatch.setattr(placer, "_spawn_lane", spawn)
    with pytest.raises(TimeoutError):
        placer.place(bench)
    assert all(p.timeouts[0] == 0 for p in processes)
    assert all(p.terminated and not p.is_alive() for p in processes)


def test_cleanup_on_join_exception(v2, bench, tmp_path, monkeypatch):
    processes = [FakeProcess(join_error=i == 0) for i in range(3)]
    placer = v2.Placer(budget_seconds=1, checkpoint_dir=str(tmp_path), verbose=False)
    monkeypatch.setattr(placer, "_spawn_lane", lambda ctx, i, *args: processes[i])
    with pytest.raises(RuntimeError, match="join failed"):
        placer.place(bench)
    assert all(p.terminated for p in processes)


def test_actual_child_terminated(v2):
    p = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(60,))
    p.start()
    v2.Placer._stop_processes([p])
    assert not p.is_alive()
    assert p.exitcode is not None


@pytest.mark.parametrize("bad_pos", [np.full((2, 2), np.nan), np.zeros((1, 2)),
                                     np.array([[2., 2.], [11., 7.]]),
                                     np.array([[3., 2.], [7., 7.]])])
def test_invalid_better_lane_rejected(v2, bench, tmp_path, monkeypatch, bad_pos):
    from thinkorplace.common.runtime_safety import atomic_pickle
    placer = v2.Placer(budget_seconds=1, checkpoint_dir=str(tmp_path), verbose=False)
    def spawn(ctx, i, name, path, log, deadline):
        atomic_pickle(path, dict(status="ok", proxy=1. if i == 0 else 2., wall=0.,
                                pos_np=bad_pos if i == 0 else bench.macro_positions.numpy()))
        return FakeProcess(alive=False)
    monkeypatch.setattr(placer, "_spawn_lane", spawn)
    assert torch.equal(placer.place(bench), bench.macro_positions)
    assert placer.last_run_metadata["picked_lane"] == "B"


def test_timed_out_lane_retains_valid_phase_checkpoint(v2, bench, tmp_path, monkeypatch):
    from thinkorplace.common.runtime_safety import atomic_pickle
    placer = v2.Placer(budget_seconds=1, lanes="A", checkpoint_dir=str(tmp_path), verbose=False)
    process = FakeProcess()
    def spawn(ctx, i, name, path, log, deadline):
        atomic_pickle(path, dict(status="ok", phase="cd1", proxy=2., wall=0.,
                                pos_np=bench.macro_positions.numpy()))
        return process
    monkeypatch.setattr(placer, "_spawn_lane", spawn)
    assert torch.equal(placer.place(bench), bench.macro_positions)
    assert placer.last_run_metadata["timed_out_lanes"] == ["A"]
    assert placer.last_run_metadata["used_fallback"] is True
    assert placer.last_run_metadata["phase_completion"]["A"]["worker_status"] == "interrupted"
    assert process.terminated


def test_returned_lane_with_phase_failure_is_not_checkpoint_recovery(v2, bench, tmp_path, monkeypatch):
    from thinkorplace.common.runtime_safety import atomic_pickle, phase_detail
    placer = v2.Placer(budget_seconds=1, lanes="A", checkpoint_dir=str(tmp_path), verbose=False)
    detail = dict(phase_detail(), attempted=True, failed=True, error="phase failed")
    def spawn(ctx, i, name, path, log, deadline):
        atomic_pickle(path, dict(status="ok", worker_status="ok", pipeline_completed=False,
                                phase="final", proxy=2., wall=0., phase_details={"cd": detail},
                                pos_np=bench.macro_positions.numpy()))
        return FakeProcess(alive=False)
    monkeypatch.setattr(placer, "_spawn_lane", spawn)
    placer.place(bench)
    assert placer.last_run_metadata["solver_outcome"] == "partial_phase_completion"
    assert placer.last_run_metadata["used_fallback"] is False
    assert placer.last_run_metadata["degraded_lanes"] == ["A"]


def test_atomic_failure_keeps_old_checkpoint(tmp_path, monkeypatch):
    from thinkorplace.common import runtime_safety as safety
    path = tmp_path / "lane0.pkl"
    safety.atomic_pickle(path, {"phase": "good"})
    def fail(*args):
        raise OSError("simulated interrupted replace")
    monkeypatch.setattr(safety.os, "replace", fail)
    with pytest.raises(OSError):
        safety.atomic_pickle(path, {"phase": "new"})
    assert pickle.loads(path.read_bytes()) == {"phase": "good"}
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("path,name", [
    ("submissions/_archive/thinkorplace-v2-lib/ablation_v2_steps.py", "safety_topology_v2"),
    ("writeup/archive/submissions/dpo/ablation_v2_steps.py", "safety_topology_v1"),
    ("thinkorplace/common/smooth_primitives.py", "safety_topology_common"),
    ("thinkorplace/v1/dpo_init.py", "safety_topology_dpo"),
])
def test_hierarchical_pin_uses_plc_owner_and_tensor_mapping(path, name):
    module = load(path, name)
    pin = SimpleNamespace(get_type=lambda: "MACRO_PIN", get_macro_name=lambda: "top/core/ram",
                          get_offset=lambda: (0.25, -0.5), get_weight=lambda: 2.)
    port = SimpleNamespace(get_type=lambda: "PORT", get_pos=lambda: (0., 3.))
    # PLC index 9 maps to tensor index 0; names deliberately aren't sortable by owner.
    plc = SimpleNamespace(nets={"top/core/ram/D": ["out"]},
                          mod_name_to_indices={"top/core/ram": 9, "top/core/ram/D": 3, "out": 7},
                          modules_w_pins={3: pin, 7: port})
    benchmark = SimpleNamespace(hard_macro_indices=[9], soft_macro_indices=[], num_hard_macros=1, num_macros=1)
    data = module._extract_net_data(benchmark, plc)
    assert data.pin_macro_idx.tolist() == [[0, 1]]
    assert data.mask.tolist() == [[True, True]]
    assert data.pin_offsets[0, 0].tolist() == [0.25, -0.5]
    assert data.weights.tolist() == [2.]


@pytest.mark.parametrize("lane_seed", [42, 142])
def test_worker_failure_preserves_descent_checkpoint(v2, bench, tmp_path, monkeypatch, lane_seed):
    import thinkorplace.common.bench_paths
    import thinkorplace.common.cd_core
    import macro_place.loader
    import macro_place.objective
    import thinkorplace.common.incremental_evaluator
    worker = v2.lane_worker
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(thinkorplace.common.bench_paths, "find_benchmark_dir", lambda name: tmp_path)
    monkeypatch.setattr(macro_place.loader, "load_benchmark_from_dir", lambda path: (bench, None))
    monkeypatch.setattr(macro_place.objective, "compute_proxy_cost", lambda *args: {"proxy_cost": 1.})
    class Descender:
        def __init__(self, **kwargs): pass
        def place(self, benchmark): return benchmark.macro_positions.clone()
    import thinkorplace.v2.smooth_global_placer_v4_gaussian as gaussian
    monkeypatch.setattr(gaussian, "SmoothGlobalPlacerV4Gaussian", Descender)
    monkeypatch.setattr(thinkorplace.common.incremental_evaluator, "IncrementalProxyEvaluator", lambda *args: None)
    def fail(*args, **kwargs):
        raise RuntimeError("CD failed after usable descent")
    monkeypatch.setattr(thinkorplace.common.cd_core, "run_cd_adaptive", fail)
    result = tmp_path / "lane0.pkl"
    worker.lane_entry(0, "synthetic", str(result), str(tmp_path / "lane0.log"), lane_seed,
                      1, .005, .005, .00005, 10., .7, "sdf", 1., .001,
                      0., 2, 50, .01, 0., 50, 100, 15, 42, [],
                      monotonic_seconds() + 60, "cpu", 1)
    record = pickle.loads(result.read_bytes())
    assert record["status"] == "ok"
    assert record["worker_status"] == "exception"
    assert record["phase"] == "descent_legalized"
    assert record["rng_seed"] == lane_seed
    assert record["effective_rng_seed"] == 42
    assert record["initializer_attempt"] == 1
    assert np.array_equal(record["pos_np"], bench.macro_positions.numpy())


def test_worker_cd_budget_reserves_time_for_final_checkpoint(v2, bench, tmp_path, monkeypatch):
    import thinkorplace.common.bench_paths
    import thinkorplace.common.cd_core
    import macro_place.loader
    import macro_place.objective
    import thinkorplace.common.incremental_evaluator
    worker = v2.lane_worker
    score_seconds = 0.2
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(thinkorplace.common.bench_paths, "find_benchmark_dir", lambda name: tmp_path)
    monkeypatch.setattr(macro_place.loader, "load_benchmark_from_dir", lambda path: (bench, None))
    def slow_score(*args):
        time.sleep(score_seconds)
        return {"proxy_cost": 1.}
    monkeypatch.setattr(macro_place.objective, "compute_proxy_cost", slow_score)
    class Descender:
        def __init__(self, **kwargs): pass
        def place(self, benchmark): return benchmark.macro_positions.clone()
    import thinkorplace.v2.smooth_global_placer_v4_gaussian as gaussian
    monkeypatch.setattr(gaussian, "SmoothGlobalPlacerV4Gaussian", Descender)
    monkeypatch.setattr(thinkorplace.common.incremental_evaluator, "IncrementalProxyEvaluator", lambda *args: None)
    seen = {}
    def capture(*args, hard_cap_s, **kwargs):
        seen["hard_cap_s"], seen["remaining"] = hard_cap_s, deadline - monotonic_seconds()
        raise RuntimeError("stop after recording the CD budget")
    monkeypatch.setattr(thinkorplace.common.cd_core, "run_cd_adaptive", capture)
    deadline = monotonic_seconds() + 15
    worker.lane_entry(0, "synthetic", str(tmp_path / "lane0.pkl"), str(tmp_path / "lane0.log"), 42,
                      1, .005, .005, .00005, 10., .7, "sdf", 600., .001,
                      0., 2, 50, .01, 0., 50, 100, 15, 42, [],
                      deadline, "cpu", 1)
    import thinkorplace.v2.worker
    reserve = thinkorplace.v2.worker.CD_FINISH_MARGIN_S + 2 * score_seconds
    assert 0 < seen["hard_cap_s"] <= seen["remaining"] - reserve + 0.05


def test_v1_expired_budget_does_not_start_cascade(bench, monkeypatch):
    import macro_place.objective
    module = load("thinkorplace/v1/stacked.py", "safety_stacked")
    clock = [0.]
    monkeypatch.setattr(time, "clock_gettime", lambda clock_id: clock[0])
    monkeypatch.setattr(module, "find_benchmark_dir", lambda name: "unused")
    monkeypatch.setattr(module, "load_benchmark_from_dir", lambda path: (bench, None))
    monkeypatch.setattr(macro_place.objective, "compute_proxy_cost", lambda *args: {"proxy_cost": 1.})
    class SlowFirstPhase:
        def __init__(self, **kwargs): pass
        def place(self, benchmark):
            clock[0] = 3400.
            return benchmark.macro_positions.clone()
    monkeypatch.setattr(module, "CDLNSSAPlacer", SlowFirstPhase)
    monkeypatch.setattr(module, "cascading_saddle_escape", lambda *a, **kw: pytest.fail("expired cascade started"))
    monkeypatch.setattr(module, "portfolio_saddle_escape", lambda *a, **kw: pytest.fail("expired portfolio started"))
    placer = module.CDLNSSACascadeStackedPlacer(budget_seconds=3090, verbose=False)
    assert torch.equal(placer.place(bench), bench.macro_positions)
    assert placer.phase_details["e25"]["completed"]
    for phase in ("e41", "cascade", "portfolio"):
        assert placer.phase_details[phase]["skipped"]
        assert placer.phase_details[phase]["reason"] == "budget_exhausted"


@pytest.mark.parametrize("outcome", ["rejected", "failed", "budget_skipped"])
def test_v1_periphery_execution_is_separate_from_acceptance(bench, monkeypatch, outcome):
    from thinkorplace.common.runtime_safety import phase_detail
    import thinkorplace.v1.placer as module
    import macro_place.objective
    class Stacked:
        def __init__(self, **kwargs):
            self.phase_details = {name: dict(phase_detail(), attempted=True, completed=True)
                                  for name in ("e25", "e41", "cascade", "portfolio")}
        def place(self, benchmark):
            return benchmark.macro_positions.clone()
    monkeypatch.setattr(module, "CDLNSSACascadeStackedPlacer", Stacked)
    monkeypatch.setattr(module, "find_benchmark_dir", lambda name: "unused")
    monkeypatch.setattr(module, "load_benchmark_from_dir", lambda path: (bench, None))
    monkeypatch.setattr(macro_place.objective, "compute_proxy_cost", lambda *args: {"proxy_cost": 1.})
    def periphery(*args, **kwargs):
        if outcome == "budget_skipped":
            pytest.fail("budget-skipped phase executed")
        if outcome == "failed":
            raise RuntimeError("periphery phase failed")
        return bench.macro_positions.clone(), 2., 0
    monkeypatch.setattr(module, "periphery_push_and_polish", periphery)
    placer = module.Placer(budget_seconds=60 if outcome == "budget_skipped" else None, verbose=False)
    assert torch.equal(placer.place(bench), bench.macro_positions)
    detail = placer.last_run_metadata["phase_details"]["periphery"]
    assert detail["attempted"] == (outcome != "budget_skipped")
    assert detail["completed"] == (outcome == "rejected")
    assert detail["failed"] == (outcome == "failed")
    assert detail["skipped"] == (outcome == "budget_skipped")
    assert placer.last_run_metadata["pipeline_completed"] == (outcome == "rejected")
    if outcome == "rejected":
        assert detail["accepted"] is False
    if outcome == "failed":
        assert "periphery phase failed" in detail["error"]


@pytest.mark.parametrize("lane_idx,phase", [(1, "saddle"), (2, "hungarian")])
def test_specialization_failure_remains_visible_after_valid_return(v2, bench, tmp_path, monkeypatch, lane_idx, phase):
    import thinkorplace.common.bench_paths as paths
    import thinkorplace.common.cd_core as cd
    import thinkorplace.common.incremental_evaluator as incremental
    import thinkorplace.v2.smooth_global_placer_v4_gaussian as gaussian
    import thinkorplace.v2.bounded_saddle as saddle
    import thinkorplace.v2.kjoint_hungarian as hungarian
    import macro_place.loader
    import macro_place.objective
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(paths, "find_benchmark_dir", lambda name: tmp_path)
    monkeypatch.setattr(macro_place.loader, "load_benchmark_from_dir", lambda path: (bench, None))
    monkeypatch.setattr(macro_place.objective, "compute_proxy_cost", lambda *args: {"proxy_cost": 1.})
    class Descender:
        def __init__(self, **kwargs): pass
        def place(self, benchmark): return benchmark.macro_positions.clone()
    class Evaluator:
        def __init__(self, benchmark, plc, placement): self.placement = placement.clone()
        def current_cost(self): return {"proxy": 1.}
    monkeypatch.setattr(gaussian, "SmoothGlobalPlacerV4Gaussian", Descender)
    monkeypatch.setattr(incremental, "IncrementalProxyEvaluator", Evaluator)
    monkeypatch.setattr(cd, "run_cd_adaptive", lambda *a, **k: None)
    def fail(*args, **kwargs): raise RuntimeError("specialization failed")
    monkeypatch.setattr(saddle, "bounded_cascading_saddle_escape", fail)
    monkeypatch.setattr(hungarian, "kjoint_hungarian_step", fail)
    result = tmp_path / f"lane{lane_idx}.pkl"
    v2.lane_worker.lane_entry(lane_idx, "synthetic", str(result), str(tmp_path / "lane.log"), 42,
                             1, .005, .005, .00005, 10., .7, "sdf", 1., .001,
                             120., 2, 50, .01, 120., 50, 100, 15, 42, [],
                             monotonic_seconds() + 60, "cpu", 1)
    record = pickle.loads(result.read_bytes())
    assert record["status"] == record["worker_status"] == "ok"
    assert record["pipeline_completed"] is False
    assert record["phase_details"][phase]["failed"]
    assert "specialization failed" in record["phase_details"][phase]["error"]
    assert record["phase_details"]["cd2"]["completed"]
    assert record["phase_details"]["final"]["completed"]


def test_worker_expired_budget_never_starts_sdf(v2, bench, tmp_path, monkeypatch):
    import thinkorplace.common.cd_core
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(thinkorplace.common.cd_core, "sdf_init", lambda *args: pytest.fail("SDF after timeout"))
    result = tmp_path / "lane0.pkl"
    v2.lane_worker.lane_entry(0, "synthetic", str(result), str(tmp_path / "lane0.log"), 42,
                             1, .005, .005, .00005, 10., .7, "sdf", 1., .001,
                             0., 2, 50, .01, 0., 50, 100, 15, 42, [],
                             monotonic_seconds() - 1, "cpu", 1)
    record = pickle.loads(result.read_bytes())
    assert record["status"] == "timeout"
    assert "pos_np" not in record


def test_deadline_clock_is_shared_with_spawned_interpreter():
    import subprocess
    before = monotonic_seconds()
    child = float(subprocess.check_output([
        sys.executable, "-c",
        "from thinkorplace.common.runtime_safety import monotonic_seconds; print(monotonic_seconds())",
    ], cwd=ROOT))
    after = monotonic_seconds()
    assert before <= child <= after


@pytest.mark.parametrize("path,name,entry", [
    ("thinkorplace/v2/bounded_saddle.py", "safety_bounded", "bounded_cascading_saddle_escape"),
    ("thinkorplace/v1/cascade.py", "safety_cascade", "cascading_saddle_escape"),
])
def test_cascade_exhausted_budget_keeps_incumbent(path, name, entry, bench, monkeypatch):
    module = load(path, name)
    monkeypatch.setattr(module, "compute_proxy_cost", lambda *args: {"proxy_cost": 1.})
    monkeypatch.setattr(module, "SmoothProxy", lambda *args: None)
    monkeypatch.setattr(module, "run_cd_adaptive", lambda *a, **kw: pytest.fail("expired polish"))
    pos, stats = getattr(module, entry)(bench.macro_positions, bench, None, total_budget_s=0, log=lambda s: None)
    assert torch.equal(pos, bench.macro_positions)


def test_bounded_eigensolver_checks_deadline_between_hvps(bench, monkeypatch):
    module = load("thinkorplace/v2/bounded_saddle.py", "safety_hvp")
    monkeypatch.setattr(module, "hessian_vector_product", lambda *a: pytest.fail("expired HVP started"))
    with pytest.raises(TimeoutError):
        module.find_softest_eigenvectors_bounded(None, bench.macro_positions,
                                                 np.ones(2, dtype=bool), k=1,
                                                 deadline=monotonic_seconds() - 1,
                                                 log=lambda s: None)


@pytest.mark.parametrize("path,name,entry,eigen", [
    ("thinkorplace/v2/bounded_saddle.py", "safety_clamp_v2", "bounded_cascading_saddle_escape", "find_softest_eigenvectors_bounded"),
    ("thinkorplace/v1/cascade.py", "safety_clamp_v1", "cascading_saddle_escape", "find_softest_eigenvectors"),
])
def test_polish_cap_accounts_for_eigensolver_and_legalizer(path, name, entry, eigen, bench, monkeypatch):
    module = load(path, name)
    clock = [0.]
    monkeypatch.setattr(time, "clock_gettime", lambda clock_id: clock[0])
    monkeypatch.setattr(module, "compute_proxy_cost", lambda *args: {"proxy_cost": 1.})
    monkeypatch.setattr(module, "SmoothProxy", lambda *args: SimpleNamespace(cw=10., ch=10.))
    def eigensolve(*args, **kwargs):
        clock[0] = 61.
        return np.array([-1.]), np.ones((2, 1))
    monkeypatch.setattr(module, eigen, eigensolve)
    def legalize(pos, benchmark):
        clock[0] = 70.
        return benchmark.macro_positions.clone(), 0
    monkeypatch.setattr(module, "project_overlaps", legalize)
    monkeypatch.setattr(module, "IncrementalProxyEvaluator", lambda b, plc, pos: SimpleNamespace(placement=pos))
    caps = []
    def polish(*args, **kwargs):
        caps.append((kwargs["min_time_s"], kwargs["hard_cap_s"]))
        clock[0] = 101.
    monkeypatch.setattr(module, "run_cd_adaptive", polish)
    getattr(module, entry)(bench.macro_positions, bench, None, total_budget_s=100., polish_budget=180., log=lambda s: None)
    assert caps == [(30., 30.)]
