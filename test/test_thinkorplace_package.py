"""Package isolation and boundary regressions; small synthetic inputs only."""
import ast
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "996193c83eaad151e5ae3fb166cedbd83372d060"


def test_legalizer_clamps_soft_clusters_and_preserves_fixed():
    from thinkorplace.v2.macro_legalizer import greedy_macro_legalize
    from macro_place.utils import validate_placement
    bench = SimpleNamespace(num_macros=3, num_hard_macros=2,
                            macro_sizes=torch.tensor([[1., 1.], [1., 1.], [.51, .64]]),
                            macro_positions=torch.tensor([[2., 2.], [7., 7.], [5., 5.]]),
                            macro_fixed=torch.tensor([True, False, False]),
                            canvas_width=10., canvas_height=10.)
    pos=bench.macro_positions.clone()
    # Legalization must build obstacles at the fixed reference position.
    pos[0] = torch.tensor([7., 7.])
    pos[1] = torch.tensor([2., 2.])
    pos[2] = torch.tensor([10.01, -0.01])
    result, _ = greedy_macro_legalize(pos, bench)
    assert validate_placement(result, bench) == (True, [])
    assert torch.equal(result[0], bench.macro_positions[0])
    assert result[2,0] < 10 and result[2,1] > 0


def test_runtime_uses_only_namespaced_and_canonical_imports():
    for path in (ROOT / "thinkorplace").rglob("*.py"):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                module=node.module or ""
                assert not module.startswith(("experiments", "submissions", "writeup")), (path, module)
                assert module not in {"macro_place.cd_core", "macro_place.sdf_init", "macro_place.bench_paths", "macro_place.incremental_evaluator"}, (path,module)


def test_runtime_validation_matches_canonical_float32_endpoint_counterexample():
    from thinkorplace.common.runtime_safety import valid_placement
    from macro_place.utils import validate_placement
    from macro_place.objective import compute_overlap_metrics
    bench = SimpleNamespace(
        num_macros=2, num_hard_macros=2,
        macro_positions=torch.tensor([[10., 10.], [17.286287307739258, 10.]]),
        macro_sizes=torch.tensor([[7.890551567077637, 1.], [6.682023525238037, 1.]]),
        macro_fixed=torch.tensor([False, False]), canvas_width=50., canvas_height=50.)
    assert compute_overlap_metrics(bench.macro_positions, bench)["overlap_count"] == 0
    assert validate_placement(bench.macro_positions, bench)[0] is False
    assert valid_placement(bench.macro_positions, bench)[0] is False


def test_runtime_validation_matches_canonical_randomized_geometry():
    from thinkorplace.common.runtime_safety import valid_placement
    from macro_place.utils import validate_placement
    generator = torch.Generator().manual_seed(314159)
    for dtype in (torch.float32, torch.float64):
        for index in range(100):
            n = 2 + index % 11
            sizes = torch.rand((n, 2), generator=generator, dtype=dtype) * 10 + .1
            positions = torch.rand((n, 2), generator=generator, dtype=dtype) * 80 + 10
            if index % 2 == 0:
                # Deliberately exercise rounded touching endpoints, plus a
                # representable step to either side of the boundary.
                positions[1] = positions[0]
                positions[1, 0] += (sizes[0, 0] + sizes[1, 0]) / 2
                direction = float("inf") if index % 4 == 0 else -float("inf")
                positions[1, 0] = torch.nextafter(positions[1, 0], positions.new_tensor(direction))
            bench = SimpleNamespace(num_macros=n, num_hard_macros=n - index % 2,
                                    macro_positions=positions.clone(), macro_sizes=sizes,
                                    macro_fixed=torch.zeros(n, dtype=torch.bool),
                                    canvas_width=100., canvas_height=100.)
            assert valid_placement(positions, bench)[0] == validate_placement(positions, bench)[0]


def test_runtime_validation_checks_pairs_across_vector_blocks():
    from thinkorplace.common.runtime_safety import valid_placement
    from macro_place.utils import validate_placement
    n = 258
    positions = torch.stack([torch.arange(n) * 2. + 1., torch.ones(n)], dim=1)
    bench = SimpleNamespace(num_macros=n, num_hard_macros=n,
                            macro_positions=positions.clone(), macro_sizes=torch.ones(n, 2),
                            macro_fixed=torch.zeros(n, dtype=torch.bool),
                            canvas_width=1000., canvas_height=10.)
    assert valid_placement(positions, bench)[0] == validate_placement(positions, bench)[0] == True
    positions[-1] = positions[0]
    assert valid_placement(positions, bench)[0] == validate_placement(positions, bench)[0] == False


def test_package_imports_with_pristine_upstream_and_without_research_tree(tmp_path):
    shutil.copytree(ROOT / "thinkorplace", tmp_path / "thinkorplace", ignore=shutil.ignore_patterns("__pycache__"))
    infra=tmp_path / "macro_place"
    infra.mkdir()
    canonical=["__init__.py", "_plc.py", "benchmark.py", "loader.py", "objective.py", "utils.py"]
    for name in canonical:
        content=subprocess.check_output(["git", "show", f"{UPSTREAM}:macro_place/{name}"], cwd=ROOT)
        (infra/name).write_bytes(content)
    # Upstream's own external dependency is allowed; solution archives are absent.
    (tmp_path / "external").symlink_to(ROOT / "external", target_is_directory=True)
    script='''
import importlib, pathlib, sys
root=pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root))
sys.meta_path=[f for f in sys.meta_path if not getattr(f,"__module__", "").startswith("__editable__")]
from thinkorplace.v1.placer import Placer as V1
from thinkorplace.v2.placer import Placer as V2
from thinkorplace.v2.worker import lane_entry
from thinkorplace.v2.smooth_global_placer_v4_gaussian import SmoothGlobalPlacerV4Gaussian
from thinkorplace.v2.bounded_saddle import bounded_cascading_saddle_escape
from thinkorplace.v2.kjoint_hungarian import kjoint_hungarian_step
from thinkorplace.v1.dpo_init import _best_of_v2_init
for name,module in list(sys.modules.items()):
    if name.startswith(("thinkorplace", "macro_place")):
        assert pathlib.Path(module.__file__).resolve().is_relative_to(root), (name,module.__file__)
assert not any(n.startswith(("experiments.", "writeup.", "submissions.")) for n in sys.modules)
assert not any(n in sys.modules for n in ["macro_place.cd_core", "macro_place.sdf_init", "macro_place.incremental_evaluator", "macro_place.bench_paths"])
V1(device="cpu",num_threads=1);V2(device="cpu",num_threads=3)
print("isolated imports passed")
'''
    result=subprocess.run([sys.executable, "-I", "-c", script, str(tmp_path)], cwd=tmp_path,
                          capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    assert "isolated imports passed" in result.stdout


def test_extracted_primitives_preserve_algorithm_ast():
    # The two original helper copies match; package extraction changes only imports.
    source=ast.parse((ROOT / "submissions/_archive/thinkorplace-v2-lib/ablation_v2_steps.py").read_text())
    extracted=ast.parse((ROOT / "thinkorplace/common/smooth_primitives.py").read_text())
    originals={n.name:ast.dump(n,include_attributes=False) for n in source.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for n in extracted.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
            assert ast.dump(n,include_attributes=False) == originals[n.name]
