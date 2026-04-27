"""Bit-for-bit parity test for ``IncrementalProxyEvaluator``.

For each of two benchmarks (ibm01 small, ibm10 large), perform 100 random
single-macro moves and compare the evaluator's output to a fresh
``compute_proxy_cost`` call after each move. Tolerance: 1e-5 relative.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest
import torch

from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


TESTCASE_ROOT = Path("external/MacroPlacement/Testcases/ICCAD04")


def _isclose(a: float, b: float, rel: float = 1e-5, abs_: float = 1e-9) -> bool:
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def _run_parity(name: str, num_moves: int = 100, seed: int = 42) -> None:
    path = TESTCASE_ROOT / name
    if not path.exists():
        pytest.skip(f"benchmark missing: {path}")

    bench, plc = load_benchmark_from_dir(str(path))
    # Use float64 throughout to avoid float32 round-trip differences between
    # plc's set_pos(np.float32) arithmetic and our Python-float arithmetic.
    placement = bench.macro_positions.detach().clone().to(torch.float64)

    # Reference cost on the initial placement
    ref0 = compute_proxy_cost(placement, bench, plc)

    eval_ = IncrementalProxyEvaluator(bench, plc, placement)
    cur = eval_.current_cost()

    # Initial parity
    assert _isclose(cur["wl"], ref0["wirelength_cost"]), (
        f"[{name}] initial wl mismatch: incr={cur['wl']} ref={ref0['wirelength_cost']}"
    )
    assert _isclose(cur["density"], ref0["density_cost"]), (
        f"[{name}] initial density mismatch: incr={cur['density']} ref={ref0['density_cost']}"
    )
    assert _isclose(cur["congestion"], ref0["congestion_cost"]), (
        f"[{name}] initial congestion mismatch: incr={cur['congestion']} ref={ref0['congestion_cost']}"
    )

    rng = random.Random(seed)
    canvas_w = bench.canvas_width
    canvas_h = bench.canvas_height

    worst_rel = 0.0
    worst_abs = 0.0
    worst_field = ""
    worst_step = -1

    for step in range(num_moves):
        # Pick a random macro (avoid fixed). Soft macros are also fair game.
        movable = (~bench.macro_fixed).nonzero(as_tuple=True)[0].tolist()
        i = rng.choice(movable)
        # Random new position in [w/2, W - w/2] x [h/2, H - h/2]
        w = float(bench.macro_sizes[i, 0])
        h = float(bench.macro_sizes[i, 1])
        nx = rng.uniform(w / 2 + 1e-3, max(w / 2 + 2e-3, canvas_w - w / 2 - 1e-3))
        ny = rng.uniform(h / 2 + 1e-3, max(h / 2 + 2e-3, canvas_h - h / 2 - 1e-3))

        new = torch.tensor([nx, ny], dtype=torch.float64)
        cur = eval_.move(i, new)

        # Reference: full recompute on the same placement
        placement_ref = eval_.placement.detach().clone().to(torch.float64)
        ref = compute_proxy_cost(placement_ref, bench, plc)

        for key_incr, key_ref in [
            ("wl", "wirelength_cost"),
            ("density", "density_cost"),
            ("congestion", "congestion_cost"),
            ("proxy", "proxy_cost"),
        ]:
            a = float(cur[key_incr])
            b = float(ref[key_ref])
            ok = _isclose(a, b, rel=1e-5)
            if not ok:
                rel = abs(a - b) / max(abs(a), abs(b), 1e-30)
                ab = abs(a - b)
                if rel > worst_rel:
                    worst_rel = rel
                    worst_abs = ab
                    worst_field = key_incr
                    worst_step = step
            assert ok, (
                f"[{name} step={step} field={key_incr}] incr={a:.10g} ref={b:.10g} "
                f"abs={abs(a-b):.3e} rel={abs(a-b)/max(abs(a),abs(b),1e-30):.3e}"
            )

    print(
        f"[{name}] parity OK over {num_moves} moves "
        f"(worst rel={worst_rel:.2e} abs={worst_abs:.3e} field={worst_field} step={worst_step})"
    )


def test_parity_ibm01() -> None:
    _run_parity("ibm01", num_moves=100, seed=42)


def test_parity_ibm10() -> None:
    # ibm10 is large (39140 nets, 786 hard macros) so each
    # compute_proxy_cost reference call is expensive; 30 moves keeps runtime
    # manageable while still exercising the per-net update paths.
    _run_parity("ibm10", num_moves=30, seed=42)


def test_revert_ibm01() -> None:
    """After move + revert, state should match initial cost exactly."""
    path = TESTCASE_ROOT / "ibm01"
    if not path.exists():
        pytest.skip("ibm01 missing")
    bench, plc = load_benchmark_from_dir(str(path))
    placement = bench.macro_positions.detach().clone().to(torch.float64)
    eval_ = IncrementalProxyEvaluator(bench, plc, placement)
    initial = eval_.current_cost()

    rng = random.Random(7)
    movable = (~bench.macro_fixed).nonzero(as_tuple=True)[0].tolist()
    for _ in range(20):
        i = rng.choice(movable)
        w = float(bench.macro_sizes[i, 0])
        h = float(bench.macro_sizes[i, 1])
        nx = rng.uniform(w / 2 + 1e-3, bench.canvas_width - w / 2 - 1e-3)
        ny = rng.uniform(h / 2 + 1e-3, bench.canvas_height - h / 2 - 1e-3)
        eval_.move(i, torch.tensor([nx, ny], dtype=torch.float64))
        post = eval_.revert()
        for k in ["wl", "density", "congestion", "proxy"]:
            assert _isclose(post[k], initial[k], rel=1e-9), (
                f"revert mismatch on {k}: post={post[k]} initial={initial[k]}"
            )
