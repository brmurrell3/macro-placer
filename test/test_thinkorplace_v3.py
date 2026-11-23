"""Tests of thinkorplace v3 (about 4 min in all on an M3 Max):
span score, seeds, legalisation, the widened congestion score, a search segment and the deadline."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macro_place.utils import validate_placement  # noqa: E402
from thinkorplace.common.bench_paths import find_benchmark_dir  # noqa: E402
from thinkorplace.v3 import seeds as seed_module  # noqa: E402
from thinkorplace.v3.chain import load_design  # noqa: E402
from thinkorplace.v3.legalize import is_legal, legalize  # noqa: E402
from thinkorplace.v3.scheduler import run_chains  # noqa: E402
from thinkorplace.common.incremental_evaluator import IncrementalProxyEvaluator  # noqa: E402
from thinkorplace.v3.search import CoordinateDescent, WideTailEvaluator  # noqa: E402
from thinkorplace.v3.seeds import HORIZONTAL_WEIGHT, cold_seed, span_score  # noqa: E402


@pytest.fixture(scope="module")
def ibm01():
    torch.set_num_threads(1)
    return load_design(str(find_benchmark_dir("ibm01")))


def test_span_score_weights_horizontal_span(monkeypatch):
    # two nets: one joins objects 0 and 1 (weight 2), the other object 1 and a port at (10, 0) (weight 1)
    nets = SimpleNamespace(
        mask=torch.tensor([[True, True], [True, True]]),
        pin_macro_idx=torch.tensor([[0, 1], [1, 2]]),  # index 2 is the common owner of ports, at the origin
        pin_offsets=torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [10.0, 0.0]]]),
        weights=torch.tensor([2.0, 1.0]),
    )
    monkeypatch.setattr(seed_module, "_extract_net_data", lambda bench, plc: nets)
    positions = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    # net 0: pins (0, 0) and (4, 4): width 4, height 4; net 1: pins (3, 4) and (10, 0): width 7, height 4
    assert seed_module.net_spans(positions, None, None) == (2 * 4 + 7, 2 * 4 + 4)
    assert span_score(positions, None, None) == pytest.approx(HORIZONTAL_WEIGHT * 15 + 12)


def test_cold_seed_descends_to_a_legal_placement(ibm01):
    from thinkorplace.v3.descent import coarse_descent, fine_descent

    bench, plc = ibm01
    seed = cold_seed(5001, bench, plc)
    assert seed.positions.shape == (bench.num_macros, 2)
    assert math.isfinite(seed.span_score) and seed.span_score > 0
    assert validate_placement(seed.positions, bench, check_overlaps=False)[0]
    raw = fine_descent(coarse_descent(seed.positions, bench, plc), bench, plc)
    legal = legalize(raw, bench)
    assert legal is not None and is_legal(legal, bench)


def test_widened_congestion_is_never_below_the_stock_term(ibm01):
    bench, plc = ibm01
    positions = bench.macro_positions.clone().to(torch.float32)
    stock = IncrementalProxyEvaluator(bench, plc, positions).current_cost()
    wide = WideTailEvaluator(bench, plc, positions).current_cost()
    assert wide["congestion"] >= stock["congestion"]
    assert wide["wl"] == stock["wl"] and wide["density"] == stock["density"]


def test_segment_keeps_placement_legal_and_stops_at_its_quota(ibm01):
    bench, plc = ibm01
    start = legalize(bench.macro_positions.clone(), bench)
    assert start is not None
    result = CoordinateDescent(bench, plc).segment(start, True, lambda done: done < 2, lambda: math.inf)
    assert result.sweeps == 2 and len(result.complete_sweep_seconds) == 2
    assert torch.equal(result.positions[:bench.num_hard_macros], start[:bench.num_hard_macros])  # soft-only sweeps
    assert is_legal(result.positions, bench)


def test_short_budget_is_honoured(tmp_path):
    t0 = time.time()
    positions, summary = run_chains(str(find_benchmark_dir("ibm01")), tmp_path, t0, budget=60.0)
    assert summary["decided_at_seconds"] <= 60.0 + 1.0
    assert all(row["published_at"] <= 60.0 for row in summary["published"])
    assert positions is not None
    bench, _ = load_design(str(find_benchmark_dir("ibm01")))
    assert is_legal(positions, bench)
