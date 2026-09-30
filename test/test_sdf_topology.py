"""Regression tests for SDF hard-macro edge extraction."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from macro_place.sdf_init import _extract_edges as local_extract_edges
from thinkorplace.common.sdf_init import _extract_edges as namespaced_extract_edges


@dataclass(frozen=True)
class _Node:
    name: str
    macro_name: str = ""

    def get_name(self) -> str:
        return self.name

    def get_macro_name(self) -> str:
        return self.macro_name


def _synthetic_case(hierarchical: bool):
    prefix = "top/cluster/" if hierarchical else ""
    macro_names = [f"{prefix}{name}" for name in ("A", "B", "C")]
    macro_indices = [8, 3, 11]  # Tensor order deliberately differs from PLC order.
    modules = {
        index: _Node(name) for index, name in zip(macro_indices, macro_names)
    }
    pin_indices = []
    next_index = 20
    pin_names = {}
    for macro_name in macro_names:
        for suffix in ("P", "Q"):
            full_name = f"{macro_name}/{suffix}"
            modules[next_index] = _Node(full_name, macro_name)
            pin_indices.append(next_index)
            pin_names[(macro_name, suffix)] = full_name
            next_index += 1

    # Ports are present in both nets and must not become graph vertices.
    modules[40] = _Node("io/in")
    modules[41] = _Node("io/out")
    plc = SimpleNamespace(
        modules_w_pins=modules,
        hard_macro_pin_indices=pin_indices,
        nets={
            pin_names[(macro_names[0], "P")]: [
                pin_names[(macro_names[1], "P")],
                pin_names[(macro_names[2], "P")],
                "io/out",
            ],
            "io/in": [
                pin_names[(macro_names[0], "Q")],
                pin_names[(macro_names[1], "Q")],
            ],
        },
    )
    benchmark = SimpleNamespace(hard_macro_indices=macro_indices)
    return benchmark, plc


@pytest.mark.parametrize(
    "extractor",
    [
        pytest.param(local_extract_edges, id="local-macro-place"),
        pytest.param(namespaced_extract_edges, id="namespaced-thinkorplace"),
    ],
)
@pytest.mark.parametrize("hierarchical", [False, True], ids=["flat", "hierarchical"])
def test_sdf_edges_use_plc_pin_ownership_and_ignore_ports(extractor, hierarchical):
    benchmark, plc = _synthetic_case(hierarchical)
    edges, weights = extractor(benchmark, plc)
    observed = {
        tuple(pair): float(weight)
        for pair, weight in zip(edges.tolist(), weights.tolist())
    }
    assert observed == {(0, 1): 1.5, (0, 2): 0.5, (1, 2): 0.5}
    assert edges.dtype == torch.long
    assert weights.dtype == torch.float32
