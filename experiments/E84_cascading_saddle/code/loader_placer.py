"""Validator-only placer that loads E84 cascading outputs per bench.

For canonical proxy verification via `uv run evaluate ... --all`. Not
for submission.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark


class CascadingLoader:
    """Load saved E84 cascading result per bench; fall back to E74 if missing."""

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        name = benchmark.name
        e84 = _ROOT / "experiments" / "E84_cascading_saddle" / "results" / f"cascade_{name}.pt"
        e74 = _ROOT / "experiments" / "E74_hessian_saddle" / "results" / f"hessian_{name}.pt"

        for pt in [e84, e74]:
            if pt.exists():
                data = torch.load(pt, weights_only=False)
                if "placement" in data:
                    return data["placement"].to(torch.float32)
                if "polished_placement" in data:
                    return data["polished_placement"].to(torch.float32)
                if "transplanted_placement" in data:
                    return data["transplanted_placement"].to(torch.float32)
        raise RuntimeError(f"No saved E84 or E74 placement for {name}")
