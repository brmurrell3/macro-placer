"""E74 cached-placement loader — returns saved best-per-bench output.

VALIDATOR ONLY (not for competition submission). Uses pre-computed
saved placements from E74 + E61V2 wave to verify aggregate via the
standard `uv run evaluate` harness.

For submission, use placer.py (CDLNSSAHessianPlacer) which runs E25 +
E41 + Hessian saddle escape from scratch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark


class CDLNSSAHessianLoader:
    """Loads pre-computed best per-bench placement (validator)."""

    # Per-bench best path (relative to repo root). Selected from the
    # 2026-05-04 → 05 overnight wave (E74 + E61V2 layered).
    _BEST_PATHS = {
        "ibm01": "experiments/E74_hessian_saddle/results/hessian_ibm01.pt",
        "ibm02": "experiments/E74_hessian_saddle/results/hessian_ibm02.pt",
        "ibm03": "experiments/E74_hessian_saddle/results/hessian_ibm03.pt",
        "ibm04": "experiments/E74_hessian_saddle/results/hessian_ibm04.pt",
        "ibm06": "experiments/E74_hessian_saddle/results/hessian_ibm06.pt",
        "ibm07": "experiments/E74_hessian_saddle/results/hessian_ibm07.pt",
        "ibm08": "experiments/E74_hessian_saddle/results/hessian_ibm08.pt",
        "ibm09": "experiments/E74_hessian_saddle/results/hessian_ibm09.pt",
        "ibm10": "experiments/E74_hessian_saddle/results/hessian_ibm10.pt",
        "ibm11": "experiments/E74_hessian_saddle/results/hessian_ibm11.pt",
        # ibm12 — E61V2-fresh + E74 layered won (1.19799 vs 1.20058 from
        # E74-only and 1.205 from E48).
        "ibm12": "experiments/E74_hessian_saddle/results/hessian_ibm12_from_e61v2.pt",
        "ibm13": "experiments/E74_hessian_saddle/results/hessian_ibm13.pt",
        "ibm14": "experiments/E74_hessian_saddle/results/hessian_ibm14.pt",
        # ibm15 — E61V2-fresh + E74 layered won (1.14163 vs 1.15612 E74-only).
        "ibm15": "experiments/E74_hessian_saddle/results/hessian_ibm15_from_e61v2.pt",
        "ibm16": "experiments/E74_hessian_saddle/results/hessian_ibm16.pt",
        "ibm17": "experiments/E74_hessian_saddle/results/hessian_ibm17.pt",
        "ibm18": "experiments/E74_hessian_saddle/results/hessian_ibm18.pt",
    }

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        name = benchmark.name
        path = self._BEST_PATHS.get(name)
        if path is None:
            raise RuntimeError(
                f"CDLNSSAHessianLoader: no cached best for bench {name}. "
                f"This is a validator only; use CDLNSSAHessianPlacer for "
                f"new benchmarks."
            )
        full = _ROOT / path
        if not full.exists():
            raise FileNotFoundError(f"Missing cached placement: {full}")
        data = torch.load(full, weights_only=False)
        placement = data["placement"]
        # Sanity check shape.
        n_total = benchmark.num_macros
        if placement.shape[0] != n_total:
            raise RuntimeError(
                f"Placement shape mismatch for {name}: "
                f"got {placement.shape[0]}, expected {n_total}"
            )
        return placement.to(torch.float32)
