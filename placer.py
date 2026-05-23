"""Submission entry placer for partcl eval_docker.

2026-05-21: routes to E171 adaptive stack (V4-Gaussian + 3-iter portfolio
for small benches, K-joint+SA for ≥400-movable benches). Falls back to
thinkorplace-v2 on any exception or budget overrun, ensuring the
submission can never regress beyond the verified V4-Gaussian floor (0.984
combined).

E171 verified components:
- E166 (multi-init + multi-seed + cascade + portfolio): ibm03 0.8870,
  --fast 0.8384
- E143 (E166 + K-joint LNS + SA polish v2): ibm10 0.95671 (-3% vs V4)
- E169 (E166 + portfolio_max_iters=3): ibm03 0.87964 (-2.7% vs V4)
- E171 itself: adaptive routing per benchmark.num_hard_macros
  (input-dimension gate, contest-legal)

See ADR-014 for the full decision.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch


def _find_repo_root():
    fallback_candidates = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parents[1],
        Path("/submission/repo"),
        Path("/submission/macro-place-challenge-2026"),
    ]
    for c in fallback_candidates:
        if (c / "macro_place").is_dir() and (c / "submissions").is_dir():
            return c
    submission_root = Path("/submission")
    if submission_root.is_dir():
        for child in submission_root.iterdir():
            if child.is_dir() and (child / "macro_place").is_dir() and (child / "submissions").is_dir():
                return child
    raise RuntimeError("Could not locate repo root")


_REPO = _find_repo_root()
sys.path.insert(0, str(_REPO))

import importlib.util  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Primary: E171 adaptive stacked cascade
_E171_PATH = _REPO / "experiments" / "E171_adaptive_kjoint_gate" / "code" / "placer.py"
# Fallback: thinkorplace-v2 (verified V4-Gaussian baseline ~0.984)
_FALLBACK_PATH = _REPO / "submissions" / "thinkorplace-v2" / "placer.py"

_e171_mod = _load_module("_e171_inner", _E171_PATH)
_fallback_mod = _load_module("_fallback_inner", _FALLBACK_PATH)


class Placer:
    """Submission entry: E171 adaptive with thinkorplace-v2 fallback.

    On any exception from E171 (broken phase, exceeded budget, overlap
    persisted), automatically falls back to thinkorplace-v2 so the
    submission never regresses past the verified V4 baseline.
    """

    def __init__(self, budget_seconds: Optional[float] = 3000.0, verbose: bool = True):
        self.budget_seconds = budget_seconds
        self.verbose = verbose

    def _log(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def place(self, benchmark) -> torch.Tensor:
        t0 = time.time()
        self._log(f"=== submission entry ({benchmark.name}) primary=E171 ===")

        try:
            primary = _e171_mod.Placer(
                budget_seconds=self.budget_seconds,
                verbose=self.verbose,
            )
            result = primary.place(benchmark)
            # Sanity verify before returning
            from macro_place.objective import compute_overlap_metrics
            ovl = compute_overlap_metrics(result, benchmark)["overlap_count"]
            if ovl > 0:
                raise RuntimeError(f"E171 returned {ovl} overlaps")
            self._log(
                f"=== primary E171 succeeded ({benchmark.name}) "
                f"wall={time.time()-t0:.0f}s ==="
            )
            return result
        except Exception as exc:
            self._log(f"=== PRIMARY E171 FAILED: {exc}; falling back to thinkorplace-v2 ===")
            # Fallback: thinkorplace-v2
            remaining = max(60.0, (self.budget_seconds or 3000.0) - (time.time() - t0))
            fallback = _fallback_mod.Placer(
                budget_seconds=remaining,
                verbose=self.verbose,
            )
            result = fallback.place(benchmark)
            self._log(
                f"=== fallback thinkorplace-v2 succeeded ({benchmark.name}) "
                f"wall={time.time()-t0:.0f}s ==="
            )
            return result
