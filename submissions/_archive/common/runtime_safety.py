"""Post-competition deadline and artifact helpers; canonical scoring is unchanged."""
from __future__ import annotations

import math
import os
import pickle
import tempfile
import time
from pathlib import Path


def monotonic_seconds():
    """A process-independent clock, including macOS on Python <3.10.

    Older CPython macOS time.monotonic() subtracts a process-local origin.
    CLOCK_MONOTONIC retains the shared kernel origin required by spawn workers.
    """
    if hasattr(time, "clock_gettime") and hasattr(time, "CLOCK_MONOTONIC"):
        return time.clock_gettime(time.CLOCK_MONOTONIC)
    return time.monotonic()


def remaining_seconds(deadline, cap=None):
    remaining = math.inf if deadline is None else max(0.0, deadline - monotonic_seconds())
    return remaining if cap is None else min(max(0.0, cap), remaining)


def require_time(deadline):
    if remaining_seconds(deadline) <= 0:
        raise TimeoutError("Placement deadline exhausted")


def atomic_pickle(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def valid_placement(pos, benchmark):
    from macro_place.utils import validate_placement
    from macro_place.objective import compute_overlap_metrics
    valid, violations = validate_placement(pos, benchmark, check_overlaps=False)
    if valid and compute_overlap_metrics(pos, benchmark)["overlap_count"]:
        valid = False
        violations.append("Hard macro overlaps")
    return valid, violations


def checkpoint(path, pos, benchmark, plc, phase, t0, **metadata):
    """Save only a fully usable float32 candidate. Invalid candidates never replace it."""
    import numpy as np
    import torch
    from macro_place.objective import compute_proxy_cost
    pos = pos.detach().cpu().to(torch.float32)
    valid, violations = valid_placement(pos, benchmark)
    if not valid:
        raise ValueError(f"Invalid {phase} placement: {violations}")
    costs = compute_proxy_cost(pos, benchmark, plc)
    proxy = float(costs["proxy_cost"])
    if not math.isfinite(proxy):
        raise ValueError(f"Nonfinite {phase} canonical score")
    record = dict(status="ok", phase=phase, wall=monotonic_seconds() - t0,
                  proxy=proxy, pos_np=pos.numpy().astype(np.float32), **metadata)
    if path is not None:
        atomic_pickle(path, record)
    return record


def clamp_to_canvas(pos, benchmark):
    """Project movable centers inside the canvas in the returned tensor's dtype.

    Legalization must include soft clusters. Correct an upper-bound rounding
    overshoot by one representable step inward, rather than weakening validation.
    """
    import torch
    result = pos.detach().cpu().clone()
    half = benchmark.macro_sizes.cpu().to(result.dtype) / 2
    canvas = result.new_tensor([benchmark.canvas_width, benchmark.canvas_height])
    upper = canvas - half
    overshoots = upper + half > canvas
    upper = torch.where(overshoots, torch.nextafter(upper, torch.full_like(upper, -float("inf"))), upper)
    movable = ~benchmark.macro_fixed.cpu().bool()
    result[movable] = torch.maximum(half[movable], torch.minimum(upper[movable], result[movable]))
    result[~movable] = benchmark.macro_positions.cpu().to(result.dtype)[~movable]
    return result
