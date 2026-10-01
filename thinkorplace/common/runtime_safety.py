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
    """Match canonical structural and endpoint-box overlap validation exactly.

    Keep tensor arithmetic in the original dtype: a center-distance predicate
    can disagree with canonical rounded float32 endpoints near touching edges.
    Blocking bounds temporary memory while avoiding the Python pair loop.
    """
    import torch
    from macro_place.utils import validate_placement
    valid, violations = validate_placement(pos, benchmark, check_overlaps=False)
    if not valid:
        return valid, violations
    n = benchmark.num_hard_macros
    lower = pos[:n] - benchmark.macro_sizes[:n] / 2
    upper = pos[:n] + benchmark.macro_sizes[:n] / 2
    indices = torch.arange(n, device=pos.device)
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        overlaps = (
            (lower[start:stop, None, 0] < upper[None, :, 0])
            & (upper[start:stop, None, 0] > lower[None, :, 0])
            & (lower[start:stop, None, 1] < upper[None, :, 1])
            & (upper[start:stop, None, 1] > lower[None, :, 1])
            & (indices[None, :] > indices[start:stop, None])
        )
        if bool(overlaps.any()):
            return False, violations + ["Hard macro overlaps"]
    return True, violations


def phase_detail():
    """Execution and selection are separate properties of a pipeline phase."""
    return dict(attempted=False, completed=False, skipped=False, failed=False,
                accepted=None, reason=None, error=None)


def phases_completed(details):
    return all((d["completed"] and not d["failed"]) or (d["skipped"] and d["reason"] == "disabled")
               for d in details.values())


CHECKPOINT_BOUNDS_SHIFT_MAX = 1e-4


def checkpoint(path, pos, benchmark, plc, phase, t0, **metadata):
    """Save only a fully usable float32 candidate. Invalid candidates never replace it.

    Coordinate descent can leave an upper edge a few float32 steps past the
    canvas. Apply the same inward projection as finalize_placement when it moves
    no coordinate by more than CHECKPOINT_BOUNDS_SHIFT_MAX; larger violations are
    validated unchanged and still fail.
    """
    import numpy as np
    import torch
    from macro_place.objective import compute_proxy_cost
    pos = pos.detach().cpu().to(torch.float32)
    clamped = clamp_to_canvas(pos, benchmark)
    bounds_shift = float((clamped - pos).abs().max()) if pos.numel() else 0.0
    if bounds_shift <= CHECKPOINT_BOUNDS_SHIFT_MAX:
        pos = clamped
        metadata["bounds_clamp_max_shift"] = bounds_shift
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


def finalize_placement(pos, benchmark, phase="placement"):
    """Restore fixed references and produce a fully valid float32 candidate.

    Bounds projection includes soft macros. Hard projection is attempted only
    when needed; canonical-equivalent validity remains the acceptance boundary.
    """
    import torch
    from types import SimpleNamespace
    fixed = benchmark.macro_fixed.cpu().bool()
    if bool(fixed.any()):
        fixed_bench = SimpleNamespace(
            num_macros=int(fixed.sum()),
            num_hard_macros=int(fixed[:benchmark.num_hard_macros].sum()),
            macro_positions=benchmark.macro_positions.cpu()[fixed],
            macro_sizes=benchmark.macro_sizes.cpu()[fixed],
            macro_fixed=torch.ones(int(fixed.sum()), dtype=torch.bool),
            canvas_width=benchmark.canvas_width, canvas_height=benchmark.canvas_height)
        fixed_valid, violations = valid_placement(fixed_bench.macro_positions, fixed_bench)
        if not fixed_valid:
            raise ValueError(f"Invalid fixed reference geometry for {phase}: {violations}")
    result = clamp_to_canvas(pos.detach().cpu().to(torch.float32), benchmark)
    valid, violations = valid_placement(result, benchmark)
    if not valid:
        # Import locally: cd_core also imports SDFPlacer.
        from .cd_core import project_overlaps
        result, _ = project_overlaps(result, benchmark)
        result = clamp_to_canvas(result, benchmark)
        valid, violations = valid_placement(result, benchmark)
    if not valid:
        raise ValueError(f"Invalid {phase} after canvas/hard-overlap projection: {violations}")
    return result
