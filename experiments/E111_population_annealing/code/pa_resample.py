"""Baumgartner & Wang (2013, PRE 87 033303) systematic resampling for
population annealing.

Given unnormalized weights and a target population size N, returns integer
counts c_i ≥ 0 with Σc_i = N such that E[c_i] = N · w_i / Σw_j and the
discretization variance is minimized (lower than multinomial resampling).
"""
from __future__ import annotations

import numpy as np


def systematic_resample(weights_unnorm: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    """Returns counts summing to N. weights_unnorm must be ≥ 0 elementwise."""
    w = np.asarray(weights_unnorm, dtype=np.float64)
    if w.sum() <= 0:
        # Degenerate (all weights zero): uniform redistribution.
        counts = np.zeros_like(w, dtype=np.int64)
        idx = rng.integers(0, len(w), size=N)
        for i in idx:
            counts[i] += 1
        return counts

    w = w / w.sum()
    expected = N * w
    floor_counts = np.floor(expected).astype(np.int64)
    remainder = expected - floor_counts
    leftover = N - int(floor_counts.sum())
    if leftover <= 0:
        return floor_counts

    # Systematic sampling on the cumulative remainder profile.
    rem_sum = remainder.sum()
    if rem_sum <= 0:
        # All exact integer expectations; distribute leftover uniformly.
        bonus = np.zeros_like(floor_counts)
        idx = rng.integers(0, len(w), size=leftover)
        for i in idx:
            bonus[i] += 1
        return floor_counts + bonus

    cum = np.cumsum(remainder)
    u = float(rng.uniform(0.0, rem_sum / leftover))
    points = u + np.arange(leftover) * (rem_sum / leftover)
    bonus = np.zeros_like(floor_counts)
    j = 0
    for p in points:
        while j < len(cum) and cum[j] < p:
            j += 1
        bonus[min(j, len(cum) - 1)] += 1
    return floor_counts + bonus
