"""Half-Cauchy probe magnitudes shared by the v1 portfolio phase."""
from typing import List

import numpy as np


def _sample_levy_magnitudes(K: int, scale: float, rng: np.random.Generator) -> List[float]:
    """K positive magnitudes from |Cauchy(0, scale)|.

    Cauchy has heavy tails (no finite variance). Median = scale.
    Typical: 50% draws in [0.41*scale, 2.4*scale], 5% > 12.7*scale.

    Returns ascending-sorted magnitudes so we can break the inner loop
    when budget runs out without missing the smallest probes.
    """
    # Half-Cauchy: |X| where X ~ Cauchy(0, scale)
    u = rng.uniform(0.0, 1.0, size=K)
    # Inverse CDF for half-Cauchy: F^-1(u) = scale * tan(pi/2 * u)
    mags = scale * np.tan(np.pi * 0.5 * u)
    # Cap at 50*scale to avoid wasting evals on absurd jumps
    mags = np.clip(mags, 0.01 * scale, 50.0 * scale)
    return sorted(mags.tolist())

