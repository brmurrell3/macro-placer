# E15 pair-swap — running notes

Working notes carried forward from the retired `findings.md`. The
structured experiment record lives in `manifest.md`; this file collects
observations that don't fit the manifest's frame.

## Findings

### Most macro pairs share exactly 1 net

Pair-swap candidate filters with `min_shared_nets ≥ 2` find zero pairs.
Use 1; rely on the `(shared_count × distance)` ranking + cap to bound work.

This was the v1 bug — `min_shared_nets=2` produced zero swap candidates and
v1's output was identical to the E16 baseline (no swaps attempted). v2
fixed the filter to `min_shared_nets=1` and finds 21–53 real swaps per
benchmark on `--fast`, but the proxy delta is below noise (avg 0.9414 vs
E16 baseline `--fast` 0.9425).
