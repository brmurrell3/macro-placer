# LP-HPWL diagnostic

## Question

Does LP-HPWL — the LP-relaxed half-perimeter wirelength of a chosen
topology — predict refined proxy cost (the cost we are scored on)?

This is the load-bearing question for the polyhedra-navigation
architecture. That architecture ranks topologies by their LP relaxation
score and walks toward better ones. If LP-HPWL tracks refined proxy,
the navigation has a usable signal. If not, every LP-based ranking is
blind to the objective and the architecture cannot work.

## Background — proxy decomposition (E8)

The companion writeup `lp_hpwl_diagnostic.md` (in this folder)
decomposes our `BestOfV2Placer` average proxy of 1.3834 as:

| Component | Value | Coefficient | Contribution | % of proxy |
|---|---|---|---|---|
| Wirelength | 0.0781 | 1.0 | 0.0781 | **5.6%** |
| Density    | 0.5583 | 0.5 | 0.2792 | **20.2%** |
| Congestion | 2.0523 | 0.5 | 1.0262 | **74.2%** |

Proxy is congestion-dominated. The maximum possible recovery from
closing the entire WL gap to LP is ~0.07 — far less than the 0.27
needed to reach the leaderboard. Anything that only optimizes WL is
mathematically capped at ~5% improvement.

## Headline finding — Miftari correlation analysis

From `writeup/evidence.md` §4.2. Across **24 feasible topologies on
ibm01** (Hamming distance 0–2249 from the base topology):

| Pair | ρ |
|------|---:|
| LP-HPWL → refined proxy | **−0.001** |
| LP-HPWL → refined WL | +0.852 |
| LP-HPWL → refined density | −0.536 |
| LP-HPWL → refined congestion | +0.072 |
| refined congestion → refined proxy | +0.825 |

Two-step chain `cheap signal → LP-HPWL → refined proxy`: the first link
holds (cheap signals predict LP-HPWL at ρ = 0.86), the second link is
zero. **LP-HPWL carries no usable information about refined proxy.**
HPWL and density physically anti-correlate (ρ = −0.54 here); their
contributions to proxy nearly cancel, and proxy ends up driven by
congestion, which HPWL is blind to.

## How to run

This probe is documented rather than scripted — `lp_hpwl_diagnostic.md`
in this folder is the formal write-up of the LP solver, the per-benchmark
gap table (17 IBM benchmarks), and the proxy decomposition derivation.
The LP solver itself lives at `scripts/lp_hpwl_lower_bound.py`:

```bash
uv run python scripts/lp_hpwl_lower_bound.py
# regenerates analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md
```

Note: `scripts/lp_hpwl_lower_bound.py` writes its output path to the old
`analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md` location; that constant should be updated
to point here when convenient (see "References elsewhere" in the
top-level handoff notes).

The Miftari ρ = −0.001 result was produced by an ad-hoc topology sweep
not preserved as a single script — its conclusion is captured in
`writeup/evidence.md` §4.2.

## Implication

This finding triggered the **first pivot (polyhedra → DPO)**. See the
ADR at `docs/decisions/001_full_proxy_over_lp_hpwl.md`. Combined with
the proxy decomposition above, it falsified every architecture that
ranks placements by an LP relaxation of any single component.

## Caveat

The ρ = −0.001 measurement was on ibm01 only. Generalizing to
ibm04/09/13 is on the writeup TODO list (see `writeup/paper.md`).

## Re-run policy

Static / one-shot. Re-run only if the proxy cost function changes
(different component weights, new component) or if the LP relaxation
formulation changes. The decomposition table in
`lp_hpwl_diagnostic.md` should be regenerated whenever a new champion
materially shifts the average proxy components.
