---
id: E8
name: lp_hpwl_diagnostic
status: graduated
parent: null
created: 2026-04-26
decided: 2026-04-26
champion_at_time: 1.3834
outcome: decomposition_diagnostic
champion_delta: null
graduated_to: ADR-001
superseded_by: null
---

# E8: lp_hpwl_diagnostic

## Hypothesis
Before committing to any architecture, decompose the proxy into its WL,
density, and congestion components on DPO-optimized placements. If WL is
the dominant term, an LP-HPWL formulation is justified. If congestion or
density dominates, LP-HPWL is structurally insufficient regardless of how
well it's solved.

## Method
Compute per-component proxy contributions across all 17 IBM benchmarks on
DPO-seed-43 placements. Cross-check against a Miftari correlation analysis
on ibm01 (LP-HPWL → refined proxy across 24 feasible topologies) to decide
whether LP-HPWL even predicts proxy.

## Kill gate
Diagnostic — no kill gate, but a structural finding (WL ≠ dominant)
falsifies any LP-HPWL-only architecture.

## Generalization check
Compute the decomposition across all 17 benchmarks, not only ibm01, to
confirm congestion-dominance generalizes.

## Outcome (filled when decided)
**Graduated to ADR-001. Falsified LP-HPWL as a sufficient objective and
triggered the DPO → CD pivot.**

Component decomposition across 17 benchmarks (DPO seed 43):

| Term | Avg fraction |
|---|---:|
| WL | **5.8 %** |
| Density | **19.4 %** |
| Congestion | **74.9 %** |

Per-bench fractions range from WL 4.0–9.2 %, density 14.5–26.4 %,
congestion 64.3–80.6 %. *Congestion dominates proxy on every benchmark*.

Miftari correlation on ibm01: LP-HPWL → refined proxy ρ = **−0.001** across
24 feasible topologies. LP-HPWL carries no usable information about refined
proxy because HPWL ↔ density anti-correlate (ρ = −0.536) and the effects
nearly cancel. Combined with the 6 / 20 / 74 decomposition, every LP-only
architecture is falsified.

This is the diagnostic that triggered both the DPO pivot and (later) the
CD pivot. The architectural conclusion was promoted to ADR-001
("congestion is the dominant term; full-proxy descent is required").

## Pointers
- Diagnostic doc: `analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md` (sibling agent is moving
  this to `analysis/lp_hpwl_diagnostic/`; reference by concept name and
  let the path resolve at read time).
- Driver: `scripts/lp_hpwl_lower_bound.py`.
- Discussion: `writeup/evidence.md` §3.3 (decomposition table) and §4.2
  (Miftari correlation); `docs/experiment_index.md` strategic-update row
  2026-04-26.
- ADR: ADR-001.
