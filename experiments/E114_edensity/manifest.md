---
id: E114
name: edensity
status: falsified
parent: E111
created: 2026-05-20
decided: 2026-05-20
champion_at_time: 1.00279
outcome: 1.59930
champion_delta: 0.596
graduated_to: null
superseded_by: null
---

# E114: Electrostatic eDensity (ePlace / Xplace / DREAMPlace formulation)

## Hypothesis
The grid-bin density used by E88/V3 has step discontinuities at cell
boundaries — Adam sees zero or stale gradient when a macro corner sits
exactly on a bin edge. Replacing it with electrostatic eDensity
(macros as charges → 2D Poisson via DCT → electric potential) gives
the descent a globally-smooth density gradient that should outperform
grid-bin density on hard benches (ibm17), where density is 22-33% of
the cost.

## Method
- `ElectrostaticDensity` (`code/edensity.py`): Gaussian charge smearing
  per macro → overflow `chi = max(0, rho - target)` → solve
  -laplacian(phi) = chi via 2D DCT-II / iDCT-II (Neumann BCs) →
  top-10% mean of phi. Fully differentiable; CPU torch.
- `DiffProxyV3eDensity` subclasses `DiffProxyV3` (E111) with eDensity
  replacing `_grid_density`. Everything else (LSE-HPWL, per-net trace
  congestion, overlap+boundary penalties) unchanged.
- `SmoothGlobalPlacerV3eDensity` subclasses the V3 placer.
- `submissions/e111_minimal_edensity/placer.py` is the drop-in placer.

## Kill gate
ibm17 single bench:
  - V3Min ovl10 720s baseline (champion): canonical proxy 1.20032
  - eDensity raw post-legalize must beat V3 raw, OR
  - eDensity +CD600s polished must beat V3Min 1.20032

If both fail by > 1% (real lift threshold per M3 variance memo), kill.

## Generalization check
If ibm17 wins: scale to --all (17 IBM) + --ng45 (4). Combined 21-bench
avg must beat thinkorplace-v2's 1.0028 / 0.6786 / combined 0.987.

## Outcome — FALSIFIED 2026-05-20

ibm17, scale=0.10, target_density=1.0, sigma_frac=0.5:
- Smooth descent: 500 steps, init loss 1.95 → final 1.08 (descent works)
- Final smooth components: wl=0.0695, d=0.174, c=1.848
- Post-legalize canonical proxy: **1.59930** (overlap=0)
- vs V3 baseline raw 1.30396 → **+22.6% WORSE**
- vs V3Min CD600s 1.20032 → +33% worse on raw

Decision: kill before CD polish. CD600s might close some gap but
+22.6% raw means polish would need to recover ~0.3 proxy units; CD
typically polishes 0.05-0.10 on similar benches. Kill gate triggered.

### Structural finding
eDensity at target_density=1.0 vanishes when no bin overflows above
1.0. For benches with mean rho ≈ 0.8 (ibm17 SDF init), Adam can
trivially zero out eDensity by spreading macros so no bin exceeds 1.0,
but this doesn't reduce the canonical top-10% bin occupation (which
remains ~0.8 because that's the natural mean density). The smooth
objective and canonical objective disagree about what "good" means.

This is the classic Xplace/DREAMPlace decoupling: they don't try to
match the canonical Circuit Training proxy — they have their own
eDensity objective that IS the target. When you graft eDensity onto
a canonical-aligned proxy framework (replacing one component), you
create a basin offset that propagates badly to the canonical score.

A meaningful eDensity-aware placer would need to:
  1. Use eDensity as the SOLE density measure (no canonical
     top-10% post-hoc); or
  2. Calibrate target_density per-bench to a value where eDensity's
     gradient remains meaningful at the canonical optimum (rho_mean *
     0.8 ≈ 0.64 for ibm17, instead of 1.0); or
  3. Use raw rho (no clamp at target) as the cost, dropping the
     ePlace overflow-only formulation — but then the smooth-vs-
     canonical bias depends on how phi distributes globally.

None of these are cheap drop-in changes to E111's pipeline.

### Implementation note
The DCT-via-FFT approach (Makhoul 1980 phase + iDCT-III matrix-mult
inverse) round-trips to <1e-5 error on float32 and runs at ~1 ms per
2D DCT on ibm17 grid (44 × 51). The DCT is NOT the bottleneck — per-
step time is dominated by per-net-trace congestion (~0.5 s/step).
Per-step wall: V3 baseline ~0.4 s/step, eDensity variant ~0.5 s/step.

## Pointers
- Code: `code/edensity.py`, `code/diff_proxy_v3_edensity.py`
- Submission: `submissions/e111_minimal_edensity/placer.py` (kept; works
  but loses to V3)
- Sanity tests: `code/test_edensity_sanity.py` (DCT round-trip),
  `code/test_edensity_bench.py` (eDensity values on real bench)
- Comparison: `code/test_edensity_ibm17_fast.py` (the falsification run)
- Results: `results/ibm17_edensity.json`
- Reference: ePlace (Lu et al, ASP-DAC 2015), DREAMPlace (Lin et al,
  TCAD 2019), Xplace (Liao et al, ASP-DAC 2022).
