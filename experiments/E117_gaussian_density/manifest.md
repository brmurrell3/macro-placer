---
id: E117
name: gaussian_density
status: in_progress
parent: E111
created: 2026-05-20
decided: null
champion_at_time: 1.0575   # Option C (stacked_periphery)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E117: Gaussian-smoothed density model

## Hypothesis
Our differentiable `_grid_density` model (used by V3 Adam descent) is
piecewise-linear in macro positions with step discontinuities at cell
boundaries: the per-cell overlap-area function is C^0 there. On dense
benches (ibm12/14/17/18) Adam sees either zero gradient (macro entirely
inside one cell) or a gradient that jumps when a macro corner crosses a
bin edge. This stalls descent at the V3 basin and forces CD polish to do
all the dense-region work.

Replacing the hard rectangle with a 2D Gaussian (sigma = half-macro-size)
and analytically integrating over each cell (via erf) gives a C^inf
smooth density field. Adam sees a continuous gradient everywhere.
Hypothesis: this lowers V3's raw smooth-proxy basin on ibm17 (from ~1.30)
and improves the +CD600s polished result (from ~1.20).

This is a SIMPLER variant of E114 electrostatic eDensity — no Poisson
FFT, just Gaussian smearing.

## Method
1. Implement `gaussian_density.py` with analytic erf-based per-cell
   integration of 2D Gaussians.
2. Subclass `DiffProxyV3` → `DiffProxyV3GaussianDensity` that swaps the
   density model. Keep V3's per-net-trace congestion and LSE-HPWL.
3. Drive via `SmoothGlobalPlacerV3GaussianDensity` (mirrors
   `SmoothGlobalPlacerV3`).
4. Build `submissions/e111_minimal_gaussian/placer.py` (mirrors
   `e111_minimal_ovl10_720s`: 720s budget, 600s CD polish).

## Kill gate
- ibm17 raw smooth proxy: must be < 1.30 (V3 baseline), else density
  model doesn't help even before polish.
- ibm17 +CD600s polished: must be < 1.20 (V3Min ovl10 720s baseline),
  else density-smoothing doesn't transfer through legalize+polish.

## Generalization check
If ibm17 wins, run on dense benches {ibm12, ibm14, ibm17, ibm18}. If
≥3/4 improve, run `--all`. Champion gate: avg < 1.068 (current Option B
floor).

## Outcome (filled when decided)

**Dense benches result (2026-05-20):**

| Bench | V3 grid baseline | V3 Gaussian (E117) | Lift |
|---|---:|---:|---:|
| ibm12 | 1.1377 | **1.0869** | **-4.47 %** |
| ibm14 | 1.0840 | **1.0771** | **-0.64 %** |
| ibm17 | 1.2003 | **1.1805** | **-1.65 %** |
| ibm18 | 1.2357 | **1.1871** | **-3.93 %** |
| **avg (4 dense)** | **1.1644** | **1.1329** | **-2.71 %** |

4/4 dense benches improved. All zero overlaps, qualified.

Cost decomposition:
- ibm12 (-4.47%): wl=0.079, den=0.527, cong=1.489
- ibm14 (-0.64%): wl=0.067, den=0.552, cong=1.469
- ibm17 (-1.65%): wl=0.072, den=0.541, cong=1.677
- ibm18 (-3.93%): wl=0.074, den=0.587, cong=1.638

ibm17 descent-only diagnostic (no CD polish):
- V3 grid descent: canonical=1.28548 (smooth=1.273)
- V3 Gauss descent: canonical=1.28077 (smooth=1.187)

The descent's canonical proxy difference is small (-0.37 % on ibm17), but
the CD-polished result shows much larger lift (-1.65 to -4.47 %). The
Gaussian basin polishes BETTER than the grid basin — likely because the
smoother gradient produces a placement with less micro-overlap structure
that legalize+CD can resolve more efficiently. Also note CD got less
budget under contention on ibm17 (172s vs baseline 600s), yet still won.

## Recommendation
**SHIP and scale to --all.** All 4 dense benches improved (-0.64% to -4.47%),
average -2.71%. The Gaussian-smeared density model gives Adam a C^inf
smooth gradient (vs C^0 grid-bin), producing basins that polish more
efficiently. The implementation is a ~50-line erf-based replacement for
`_grid_density` — much simpler than Poisson-FFT eDensity (E114). No
external dependencies. Pure CPU/torch.

Status: pending --all verification (~14 hr serial). Likely candidate
for new Tier-1 floor pending the multi-bench result.

## Pointers
- Code:
  - `code/gaussian_density.py` (erf-based density model)
  - `code/diff_proxy_v3_gaussian_density.py` (V3 subclass + driver)
- Submission: `submissions/e111_minimal_gaussian/placer.py`
- Parent baseline: `submissions/_archive/e111_minimal_ovl10_720s/placer.py`
- Comparable experiment: E114 (electrostatic eDensity, FFT-based —
  not yet validated, more complex)
