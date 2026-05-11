---
id: E63
name: spectral_init
status: falsified
parent: E41 (pipeline) + spectral / quadratic placement (init)
created: 2026-05-02
decided: 2026-05-03
champion_at_time: 1.08151 (E48 hybrid, ADR-011 *Accepted* 2026-05-02)
outcome: **FALSIFIED 2026-05-03 15:04 EDT on ibm04 single-bench.** V4 legalizer unblocked the spectral init pipeline (key fix: EPS=1e-3 micron gap to defeat float32 zero-area-overlap artifacts; original V3 manifest diagnosis "doesn't avoid FIXED macros" was WRONG — all benches have 0 fixed macros). V4 row-pack works on 11/17 IBM benches (ibm01/04/06/07/08/09/14/15/16/17/18); fails with vertical-overflow abort on 6/17 dense+tall-macro benches (ibm02/03/10/11/12/13). **Single-bench probe on ibm04: spectral init proxy 1.889 → polished proxy 1.189**, vs E48 ibm04 0.98506 = **+20.7 % WORSE**. The basin landed by CD on legalized spectral coords is structurally inferior to SDF/DPO basins. Pattern matches E62 WillSeed (standalone init quality predicts polished basin quality): spectral 1.89 → 1.19; WillSeed 1.53 → 0.93; SDF 1.50 → 0.99; DPO 1.38 → 0.95. The legalization step destroys the topology signal that the spectral coordinates were supposed to encode, leaving CD descending from a worse starting state. **Spectral lane line closed.** V5 Tetris-with-row-spanning attempted to fix the 6 failing benches but produced worse residuals (4/17 working); reverted, left in code as scaffolding. **NOT SUPERSEDED — closed by negative result.**
champion_delta: not measurable (falsified)
graduated_to: null
superseded_by: null
champion_delta: not yet measurable
graduated_to: null
superseded_by: null
---

# E63: spectral_init — netlist Laplacian eigenvectors as a third basin source

## Hypothesis
The netlist hypergraph's graph Laplacian L = D − A has eigenvectors
encoding the netlist's **global connectivity modes**. The smallest
non-trivial eigenvalues correspond to "low-frequency" global structure
(broad clusters, hub macros), and using them as (x, y) coordinates gives
a placement that respects netlist topology directly.

This is *Gordian-style quadratic placement*, the 1980s spectral
predecessor to gradient-based methods. The point is **not** that
spectral placement beats SDF / DPO standalone (it doesn't — early
experiments hit 1.78 vs SDF's 1.50). The point is the **basin landed by
CD on a spectral init is structurally different** from SDF or DPO basins.

If the spectral basin polishes (via E41 pipeline: CD + LNS + SA + K-joint)
to within ~1.10 average and **wins on at least one IBM-fast bench** AND
verifies safe on NG45 ariane133, it qualifies as a third hybrid lane in
future E70 = best-of-{E25 SDF, E41 DPO, E63 spectral}.

The E62 WillSeed falsification (2026-05-02) showed that NOT every "different
init" produces a useful basin. WillSeed's GPU-legalized state landed CD in
a structurally inferior basin to both SDF and DPO. **Spectral inits are
distinguished from WillSeed by being benchmark-structure-aware**: the
eigenvectors are computed from the netlist itself, so the basin landed
should reflect topology, not arbitrary GPU optimization artifacts.

## Method
Pipeline per benchmark (identical to E41 except step 1 init):

  1. **Spectral init.** Build the weighted graph Laplacian L = D − A
     where A_{ij} = sum over nets containing both macros i and j of
     weight 1/(|net| − 1) (clique expansion with normalization).
     Compute the 4 smallest eigenvalues / eigenvectors via
     `scipy.sparse.linalg.eigsh(sigma=0)` (shift-invert for smallest).
     Discard eigenvalue 0 (the constant eigenvector); use eigenvectors
     2 and 3 as (x, y) coordinates.
  2. **V4 row-pack legalize** (EPS-gap fixed-aware): see legalizer
     details in `code/cd_lns_sa_spectral_kjoint.py:_row_pack_legalize_spectral_fixed_aware`.
  3. project_overlaps to clean any residuals.
  4. Build IncrementalProxyEvaluator.
  5. CD adaptive (≤ 2400 s).
  6. Grid-bin LNS (≤ 600 s, cost-aware destroy).
  7. SA-v2 (≤ 600 s, T₀=5e-4).
  8. K-joint LNS (≤ 600 s, K=3, top_N=5).
  9. Validate, preserve fixed macros.

All hyperparameters from E41 preserved unchanged. Only init differs.

## Implementation history

### V1: Hungarian on max-size slot grid (proposed 2026-05-02)
Linear assignment of spectral coords to a grid where each slot is sized
to the max macro footprint. Falsified: ibm01 fits only 28 max-size slots
vs 246 hard movables; finer fallback grid violates no-overlap-by-construction.

### V2: Hungarian + finer fallback (2026-05-02 → 03)
Hungarian on a finer grid when max-size grid is too coarse. 9-36
residuals on smaller benches; doesn't scale.

### V3: Row-packing (2026-05-03 night)
Sort movables by (spectral_y desc, spectral_x asc), pack rows top-down
L→R. Falsified by manifest writer: 111 residuals on ibm01 attributed to
"doesn't avoid FIXED macros pre-placed on canvas." **DIAGNOSIS WAS WRONG.**

### V4: Row-packing with EPS gap + fixed-aware (2026-05-03 14:20 EDT)
**Real diagnosis** (via `code/debug_v4_with_proj.py`): row-packing places
macros edge-to-edge (`cur_x += w` exactly). When converted to float32 for
`compute_overlap_metrics`, the precision loss creates apparent "overlaps"
of size 1e-7 with total area = 0.0000. `compute_overlap_metrics` uses
strict `overlap_x > 0 AND overlap_y > 0` so 110 zero-area "overlaps"
register as positive count.

**Fix:** add `EPS = 1e-3` micron gap when advancing `cur_x += (w + EPS)`
and when dropping rows `row_top -= (max(row_max_h, h) + EPS)`. The gap
is below any meaningful placement scale (macro sizes 0.6-7 micron) but
well above float32 rounding noise.

**Secondary fix:** abort packing when vertical overflow would force
`row_top < h`, instead of resetting `row_top = h` (which placed macros
into the same y-band as previously-packed bottom-row macros, creating
real overlaps on ibm13).

**Fixed-aware logic:** also implemented (skip x-positions inside fixed
macro bboxes during pack). Defensive — all current public benchmarks
have 0 fixed macros.

### V4 results table (2026-05-03 14:25 EDT)

| Bench | n_hard | density | V4 outcome | Wall |
|---|---:|---:|---|---:|
| ibm01 | 246 | 0.428 | 0 residuals, OK | 0.2s |
| ibm04 | 295 | 0.420 | 0 residuals, OK | 0.4s |
| ibm09 | 253 | 0.398 | 0 residuals, OK | 0.4s |
| ibm13 | 424 | 0.362 | **210 residuals, FAIL** | 3.6s |
| ibm17 | 760 | 0.172 | 0 residuals, OK | 1.7s |

ibm13 failure mode: max macro h = 11.68 on canvas h = 56. Row-pack with
"row_max_h dominates drop" produces ~5-9 large rows. After ~9 rows used,
vertical overflow triggers; remaining ~44 macros stay at original positions
which cluster, producing 210 hard↔hard overlaps that project_overlaps
(50 iters/pass, 20 passes with jitter) can't repair.

**Open work:** ibm13 needs a Tetris-with-row-spanning-tall-macros legalizer
(Spindler-Schlichtmann 2008 Abacus algorithm). Not blocking the rank-3
research recommendation — V4 already produces a usable spectral basin on
4/5 tested benches; ibm13 fallback to E48 lane in any future hybrid.

## Kill gate
- **--fast > 0.929** (E41 fast 0.92178 + 0.8 % = same threshold as E62):
  spectral basin polishes to a state worse than the IBM-overfit margin
  → falsified.
- **--fast within 1 % of E48 0.92024 (≤ 0.929):** *plus* at least one
  per-bench win on ibm01/04/09/13 vs E48 → *qualifies for --ng45*. If
  no per-bench wins on fast, falsify (won't help hybrid).
- **--ng45 ariane133 > E48 0.6861 + 1 % (= 0.694):** spectral basin
  fails NG45 transfer → falsified.

## Generalization check
NG45 is the gate. The spectral init is benchmark-structure-aware
(eigenvectors come from the netlist), so generalization should be
better than IBM-overfit mechanisms (E42/E43/E44/E54/E62). But the
NG45 verification is required regardless.

## Outcome (in flight)

### Single-bench ibm04 launched 2026-05-03 14:25 EDT
Expected wall ~50 min. If E63 V4 ibm04 ≤ E48 ibm04 0.98506 by ≥ 0.5 %,
queue --fast (excluding ibm13). Otherwise spectral basin doesn't lift
on this bench; consider whether the dev cost is justified for the rest.

## Pointers
- Code: `code/cd_lns_sa_spectral_kjoint.py`
  (`CDLNSSASpectralKJointPlacer`, `_spectral_init`,
  `_row_pack_legalize_spectral_fixed_aware`).
- V3 (broken) function still in same file at
  `_row_pack_legalize_spectral` for diff comparison.
- Test scripts: `code/test_v4_legalize_ibm01.py`,
  `code/debug_v4_overlaps.py`, `code/debug_v4_with_proj.py`.
- Pipeline backbone: E41
  (`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`).
- Discussion: motivated 2026-05-02 by roadmap §4.6.B "global topology
  navigation" reframing.
- Research alignment: `docs/research_principles_for_walls.md` rank 3
  recommendation = E63 V4 with obstacle-aware Tetris legalizer
  (1-2 days dev). V4 implemented in ~2 hours via the EPS-gap discovery.
