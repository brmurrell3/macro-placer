---
id: E12
name: grid_bin_lns
status: graduated
parent: E3
created: 2026-04-27
decided: 2026-04-28
champion_at_time: 1.1055
outcome: 1.0990
champion_delta: -0.0065 (-0.59%)
graduated_to: submissions/cd_lns_gridbin/placer.py
superseded_by: null
---

# E12: grid_bin_lns

## Hypothesis
The earlier LNS attempts (E3 v1 / v2) failed because their reinsertion step
used the same per-axis CD move type, so they could only re-discover CD's
fixed point. A *different move type* — full-canvas grid-bin search, where a
destroyed macro is re-evaluated at every (col, row) cell center — is outside
CD's reachable set and should be able to escape converged plateaus. This is
also the recipe implied by the leaderboard's "Incremental CD+LNS"
description.

## Method
Run CDAdaptive to plateau, then run grid-bin LNS as a polish phase. Adaptive
parameters: destroy size = 5 % of movable hard macros, capped at 30; destroy
selection via cost-aware ranking (with a random-destroy ablation in
`cd_lns_gridbin_random.py`). For each destroyed macro, enumerate all
`(grid_col, grid_row)` cell centers (~50–2 500 candidates per benchmark) and
take the globally best position via the incremental evaluator. Iterate up
to a per-benchmark wall budget (CD ≤ 3 000 s + LNS ≤ 600 s, total ≤ 3 600 s
hard cap). All hyperparameters are global; nothing is keyed on benchmark
name.

## Kill gate
0 / 8 LNS samples improve baseline by ≥ 0.5 % on the **median** benchmark
(deliberately not the worst or easiest, to avoid overfitting). If LNS shows
positive samples on the median, validate on `--fast` and at least one NG45
design before promoting to `--all`.

## Generalization check
After `--fast` passes, validate on at least one NG45 design (ariane133)
before declaring this the new champion. The cost-aware destroy ranking ablation
(`cd_lns_gridbin_random.py`) tests whether ranking is load-bearing or noise.

## Outcome (graduated 2026-04-28)

**Final result: avg proxy 1.0990 on `--all`, 17/17 IBM benchmarks, zero
overlaps.** Source JSON: `results/CDLNSGridBinPlacer_20260428_155739.json`.
Total wall 28 256 s = 7.85 hr (vs CDAdaptive 4.85 hr; the +3 hr is the LNS
phase running on every benchmark within the 600 s LNS budget).

Beats CDAdaptive (E9, 1.1055) by **−0.0065 (−0.59 %)**, beats leaderboard
1.1172 by **−1.63 %**, beats RePlAce 1.4578 by **−24.6 %**. All 17 benchmarks
improved over E16 — no regressions. Biggest wins on basin-locked /
high-density benches: ibm10 −1.51 %, ibm02 −0.81 %, ibm01 −0.99 %.

**Decision: promoted to champion 2026-04-28; supersedes E9 CDAdaptive.**
Production placer code at `submissions/cd_lns_gridbin/placer.py`. The
governing ADR is `docs/decisions/007_cd_lns_gridbin_promotion.md`.

The random-destroy ablation `cd_lns_gridbin_random.py` stays in `code/` as
the experiment's evidence that **cost-aware destroy ranking is not
load-bearing**: random destroy on `--fast` matched cost-aware within noise
(ibm09 random 0.8541 actually beat cost-aware 0.8591). Cost ranking adds
~10 % wall per LNS sample but does not change quality. Future
simplification: drop the ranking, use random destroy. Stays cost-aware in
the production placer for now to avoid mid-deadline changes.

Earlier in-flight datapoints (preserved for the lineage):

- ibm12 production smoke: CD plateau 1.20564 → LNS converged 1.20466 over
  5 samples (zero overlaps).
- Partial `--all` snapshot (12 / 17): avg 1.0362 vs E16 1.0291 over the
  same 12. Source: `results/partial/CDLNSGridBinPlacer_PARTIAL.json`.
- Random-destroy ablation (`--fast`): avg 0.9372 over ibm01/04/09/13.
  Source: `results/CDLNSGridBinRandomPlacer_20260428_093846.json`.

## Pointers
- Code (production): `submissions/cd_lns_gridbin/placer.py` (post-2026-04-28
  promotion).
- Code (ablation, kept here): `code/cd_lns_gridbin_random.py` — random-destroy
  variant, `--fast` only. Imports the production placer from
  `submissions/cd_lns_gridbin/placer.py` after the move.
- Results: `results/CDLNSGridBinPlacer_20260428_155739.json` (full `--all`,
  the verified 1.0990 champion result);
  `results/CDLNSGridBinRandomPlacer_20260428_093846.json` (random ablation,
  `--fast`); `results/partial/CDLNSGridBinPlacer_PARTIAL.json`
  (12-of-17 snapshot, superseded by full result).
- Decision: `docs/decisions/007_cd_lns_gridbin_promotion.md`.
- Discussion: `writeup/evidence.md` §7.7; `notes.md` ("Algorithmic findings"
  #4–5); `experiments/E12_grid_bin_lns/notes.md`.
- Parent: `experiments/E3_lns_v1/manifest.md` (subset-CD reinsert, falsified).
