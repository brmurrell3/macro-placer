---
id: E14
name: sa_polish
status: falsified
parent: ADR-003
created: 2026-04-28
decided: 2026-04-29
champion_at_time: 1.0990
outcome: 0.9962 avg (--fast); +5.7% WORSE than E16 baseline 0.9425 on every benchmark; SA actively destroyed CD's progress
champion_delta: null
graduated_to: null
superseded_by: null
---

# E14: sa_polish

## Hypothesis
CD plateaus when every macro sits at its single-macro per-axis fixed point
(ADR-003: plateau-bound, not budget-bound). E15 (pair_swap) and E12's LNS
test the *structural* lever — different move types. E14 tests the
*acceptance-rule* lever instead: same per-axis breakpoint move set CD
already searched, but Metropolis acceptance with a global temperature
schedule. Worsening moves can be accepted probabilistically — explicit
tunneling out of CD's fixed point with the same primitives, just under
SA rather than greedy.

## Method
After CDAdaptive converges (cap=3000s, plateau=0.001 — same as E12), run
SA polish for 600s:
- Pick a random hard-movable macro and a random axis (uniform).
- Build legal range via `cd_core.legal_axis_range`; build candidate set
  via `cd_core.axis_breakpoints` (identical to CD's per-axis search).
- Pick one breakpoint uniformly at random (excluding current position).
- `evaluator.move(idx, new_xy)`; query proxy; Metropolis: accept if
  Δ ≤ 0; else accept with probability `exp(-Δ / T)`. On reject,
  `evaluator.revert()` (single-step revert stack — must revert
  immediately).
- Geometric cooling against `elapsed / budget`:
  `T(t) = T0 * (Tf / T0) ** (t / sa_budget_s)`.

Hyperparameters (global, no per-benchmark tuning):
- CD: `cd_hard_cap_s=3000`, `cd_min_time_s=300`, `cd_patience=3`,
  `cd_plateau_threshold=0.001`.
- SA: `sa_budget_s=600`, `sa_T0=0.01`, `sa_Tf=1e-5`, `sa_seed=42`,
  `sa_breakpoint_budget=12`.

T0 / Tf chosen against observed CD-plateau Δ-magnitudes (~1e-3): early
acceptance of typical worsening moves is `exp(-1e-3 / 1e-2) ≈ 0.9`; late
acceptance is `exp(-1e-3 / 1e-5) ≈ 0`. About three decades of cooling.

## Kill gate
Per `docs/roadmap.md` E14 entry: "SA polish on `--fast` shows no
improvement over CDAdaptive baseline → kill." Concretely: if the
`--fast` avg ≥ E16's `--fast` baseline of 0.9425 (within float-noise
band of ~0.001), kill.

## Generalization check
If `--fast` shows ≥ 0.5% improvement over the E16 baseline, validate on
NG45 ariane133 before scaling to `--all`.

## Outcome (filled when decided)
**Falsified 2026-04-29.** SA polish at the configured temperature
schedule actively *destroys* CD's progress instead of escaping its
plateau.

`--fast` numbers (zero overlaps everywhere):

| Benchmark | E14 SA | E16 baseline | Δ |
|---|---:|---:|---:|
| ibm01 | 0.9661 | 0.9135 | **+5.8 %** |
| ibm04 | 1.0801 | 1.0179 | **+6.1 %** |
| ibm09 | 0.9056 | 0.8605 | **+5.2 %** |
| ibm13 | 1.0330 | 0.9785 | **+5.6 %** |
| **AVG** | **0.9962** | 0.9425 | **+5.7 %** |

Kill gate (`--fast` avg ≥ 0.9425): **CLEARLY HIT** at 0.9962. Worse than
baseline on every single benchmark.

**Failure mechanism (visible in SA log on ibm13).**
- CD phase plateaus at proxy 0.97136 (sweep 13, wall 1616 s).
- SA budget = 600 s. Within 30 s of SA start, proxy has *risen* to
  1.24631 — a +28 % regression from the CD plateau.
- Throughout SA: `accepted_better ≈ accepted_worse` (177 845 vs 172 062
  on ibm13). Net SA delta is **+0.05578** (positive — proxy went UP).
- Cooling pulls proxy back to 1.02713 by t=600 s, but never recovers
  below the CD plateau.

**Two compounding implementation/design issues:**

1. **No best-so-far tracking.** SA returns whatever placement happens to
   be at `evaluator.placement` when the budget expires — not the best
   ever seen. This is a textbook SA bug; standard implementations track
   `best_placement` separately and restore it before return.

2. **T₀ = 0.01 is too high relative to per-move Δ scale.** The manifest's
   own design rationale was "early acceptance of typical worsening moves
   is exp(−1e−3 / 1e−2) ≈ 0.9." That's not "exploration with occasional
   tunneling" — it's a 50/50 random walk. The SA log confirms: better and
   worse acceptance rates are roughly equal across the entire schedule.

Both issues need fixing before SA is a fair test of the
acceptance-rule-vs-greedy lever; either alone could explain the
regression. **Status: falsified as configured.** A revised SA with
best-so-far tracking and T₀ ~ 1e-4 (so exp(−1e-3 / 1e-4) ≈ exp(−10)
≈ 5e-5 — worsening moves rare, real tunneling) is a separate
hypothesis; do not credit this attempt as testing it.

**Lesson for the writeup.** The roadmap (Tier-1 E14) cited SA as "a
known-strong method"; this run shows the strength is implementation-
sensitive. The candidate-set lever (E12 grid-bin LNS, 1.0990) and the
acceptance-rule lever (this attempt, 0.9962 — degraded) are not symmetric
— same move set under Metropolis is dominated by greedy CD here.

**Source JSON:** `results/CDSAPolishPlacer_20260429_001135.json`. Run
log: `experiments/E14_sa_polish/run.log`. Logged with hypothesis tag
`e14_sa_polish_fast` in `results/experiment_log.jsonl`.

## Pointers
- Code: `code/cd_sa_polish.py` (defines `CDSAPolishPlacer`).
- Reference: `submissions/cd_lns_gridbin/placer.py` (E12 — pattern this
  mirrors), `submissions/cd_adaptive/placer.py` (E9 — CD-only baseline),
  `macro_place/cd_core.py` (`legal_axis_range`, `axis_breakpoints`,
  `_grid_lines`, `run_cd_adaptive`).
- Sibling: `experiments/E15_pair_swap/manifest.md` (different move-type
  lever — falsified, same parent).
- Parent: ADR-003 ("CD plateau-bound, not budget-bound").
