---
id: E24
name: sa_polish_v2
status: marginal
parent: E14
created: 2026-04-29
decided: 2026-04-29
champion_at_time: 1.0990
outcome: 0.9366 (--fast); +0.63 % lift over E16 baseline 0.9426 — passes gen-check; ties E12 random-destroy ablation 0.9372 within noise; ibm13 saw zero SA lift (best == CD plateau)
champion_delta: null
graduated_to: null
superseded_by: null
---

# E24: sa_polish_v2

## Hypothesis
E14's SA polish was falsified on `--fast` at avg 0.9962 (+5.7 % WORSE
than E16 baseline 0.9426 on every benchmark) due to two implementation
issues that compounded:

1. **No best-so-far tracking.** v1 returned whatever placement was at
   `evaluator.placement` when budget expired — not the best ever visited.
2. **T₀ = 0.01 was too high vs Δ-scale ~1e-3.** Early acceptance of
   typical worsening moves was `exp(-0.1) ≈ 0.9`, i.e. a 50/50 random
   walk, not tunneling.

E24 is the principled retest of the E14 hypothesis — *same per-axis
breakpoint move set CD already searched, but Metropolis acceptance lets
the search jump out of the basin CD is stuck in* — with both issues
fixed. With best-tracking the worst-case SA run is "ties with CD
plateau"; with low T₀ + best-tracking, any improvement found is a real
escape from CD's per-axis fixed point, not a lucky end-of-chain state.

## Method
After CDAdaptive converges (cap = 3000 s, plateau = 0.001 — same as E12
and E14), run SA polish v2 for 600 s:

- Pick a random hard-movable macro and a random axis (uniform).
- Build legal range via `cd_core.legal_axis_range`; build candidate set
  via `cd_core.axis_breakpoints` (identical to CD's per-axis search).
- Pick one breakpoint uniformly at random (excluding current position).
- `evaluator.move(idx, new_xy)`; query proxy; Metropolis: accept if
  Δ ≤ 0; else accept with probability `exp(-Δ / T)`. On reject,
  `evaluator.revert()`.
- **Track `best_proxy` and `best_placement` across the chain.** On every
  accepted move, if `cur_proxy < best_proxy` snapshot
  `evaluator.placement.clone()`.
- Geometric cooling against `elapsed / budget`:
  `T(t) = T₀ · (T_f / T₀) ^ (t / sa_budget_s)`.
- **At budget exhaustion, restore the evaluator to `best_placement`** by
  walking from chain-last to best via per-macro `move()` calls (keeps
  the V/H congestion cache in sync — direct mutation of
  `evaluator.placement` would desync).

Hyperparameters (global, no per-benchmark tuning):
- CD: same as E12/E14 (`cd_hard_cap_s=3000`, `cd_min_time_s=300`,
  `cd_patience=3`, `cd_plateau_threshold=0.001`).
- SA: `sa_budget_s=600`, **`sa_T0=5e-4`**, **`sa_Tf=1e-6`**, `sa_seed=42`,
  `sa_breakpoint_budget=12`.

T₀ rationale (Δ-scale ~1e-3 from E14's data): early acceptance of
typical worsening moves is `exp(-1e-3 / 5e-4) = exp(-2) ≈ 13.5 %` —
modest exploration with downhill bias, not random walk. Late
acceptance is `exp(-1e-3 / 1e-6) ≈ 0` — greedy. Three decades of
cooling.

## Kill gate
- **Strict gate (avg `--fast` ≥ 0.9425):** if SA-v2 doesn't beat the E16
  baseline on average, kill — same as E14's gate.
- **Non-regression gate (avg `--fast` ≥ 0.9426):** with best-tracking,
  the worst case should be "matches CD plateau." If avg is *worse* than
  CD-only (E16 baseline 0.9426), there's a bug in best-restore or the
  per-macro walk. Kill and inspect.

## Generalization check
If `--fast` shows ≥ 0.5 % improvement over the E16 baseline (avg ≤
0.9379), validate on NG45 ariane133 before queueing `--all`.

## Outcome (filled when decided)
**Marginal — passes gen-check threshold but doesn't beat E12-random
ablation (2026-04-29).**

`--fast` numbers (zero overlaps everywhere):

| Benchmark | CD plateau | E24 SA-v2 best (internal) | Internal lift | Eval result | E13 LNS | E12-random | E16 baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| ibm01 | 0.91293 | 0.89805 | **−1.63 %** | 0.8989 | 0.9049 | 0.9073 | 0.9135 |
| ibm04 | 1.01468 | 1.01134 | −0.33 % | 1.0128 | 1.0150 | 1.0150 | 1.0179 |
| ibm09 | 0.85680 | 0.85317 | −0.42 % | 0.8561 | 0.8573 | 0.8541 | 0.8605 |
| ibm13 | 0.97136 | 0.97136 | 0.00 % | 0.9785 | 0.9763 | 0.9724 | 0.9785 |
| **AVG** | — | — | — | **0.9366** | 0.9384 | 0.9372 | 0.9426 |

**Lift = +0.63 % vs E16 baseline (0.9426).** Above the 0.5 % gen-check
threshold; below the 1 % "this is a real win" bar.

**SA found best near end of budget on every winning bench.** ibm01 best
at t = 599.4 s, ibm04 at 599.4 s, ibm09 at 598.9 s. The 600 s SA budget
was *barely enough* — proxy was still actively decreasing when budget
expired. More time would help.

**ibm13 saw zero SA lift** (best == CD plateau, found at t = 0.0 s). The
SA chain explored 458k proposals over 600 s and never found anything
below the CD plateau. ibm13's plateau is genuinely robust to per-axis
breakpoint Metropolis moves. This is the same pattern likely to hold for
the harder `--all` benchmarks (ibm17, ibm18) — which is why the
expected `--all` lift is smaller than the `--fast` 0.63 %.

**Best-so-far + restore mechanics worked correctly.** ibm13 SA chain
ended at proxy 0.97268 (worse than CD plateau); the best-restore loop
walked the evaluator from chain-last to the t=0 snapshot via per-macro
moves, returning evaluator-final = 0.97136 (= CD plateau, as expected).
The bug that hosed E14 (returning chain-final not best) is fixed.

**Vs E12 random-destroy ablation (the closer baseline):** E24 SA-v2 is
−0.06 % better — within noise. SA-on-breakpoints and LNS-grid-bin both
provide ~0.6 % lift over CD plateau on `--fast`, but they may target
DIFFERENT improvements. Compositionality is testable: E25 = CD + LNS +
SA-v2 might add their lifts. Surfaced separately as
`experiments/E25_lns_sa_compose/`.

**Decision: marginal — did not queue `--all`.** The expected `--all`
lift after attenuation (~0.2–0.4 %) puts E24-only at ~1.095 vs champion
1.0990 — a tie at best, not a champion. Higher EV is testing
compositionality (E25). Surface to human for verdict on whether to
revisit with a longer SA budget (e.g., 1200 s) or more aggressive cooling
schedule.

**Source JSON:** `results/CDSAPolishV2Placer_20260429_030032.json`. Run
log: `experiments/E24_sa_polish_v2/run.log`. Logged with hypothesis tag
`e24_sa_polish_v2_fast` in `results/experiment_log.jsonl`.

## Pointers
- Code: `code/cd_sa_polish_v2.py` (defines `CDSAPolishV2Placer`).
- Predecessor: `experiments/E14_sa_polish/` (v1 — falsified due to no
  best-tracking + T₀ too high; see `experiments/E14_sa_polish/manifest.md`
  Outcome section).
- Reference: `submissions/cd_lns_gridbin/placer.py` (E12 pattern this
  mirrors), `macro_place/cd_core.py` (`legal_axis_range`,
  `axis_breakpoints`, `_grid_lines`, `run_cd_adaptive`).
- Parent: ADR-003 ("CD plateau-bound, not budget-bound") via E14.
- Lesson source: `writeup/evidence.md` §9.B (E14 falsification) and
  `~/.claude/projects/.../memory/feedback_sa_implementation.md`.
