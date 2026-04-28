# ADR-007: Promote CD + grid-bin LNS as champion (E12 → 1.0990)

**Status:** Accepted. Supersedes ADR-003 as the champion-bearing decision (ADR-003
still describes the CD foundation; ADR-007 adds the LNS overlay on top of it).
**Date:** 2026-04-28
**Deciders:** project owner

## Context

CDAdaptive (ADR-003) hit 1.1055 on `--all` — beat the public leaderboard
1.1172 by −1.05 % — but the run signature was striking: every one of the 17
benchmarks exited via plateau detection, none hit the 1-hour cap. The
champion was *plateau-bound*, not budget-bound. The plateau is the per-axis
fixed point of CD's breakpoint enumeration: with the same move type, more
wall-clock cannot escape it.

Three candidate escape mechanisms were tested and falsified before ADR-007:

- **E3 LNS rip-up-and-reinsert.** v1 (full-canvas, 2 244 grid candidates, 428 s
  per LNS iteration) and v2 (5×5 local-window, 90× faster) both produced
  flat ibm17 results vs CDOnly (1.3846 / 1.3824 vs 1.3830). Single-macro
  destroy/reinsert with a *single-axis* reinsertion step does not escape
  CD's local minimum. (`experiments/E3_lns_v1/`,
  `experiments/E3_lns_v2/`.)
- **Multi-init via SDF jitter.** 8 jittered SDF inits on ibm09
  (σ ∈ {0.02, …, 0.10}·canvas) — 0/8 improved the unjittered baseline.
  SDF→CD is contractive; perturbing strictly hurts.
  (`analysis/multi_init_probe/`.)
- **Subset-CD destroy-and-reinsert.** 24 random destroy-and-reinsert samples
  across ibm09 + ibm12 (varying K, jitter, destroy strategy) — 0/24
  improved baseline by ≥ 0.1 %. The reinsertion uses CD's per-axis move
  type, so it finds the same per-axis fixed point.
  (`analysis/lns_escape_probe/`.)

The diagnosis: escaping CD requires a *different move type*, not more time
spent on the same move type. Grid-bin LNS searches all `(col, row)` cell
centers — a different candidate set than CD's per-axis breakpoints, and
exactly the move-type the leaderboard winner's "Incremental CD+LNS"
description implies.

## Decision

Adopt CD + grid-bin LNS as the champion. CD runs to plateau (per ADR-003),
then grid-bin LNS layers on top: destroy 5 % of movable hard macros (capped
at 30), enumerate `(grid_col × grid_row)` candidate positions for each
destroyed macro, accept the globally proxy-minimizing legal reinsertion via
the incremental evaluator (ADR-002). LNS iterates until a sample produces
no improvement or the wall budget is exhausted.

Production budget: CD ≤ 3 000 s + LNS ≤ 600 s = 3 600 s total per benchmark
— at the contest 1-hour-per-bench cap. Defaults are global (no
per-benchmark tuning).

## Consequences

- **Champion:** 1.1055 → **1.0990** (−0.0065, −0.59 %), −24.6 % vs RePlAce
  1.4578, −1.63 % vs leaderboard 1.1172. Verified `--all`, all 17 IBM
  benchmarks, zero overlaps. All 17 improved over E16 (no regressions);
  biggest wins on basin-locked / high-density benchmarks (ibm10 −1.51 %,
  ibm02 −0.81 %, ibm01 −0.99 %).
- **Total wall:** 4.85 hr (CDAdaptive) → 7.85 hr (E12) on `--all`. Inside
  the 17-hr competition envelope (17 × 1 hr) but with less margin than
  CDAdaptive. The added cost is the LNS phase running on every benchmark,
  including those CD already plateaued comfortably on; the LNS wall is
  bounded by the 600 s phase budget.
- **Algorithmic interpretation.** Validates the leaderboard winner's
  "Incremental CD+LNS" architecture without needing to reproduce theirs.
  The escape mechanism is move-type, not random restart, not basin-hopping:
  grid-bin candidates exist outside CD's per-axis breakpoint set.
- **Random destroy on `--fast` matched cost-aware within noise** (random
  ibm09 0.8541 beat cost-aware 0.8591). Cost ranking adds ~10 % wall per
  LNS sample but doesn't change quality. Future simplification: drop the
  ranking, use random. Stays cost-aware in the production placer for now
  to avoid mid-deadline changes; ablation at
  `experiments/E12_grid_bin_lns/code/cd_lns_gridbin_random.py`.
- **Open follow-up.** ADR-005's SDF-init transfer reasoning carries over;
  ADR-007 needs its own NG45 datapoint before treating LNS-overlay
  transfer as empirically confirmed.
- **Alternatives ruled out.** Single-macro local LNS, SDF-jitter multi-init,
  subset-CD destroy/reinsert — all three (cited in Context) failed by
  attempting to escape CD with the same move type.

## Evidence

- `submissions/cd_lns_gridbin/placer.py` — champion code (promoted
  2026-04-28).
- `experiments/E12_grid_bin_lns/manifest.md` — graduated 2026-04-28.
- `experiments/E12_grid_bin_lns/notes.md` — algorithmic findings,
  cost-aware-vs-random ablation.
- `results/CDLNSGridBinPlacer_20260428_155739.json` — verified `--all`
  result: avg 1.0990, 0 overlaps, 28 256 s total wall.
- `analysis/multi_init_probe/`, `analysis/lns_escape_probe/`,
  `experiments/E3_lns_v1/`, `experiments/E3_lns_v2/` — escape-mechanism
  falsifications cited in Context.
- `writeup/evidence.md` §7.5, §9.6, §9.7.
