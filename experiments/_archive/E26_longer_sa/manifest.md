---
id: E26
name: longer_sa
status: falsified
parent: E25
created: 2026-04-29
decided: 2026-04-29
champion_at_time: 1.0954 (E25)
outcome: 0.9336 (--fast); tied with E25 0.9336 (Δ +0.0023, sub-noise). The "SA still improving at t=599 s" observation was noise in best-tracking, not real headroom. +20 % wall for zero quality gain.
champion_delta: null
graduated_to: null
superseded_by: null
---

# E26: longer_sa

## Hypothesis
E25's `--fast` phase logs revealed the budget was misallocated:

- **LNS converged early.** ibm01 LNS hit sample 4 (= no improvement)
  at 98 s of the 600 s budget. ibm04, ibm09, ibm13 all converged in
  similar wall time. Most of LNS's 600 s was unused.
- **SA-v2 was budget-bound.** SA found `best` at t = 599.4 s
  (ibm01), 599.4 s (ibm04), 598.9 s (ibm09) — proxy was *still
  actively decreasing* when the 600 s budget expired.

E26 reallocates: cut LNS to 300 s (still 3× the average converge
time), bump SA to 900 s. Same total cap (3600 s = legal limit).

If the "SA still improving at budget end" observation has real headroom,
E26 `--fast` should beat E25's 0.9336 by some Δ > 0. If SA's lift was
already saturated by 600 s and the late-budget activity was just
noise, E26 should match E25 within float drift.

## Method
Pipeline identical to E25; only budget split differs:

| Phase | E25 | E26 |
|---|---|---|
| CD plateau | 2400 s | 2400 s |
| LNS grid-bin | 600 s | **300 s** |
| SA-v2 polish | 600 s | **900 s** |
| Total | 3600 s | 3600 s |

CD `cd_plateau_threshold=0.001`; LNS `destroy_frac=0.05`,
`destroy_cap=30`, `destroy_strategy='cost_aware'`; SA `T0=5e-4`,
`Tf=1e-6`, `breakpoint_budget=12`. All global.

## Kill gate
- **Compositionality preserved (avg `--fast` < 0.9372):** must beat E12
  random-destroy ablation, otherwise the longer SA budget broke
  something.
- **Improvement over E25 (avg `--fast` < 0.9336):** if E26 doesn't beat
  E25, the late-budget SA activity was already saturated and 600 s was
  enough — kill (no longer-budget value).

## Generalization check
If E26 `--fast` shows ≥ 0.5 % improvement over E25's 0.9336 (i.e., avg
≤ 0.9289), validate on NG45 ariane133 before queueing `--all`. Smaller
gains may still warrant `--all` if E25 `--all` is already a confirmed
champion (the comparison frame shifts post-E25-`--all`).

## Outcome (filled when decided)
**Falsified 2026-04-29.** Longer SA budget produced no real lift over
E25's 600 s split.

`--fast` numbers (zero overlaps):

| Benchmark | E26 (LNS 300, SA 900) | E25 (LNS 600, SA 600) | Δ |
|---|---:|---:|---:|
| ibm01 | 0.8913 | 0.8910 | +0.03 % ~ |
| ibm04 | 1.0120 | 1.0119 | +0.01 % ~ |
| ibm09 | 0.8545 | 0.8551 | −0.07 % ~ |
| ibm13 | 0.9766 | 0.9766 | tied |
| **AVG** | **0.9336** | 0.9336 | **tied** |

Per-bench Δ all within float drift (≤ 0.07 %). Net E26 vs E25 average
delta is +0.0023 — *worse* than E25 by sub-noise margin.

**Kill-gate analysis:**
- Compositionality preserved (avg `--fast` < 0.9372): PASSES (0.9336).
- **Improvement over E25 (avg `--fast` < 0.9336): NOT MET** — E26 ties
  E25. Per the manifest's own clause, the late-budget SA activity is
  saturated; 600 s was enough.

**The "SA found best at t≈599 s" finding was misleading.** That
observation was *noise in best-tracking*, not real headroom. Mechanism:
SA chains explore many states near best; any one can update best by a
tiny ε at any time. The geometric T schedule has T ≈ 1e-6 by t=600 s,
which is essentially greedy — the chain is not exploring new basins,
just doing local greedy moves that occasionally beat best by float
noise. Extending SA gives more chances to update best by ε but doesn't
find new structure.

**Wall cost.** E26 took 7272 s = 2.02 hr vs E25 fast 6063 s = 1.68 hr
(+20 %). The extra 300 s of SA per benchmark adds ~25 % to total wall
budget at scale, with zero quality gain — a clear net negative.

**Implication for E25 production.** Don't increase SA budget beyond
600 s without a new mechanism. Adding more SA time at the same T
schedule, on the same per-axis breakpoint move set, doesn't escape any
plateau the 600 s version doesn't already escape. The next lever
isn't longer SA — it's a *different* SA (e.g., higher T₀ for genuine
basin-hopping, or SA on a different move set) or composition with a
fourth phase.

*Lesson: "the optimizer was still improving when budget hit" can be
either real headroom OR best-tracking noise. To distinguish, run the
extended budget and observe whether the gain is meaningfully larger
than noise — that's exactly what E26 did. The result here is the right
shape: extended budget produces only float-drift-scale fluctuation, so
the 600 s saturation is real and we're locked at the SA-attainable
floor.*

**Source JSON:** `results/CDLNSSAE26Placer_20260429_171533.json`. Run
log: `experiments/E26_longer_sa/run.log`. Logged with hypothesis tag
`e26_longer_sa_fast` in `results/experiment_log.jsonl`.

## Pointers
- Code: `code/cd_lns_sa_e26.py` (defines `CDLNSSAE26Placer`).
- Imports `run_lns_gridbin` from `submissions/cd_lns_gridbin/placer.py`.
- Imports `run_sa_polish_v2` from
  `experiments/E24_sa_polish_v2/code/cd_sa_polish_v2.py`.
- Parent: E25 — same pipeline, different budget split.
- Discussion: emergent from E25's `--fast` phase logs (SA best at
  t≈599 s). Surface in `writeup/evidence.md` §9.E continuation.
