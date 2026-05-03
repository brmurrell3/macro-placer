# ADR-008: Promote CD + LNS + SA-v2 as champion (E25 → 1.0954)

**Status:** **Superseded by ADR-011** 2026-05-02. Never reached *Accepted*.
The verified `--all` result (1.0954, zero overlaps, 11/17 wins, 4/17 ties,
2/17 sub-noise regression) cleared all decision-rule thresholds at
2026-04-29, but the human elected to flag E25 as a champion *candidate*
rather than auto-promote. Subsequent candidates E18 (ADR-009) at 1.08979,
E41 (ADR-010) at 1.0848, and finally E48 hybrid (ADR-011) at **1.08151**
each strictly improved on E25, and ADR-011 is the accepted promotion.
E25 code remains at `submissions/cd_lns_sa/placer.py` because E48 calls
it as a component pipeline (one of two best-of lanes).
**Date:** 2026-04-29 (proposed) → 2026-05-02 (superseded by ADR-011)
**Deciders:** project owner

## Context

E12 CDLNSGridBin (ADR-007) hit 1.0990 on `--all` — beat the public
leaderboard 1.1172 by −1.63 % — with the structural insight that CD's
per-axis fixed point is escaped by a *different move type*: grid-bin
`(col, row)` cell centers, outside CD's per-axis breakpoint set.

Two follow-up questions remained:

1. **Is the *acceptance rule* a complementary lever, separate from the
   move type?** CD greedily accepts only Δ ≤ 0; perhaps Metropolis
   acceptance over the *same* per-axis breakpoint set finds breakpoint
   moves CD's sweep order missed.
2. **Are LNS and SA-on-breakpoints compositional?** If LNS-grid-bin and
   SA-on-breakpoints target *different* improvements, running them in
   sequence after CD adds their lifts.

E14 (SA polish v1) was falsified at avg `--fast` 0.9962 (+5.7 % WORSE
than baseline 0.9426) due to two implementation bugs: no best-so-far
tracking + T₀ = 0.01 too high (50/50 random walk, not tunneling).

E24 (SA polish v2) fixed both: best-so-far placement tracking with a
restore-via-per-macro-move walk that keeps the V/H congestion cache in
sync, plus T₀ = 5e-4 (early acceptance of typical worsening moves at
exp(−2) ≈ 13.5 %, modest exploration with downhill bias). E24
`--fast` = 0.9366: matched E12-random-destroy ablation 0.9372 within
noise. SA-on-breakpoints is *not worse* than LNS-grid-bin, and on the
easier benches (ibm01, ibm04, ibm09) it found wins LNS-alone left on
the table.

E25 composed E24's SA-v2 with E12's LNS overlay: CD ≤ 2400 s + LNS
≤ 600 s + SA ≤ 600 s = 3600 s total. `--fast` = **0.9336** (best ever),
beating E12-random ablation 0.9372 by 0.39 % — clear evidence of
compositionality. `--all` = **1.0954**, beating E12 1.0990 by 0.33 %,
zero overlaps everywhere.

## Decision (proposed)

If accepted: adopt CD + grid-bin LNS + SA-v2 polish as the champion. The pipeline
runs CD to plateau (per ADR-003), then layers grid-bin LNS (per
ADR-007), then SA-v2 polish on per-axis breakpoints with best-so-far
tracking and low T₀ (E24's fix). The three phases target distinct
candidate sets:

| Phase | Candidate set | Acceptance |
|---|---|---|
| CD | per-axis breakpoints | greedy (Δ ≤ 0 only) |
| LNS | grid `(col, row)` cell centers | greedy (best of cell candidates) |
| SA-v2 | per-axis breakpoints | Metropolis, geometric T₀ → T_f |

Production budget: CD ≤ 2 400 s + LNS ≤ 600 s + SA ≤ 600 s = 3 600 s
per benchmark — at the contest 1-hour-per-bench cap. CD cap is tightened
from E12's 3 000 s to 2 400 s to make room for SA; on `--all` no
benchmark hit the per-bench cap. Defaults are global (no per-benchmark
tuning).

## Consequences

- **Champion:** 1.0990 → **1.0954** (−0.0036, **−0.33 %**), −24.9 % vs
  RePlAce 1.4578, **−1.95 % vs leaderboard 1.1172** (vs E12's −1.63 %).
  Verified `--all`, all 17 IBM benchmarks, zero overlaps. Wins on 11/17,
  ties on 4/17 (ibm11/13/14/15), sub-noise regression on 2/17 (ibm12
  +0.02 %, ibm17 +0.09 %; both below the documented 0.5–1.5 % drift
  between incremental and reference evaluators). The wins concentrate
  on the easier benches (ibm01 −1.58 %, ibm10 −0.97 %, ibm08 −0.69 %,
  ibm09 −0.68 %) where SA-after-LNS extracts wins LNS-alone leaves on
  the table.
- **Total wall:** 7.85 hr (E12) → 10.33 hr (E25) on `--all`. The added
  cost is the 600 s SA phase running on every benchmark. Still 6.7 hr
  of headroom under the 17-hr competition envelope (17 × 1 hr); no
  per-bench cap violation observed.
- **Compositionality observed.** ibm01 saw the largest improvement
  (−1.58 %), confirming the `--fast` finding that SA-after-LNS finds
  breakpoint-set wins LNS-alone misses. The hardest benches (ibm14,
  ibm15, ibm17) tied with E12 — both pipelines hit the same floor; the
  SA phase finds nothing on top of LNS on those.
- **Algorithmic interpretation.** Three phases on three distinct
  candidate sets is the right architecture. Adding a fourth phase is
  speculative; E26 (longer SA budget, 300/900 split) is queued to test
  whether SA's "best at t≈599 s of 600 s" observation has further
  headroom.
- **The E14 lesson is in the placer's docstring.** The SA primitive
  must (a) track best-so-far placement and (b) use T₀ producing
  ≤ 5–10 % early acceptance of typical worsening moves. v1 violated
  both and lost 5.7 % vs baseline; v2 obeys both and matches LNS.
- **Open follow-up.** NG45 sanity check on E25 hasn't been run. ADR-007
  validated NG45 transfer for the CD + LNS pipeline (E23, avg 0.7037 on
  4 NG45 designs, zero overlaps); the SA-v2 phase is mechanically
  identical across benchmark families (per-axis breakpoints; same
  evaluator), so transfer is expected to hold but should be confirmed.
- **Alternatives ruled out.** E14 (broken SA), E13 (congestion-region
  destroy, marginal), E24-only (CD + SA, no LNS — matches LNS-only,
  doesn't beat champion alone). Compositionality is necessary to clear
  the champion.

## Evidence

- `submissions/cd_lns_sa/placer.py` — candidate code (in tree
  2026-04-29; not yet promoted as champion).
- `experiments/E25_lns_sa_compose/manifest.md` — graduated 2026-04-29.
- `results/CDLNSSAComposePlacer_20260429_044619.json` — `--fast` result:
  avg 0.9336, 0 overlaps.
- `results/CDLNSSAComposePlacer_20260429_151328.json` — verified `--all`
  result: avg 1.0954, 0 overlaps, 37 188 s total wall.
- `experiments/E14_sa_polish/manifest.md` — falsified v1 (lesson source).
- `experiments/E24_sa_polish_v2/manifest.md` — fixed v2 (component).
- `writeup/evidence.md` §1, §7, §9.B (E14), §9.D (E24), §9.E (E25).
