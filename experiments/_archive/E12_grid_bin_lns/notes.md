# E12 grid-bin LNS — running notes

Working notes carried forward from the retired `findings.md` and
`experiments_overnight.md`. The structured experiment record lives in
`manifest.md`; this file collects observations and ablations that don't fit
the manifest's frame.

## Findings

### Grid-bin LNS DOES find escapes (vs. subset-CD LNS which doesn't)

Different move type — searches all (col, row) cell centers, not just per-axis
breakpoints. ibm12 production smoke: CD plateau at 1.20564 → LNS converged at
1.20466 over 5 samples. The full `--all` run shows real per-bench gains
(verified 1.0990 final per `findings.md`, source
`results/CDLNSGridBinPlacer_20260428_155739.json`). Contrast with subset-CD
LNS escape probe (24 random destroy-and-reinsert samples across ibm09 +
ibm12, 0/24 improved baseline by ≥0.1%) where the reinsertion uses the same
move type as CD and finds the same fixed point.

### Cost-aware destroy ranking is NOT load-bearing

Random destroy on `--fast` matched cost-aware within noise — ibm09 random
actually BEAT cost-aware (0.8541 vs 0.8591). Cost ranking adds ~10% wall per
LNS sample but doesn't change quality. **Future simplification: drop
ranking, use random.** Source: `results/CDLNSGridBinRandomPlacer_20260428_093846.json`
(random ablation, `--fast`) vs the cost-aware run on the same set.

## Production-config plans (carried forward 2026-04-28 07:14)

Production-config smoke on ibm12 launched with CD cap 3000s + LNS budget
600s, ~60 min wall. Plan: if smoke lands ≤ E16's ibm12 (1.2091) with zero
overlaps → kick off `--all` overnight.

**Earlier short-CD smoke result:** LNS phase found −0.24% improvement on a
non-converged baseline (ibm12 1.30891 → 1.30574 over 7 samples). Code path
verified, zero overlaps. Per-sample wall ~20s.

**Verified `--all` result (per `findings.md`):** avg 1.0990 in 7.85 hr
runtime; beats E16 by −0.0035 (−0.32%) and beats prior champion by −0.0065
(−0.59%). All 17 benchmarks improved over E16 (no regressions). Zero
overlaps. Zero per-benchmark tuning. Biggest wins on basin-locked /
high-density benches: ibm10 −0.0162 (−1.51%), ibm02 −0.0092 (−0.81%), ibm01
−0.0090 (−0.99%). Promotion to champion is a separate decision held by the
parent agent — manifest stays at `status: in_progress` until that
reconciles.

**Position vs verified leaderboard at this score:** −1.63% below vmallela
#1 (1.1172 unverified), −10.1% below Cezar #2 verified (1.2224), −14.3%
below MTK #3 verified (1.2818).
