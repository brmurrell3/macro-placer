---
id: E23
name: ng45_sanity
status: validated
parent: E12
created: 2026-04-28
decided: 2026-04-28
champion_at_time: 1.0990
outcome: 0.7037 avg on 4 NG45 designs; zero overlaps; all under per-bench cap
champion_delta: n/a (defensive — different benchmark set)
graduated_to: null
superseded_by: null
---

# E23: ng45_sanity

## Hypothesis
Defensive: confirm the E12 champion (`submissions/cd_lns_gridbin/placer.py`,
1.0990 on `--all` IBM) runs end-to-end on the four public NG45 commercial
designs (ariane133, ariane136, mempool_tile, nvdla) and produces zero-overlap
placements. ADR-007 needs its own NG45 datapoint before treating LNS-overlay
transfer as confirmed; this run also fills the "Tier 2 robustness check on
E12" open follow-up.

## Method
No algorithm change. Run `uv run evaluate submissions/cd_lns_gridbin/placer.py
--ng45 --json --hypothesis e23_ng45_e12`. Same plateau detection, same LNS
overlay, same global hyperparameters as the IBM run. NG45 designs are
present locally at `benchmarks/processed/public/{ariane133,ariane136,
mempool_tile,nvdla}_ng45.pt` — the roadmap's "we don't have the public NG45
designs locally" entry is stale.

## Kill gate
Diagnostic only — no kill gate. Failure modes that warrant action:
- Any benchmark produces overlaps (DQ on hidden test) → bug, fix in placer.
- Any benchmark blows the 1-hr-per-bench cap → reduce CD cap or LNS budget.
- Any benchmark produces NaN/inf proxy → loader / evaluator bug.

## Generalization check
Each NG45 design's proxy should be in a comparable range to IBM (i.e., not
catastrophically off the top of the distribution). If proxy is wildly out of
band, that's signal of a structural problem with how E12 handles commercial
designs.

## Outcome (filled when decided)
**Validated 2026-04-28.** E12 runs clean on all four public NG45 designs.

| Design | Proxy | WL | Density | Congestion | Overlaps | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|
| ariane133 | 0.7061 | 0.064 | 0.525 | 0.759 | 0 | ~615 |
| ariane136 | 0.6840 | 0.060 | 0.540 | 0.708 | 0 | ~767 |
| mempool_tile | 0.7438 | 0.066 | 0.653 | 0.704 | 0 | 686 |
| nvdla | 0.6807 | 0.069 | 0.510 | 0.712 | 0 | 1053 |
| **AVG** | **0.7037** | 0.065 | 0.557 | 0.721 | **0** | — |

**Total runtime: 3122 s = 52 min** (vs 28 256 s = 7.85 hr on `--all` IBM).

**Per-design plateau detection.** Every design hit `exit=plateau` within
9–10 sweeps; none came near the 1-hr-per-bench legal cap (max 1053 s on
nvdla). The plateau-detection policy transfers to commercial designs
without per-benchmark tuning — the same `cd_plateau_threshold=0.001,
cd_hard_cap_s=3000, lns_budget_s=600` configuration as IBM.

**LNS phase observations.**
- mempool_tile has only 20 hard movables → K = max(1, min(30, 0.05·20)) = 1.
  Single-macro destroy + grid-bin reinsert produced no improvement;
  converged at sample 1 with Δ=0. Expected for tiny hard-macro counts.
- nvdla (128 hard movables, K=6) found Δ=−0.00201 on sample 1, then
  three Δ=0 samples → converged at sample 4. Same convergence shape as
  IBM benchmarks.
- ariane133 / ariane136 LNS phases converged similarly fast on cost-aware
  destroy.

**Zero overlaps everywhere.** Hard validation in placer raises
`RuntimeError` on overlap; not triggered. ADR-007's "transfer to
commercial designs" claim is now supported by direct measurement, not
just by the structural argument.

**No bugs surfaced.** No NaN/inf proxies, no crashes, no per-bench cap
violations. The Tier-2 robustness check open follow-up
(`docs/roadmap.md` "Open follow-ups") is now closed.

**Source JSON:** `results/CDLNSGridBinPlacer_20260428_223405.json`. Run
log: `experiments/E23_ng45_sanity/run.log`.

## Pointers
- Code: re-uses `submissions/cd_lns_gridbin/placer.py` unchanged.
- Results: `results/CDLNSGridBinPlacer_*.json` (ng45 mode).
- Discussion: `docs/decisions/007_cd_lns_gridbin_promotion.md` (ADR-007 needs
  this datapoint); `docs/roadmap.md` Risk register row 1.
- Parent: E12 (champion).
