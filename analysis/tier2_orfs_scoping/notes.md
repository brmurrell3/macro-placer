# Tier 2 ORFS Verification Scoping — Task 18

Created 2026-05-05. References: `SCORING.md` v2.0.

## Eligibility

Top **7 by Tier 1 proxy cost** get evaluated through full OpenROAD flow on
NG45 designs. Current state: E74 verified `--all` 1.0666 beats every
verified leaderboard entry (best previously verified MTK 1.2818). Only
Cezar 1.037 and vmallela 1.1 are unverified-lower. We are very likely to
make top 7. **Estimated probability of Tier 2 entry: high (≥ 90 %).**

## Designs to verify

- `ariane133` (public, NG45 0.6641 currently)
- `ariane136` (public, 0.6518)
- `mempool_tile` (public, 0.7376)
- `nvdla` (public, 0.6716)
- 1–2 hidden NG45 designs (cannot test locally)

## Two-stage evaluation

### Stage 1 — Feasibility (gate)

For each design:
```
WNS_sub ≥ min(WNS_SA, WNS_RP)
TNS_sub ≥ min(TNS_SA, TNS_RP)
```

Fail on any design → disqualified from Grand Prize. **This is the dominant
risk:** we have zero local visibility into WNS/TNS. We need either:
1. Run ORFS locally on each public design
2. Or trust that low proxy cost → reasonable timing (correlation, not
   guarantee).

### Stage 2 — Scoring

Weighted geometric mean of improvement ratios with weights (WNS:3, TNS:2,
Area:1). Higher is better. Score > 1.0 means better than average baseline.

WNS dominates — a 1 % WNS improvement is worth 3× a 1 % Area improvement.

## Auto-handling at Tier 2

The evaluator auto-handles:
1. Snap to manufacturing grid
2. **Push macros apart for ≥ 12 μm clearance** (PDN channel routing).
   *Only at Tier 2*; Tier 1 proxy uses our coords unchanged.
3. Instance name escaping for Genus netlists

Per-pre/post-push diff logged to `macros.tcl.spacing_diff.txt`.

## Action items (priority order)

### High priority (gates submission integrity)

1. **Verify minimum macro clearance.** If our placements have ≥ 12 μm
   between every pair of macros, the Tier 2 push won't move anything and
   we keep full control. If not, the push moves macros — which can change
   WL / density / congestion at the routed-design level. *Action*: write
   a quick `analysis/macro_clearance_diagnostic` that loads each design's
   E74/E79 placement and checks min pairwise edge-to-edge distance.
2. **Set up local ORFS evaluation** for the 4 public designs. Without this
   we ship blind. *Effort*: ~1 day to set up if `external/MacroPlacement`
   has the OpenROAD recipe wired in; possibly multiple days if not.

### Medium priority (improves score within Tier 2)

3. **Re-tune for WNS-dominant scoring.** Our proxy cost is 0.5*WL +
   0.5*density + 0.5*congestion (no explicit timing). Tier 2 rewards
   1 % WNS improvement at 3× the weight of 1 % Area. There's a possible
   gap between proxy-optimal and timing-optimal placements. *Action*:
   when ORFS infra is up, sweep alpha weights in our internal cost and
   correlate proxy diffs vs WNS deltas.

### Low priority (nice-to-have)

4. **Hidden-design robustness.** 1–2 hidden NG45 designs we can't see.
   Best mitigation: ensure our placer doesn't have benchmark-specific
   tuning. ✓ Already audited — all hyperparameters are global. No
   per-bench logic in the champion.

## Verdict — for the submission deadline

- **Highest-leverage Tier 2 action**: macro-clearance diagnostic
  (cheap, gates submission integrity).
- **Next**: ORFS local setup for sanity check on public designs.
- **Nice-to-have**: WNS-correlation study.
- We can submit even without local Tier 2 verification — Tier 1 proxy
  prize is a separate $5k that doesn't require ORFS pass.
