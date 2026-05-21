# E122 — next steps not pursued

E122 closed +6.4 pp of canonical bias on ibm17 but **lost** +0.98 % on
the end-to-end CD600s pipeline (1.2003 → 1.2121). This documents
unexplored directions that could potentially recover the calibration
benefit without the basin-quality hit.

## Option A — Two-stage Adam: L-route → Steiner

Run 80% of Adam steps with E111 (L-only) for the bulk gradient signal,
then switch to E122 (Steiner) for the last 20% to fine-tune toward
canonical. Hypothesis: the L-only objective gives the "right" rough
gradient field, and Steiner gives a "true canonical" refinement.

Implementation: a hybrid `SmoothGlobalPlacerV3Hybrid` that swaps the
proxy mid-descent at step `int(num_steps * 0.8)`.

Risk: the discontinuity at swap point may break Adam momentum.
Mitigation: linear blend the two proxy outputs over a few steps.

## Option B — Steiner as 0.1× regularizer

Use BOTH proxies simultaneously:
```
loss = wl + 0.5*density + 0.5*(0.9*L_cong + 0.1*Steiner_cong)
```

The L-cong dominates gradient direction, but the Steiner term nudges
toward canonical-faithful basins. May give the best of both worlds.

## Option C — Full 4-case 3-pin Steiner

Survey IMPROVEMENT 1 §3 "case-split full" variant: implement all 4
sub-cases of the canonical `__three_pin_net_routing`:
- L-route variant A (Z-pattern)
- L-route variant B (y2==y3 degenerate)
- L-route variant C (x2==x3 degenerate)
- T-route (default)

with soft sigmoid indicators distinguishing each case. This should
preserve gradient direction better than the single-case T-only.

Effort: M (1-3 days). Worth pursuing only if Options A/B fail.

## Option D — Test on cascade as fallback Lane

Even if E122 doesn't help V3+CD as a standalone pipeline, it might
contribute as a 4th lane in the cascade plateau pick (alongside SDF,
DPO, E25, E41 lanes). The basin is *different* from E111's, even if
not better — diversity may help in ensemble.

Implementation: copy `submissions/cd_lns_sa_cascade_stacked_periphery/`
and add E122 as a 4th plateau candidate. Compare combined 21-bench
avg on --all + --ng45.

## What we learned (independent of these options)

1. **Pin-degree distribution**: 22-37% of nets are 3-pin across IBM
   benchmarks. This validates the survey's claim that 3-pin nets are
   a major contributor to canonical bias.

2. **Steiner T-route is implementable differentiably** via soft sort
   (softmax(-βx) for left, softmax(βx) for right, complement for mid).
   The math is clean and the implementation vectorizes well.

3. **Scalar fidelity ≠ basin quality**. This is the second confirmation
   of this principle in our project (first was E111 itself, where bbox-
   uniform had +260% bias but the L-trace version "only" +14%, yet the
   L-trace placer did better on hard benches; here Steiner has +8% bias,
   even less, but the placer is worse). The relationship is non-monotonic.

4. **The bias is structurally from 3-pin nets**: closure happens
   strictly in the 3-pin-net contribution. The 2-pin and 4+pin nets
   are already canonical-faithful.
