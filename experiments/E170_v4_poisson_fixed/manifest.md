---
id: E170
name: v4_poisson_fixed
status: in_progress
parent: E167
created: 2026-05-21
decided: null
champion_at_time: 0.984
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E170: v4_poisson_fixed — fixed DCGP-lite Poisson refinement

## Hypothesis

Same as E167: Poisson-FFT congestion adds globally-smooth pressure
signal that complements canonical's local ABU-5%. Adam descending the
combined loss should find lower-pressure neighbors than V4 alone.

E167 bug: refinement loss omitted WL/density/canonical-cong, so Adam
optimized Poisson alone and destroyed the basin (+39% on ibm03).

Fix: include FULL V4 loss in refinement, ADD Poisson as extra penalty:
    loss = WL + 0.5*density + 0.5*canonical_cong + λ*overlap + α*poisson

## Method

Same pipeline as E167 but refinement loop now uses
`fast_loss_with_penalty` (V4's standard loss) plus Poisson term added.
Compute Poisson from separate PerNetTraceCongestion instance (its
internal V/H demand grids are differentiable w.r.t. positions).

Conservative α=0.1 (was 0.5 in E167 — too aggressive).
Conservative steps=50 (was 100).

## Kill gate

ibm03 smoke: refined basin ≤ V4 basin + 0.005 (no >0.5% regression).
If refined proxy diverges → α still too aggressive or signal mismatch.

If refined ≤ V4 basin - 0.001 → Poisson penalty helps; promote to test
on top of E166 stack.

## Generalization check

If positive: rebuild as full pipeline (V4 multi-init + Poisson refine +
cascade + portfolio) and run --fast.

## Outcome (filled when decided)

**Smoke ibm03 FALSIFIED** (2026-05-21):
- V4 basin: 0.89860 ovl=0
- Refine (FULL V4 loss + α=0.1 Poisson): proxy=1.04312 ovl=4 (Δ=+0.14452)
- "refine made things worse; reverting to basin"
- (CD polish from V4 basin will likely land in 0.89 range)

Even with the bug fixed, Poisson penalty PULLS THE OPTIMIZER AWAY from
canonical-good basins. The Poisson signal (FFT solve on existing
per-net-trace demand grids) is not aligned with canonical congestion's
ABU-5% on V4-Gaussian basins.

Possible reasons:
- Poisson computes pressure on demand grid; canonical computes top-5%
  congestion. Different mathematical objects with weak correlation here.
- α=0.1 may still be too aggressive on a signal that's misaligned.
- Refinement from V4 basin (already near canonical local-min) is the
  hardest place to add a different objective — no slack to give.

DCGP route dead for V4-context in current form. A true DCGP integration
would require:
- Replacing congestion entirely (not adding as penalty)
- Re-tuning V4's other hyperparameters around the new objective
- Multi-day effort


## Pointers

- Code: `code/placer.py`
- Parent: E167 (falsified due to bug)
- Poisson primitive: `experiments/E167_v4_poisson_congestion/code/poisson_congestion.py`
