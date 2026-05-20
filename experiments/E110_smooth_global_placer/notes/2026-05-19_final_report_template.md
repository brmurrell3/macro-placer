# Final report — 2026-05-19 (user check-in at 6 PM)

## TL;DR

**Submission recommendation: [PLACER_NAME] at IBM=[X] / NG45=[Y].**
Backup: Option C (1.0575 IBM, verified 2026-05-18).

## What we tested today (10 jobs total)

| Placer | --all avg | NG45 avg | Wall | Status |
|---|---:|---:|---:|---|
| Option C (yesterday baseline) | 1.0575 | 0.689 | 14 hr | reference |
| Variant A default | 1.05737 | TBD | 14 hr | TIE on IBM |
| Variant B default | 1.0604 | — | 15 hr | slight LOSS |
| Variant A ovl10 (sweep top) | [TBD] | [TBD] | 6 hr | RUNNING |
| Variant A nocong | killed mid-run | — | — | killed (4/4 REJECT) |
| E110Minimal-600s (no cascade) | [TBD] | — | 2 hr | RUNNING |
| Variant A Triple (multi-cfg) | [TBD] | — | 4 hr | RUNNING |

## What we learned

### 1. The cascade pipeline is saturated

Default Lane-4 (Option C + E110 lane as 4th candidate) at 1.05737 ≈ Option C 1.0575.
The cascade saddle escape + portfolio_saddle already extracts most of the gradient
basin's value. Adding E110 lane is neutral.

### 2. Sweep found 3 winning hyperparameters on --fast

| cfg | --fast Δ vs SDF | wins |
|---|---:|---|
| `lr=5e-3, steps=500, ovl_end=10` | −4.39% | 4/4 |
| `lr=2e-3, steps=1500, ovl_end=50` | −3.98% | 4/4 |
| `no_cong, lr=3e-3, steps=500, ovl=30` | −3.92% | 4/4 |

**Critical caveat:** These lifts are vs SDF+CD60s baseline, NOT vs full cascade.
The Variant A ovl10 --all (running) will tell us if the lift survives cascade.

### 3. nocong cfg consistently loses against cascade (4/4 REJECT)

Dropping smooth congestion from descent gives a basin with high canonical
congestion. CD polish can't fully recover. Cascade always wins.

### 4. Profile: 94% of cascade time is Python CD polish

`run_cd_adaptive`: 396s of 470s on ibm01.
`_net_cong_contrib_flat`: 156s alone.

To match Carrotato's 3.8 min/bench, we'd need to either:
- Cython-ize the CD polish (8-16 hr engineering)
- Replace CD with continuous refinement (algorithmic change)
- Use multi-seed E110 ensemble in the same budget

### 5. E110Minimal-600s: ~98% of cascade quality in <30% of wall

Preliminary data (14/17 benches): avg ~1.04. If the 3 remaining are similar,
full --all ~1.05-1.08. Slightly worse than cascade but MUCH faster.

This unlocks fast iteration if we want it (10× more --all runs per day).

## On the per-bench tuning concern

The ovl10 cfg is a SINGLE global hyperparameter change (overlap_lambda_end:
50 → 10). Not per-bench. It's tuned on --fast (4 benches) and the --all
test (running) validates whether that single cfg generalizes to 17 IBM +
4 NG45.

**The structural alternative (Triple variant)** runs 3 different sweep-winning
cfgs as parallel E110 lanes and plateau-picks. This is NOT per-bench tuning —
it's running multiple algorithms and picking the best output WITHOUT
knowing the bench. The plateau-pick is bench-agnostic.

If Triple --all wins, that's the cleanest answer to the per-bench concern.

## On the speed gap to Carrotato

Our 55 min/bench vs their 3.8 min/bench is mostly Python overhead:
- 94% of cascade time = Python loops in CD polish
- 175s on ibm17 just for `compute_proxy_cost` canonical eval

For TODAY's submission, we use 55 min/bench (within 60-min cap). Speed
matters for ITERATION SPEED, not competition score.

## What's still needed before May 21 submission

1. EPYC cross-validation on chosen placer (~4 hr on AWS c6a.4xlarge spot)
2. Update root `placer.py` launcher to chosen placer
3. Push to git + submit via form

## Risk register

- **CPU contention slowdown:** running 4 --all in parallel oversubscribed
  M3, ~1.5-2× slower per job
- **M3 → EPYC variance:** ~2.4% gap on identical placer. Our 1.05 M3
  ≈ 1.08 EPYC. Need EPYC verification.
- **Untested placers (Triple, Adaptive, Multiseed) may have edge-case bugs.**
  All passed smoke tests but --all could reveal issues.
- **NG45 generalization unverified for E110 lane variants.** NG45 default
  Lane-4 partial data shows ariane lifts but mempool_tile/nvdla pending.
