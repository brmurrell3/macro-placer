---
id: E62
name: will_seed_init
status: falsified
parent: E41 (pipeline) + WillSeed v4 (init)
created: 2026-05-02
decided: 2026-05-02
champion_at_time: 1.08151 (E48 hybrid, ADR-011 *Accepted* 2026-05-02)
outcome: **falsified --fast 2026-05-02 16:02 EDT** — avg 0.93347 vs E48 fast 0.92024 = +1.44 % regression, exceeds 0.929 kill gate. Per-bench: ibm01 0.9060 (+1.54 % vs E48 0.8923), ibm04 1.0181 (**+3.35 %** vs E48 0.9851), ibm09 0.8496 (+0.99 % vs E48 0.8413), ibm13 0.9602 (+1.31 % vs E48 0.9478). **WillSeed loses on every fast bench** → 0 per-bench wins for a hybrid extension. NG45 not queued (kill gate fired on fast). Wall 8711 s = 2.42 hr.
champion_delta: +0.01324 (+1.44 %) vs E48 fast (regression)
graduated_to: null
superseded_by: null
---

# E62: will_seed_init — third basin lane via WillSeed → E41 pipeline

## Hypothesis
E48 hybrid extracts −0.30 % over E41 by per-bench best-of-{E25 SDF basin,
E41 DPO basin}. The overnight 2026-05-01 → 02 wave confirmed:
- Multi-seed within DPO (E53m) is dead-end at --all aggregate scale.
- Mechanism-aligned destroy heuristics (E54) break NG45 transfer.
- GPU DPO basin polish (E53) is non-additive on top of CD-LNS-SA.

The roadmap §4.6 Tier A direction is **adding more independent basins** as
hybrid lanes, where each lane is individually NG45-safe. Will's seed v4
(`submissions/will_seed/placer.py`, 1.5338 standalone) uses minimal
legalization + GPU gradient refinement on a different cost — a basin
class that's neither SDF-driven (CD's E25 lane) nor DPO-driven (CD's
E41 lane).

**Claim:** WillSeed → CD-LNS-SA-K-joint (E41 backbone with WillSeed
replacing the DPO best_of_v2 init) lands a third basin distinct from
both. If verified safe (--ng45 ariane133 doesn't regress past E41), it
qualifies as a 3rd hybrid lane in a future E63 = best-of-{E25, E41, E62}.

## Method
Pipeline per benchmark (identical to E41 EXCEPT step 1 init):

  1. **WillSeed v4 init** (replaces DPO best_of_v2; uses
     `submissions/will_seed/placer.py` `WillSeedPlacer.place(benchmark)`).
  2. project_overlaps to clear any residuals (WillSeed should produce
     legal placements but defensive).
  3. Build IncrementalProxyEvaluator.
  4. CD adaptive (≤ 2400 s).
  5. Grid-bin LNS (≤ 600 s, cost-aware destroy — the safe baseline).
  6. SA-v2 (≤ 600 s, T₀=5e-4).
  7. K-joint LNS (≤ 600 s, K=3, top_N=5).
  8. Validate, preserve fixed macros.

All hyperparameters from E41 preserved unchanged. Only init changes.

## Kill gate
- **--fast > 0.929** (E41 fast 0.92178 + 0.8 %): WillSeed basin is
  worse than DPO basin and not worth keeping. Falsify.
- **--fast lift over E41 = NICE-TO-HAVE not required**: even tied --fast
  is acceptable IF NG45 verifies, because per-bench best-of will pick
  WillSeed lane only on benches where it wins.
- **--ng45 ariane133 > E48 0.6861 + 1 %** (= 0.694): WillSeed basin
  fails NG45 transfer. Joins E42/E43/E44/E54 in the IBM-aware/NG45-blind
  failure class. Falsify.
- **NG45 avg > E48 0.6922 + 0.5 %** (= 0.696): same falsification.

## Generalization check
NG45 is the gate. WillSeed's basin must be NG45-safe to be useful as a
hybrid lane.

## Outcome (filled when decided)

### --fast result (2026-05-02 16:02 EDT)

**FALSIFIED.** WillSeed-init pipeline (`CDLNSSAWillKJointPlacer`) tested
on fast 4 benches under `--jobs 4`, wall 8711 s = 2.42 hr.

```
avg_proxy_cost = 0.93347 (vs E48 fast 0.92024 = +1.44 %)
total_overlaps = 0
per_benchmark:
  ibm01 = 0.9060   (E48: 0.8923; +1.54 %)
  ibm04 = 1.0181   (E48: 0.9851; +3.35 %)
  ibm09 = 0.8496   (E48: 0.8413; +0.99 %)
  ibm13 = 0.9602   (E48: 0.9478; +1.31 %)
```

**Decision rule fired**: kill gate "--fast > 0.929" exceeded by +0.4 %.
WillSeed init lands a *worse* basin than DPO best_of_v2 init across all
fast benches, with the worst loss on ibm04 (+3.35 %). Loses on every
bench — would contribute zero per-bench wins to any hybrid extension.

**Why this matters structurally**: this is the **fourth falsification**
of the "find a new productive basin via init-class swap" attack:
- E17 random init (ADR-005): worse-or-equivalent basin after CD.
- E32 SAM-CD perturbed init: +5.5 % on --fast.
- E53 GPU DPO basin polish (overnight 2026-05-01 → 02): 0/350 accepts.
- **E62 WillSeed init**: regression on every bench.

The SDF + DPO basin pair is not arbitrary — they appear to be specifically
near-optimal init classes for this evaluator. Constructing a third class
that competes with them on the IBM proxy is a hard, structurally constrained
problem. **Pivot direction**: refining within the existing 2-basin envelope
rather than searching for new init classes.

NG45 not queued (kill gate on fast pre-empted further runs).

### Implications for active work

- **E61 V2 (in flight on ibm12)** — recombination of the existing 2
  basins remains a live attack. Not a new basin source, but a polish-
  from-mixed-start attempt; doesn't share E62's failure mode.
- **No new "more independent basins" experiment is queued.** The Tier A
  roadmap direction "init class diversification" is exhausted barring
  a fundamentally different idea (LP-relaxation init, learned policy
  init, spectral connectivity init — high-cost, deferred).

## Pointers
- Code: `code/cd_lns_sa_will_kjoint.py` (defines `CDLNSSAWillKJointPlacer`).
- Init component: `submissions/will_seed/placer.py` (`WillSeedPlacer`).
- Pipeline backbone: E41
  (`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`).
- Discussion: motivated 2026-05-02 by E48 promotion (ADR-011 *Accepted*)
  + roadmap §4.6 Tier A "non-DPO inits as third basin lane".
