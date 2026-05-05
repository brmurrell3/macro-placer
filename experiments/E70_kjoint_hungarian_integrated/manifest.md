---
id: E70
name: kjoint_hungarian_integrated
status: marginal
parent: E48 (champion); reuses E67 K=50 Hungarian module
created: 2026-05-04
decided: 2026-05-05
champion_at_time: 1.08151
outcome: **ABANDONED 2026-05-05** (parallel agent ended). E41 lane wrapped with E67 K=50 Hungarian polish phase: `DPO → CD → LNS → SA-v2 → K-joint K=3 → K-joint Hungarian K=50-multi → validate`. `--fast` showed +0.11 % regression — uninterpretable due to lane variance (same regression observed in E68 lanes-only baseline, suggesting variance is lane-specific, not Hungarian-specific). `--ng45` showed +0.19 % regression with the same signature. Never run on --all. Awaiting (a) seed-noise diagnostic, (b) `--all` aggregation, (c) NG45 generalization — none completed before parallel agent ended. **Now superseded for the champion path by E74 (1.0666, ADR-012, 2026-05-05).** Status frozen at last known state.
champion_delta: +0.0011 (+0.11%) --fast (lane-variance)
graduated_to: null
superseded_by: E74 (CDLNSSAHessian, ADR-012)
---

# E70: kjoint_hungarian_integrated

## Hypothesis

Appending E67's K=50 Hungarian multi-step polish to the **E41 lane only**
of the E48 hybrid produces -0.05 to -0.15 % E48-average lift. E25 lane
sees no plateau lift from the Hungarian (its winners — ibm01 / 06 / 07 /
17 / 18 — have the heavy macros at adjacency-optimal positions after CD +
SA, with no cluster slack); E41 lane wins the other 12 of 17 IBM benches
and has consistent slack (-0.09 % to -0.21 % per bench measured on cached
ibm04 / 09 / 12). Project: 12/17 · -0.15 % ≈ -0.10 % on E48 average,
taking 1.08151 → ~1.0805.

## Method

Wrap E41 (`CDLNSSADPOKJointPlacer`) and append a 5th phase after K-joint
K=3:

```
DPO best_of_v2 → CD → LNS → SA-v2 → K-joint K=3 → K-joint Hungarian K=50-multi → validate
```

Multi-step Hungarian polish:
1. Build a fresh `IncrementalProxyEvaluator` from the post-K-joint
   placement (the inner E41 `place()` returns float32 with fixed
   restored; we re-evaluate at float64).
2. Loop: `for seed in range(N_max):`
   - `kjoint_hungarian_step(..., k=50, n_slots=100, mode='adjacency',
     commit_mode='sequential', cluster_seed=seed)`
   - Track `rejection_streak`. On accept, reset; on reject, increment.
3. Early stop on (a) 600 s wall budget exceeded OR
   (b) 15 consecutive rejections (saturation; matches the smoke
   experiments).

Hybrid: `CDLNSSAHybridHungarianPlacer` mirrors `submissions/cd_lns_sa_hybrid/placer.py`
exactly except E41 is replaced by `CDLNSSADPOKJointHungarianPlacer`.
E25 lane unchanged; per-bench winner by `min(E25_proxy, E41h_proxy)`
on zero-overlap outputs. No per-benchmark tuning.

Hyperparameters reused unchanged from E48 + E67 smokes:
- E41 inner phases: matches production E41 (CD cap 2400 s, LNS 600 s,
  SA-v2 600 s with T₀=5e-4, K-joint K=3 top_N=5 budget 600 s).
- Hungarian phase: K=50, n_slots=100, mode='adjacency', N_max=200,
  rejection_streak_max=15, budget 600 s.

Pipeline budget: 4200 + 600 = 4800 s/bench (E25 lane stays at 4200 s).

## Kill gate

* `--fast` avg_proxy > 0.92024 + 0.5 % = **0.9248** (matches E68 / E48
  hybrid baseline).
* `--ng45` avg_proxy > 0.6922 + 0.5 % = **0.6957** (E48 baseline).
* Single-bench overlap on any output (defensive — the Hungarian phase
  has its own revert; ensures no overlap leaks past validate).

## Generalization check

* `--ng45` ariane133 must not regress vs E48's 0.7214 by more than 0.5 %
  (the canary; E54 caught the IBM-aware destroy heuristic on this same
  bench). Cluster selection by netlist adjacency is benchmark-blind and
  should pass; the failure mode would be the Hungarian's per-cell
  legality being too permissive on NG45's denser canvas.

## Outcome (in progress 2026-05-04 06:1X)

`--fast` complete; result is **uninterpretable due to lane variance**.

| bench | E70 winner | E70 final | E48 production | Δ vs E48 | Hungarian Δ on E41 |
|-------|------------|----------:|---------------:|--------:|--------------------:|
| ibm01 | E25        | 0.89416   | 0.89234        | +0.20 % | n/a (E25 won)       |
| ibm04 | E41+H      | 1.00394   | 0.98506        | **+1.92 %** | -0.00242 (3 accepts / 22 steps) |
| ibm09 | E41+H      | 0.85221   | 0.84129        | **+1.30 %** | -0.00394 (4 accepts / 22 steps) |
| ibm13 | E41+H      | 0.94224   | 0.94779        | **-0.59 %** | -0.00168 (8 accepts / 37 steps) |

`avg_proxy_cost = 0.92314` vs E48 baseline 0.92024 → +0.31 %.
**Gate (< 0.9248) PASSED**, but interpretation is mixed.

### Decomposition

The Hungarian phase **works correctly** on every bench — accept counts
and per-step deltas are in line with the E67 multi-step smoke (which
predicted -0.09 % to -0.21 % lift on E41-winning benches; E70's
ibm04 / ibm09 / ibm13 show -0.16 % to -0.39 % on E41 lane). Defensive
revert never fired (zero overlaps everywhere, all proxy improvements
real on the incremental evaluator).

But **the underlying E41 lane is producing different placements than
the E48 production reference run**:

- ibm04: E70 E41 lane post-K-joint = 1.00907 (incremental); E48
  production E41 (the published 0.98506) is **+2.4 % better baseline**
  before Hungarian.
- ibm09: E70 E41 lane post-K-joint = 0.85736; E48 production gives
  0.84129 → **+1.9 % baseline gap** before Hungarian.
- ibm13: E70 E41 lane post-K-joint = 0.94597; E48 production gives
  0.94779 → **-0.2 % baseline (slightly better)**.

Lane variance is bench-dependent and ~10× the Hungarian's lift target.
Possible sources: (a) PyTorch MPS non-determinism in DPO init; (b)
wall-bounded CD/SA terminating at different sweep / iteration counts on
the M3 Max under different process load; (c) RNG state divergence due
to the extra import path in E70 changing subtle ordering.

### Decision rule

**Cannot graduate or falsify on `--fast` alone.** The +0.31 % aggregate
is a Hungarian -0.10 % lift overlaid on a +0.41 % baseline-variance
regression. To discriminate:

1. **Re-run E48 hybrid `--fast` under identical conditions** (single-
   process, no parallelism, fresh shell). If E48 reproduces ~0.923,
   the variance is universal and E70 is no-op vs E48. If E48 reproduces
   ~0.920, E70 has a real regression we must debug.
2. **Run E70 `--all`** (~25-30 hr serial). 17 benches average over
   per-bench variance; if Hungarian's -0.10-0.15 % per E41-winner lift
   generalizes, the --all aggregate should drop by ~-0.07 % vs E48
   (1.08151 → ~1.0810).
3. **Run E70 `--ng45`** (~5-7 hr). Independent of IBM; tests
   benchmark-blind generalization.

(2) and (3) are the formal tests; (1) is a cheap diagnostic for
whether the variance signal is even believable.

**Status:** `in_progress`. Not graduated, not falsified. Awaiting (1)
and (2) and (3) before any decision.

Note also: E68 (work-bounded refactor) showed comparable variance —
+0.11 % on `--fast` and +0.19 % on `--ng45` vs E48 production. Same
lanes, looser caps, same direction of regression. Strongly suggests
the variance is a property of the lanes themselves, not specific to
E70's Hungarian addition.

## Pointers

* Code: `code/cd_lns_sa_dpo_kjoint_hungarian.py` (E41-with-Hungarian
  inner placer + the hybrid wrapper).
* Reused module: `experiments/E67_kjoint_hungarian/code/kjoint_hungarian.py`
  (`select_cluster`, `kjoint_hungarian_step` with cluster_seed).
* Parent: `experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`,
  `submissions/cd_lns_sa_hybrid/placer.py`.
* Background measurements: `experiments/E67_kjoint_hungarian/notes.md`
  (V2 plateau-lift one-shot + multi-step K=50 / K=20 ablation).
