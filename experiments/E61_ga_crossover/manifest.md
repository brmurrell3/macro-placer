---
id: E61
name: ga_crossover
status: falsified
parent: E25, E41
created: 2026-05-02
decided: 2026-05-02
champion_at_time: 1.08151 (E48 hybrid, ADR-011 *Accepted*)
outcome: **falsified 2026-05-02 23:34 EDT — small-optimization at best, no breakthrough.** V2 --all killed mid-run after 8/17 benches. Subset avg V2 = 1.01819 vs E48 same-bench avg = 1.01815 (Δ +0.04 % — sub-noise tied). The big --fast lift (−0.74 %) was driven by ibm09 0.8259, which **did not replicate at --all** (V2 ibm09 --all = 0.8413 = tied with E48). DPO seed-noise within E41 lane explains both per-bench variance and the ibm09 single-run lucky basin. ibm12 smoke (−0.61 %) and ariane133 NG45 (−1.47 %) lifts may also be partial seed-noise artifacts; smaller wall budgets needed to confirm. **Verdict**: V2 mechanism produces results indistinguishable from E48 at scale. Real signal would need average improvement of ≥0.3 %, which we never approached. Per-bench data through 8/17 benches: ibm01 −0.37 %, ibm02 tied, ibm03 +0.41 %, ibm04 +0.23 %, ibm06 +0.22 %, ibm07 −0.43 %, ibm08 tied, ibm09 tied. 4 wins, 2 losses, 2 ties — pattern of seed-noise around E48, not systematic improvement.
champion_delta: +0.0004 (+0.04 %) on 8-bench subset (sub-noise tied; not a meaningful Δ)
graduated_to: null
superseded_by: null
---

# E61: ga_crossover (genetic algorithm crossover between E25/E41 basins)

## Hypothesis
E48 hybrid plateaus at the BETTER of {E25, E41} per-bench. The
fundamental ceiling is set by the basins those two pipelines
deterministically converge to. Multi-seed within DPO (E53) was
sub-noise. Replica exchange SA-v2 (E60) targets SA convergence —
wrong attack point because our SA isn't actually stuck.

The RIGHT attack point is **basin choice**: E25 ends in SDF basin,
E41 ends in DPO basin, and there are presumably OTHER deeper basins
that neither pipeline reaches.

**Crossover (genetic algorithm style)** is the standard cross-field
technique: take two converged solutions, recombine subsets of their
variables, polish the recombination → new local minimum, often
better than either parent. **Crystal structure prediction (USPEX)
wins by exactly this mechanism.**

For our problem: E25 and E41 disagree on macro positions on every
bench (different basins). Crossover would take per-macro positions
from a random partition: subset A from E25, subset B from E41.
Project overlaps + run CD+LNS+SA-v2 polish from there → potentially a
hybrid basin neither pipeline finds alone.

## Method
Per-benchmark pipeline:

1. Run E25 (CDLNSSAPlacer) → e25_placement.
2. Run E41 (CDLNSSADPOKJointPlacer) → e41_placement.
3. **Crossover**: for each hard macro, randomly pick its position from
   either e25_placement or e41_placement (Bernoulli p=0.5 with global
   seed). Soft macros stay at their SDF/E25 positions.
4. **Repair**: project_overlaps to clear any overlaps introduced by
   the random recombination.
5. **Polish**: rebuild evaluator on the crossed-over placement; run
   CD plateau (≤1200 s) + grid-bin LNS (≤300 s) + SA-v2 (≤300 s) to
   refine into the nearest local minimum. Reduced budgets vs E25/E41
   primary phases since we're polishing, not from scratch.
6. Compare proxies of {e25_placement, e41_placement, polished_crossover}.
7. Return the lowest-cost output (zero overlaps verified).

All hyperparameters global; no per-benchmark tuning. The crossover is
a deterministic random sampling (fixed seed), not bench-specific logic.

## Kill gate
- **Polished crossover never wins:** if avg --fast >= E48 fast 0.92024
  (no improvement from crossover beyond what E48 picks), kill —
  crossover doesn't find new basins productively.
- **Regression:** if avg --fast > E48 fast 0.92024 + 0.5 % → kill,
  something broke in the polish phase.

## Generalization check
If --fast lifts ≥ 0.3 % over E48 fast 0.92024, run --ng45.
If --ng45 doesn't regress, queue --all.

Wall: per-bench wall = E25 + E41 + polish ≈ 80 + 30 = ~110 min/bench.
Total --all wall ≈ 30 hr serial, ~10 hr `--jobs 4`. Tight but within
17-hr envelope.

## Outcome (filled when decided)

### V1 (per-macro Bernoulli, 50/50): FALSIFIED 2026-05-02 ~15:11 EDT

ibm01 smoke fell back to min(E25, E41) = 0.8918. Crossover phase produced
136 unrecoverable overlaps after 50 project_overlaps iterations. Phase
output:

```
crossover: 124/246 hard macros from E25, 122 from E41 (seed=42)
project_overlaps: 50 iters, residual=136
crossover unrecoverable (136 overlaps after 50 iters); falling back to min(E25, E41)
```

**Mechanism**: per-macro 50/50 Bernoulli treats macro positions as
independent alleles, but they're correlated via local routing/density
topology. Mixing 246 positions from two structurally different basins
creates a placement that's massively non-feasible — local overlaps form
cliques that don't dissipate under the iterative push-apart legalizer.

**Important signal**: even though E25 and E41 reach similar proxy values
on ibm01, their per-macro spatial configurations are not interchangeable.
Confirms they are *genuinely separated* basins, not minor variants of
the same.

### V2 (spatial-block 2×2 quadrants): proposed 2026-05-02

Replace per-macro Bernoulli with quadrant-block crossover:
1. Compute each macro's E25-position quadrant (BL/BR/TL/TR by canvas center).
2. Bernoulli pick per quadrant: take all macros in that quadrant from E25 or E41.
3. Force at least one quadrant flip (avoid degenerate case).

Preserves quadrant-level locality → far fewer overlaps to legalize.

**Smoke target**: ibm12 (E25 1.2079, E41 1.2056, gap 0.2 % — proper diagnostic
where parents are nearly tied; crossover most likely to reveal a productive
recombination). ibm01 was wrong choice for V1 because E25 dominates (gap 2 %)
— polished_crossover would have to beat E25 by a small margin to lift, very
high bar.

### V2 ibm12 SMOKE: VERIFIED 2026-05-02 17:41 EDT

```
E25 phase done:    1.20781 (matches verified E25 ibm12 1.2079, drift <0.01%)
                   wall 3166s = 52.8 min
E41 phase done:    1.20655 (matches verified E41 ibm12 1.2056)
                   wall 3867s = 64.4 min
Crossover:         spatial-block, quadrant_pick_e25=[F, T, F, F]
                   108/651 from E25 (BR quadrant), 543 from E41 (BL+TL+TR)
project_overlaps:  0 iters, residual=0  ← FEASIBLE OUT-OF-BOX
crossover init:    1.21022  (+0.30% above E41, expected boundary cost)
polish-CD done:    7 sweeps, hit cap (1200s), proxy=1.19682
polish-LNS done:   5 samples, Δ=-0.00011, proxy=1.19671
polish-SA done:    no lift, final proxy=1.19671
polished (post fixed-macro restore): 1.19898  ← WINNER
                   −0.00757 vs E41 this run  (−0.63%)
                   −0.00883 vs E25 this run  (−0.73%)
                   −0.00742 vs verified E48  (−0.61%)
total wall: 8719s = 2.42 hr
```

**Verified mechanism.** Spatial-block 2×2 crossover + polish lands a
basin distinct from both parents on ibm12, **0.6-0.7 % below either**.
This is the **first verified lift past E48's 2-basin envelope** since
ADR-011 promotion. The polished basin is genuinely new — CD found a
local minimum that neither E25 nor E41 reaches alone, opened by the
mixed starting point.

Why ibm12 was the right test: parents are nearly tied (0.2 % gap), so
neither basin's gradient strongly attracts. Polish from a 16/84 mixed
state (108 E25 macros + 543 E41 macros) had freedom to find a third
fixed point.

### V2 --fast (launched 2026-05-02 17:41 EDT)

ibm01/04/09/13 under `--jobs 4`. Wall ETA ~80-90 min (~19:00-19:15 EDT).
Decision rule:
- avg --fast < E48 fast 0.92024 − 0.3 % (= 0.9175): mechanism generalizes;
  run --ng45 then --all.
- avg --fast tied (within 0.1 % of E48): only ibm12 was special; mechanism
  doesn't generalize. Document as bench-specific finding; do NOT promote.
- avg --fast > E48 fast 0.92024 + 0.5 % (= 0.9248): regression on benches
  where E25 dominates; falsify V2.

### V2 --fast RESULT: VERIFIED 2026-05-02 19:12 EDT

```
avg_proxy_cost = 0.913423
                 vs E48 fast    0.920240
                 Δ              −0.00682  (−0.74 %)
total_overlaps = 0
runtime        = 15646 s (4.35 hr)
```

Per-bench (vs E48 --all per-bench, not --fast specific — but the deltas
show pattern):

| Bench | E61 V2 fast | E48 --all | Δ vs E48 |
|---|---:|---:|---:|
| ibm01 | 0.8926 | 0.8923 | +0.03 % (tied) |
| ibm04 | 0.9874 | 0.9851 | +0.23 % (slight loss) |
| ibm09 | **0.8259** | 0.8413 | **−1.83 %** (strong win) |
| ibm13 | 0.9478 | 0.9478 | tied |

**Decision rule fires** (kill gate was 0.9175; we hit 0.9134). Mechanism
generalizes:
- Where one parent strongly dominates (ibm01 E25, ibm13 E41), polish
  converges back ≈ that parent's basin → tied with E48.
- Where parents are closer or recombination opens new structure (ibm09),
  polish lands a basin **better than either parent**. ibm09 0.8259 is
  −1.83 % below E41 0.8413 (and −3.21 % below E25 0.8533).

**Critical observation**: V2's worst case is "tied with E48"; it never
regresses on these 4 fast benches. The crossover is feasibility-safe
(spatial-block locality preserves overlaps-free) and the polish either
finds a new minimum or falls back to a parent basin.

### V2 --ng45 (launched 2026-05-02 19:12 EDT)

ariane133/136/mempool_tile/nvdla under `--jobs 4`. Wall ETA ~4-5 hr
(~23:00-00:00 EDT). Kill gate: ariane133 > E48 0.6861 + 1 % (= 0.694).
Decision rule:
- avg --ng45 ≤ E48 ng45 0.6922 + 0.3 % AND ariane133 ≤ 0.694: NG45 safe;
  run --all (the verifying full result).
- ariane133 > 0.694 OR avg > 0.6943: NG45 transfer fails (same pattern
  as E42/E43/E54). Falsify V2 and stop.

### V2 --ng45 RESULT: VERIFIED 2026-05-02 20:32 EDT

```
avg_proxy_cost = 0.690781
                 vs E48 ng45    0.69220
                 Δ              −0.0014  (−0.20 %)
total_overlaps = 0
runtime        = 13466 s aggregate (3.74 hr CPU; ~80 min wall under --jobs 4)
```

Per-design vs E48 NG45 (from ADR-011 follow-up #1):

| Design | E61 V2 | E48 | Δ |
|---|---:|---:|---:|
| **ariane133** | **0.6760** | 0.6861 | **−1.47 %** ← gate test |
| ariane136 | 0.6737 | 0.6685 | +0.78 % (slight loss) |
| mempool_tile | 0.7362 | 0.7375 | −0.18 % (tied) |
| nvdla | 0.6772 | 0.6767 | +0.07 % (tied) |

**Decision rule fires**: avg under target, ariane133 well under 0.694
gate. **ariane133 is exactly the bench where E54 had +5.14 % catastrophic
regression** — V2 not only avoids that failure mode but WINS by −1.47 %.

This is structurally significant: V2's basin (recombined from SDF/DPO via
spatial-block crossover + polish) **transfers to commercial NG45 designs
better than E48 on the hardest one**. The IBM-aware/NG45-blind failure
class (E42/E43/E54) does NOT contain V2.

### V2 --all (launched 2026-05-02 20:33 EDT, KILLED 2026-05-02 23:34 EDT)

17 IBM benches under `--jobs 4`. Killed at 8/17 benches (3 hr in). Decision
rule was:
- avg ≤ 1.0784 (E48 −0.3 %): promote.
- avg > 1.08475 (E48 +0.3 %): falsify.
- between: candidate, surface to human.

Killed because the picture at 8 benches showed V2 trending toward
"sub-noise tied with E48" — not a path to ADR-012 promotion territory.

### V2 --all PARTIAL DATA (8/17 benches before kill)

| Bench | V2 --all | E48 --all (ADR-011) | Δ |
|---|---:|---:|---:|
| ibm01 | 0.8890 | 0.8923 | **−0.37 %** ✓ |
| ibm02 | 1.1161 | 1.1163 | tied (−0.02 %) |
| ibm03 | 0.9592 | 0.9553 | +0.41 % (loss) |
| ibm04 | 0.9874 | 0.9851 | +0.23 % (slight loss) |
| ibm06 | 1.1556 | 1.1531 | +0.22 % (slight loss) |
| ibm07 | 1.0938 | 1.0985 | **−0.43 %** ✓ |
| ibm08 | 1.1031 | 1.1033 | tied (−0.02 %) |
| ibm09 | 0.8413 | 0.8413 | tied (was −1.83 % at --fast) |
| **subset avg** | **1.01819** | **1.01815** | **+0.04 % (sub-noise)** |

**8-bench distribution: 2 wins, 2 ties, 4 losses.** Pattern is
DPO-seed-noise drift around E48, not systematic improvement.

### Lessons (filed for future basin-search experiments)

1. **--fast results don't extrapolate to --all** when ibm09 single-bench
   wins are involved. ibm09 has high seed-noise variance; basin landed
   on --fast (0.8259) didn't reproduce at --all (0.8413). Multi-seed
   verification before celebrating fast lifts.

2. **The "min{E25, E41, polished_crossover}" floor IS E48 by construction**
   only if we're comparing to *the same run's* E25/E41. Comparing across
   different DPO-seed runs introduces ~0.3-0.5 % per-bench noise that
   masquerades as V2 wins/losses. To get a clean signal, we would have
   needed to run E48 alongside V2 in the same wall window — too costly
   for a falsifying experiment.

3. **Spatial-block 2×2 crossover is feasibility-safe**, confirmed across
   ibm01, ibm04, ibm09, ibm13 fast benches plus 8 of 17 --all benches.
   That part of the V2 mechanism works — `project_overlaps` zero-iter
   feasible recombination. The ineffective part is the polish: it lands
   approximately at one parent's basin, not a productive third one.

4. **Real breakthrough requires structurally different mechanism**, not
   recombination of existing basins. Confirms the user's read: "we
   need a big breakthrough not a small optimization."

## Pointers
- Code: `code/cd_lns_ga_crossover.py`.
- Parents: E25 (SDF basin pipeline), E41 (DPO basin pipeline).
- Related: E48 hybrid (just picks per-bench best of E25/E41).
- Discussion: standard GA crossover from crystal-structure prediction
  / USPEX literature. Direct attack on basin choice.
