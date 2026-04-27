# Next Experiments — Prioritized

Date: 2026-04-26 (DPO era). Updated 2026-04-27 (CD era).

**Status:** All DPO-era experiments (1-6) completed. CDOnlyPlacer
(1.1193 avg, 23.2% over RePlAce) supersedes DPO; CDAdaptive (E9, 1.1055)
beats the leaderboard. See `closing_the_gap.md` (this directory) for
CD-era experiments (E1-E11), and `experiment_notes.md` §17 for E9.

---

## The core diagnosis (DPO era — SUPERSEDED)

DPO's differentiable proxy has a **3.1x congestion gap** (RUDY vs real
L-routing, confirmed by §11 analysis). The error is in gradient
*direction*, not magnitude (§10 congestion weight sweep killed).
Top-5% hotspot overlap: 10.9%. This motivated the CD pivot.

**Resolution:** Bypass RUDY entirely via incremental evaluator (4657x
speedup) + full-proxy coordinate descent. See `experiment_notes.md` §15.

---

## Experiment 1: Scale congestion weight to compensate (DONE — KILLED)

**What:** Change DPO loss from `wl + 0.5*C` to `wl + 1.0*C` (or sweep
0.5, 0.75, 1.0, 1.25, 1.5).

**Why it might work:** If RUDY consistently underestimates by 2x,
doubling the weight makes the effective objective match the real proxy.
The gradient direction stays the same; only the magnitude changes
relative to density/WL.

**Why it might not:** RUDY doesn't just underestimate uniformly — it
misses L-routing patterns entirely. Scaling the weight amplifies both
the signal AND the noise. On benchmarks where RUDY is most wrong
(ibm01, ibm06), this could make DPO worse.

**How to test:**
```bash
# Create ablation variants with different congestion weights
# Modify line 348 in submissions/dpo/placer.py:
#   proxy = wl + 0.5 * density + 0.5 * congestion
# to:
#   proxy = wl + 0.5 * density + CONG_WEIGHT * congestion

# Test on --fast first, then --all for promising variants
uv run evaluate submissions/dpo/ablation_cong_weight_X.py --fast --json
```

**Expected:** 0.05-0.15 improvement if RUDY error is mostly scaling.
Neutral or worse if RUDY error is directional (wrong gradient direction).

**Kill gate:** If no weight in [0.5, 1.5] beats 1.4246 on --fast, the
issue is RUDY's gradient direction, not magnitude. Stop here.

**RESULT:** Kill gate triggered. All weights monotonically worse. Problem
is gradient direction, not magnitude. See experiment_notes.md §10.

---

## Experiment 2: Best-of(SDF, DPO) submission (DONE)

**What:** Evaluate both SDF init and DPO output per benchmark, submit
the better one.

**Why:** DPO worsens 4/17 benchmarks vs SDF init (ibm01, ibm02, ibm06,
ibm12). Already measured: best-of-two gives 1.4135 (3.0% over RePlAce)
vs DPO-only 1.4246 (2.3%).

**How:** Add a wrapper placer that runs both, evaluates both with real
proxy cost, returns the better.

**Expected:** +0.8% improvement, guaranteed. No risk.

**RESULT:** 1.4145 avg (+3.0% over RePlAce). SDF wins on ibm01, ibm02,
ibm06, ibm12 as expected. See experiment_notes.md §12.

---

## Experiment 3: Restore v2 config (more optimizer steps) (DONE)

**What:** The experiment log shows dpo_v2_moresteps at 1.4107 (3.2%
over RePlAce), systematically better than v3 on 15/17 benchmarks.
The current placer.py is v3 with reduced step counts for runtime.

**Why:** v2 ran more steps. The optimizer hasn't converged — more
compute = better quality. The step_scale variable in _optimize()
reduces steps for large benchmarks. Removing or relaxing this gives
v2-like behavior.

**How:** In submissions/dpo/placer.py, the `step_scale` logic (lines
283-293) aggressively reduces steps for high-complexity benchmarks.
Either remove step_scale entirely or set a minimum (e.g., step_scale
= max(0.6, current_scale)).

**Expected:** +1.0% (already demonstrated). Combined with experiment 2,
could reach ~1.39.

**RESULT:** 1.3888 avg (+4.7% over RePlAce). step_scale = max(0.6,
current). See experiment_notes.md §12.

---

## Experiment 4: v2 steps + best-of combined (DONE)

**What:** Best congestion weight from Exp 1 + v2-style step counts +
best-of(SDF, DPO).

**Why:** These improvements are orthogonal. Weight fixes the gradient
magnitude. More steps let the optimizer converge further. Best-of
catches benchmarks where DPO regresses.

**Expected:** 1.37-1.40 range if congestion weight helps.

**RESULT:** 1.3834 avg (+5.1% over RePlAce). New champion. SDF wins
only ibm02 and ibm12 (v2 steps fixed ibm01 and ibm06). See §12.

---

## Experiment 5: Multi-seed with best config (DONE)

**What:** Run 5-10 seeds with the best config from Exp 4, take best
per benchmark.

**Why:** 5 seeds showed 0.45% range. With a better congestion weight,
different seeds might explore more diverse congestion landscapes,
potentially widening the range.

**Caveat:** The diminishing returns analysis suggests this adds only
~0.1%. But it's free if compute is available.

**RESULT:** 5-seed mean=1.3831, stdev=0.0056, range=1.0%. Best single
seed=1.3790 (seed 46). Best-of-5 per benchmark=1.3703 (6.0% over
RePlAce). 4/5 seeds contribute benchmark-bests. ibm02/ibm12 zero
variance (SDF deterministic). See experiment_notes.md §13.

---

## Experiment 6: Investigate RUDY vs real congestion (DONE)

**What:** For ibm01 (where RUDY is worst), compare RUDY congestion
map vs PlacementCost congestion map cell by cell. Identify where they
diverge and why.

**Why:** Understanding the SYSTEMATIC error pattern (not just the
2x scaling) could suggest a better differentiable congestion model.
If RUDY is wrong in predictable ways (e.g., always misses T-junctions,
always underestimates at macro corners), we could add correction terms.

**How:** Extract both congestion grids from DPO output, compute
per-cell ratio, visualize.

**Expected outcome:** Either (a) RUDY is ~2x everywhere (scaling fix
is sufficient) or (b) RUDY is wildly wrong in specific regions (need
a better model). This determines whether Exp 1 has legs.

**RESULT:** Answer is (b). Gap is 3.1x not 2x. Top-5% hotspot overlap
only 10.9% (near-random). Three sources: L-routing vs bbox (2.74x),
macro blockage (28% missing), spatial smoothing. See §11.

---

## Experiment 7: Differentiable Steiner-tree congestion (HIGH EFFORT)

**What:** Replace RUDY (uniform bbox distribution) with a RSMT-based
congestion model that estimates actual routing topology.

**Why:** RUDY distributes routing demand uniformly across net bounding
boxes. Real routing uses Steiner trees — demand is concentrated along
tree edges, not uniform. This is probably the main source of the 2x
error.

**How:** For each net, estimate RSMT (Rectilinear Steiner Minimum
Tree) structure. Distribute demand along tree edges instead of
uniformly across bbox. The RSMT estimation itself can be differentiable
(use soft assignments to Hanan grid points).

**Expected:** This is the path to closing the gap to 1.22. But it's
a multi-day implementation effort with uncertain payoff.

---

## What we know works (full trajectory)

| Finding | Evidence | File |
|---|---|---|
| SDF init is essential (+247%) | ablation_random_init | experiment_log.jsonl |
| Density gradient: +21.7% | ablation_no_dens | experiment_log.jsonl |
| Congestion gradient: +5.9% | ablation_no_cong | experiment_log.jsonl |
| Multi-phase: +2.5% | ablation_phase1 | experiment_log.jsonl |
| 5-seed stability: 0.45% range | seeds 42-46 | experiment_log.jsonl |
| Congestion = 75% of proxy | component breakdown | experiment_notes.md §2 |
| RUDY gap is 3.1x, direction wrong | cell-by-cell analysis | experiment_notes.md §11 |
| Barrier crossing: 5-12% pairs | traversal analysis | polyhedra_traversal.txt |
| v2 (more steps) +2.5% | experiment log | experiment_log.jsonl |
| **Incremental evaluator: 4657x speedup** | **ibm10 benchmark** | **closing_the_gap.md E1 (this directory)** |
| **Full-proxy CD: 1.12 avg (+23%)** | **all 17 benchmarks** | **closing_the_gap.md E2 (this directory)** |
| **CD breaks ibm02 basin lock (-32%)** | **ibm02 1.689→1.153** | **closing_the_gap.md (this directory)** |
| **CDAdaptive E9: 1.1055 (-24%)** | **all 17 benchmarks, beats leaderboard** | **experiment_notes.md §17** |

## What we know doesn't work

| Approach | Why | Evidence |
|---|---|---|
| Hierarchical decomposition | Objectives coupled at all scales | 0/190 swaps, §7 |
| Alternative inits (spectral, partitioning) | SDF already near-optimal coarsely | Overnight SP1 |
| Better surrogate ranking | Only 1-7 good candidates exist | Overnight SP3 |
| LP-based congestion optimization | LP-HPWL uncorrelated with proxy | rho=-0.001 |
| Congestion weight scaling | RUDY direction wrong, not magnitude | Exp 1, §10 |
| Multi-start brute force (DPO) | Diminishing returns (0.45% range) | Seed analysis |
| Batched GPU seeds (64x) | Same-basin collapse | E5, §14 |
| Diverse priors (4-init best-of) | Flat on --all, ibm02/12 worse | E11, §14 |
| Congestion-only DPO refinement | Marginal -0.33%, basin lock deepens | E10, §14 |
| Pulling clusters together | Increases density (WL-density anti-correlation) | Hierarchical v2 |
