# Next Experiments — Prioritized

Date: 2026-04-26. Context: DPO achieves 1.4246 avg (2.3% over RePlAce).
Leading teams at ~1.22 (16% over RePlAce). The gap is almost entirely
congestion — see `writeup/experiment_notes.md` §9 for full analysis.

---

## The core diagnosis

DPO's differentiable proxy has a **2x congestion underestimate** (RUDY
vs real L-routing). This means the optimizer effectively minimizes:

    f ≈ WL + 0.5*D + 0.25*C_real   (what DPO actually optimizes)

instead of:

    f = WL + 0.5*D + 0.5*C_real    (the competition metric)

Congestion is 75% of the proxy cost. The gradient on the dominant
component is wrong by 2x. This is the single biggest lever available.

---

## Experiment 1: Scale congestion weight to compensate (TRIVIAL)

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

---

## Experiment 2: Best-of(SDF, DPO) submission (ZERO EFFORT)

**What:** Evaluate both SDF init and DPO output per benchmark, submit
the better one.

**Why:** DPO worsens 4/17 benchmarks vs SDF init (ibm01, ibm02, ibm06,
ibm12). Already measured: best-of-two gives 1.4135 (3.0% over RePlAce)
vs DPO-only 1.4246 (2.3%).

**How:** Add a wrapper placer that runs both, evaluates both with real
proxy cost, returns the better.

**Expected:** +0.8% improvement, guaranteed. No risk.

---

## Experiment 3: Restore v2 config (more optimizer steps) (TRIVIAL)

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

---

## Experiment 4: Congestion weight + more steps combined

**What:** Best congestion weight from Exp 1 + v2-style step counts +
best-of(SDF, DPO).

**Why:** These improvements are orthogonal. Weight fixes the gradient
magnitude. More steps let the optimizer converge further. Best-of
catches benchmarks where DPO regresses.

**Expected:** 1.37-1.40 range if congestion weight helps.

---

## Experiment 5: Multi-seed with best config (LOW EFFORT)

**What:** Run 5-10 seeds with the best config from Exp 4, take best
per benchmark.

**Why:** 5 seeds showed 0.45% range. With a better congestion weight,
different seeds might explore more diverse congestion landscapes,
potentially widening the range.

**Caveat:** The diminishing returns analysis suggests this adds only
~0.1%. But it's free if compute is available.

---

## Experiment 6: Investigate RUDY vs real congestion (RESEARCH)

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

## What we know works (from today's experiments)

| Finding | Evidence | File |
|---|---|---|
| SDF init is essential (+247%) | ablation_random_init | experiment_log.jsonl |
| Density gradient: +21.7% | ablation_no_dens | experiment_log.jsonl |
| Congestion gradient: +5.9% | ablation_no_cong | experiment_log.jsonl |
| Multi-phase: +2.5% | ablation_phase1 | experiment_log.jsonl |
| 5-seed stability: 0.45% range | seeds 42-46 | experiment_log.jsonl |
| Congestion = 75% of proxy | component breakdown | experiment_notes.md §2 |
| RUDY underestimates 2x | DPO internal vs real | experiment_notes.md §9 |
| DPO worsens 4/17 benchmarks | SDF vs DPO per-bench | experiment_notes.md §8 |
| Decomposition fails | 22 sweep + 0/190 swaps + hierarchical | experiment_notes.md §7 |
| Legal-state methods trapped | polyhedra nav at ceiling | results.md |
| Barrier crossing: 5-12% pairs | traversal analysis | polyhedra_traversal.txt |
| v2 (more steps) +1.0% | experiment log | experiment_log.jsonl |

## What we know doesn't work

| Approach | Why | Evidence |
|---|---|---|
| Hierarchical decomposition | Objectives coupled at all scales | 0/190 swaps, §7 |
| Alternative inits (spectral, partitioning) | SDF already near-optimal coarsely | Overnight SP1 |
| Better surrogate ranking | Only 1-7 good candidates exist | Overnight SP3 |
| LP-based congestion optimization | LP-HPWL uncorrelated with proxy | rho=-0.001 |
| Multi-start brute force | Diminishing returns (0.45% range) | Seed analysis |
| Pulling clusters together | Increases density (WL-density anti-correlation) | Hierarchical v2 |
