# Experiment Notes for Writeup

Date: 2026-04-26

## 1. DPO Ablation Study (all 17 benchmarks)

### Results

| Variant | Avg Proxy | vs DPO | vs RePlAce |
|---|---|---|---|
| **Full DPO (seed 42)** | **1.4246** | baseline | **+2.3%** |
| Seed 43 | 1.4301 | +0.4% | +1.9% |
| Seed 44 | 1.4267 | +0.1% | +2.1% |
| Seed 45 | 1.4237 | -0.1% | +2.3% |
| Seed 46 | 1.4268 | +0.2% | +2.1% |
| No congestion gradient | 1.5092 | +5.9% | -3.5% |
| No density gradient | 1.7342 | +21.7% | -19.0% |
| Phase 1 only | 1.4605 | +2.5% | -0.2% |
| Random init (no SDF) | 4.9415 | +247% | -239% |

### Multi-seed stability and what it proves

- Mean: 1.4264, Stdev: 0.0025, Range: 0.0064 (0.45%)
- All 5 seeds beat RePlAce (1.4578). Result is stable.
- Under a normal model, best-of-200 seeds gains only ~0.006 (0.4%)
  over best-of-5. Diminishing returns are steep.

**The low variance is evidence, not a limitation.** A method trapped
in random local minima would show HIGH variance — each seed landing
in a different basin of unpredictable quality. DPO's 0.45% range
means the penalty continuation reliably guides the optimizer to the
same region of configuration space regardless of the stochastic path.
This is the practical signature of an effective continuation method:
the landscape, as smoothed by the penalty schedule, has a consistent
attractor that dominates the optimization dynamics.

Contrast with what random init shows: seed 42 from random positions
gives 4.94 avg proxy — the landscape without SDF's basin has many
terrible local minima. SDF + DPO's penalty continuation collapses
this chaos to a 0.45% range, meaning the combined method has
effectively solved the exploration problem for these benchmarks.

**Multi-start has steep diminishing returns.** The normal model
predicts best-of-N improvement:

| N seeds | Expected best | vs RePlAce | Marginal gain |
|---|---|---|---|
| 5 | 1.4219 | +2.5% | — |
| 50 | 1.4195 | +2.6% | +0.002 |
| 200 | 1.4183 | +2.7% | +0.001 |

Going from 5 to 200 seeds buys ~0.1% additional improvement. The
penalty continuation already does the heavy lifting — brute-force
parallelism adds little on top. This suggests that fundamentally
different exploration mechanisms (HMC, parallel tempering, learned
congestion models) would be needed to access better basins, if they
exist at all.

### Key findings for writeup

1. **SDF init is essential** (+247% without it). DPO cannot converge from
   random positions — the landscape has too many bad local minima. SDF
   provides the basin; DPO optimizes within and across it.

2. **Density gradient contributes the most** (+21.7%). This is the primary
   signal that pushes macros away from top-10% hotspots. Without it, macros
   cluster and density cost explodes.

3. **Congestion gradient has real value** (+5.9%). This confirms dC/dp is
   not noise — it provides a meaningful signal that HPWL optimization alone
   misses. The 5.9% gap is the difference between beating RePlAce (1.42 vs
   1.46) and losing to it (1.51 vs 1.46).

4. **Multi-phase continuation helps** (+2.5%). Phase 1 alone gets close
   (1.46) but doesn't fully resolve overlaps (60 remaining, 8 invalid).
   Phases 2-3 with higher lambda eliminate residual overlaps.

5. **Low seed sensitivity** (0.45% range). The optimization is robust —
   not dependent on a lucky initialization seed.

6. **DPO worsens 4/17 benchmarks vs SDF init.** On ibm01, ibm02, ibm06,
   ibm12, the SDF init is BETTER than DPO output. DPO's optimization
   (not just congestion gradient — no-cong ablation is even worse)
   degrades these benchmarks. Best-of(SDF, DPO) per benchmark gives
   **1.4135 avg (3.0% better than RePlAce)** vs DPO-only 1.4246 (2.3%).
   This is a free +0.8% improvement from simply evaluating both and
   keeping the winner.

---

## 2. Component Breakdown Across All Benchmarks

Computed on DPO-optimized placements (seed 43):

| Benchmark | Proxy | WL% | Density% | Congestion% |
|---|---|---|---|---|
| ibm01 | 1.280 | 9.2% | 26.4% | 64.3% |
| ibm02 | 1.763 | 6.2% | 16.0% | 77.7% |
| ibm03 | 1.278 | 8.2% | 19.8% | 71.9% |
| ibm04 | 1.376 | 6.9% | 18.8% | 74.3% |
| ibm06 | 1.767 | 4.9% | 14.5% | 80.6% |
| ibm07 | 1.424 | 6.0% | 19.1% | 74.8% |
| ibm08 | 1.434 | 6.3% | 18.1% | 75.6% |
| ibm09 | 1.063 | 6.6% | 25.4% | 68.0% |
| ibm10 | 1.279 | 6.1% | 21.4% | 72.5% |
| ibm11 | 1.092 | 6.3% | 23.3% | 70.4% |
| ibm12 | 1.706 | 4.4% | 15.4% | 80.2% |
| ibm13 | 1.245 | 5.5% | 20.9% | 73.7% |
| ibm14 | 1.503 | 4.4% | 17.7% | 77.9% |
| ibm15 | 1.402 | 4.9% | 19.2% | 76.0% |
| ibm16 | 1.430 | 4.2% | 19.7% | 76.1% |
| ibm17 | 1.615 | 4.1% | 16.1% | 79.7% |
| ibm18 | 1.655 | 4.0% | 17.0% | 78.9% |
| **AVG** | | **5.8%** | **19.4%** | **74.9%** |

**Generalizes the ibm01 finding:** Congestion dominates proxy cost on
ALL benchmarks (64-81%, avg 74.9%). Not just ibm01. The objective
mismatch (LP-HPWL is blind to congestion) is structural across the
entire benchmark suite.

Note: these fractions are *after* DPO optimization. DPO successfully
reduces density (from ~0.9 to ~0.5), which makes congestion's share
even larger. The pre-DPO (SDF init) fractions are more balanced
(ibm01: WL 6.1%, D 39.4%, C 54.5%).

---

## 3. Improvement Potential Assessment

### Can DPO be improved?

**Yes, moderately.** The ablation study suggests several paths:

1. **Better congestion model.** RUDY underestimates real congestion by
   ~2x. A more accurate differentiable congestion model (e.g., net-by-net
   L-routing approximation) would directly improve the congestion gradient.
   This is the most promising direction.

2. **Adaptive phase scheduling.** Current phases use hardcoded step counts.
   Convergence-based transitions could help weaker benchmarks (ibm01, ibm06)
   where the fixed schedule may cut off too early.

3. **Multi-seed + take best.** 5 seeds show 0.45% range. Taking the best
   of 3-5 seeds gives a free ~0.1-0.2% improvement for the competition.

4. **SA polish.** 5-10s of SA refinement after legalization could recover
   quality lost during overlap repair. Expected +0.2-0.5%.

Realistic target with these improvements: **1.38-1.40** avg proxy
(3-5% better than RePlAce).

### Can polyhedra navigation be improved?

**Unlikely beyond current ceiling.** The 22-experiment overnight sweep
exhaustively tested surrogate accuracy, initial topology, and LP
formulation. All within noise. The fundamental bottleneck is the
objective mismatch: LP-HPWL is uncorrelated with proxy cost.

The only path to improving polyhedra navigation would be solving a
non-convex LP (minimizing congestion + density + HPWL jointly) within
each polyhedron. This is possible in principle (successive LP
approximation) but would be a major engineering effort with uncertain
payoff.

### Implications for writeup framing

The ablation results strengthen the writeup significantly:

1. **Contribution 3 (DPO) is quantified.** Each component has a
   measured contribution. The congestion gradient provides 5.9% — this
   is the specific innovation over HPWL-only optimization.

2. **The SDF → DPO pipeline is validated.** SDF provides the starting
   basin (essential), DPO refines it (density + congestion gradients).
   Neither alone is sufficient.

3. **Multi-seed stability validates the result.** 1.4264 ± 0.0025 is
   not a lucky seed. All 5 seeds beat RePlAce.

4. **Congestion dominance is universal** (74.9% avg, not just ibm01).
   This strengthens the barrier diagnosis: any HPWL-based method is
   optimizing <6% of the objective.

---

## 4. Polyhedra Traversal (barrier-crossing quantification)

| Benchmark | Movable Pairs | Changed | % Changed | Mean Displacement |
|---|---|---|---|---|
| ibm01 (small) | 30,135 | 3,556 | **11.8%** | 1.5 um (4.5% diagonal) |
| ibm10 (large) | 308,505 | 15,539 | **5.0%** | 2.3 um (2.1% diagonal) |

**Barrier crossing is real.** DPO changes 3,500-15,500 pairwise L/R/A/B
assignments between SDF init and final output. This means DPO traverses
thousands of polyhedra boundaries via the overlap penalty's barrier-
crossing mechanism. The smaller benchmark (ibm01) shows more aggressive
crossing (11.8%) because the tighter canvas requires more rearrangement.

Top transition types are overwhelmingly L↔B and R↔A (perpendicular
flips), not L↔R or A↔B (same-axis flips). This means DPO is rotating
relative macro positions, not just sliding them.

---

## 5. Literature Check: Congestion Gradient Novelty

**dC/dp is NOT novel in standard-cell placement, but appears novel for
macro placement.** Key prior works (all standard-cell):

1. **DREAMPlace-Cong (DATE 2021):** Additive C(p) term in objective,
   but gradients computed **numerically/discretely** — not autograd.
   46% of runtime spent on numerical gradient computation.

2. **DCGP (DAC 2025):** Differentiable congestion from Poisson equation.
   True differentiable congestion term for **net movement**, not cell
   positions directly.

3. **C3PO / NV-Place (ASP-DAC 2026):** NVIDIA. Claims "first fully
   differentiable routability kernel" with exact RUDY gradients via
   custom CUDA + PyTorch autograd. This is the closest prior work to
   ours — direct dC/dp via backprop for standard-cell placement.

4. **DREAMPlace 4.0 (DATE 2022):** Timing-driven, not congestion. No dC/dp.

5. **Lee et al. (ICML 2025, diffusion):** No congestion in score model.

**Critical distinction:** All prior dC/dp work targets **standard-cell**
global placement. For **macro placement** specifically, no paper appears
to compute dC/dp directly — macro placement literature (AutoDMP, EGPlace,
diffusion-based) uses WL + density + overlap, with congestion at most as
a post-hoc evaluation metric.

**Reframing:** "First direct differentiable congestion gradient for macro
placement" is defensible. "First dC/dp in any placement" is not. The
technique (differentiable RUDY) is not novel; the application to macro
placement with the competition metric is.

---

## 6. DPO Version Confusion

| Version | Avg Proxy | Runtime | Wins |
|---|---|---|---|
| v2 (more steps) | **1.4107** | 311s | **15/17** |
| v3 (final) | 1.4246 | 288s | 2/17 |

**v2 is systematically better** by 1.0% — this exceeds seed variance
(0.45%). v2 beats v3 on 15/17 benchmarks. v3 was likely chosen to
reduce runtime (288s vs 311s) but sacrificed quality.

**Implication for writeup:** Should report v2's 1.4107 as the best
result, or re-run v2's configuration with multi-seed. v2 at 1.4107
beats RePlAce by **3.2%**, not 2.3%. This is a meaningfully stronger
result.

**Implication for improvement:** Simply running more optimization steps
(v2's configuration) improves results. This suggests the optimizer
hasn't converged — more compute = more quality. The "adaptive phases"
TODO item (item 7 in todo.md) would address this directly.

---

## 7. Hierarchical Placement Experiments (cluster → coarse → fine)

Tested whether decomposing the problem hierarchically — deciding global
macro group arrangement first, then refining individual positions —
could improve on flat DPO.

### Three approaches tested

| Approach | ibm01 | ibm04 | ibm09 | ibm13 | vs Flat DPO |
|---|---|---|---|---|---|
| Coarse DPO → expand → fine DPO | 1.43 | 1.70 | 1.34 | 1.51 | +18-28% worse |
| Cluster-pull refinement → DPO | 1.25 | 1.47 | 1.27 | 1.53 | +3-24% worse |
| Cluster-swap search → DPO | 1.21 | — | — | — | 0/190 swaps accepted |
| **Flat DPO (baseline)** | **1.21** | **1.36** | **1.05** | **1.24** | — |

### Why decomposition fails

The proxy cost f(p) = WL + 0.5*D + 0.5*C couples all macro positions
through shared grid cells. Density and congestion are *global, non-
decomposable* properties — moving one cluster changes congestion on
every grid cell between its old and new location. You cannot optimize
"where do groups go" independently of "where do individual macros go"
because the objectives are coupled at every scale.

Specific failure modes:
1. **Tight packing** (approach 1): Shelf-packing into super-macro boxes
   creates artificial density hotspots. Fine DPO cannot undo the damage.
2. **Cluster attraction** (approach 2): Pulling connected macros together
   improves WL but *increases* density — the same WL-density anti-
   correlation (rho=-0.536) that defeats the polyhedra LP.
3. **Cluster swaps** (approach 3): 0/190 pairwise cluster swaps on ibm01
   improved proxy cost. SDF's force-directed spreading already finds a
   near-optimal coarse arrangement. The coarse level is not the bottleneck.

### Key insight for writeup

**Decomposition is counterproductive for this objective.** Every approach
that separates "coarse arrangement" from "fine positioning" loses
information about the coupling between WL, density, and congestion. The
polyhedra decomposition (HPWL LP within each polyhedron) optimizes one
component. Hierarchical clustering optimizes at one scale. Both fail
because the proxy cost requires joint optimization of all components
at all scales simultaneously.

DPO succeeds precisely because it does NOT decompose: it differentiates
through the full f(p) = WL + 0.5*D + 0.5*C, moving all macros at once,
seeing all three cost components in every gradient step. This is the
practical resolution of the objective mismatch — not a better
decomposition, but abandoning decomposition entirely.

This also explains why the polyhedral decomposition, despite being the
correct structural description of the feasible region, doesn't translate
to algorithmic advantage. The structure is real (the feasible region IS
a union of polyhedra) but the useful decomposition (separate topology
from positions) doesn't align with the objective's structure (all
components are coupled through positions).

---

## 8. Assessment: Can We Improve for the Writeup?

### DPO improvement potential: YES

| Improvement | Expected Impact | Effort |
|---|---|---|
| **Best-of(SDF, DPO) per benchmark** | **+0.8% (1.4135, free)** | **Zero** |
| Use v2 config (more steps) | +1.0% (1.41 → already demonstrated) | Trivial |
| Multi-seed best-of-5 | +0.2-0.3% | Trivial |
| Better RUDY model | +1-2% (uncertain) | 1-2 days |
| SA polish after legalization | +0.2-0.5% | Half day |
| Adaptive phase scheduling | +0.5-1% (uncertain) | 1 day |

**Lowest-hanging fruit: best-of(SDF, DPO).** DPO worsens 4/17
benchmarks vs SDF init. Simply evaluating both and keeping the winner
per benchmark gives 1.4135 (3.0% over RePlAce). This is zero-effort
and reveals something important: DPO's differentiable proxy is
imperfect, and the optimization sometimes moves macros in the wrong
direction. The benchmarks where this happens (ibm01, ibm02, ibm06,
ibm12) are NOT just the small ones — ibm12 has 651 macros.

**Combined low-hanging fruit:** best-of(SDF, DPO) + v2 config +
best-of-5 seeds could give ~1.37-1.39 avg (4-6% better than RePlAce).

### Init quality findings

DPO worsens 4/17 benchmarks vs SDF init (ibm01, ibm02, ibm06, ibm12).
On these benchmarks, the SDF init is better than DPO output AND better
than no-congestion DPO. The entire DPO optimization — not just the
congestion gradient — degrades these benchmarks.

| Benchmark | SDF init | DPO output | No-cong DPO | Best |
|---|---|---|---|---|
| ibm01 | **1.1953** | 1.2105 | 1.5026 | SDF |
| ibm02 | **1.6888** | 1.7560 | 1.7144 | SDF |
| ibm06 | **1.7150** | 1.7553 | 1.9843 | SDF |
| ibm12 | **1.6497** | 1.7166 | 1.7174 | SDF |

**This means:** the differentiable proxy is an imperfect model of the
real proxy cost. On some benchmarks the approximation error actively
misleads the optimizer. The practical response: evaluate both SDF and
DPO, keep the winner per benchmark. This is itself evidence that the
optimization landscape has benchmark-dependent structure that a single
algorithm cannot universally exploit.

**Improving init is not the path forward.** SDF is already at or near
the coarse-level optimum (0/190 cluster swaps helped). Alternative
inits all tested worse (overnight SP1). The init isn't the bottleneck —
the differentiable proxy's fidelity is.

### Polyhedra improvement potential: NO

The polyhedra system is at ceiling (1.49). The 22-experiment sweep
proved this. The objective mismatch is structural. No parameter tuning
can overcome it.

### Writeup framing strengthened by these experiments

1. **Ablation quantifies every claim.** Density gradient (+21.7%),
   congestion gradient (+5.9%), multi-phase (+2.5%), SDF init (+247%).
2. **Congestion dominance generalized.** 74.9% avg across all 17
   benchmarks, not just ibm01.
3. **Barrier crossing quantified.** 5-12% of pairs change assignment.
4. **Multi-seed stability validated.** 0.45% stdev across 5 seeds.
5. **Congestion gradient claim correctly scoped.** Not "first" but
   "direct analytical RUDY applied to competition metric."
6. **v2 result available.** Can report 1.4107 (3.2% over RePlAce) if
   we validate with multi-seed.

---

## 9. Gap Analysis: What Would 1.22 Avg Require?

Leading teams achieve ~1.22 avg proxy — 16% better than RePlAce, vs
our 3%. The gap is 0.19 proxy cost points on average.

### Where the gap comes from

| Component | Our avg | Fraction | Headroom |
|---|---|---|---|
| WL | 0.081 | 5.7% | None — already minimal |
| 0.5*Density | 0.271 | 18.9% | Some — density ~0.54, floor ~0.3 |
| **0.5*Congestion** | **1.078** | **75.4%** | **Most — congestion ~2.16** |

To reach 1.22, congestion must drop by ~19% (from 2.16 to 1.74). This
is the ONLY component with enough headroom. WL is already 0.081 — you
can't subtract 0.21 from it. Density is 0.54 — even halving it only
saves 0.13.

### What 1.22 teams likely do differently

1. **Routing-aware congestion model.** RUDY underestimates real
   L-routing congestion by ~2x. A model that accurately predicts
   post-routing congestion — using actual global routing, learned
   surrogates from routing data, or Steiner-tree estimation — would
   give much better congestion gradients. This is our biggest
   weakness: the ablation shows the congestion gradient contributes
   +5.9%, but it's working with a 2x-inaccurate signal.

2. **Per-benchmark tuning.** Our approach uses identical hyperparameters
   across all 17 benchmarks. DPO worsens 4/17 benchmarks — a team with
   per-benchmark tuned parameters (phase lengths, learning rates,
   congestion weight) could avoid these regressions.

3. **GPU + massive search.** With 1hr + GPU, you could run thousands
   of DPO variants (different hyperparams, different inits, different
   congestion models) and take the best per benchmark. Even our
   5-seed range (0.45%) suggests this has diminishing returns, but a
   team running 10,000 configurations would have a wider search.

4. **Global router in the loop.** Replace RUDY with an actual fast
   global router (or differentiable router like DGR). Evaluate real
   congestion at each step or every K steps. This is expensive but
   eliminates the RUDY approximation error entirely.

### The honest assessment

Our congestion model (RUDY) is the binding constraint. The 0.19 gap
to 1.22 requires a ~19% congestion reduction that RUDY gradients
cannot drive because RUDY doesn't see the real congestion landscape.
Everything else in our pipeline — the objective formulation, the
penalty continuation, the SDF init — is sound. The ceiling is set by
how accurately we model congestion.

---

Sources (literature check):
- [DREAMPlace routability DATE 2021](https://www.cse.cuhk.edu.hk/~byu/papers/C112-DATE2021-DREAMPlace-Cong.pdf)
- [DCGP DAC 2025](https://ieda.oscc.cc/res/papers/25-DAC25-DCGP.pdf)
- [DGR Differentiable Global Router](https://dl.acm.org/doi/10.1145/3649329.3656530)
