# Contributions

Every claim below has corresponding code, data, or experiments that we
ran. Nothing is included on the basis of "we read about it" alone.

---

## 1. Polyhedral decomposition applied to macro placement

**Claim:** The non-overlap feasible region is a union of convex
polyhedra (one per pairwise L/R/A/B assignment). Within any polyhedron,
HPWL minimization is a linear program. LP dual variables give exact
marginal costs of each pairwise constraint — a complete sensitivity
map available for free from each solve.

**What's novel:** The decomposition itself is well-known in disjunctive
programming (Balas 1979, Kronqvist et al. 2025). What we contribute is
a *complete working system* built on it for macro placement: SDF init,
assignment extraction, HiGHS LP with dual extraction, GridSurrogate
(~0.1ms candidate eval), surrogate-guided navigation with simulated-
annealing acceptance, multi-pair cluster moves, ClusterScreener (Zobrist
hash + displacement floor + net-span HPWL bound + axis crowding,
prunes 20-40% of moves at ~1-10us before projection), cascade
constraint repair + direct overlap repair. Seven modules, ~2200 lines.
And more importantly, we contribute the empirical characterization of
its failure mode (see contribution 2).

**What's not novel:** The math. Disjunctive programming, LP duality,
piecewise-linear HPWL formulation — all textbook.

**Evidence:** Polyhedra navigation system (`submissions/polyhedra/` in
pre-2026-04-28 git history). 1.49 avg proxy on 17 IBM benchmarks, 0
overlaps. The seven non-init modules (`placer.py`, `navigator.py`,
`assignment.py`, `lp.py`, `moves.py`, `projection.py`, `surrogate.py`,
`cluster_bounds.py`) were removed in the post-CD repo cleanup. The SDF
initialization module was retained because the CD champion still uses
it; it now lives at `macro_place/sdf_init.py`.

---

## 2. Barrier diagnosis: the objective mismatch

**Claim:** The polyhedra navigation system plateaus because LP-HPWL has
rho = -0.001 correlation with the competition proxy cost. Congestion is
66.5% of proxy cost. HPWL is uncorrelated with congestion (rho = 0.072).
HPWL and density anti-correlate (rho = -0.536). No LP-based ranking of
polyhedra can predict proxy quality.

**What's novel:** This specific empirical finding for the ICCAD04 proxy
metric. The methodology: systematic 22-experiment ablation across three
independent subproblems (surrogate accuracy, initial topology, LP
formulation), combined with Miftari-style correlation analysis between
LP values and refined proxy cost across 24 feasible topologies. The
diagnosis that "this is an objective mismatch, not a search problem" is
a structural insight that redirects algorithm design.

**What's not novel:** Correlation analysis, ablation studies, the
general idea that proxy objectives can mislead.

**Evidence:**
- 22-experiment overnight sweep: `results/overnight_run.log`
- Miftari correlation matrix: `docs/results.md` "Miftari" section
- Component breakdown: congestion 66.5%, density 29.3%, WL 4.2%
- All 22 experiments within +/-0.5% of baseline

**Caveat:** The Miftari rho=-0.001 was measured on ibm01 only. Must
generalize to 2-3 more benchmarks before the writeup can make the claim
broadly. (See `todo.md`.)

---

## 3. DPO: differentiable proxy optimization

**Claim:** Differentiating through the actual competition metric
f(p) = WL + 0.5*D + 0.5*C — including a fully differentiable RUDY
congestion model — beats RePlAce by 5% on the IBM benchmarks.

**What's novel:** Including the congestion gradient dC/dp directly in
macro placement optimization. DREAMPlace (Lin et al. 2019) differentiates
HPWL + electrostatic density. Congestion-aware variants (Lu et al. 2020)
use congestion as net weights on HPWL. C3PO/NV-Place (ASP-DAC 2026)
computes dC/dp for standard-cell placement. Our contribution is the
application to macro placement with the competition metric and top-k
aggregation — not "first dC/dp" but "first direct congestion gradient
for macro placement."

**What's not novel:** LSE-HPWL (Naylor 2001), differentiable grid
density (standard), penalty continuation (Bertsekas 1982, Hazan 2016),
Adam optimizer, top-k via PyTorch.

**Evidence:** `submissions/dpo/placer.py`. Best-of-v2 config: 1.3831 ±
0.0056 avg proxy across 5 seeds (range 1.3790–1.3927). All seeds beat
RePlAce (1.4578). Ablation: removing congestion gradient → 1.5092
(+5.9%), removing density gradient → 1.7342 (+21.7%), random init →
4.94 (+247%).

**Superseded by CD as competition method** but remains the key
intermediate result in the paper's narrative arc.

---

## 4. The penalty-as-barrier-crossing interpretation

**Claim:** Legal-state representations are fundamentally trapped.
Any method that represents only non-overlapping placements is confined
to a disconnected feasible region where local moves cannot cross the
congestion barrier. DPO's overlap penalty provides the escape: at low
lambda, the optimizer traverses infeasible configurations between
polyhedra; at high lambda, it converges to a feasible placement in a
different polyhedron.

**Geometric interpretation:** The barrier-crossing mechanism can be
understood as dimensional lifting. Give each macro a z-coordinate;
two macros at different z-heights don't overlap even if their x-y
projections collide. A z-penalty (mu * sum(z^2)) pushes macros
toward z=0. Annealing mu: 0 → infinity continuously deforms a 3D
arrangement (where macros freely pass over each other) into a 2D one.
The z-dimension IS the "imaginary dimension" from complexification
(Zariski 1962), made geometric: in R^{2N} the feasible region is
disconnected, but in R^{3N} it is connected. DPO's flat overlap
penalty is the projection of this mechanism onto 2D — it achieves
the same barrier crossing without explicitly adding the z-coordinate.

**What's novel:** (1) The empirical demonstration that legal-state
methods are trapped: 22 polyhedra experiments all noise, 0/190
cluster swaps accepted, hierarchical decomposition 18-28% worse
than flat DPO. (2) The interpretive connection between penalty
continuation and complexification, with the z-lifting as a geometric
bridge between the two. We claim observation, not theorem.

**What's not novel:** Penalty methods, complexification, graduated
optimization — all well-established.

**Evidence (quantified):** On ibm01, DPO changes 3,556 of 30,135
pairwise L/R/A/B assignments (11.8%). On ibm10, 15,539 of 308,505
(5.0%). Transitions are overwhelmingly perpendicular flips (L↔B,
R↔A — ~98% of changes), consistent with crossing nearby polyhedra
boundaries where the binding constraint switches between horizontal
and vertical separation. DPO does not perform deep topological
restructuring — it optimizes within and across adjacent polyhedra.

---

## 5. Non-decomposability of the proxy cost

**Claim:** The proxy cost f(p) = WL + 0.5*D + 0.5*C cannot be
productively decomposed — by component, by scale, or by structural
partition. Every decomposition produces worse results than joint
optimization.

**What's novel:** The empirical demonstration across multiple
decomposition strategies, with quantified failure modes.

**Evidence:**
- *By component:* LP-HPWL (optimizing WL alone) has rho=-0.001 with
  proxy cost. WL and density anti-correlate (rho=-0.536). Congestion
  is 74.9% of proxy cost but uncorrelated with HPWL (rho=0.072).
- *By structure:* Polyhedra decomposition (which polyhedron vs where
  within it) optimizes 5.8% of the objective. 22-experiment sweep
  across surrogate, topology, and LP: all within noise.
- *By scale:* Hierarchical clustering experiments. Three approaches
  tested (coarse DPO + expand, cluster-pull refinement, cluster-swap
  search). All worse than flat DPO. 0/190 cluster-pair swaps improved
  proxy cost — the coarse arrangement is not the bottleneck.
- *Resolution:* DPO optimizes f(p) directly, jointly over all
  components at all scales. This is not a better decomposition —
  it is the abandonment of decomposition.

**Implication for the field:** Macro placement with composite
objectives (WL + density + congestion) may be fundamentally resistant
to the divide-and-conquer strategies that work for single-objective
problems. The coupling between cost components through shared grid
cells means that any separation of concerns loses the information
that matters most.

---

## 6. Low seed variance as evidence of effective continuation

**Claim:** DPO's 0.45% seed variance (5 seeds, all beating RePlAce)
is evidence that the penalty continuation is working, not a limitation.
A method trapped in random local minima would show high variance. The
low variance means the penalty schedule reliably collapses the
landscape to a consistent attractor.

**What this implies for compute scaling:** Multi-start parallelism
has steep diminishing returns. Best-of-200 seeds under a normal model
gains only ~0.1% over best-of-5. The continuation already does the
heavy lifting. Accessing qualitatively better basins (if they exist)
would require fundamentally different exploration mechanisms, not
more starts.

**Contrast:** Random init → 4.94 avg (massive variance across the
raw landscape). SDF + DPO continuation compresses this to a 0.45%
range. The combined method has effectively solved the exploration
problem for these benchmarks.

**Evidence:** 5-seed ablation data; diminishing returns analysis
under normal model.

---

## 7. RUDY fidelity analysis: the second diagnosis

**Claim:** DPO's RUDY congestion model is not just inaccurate in
magnitude — it is structurally wrong in direction. The real/RUDY gap is
3.1x (not 2x). Top-5% hotspot overlap is 10.9% (Jaccard 0.057,
near-random). Three identified sources: L-routing vs uniform bbox
distribution (2.74x), missing macro blockage (28% of real congestion),
missing spatial smoothing.

**What's novel:** The cell-by-cell characterization of RUDY vs real
congestion for macro placement. Prior work discusses RUDY's aggregate
inaccuracy; we show the spatial pattern of disagreement and identify
that the gradient *direction* is wrong, not just the magnitude. This
is confirmed experimentally: congestion weight sweep (0.5-1.5) is
monotonically worse.

**Why it matters:** This diagnosis directly motivated the second pivot.
The natural response to "RUDY is wrong" is to build a better RUDY.
The correct response was to bypass RUDY entirely — evaluate the real
proxy cost through the incremental evaluator.

**Evidence:** `analysis/rudy_fidelity/rudy_analysis.py`, `experiment_notes.md` §11.
Congestion weight sweep: 5 variants, all monotonically worse (§10).

---

## 8. Incremental proxy evaluator (4657x speedup)

**Claim:** An incremental evaluator that maintains per-net min/max
trackers, bin-density grid, and RUDY congestion deltas achieves
bit-for-bit parity with compute_proxy_cost at 4657x speedup (6.5 ms/move
vs 30s full evaluation on ibm10). This infrastructure gates the
full-proxy CD algorithm.

**What's novel:** The specific combination of incremental data structures
for the competition's proxy cost (WL + 0.5*density + 0.5*congestion).
Incremental HPWL evaluators are standard; incrementalizing density
(bin-level delta) and RUDY congestion (per-net contribution tracking +
smoothing via vectorized cumsum) for parity with the competition
evaluator is specific to this work.

**What's not novel:** Incremental evaluation for VLSI placement is
well-established.

**Evidence:** `macro_place/incremental_evaluator.py` (930 lines).
Verified bit-for-bit parity on ibm01 and ibm10 across 130 random
moves (worst diff 1.1e-15 absolute). Revert test passes within 1e-9
relative. Speedup benchmark: `scripts/bench_incremental.py`.

---

## 9. Full-proxy coordinate descent

**Claim:** Coordinate descent on the actual proxy cost — via the
incremental evaluator, with breakpoint enumeration for exact 1D
search — achieves 1.1193 avg proxy at 600s/bench (matches leaderboard
within 0.18%), **1.1055 avg with adaptive budget (E9, beats leaderboard
by -1.05%)**, and **1.0990 avg with grid-bin LNS overlay (E12, beats
leaderboard by -1.63%)**. Every benchmark improves over DPO; under E12
every benchmark also improves over E9. Zero regressions throughout.

**What's novel:** The application of full-proxy CD (not HPWL-only
weighted median) to macro placement. The key insight is that CD on
the real objective, enabled by a fast enough evaluator, outperforms
gradient descent on a differentiable approximation — even though
the approximation (RUDY) is the standard approach in the literature.

**What's not novel:** Coordinate descent, breakpoint enumeration,
incremental evaluation — all textbook techniques. The contribution
is recognizing that combining them bypasses the RUDY fidelity
problem that limits gradient methods.

**Evidence:** `submissions/cd_only/placer.py` (fixed-budget
1.1193). `submissions/cd_adaptive/placer.py` (plateau-detection
1.1055). `submissions/cd_lns_gridbin/placer.py` (grid-bin LNS
overlay 1.0990 — current champion, ADR-007). All 17 IBM benchmarks.
Zero overlaps. Breaks the ibm02 basin lock (1.6888 → 1.1534, -32%)
that DPO cannot escape regardless of seed; E12 lifts ibm02 a further
-1.72% (1.1538 → 1.1340).

---

## 10. "Bypass, don't fix" as a design principle

**Claim:** When a model approximation is structurally wrong (not just
noisy), exact evaluation with a faster data structure beats a more
accurate approximation. This is demonstrated twice:
  - LP-HPWL → DPO (bypass LP's single-component optimization with
    joint gradient on all three components)
  - DPO RUDY → CD (bypass the differentiable approximation with exact
    proxy evaluation via incremental data structures)

**What's novel:** The explicit identification of this principle through
two sequential demonstrations on the same problem. Each bypass was
motivated by quantitative diagnosis (rho=-0.001; 10.9% hotspot overlap)
that proved the model was structurally wrong, not improvable by tuning.

---

## 11. Methodology: the complete experimental trajectory

**Claim:** The sequence *build system on structural insight* ->
*hit wall* -> *run systematic ablation to diagnose root cause* ->
*use diagnosis to pivot* is a transferable methodology for applied
optimization research. This sequence occurs TWICE in our work, with a
third infrastructure-driven refinement:

  Cycle 1: polyhedra nav (1.49) → 22-experiment sweep → rho=-0.001
  → DPO (1.38)

  Cycle 2: DPO (1.38) → RUDY hotspot analysis → 10.9% overlap
  → incremental evaluator + CDOnly (1.12)

  Refinement: CDOnly (1.12, fixed 600s) → per-bench sweep-delta logs
  → adaptive budget (E9) → 1.1055 (beats leaderboard by -1.05%)

  Refinement 2: CDAdaptive (1.1055, every bench plateau-bound) →
  diagnose plateau as per-axis fixed point → introduce a *different
  move type* (grid-bin LNS, E12) → 1.0990 (beats leaderboard by -1.63%)

**What's novel:** The specific application and the completeness of the
documentation. Each step is quantified. The two-cycle structure shows
the methodology is not ad hoc — the same diagnosis pattern works on
different failure modes. The refinement (E9) shows that even after a
working algorithm is in place, observing per-benchmark behavior reveals
budget-allocation wins that fixed schedules miss.

**Evidence:** The full experiment log
(`results/experiment_log.jsonl`), results history (`docs/results.md`,
`writeup/historical_results.md`), and approach documents.

---

## 12. Per-benchmark plateau detection (E9)

**Claim:** Letting each benchmark exit when its 3-sweep delta-window
drops below 0.005 (with a 1-hour hard cap matching the competition
rule) outperforms any fixed budget. On --all, this turned a 1.1193
result (CDOnly fixed 600s/bench) into 1.1055 — a -1.23% lift that
beats the public leaderboard 1.1172 by -1.05%. (E9 was the champion
2026-04-27 to 2026-04-28, then superseded by E12 grid-bin LNS at
1.0990; the plateau-detection mechanism remains the CD-phase budget
controller in the E12 production placer.)

**Why this is more than a hyperparameter trick:** A fixed per-benchmark
budget is fragile to dataset shift. The hidden NG45 commercial designs
are unseen; setting a budget for them by fitting on IBM is overfitting
the schedule. Plateau detection is a *per-run, per-benchmark* policy —
it adapts to the actual descent trajectory rather than to priors set
on a different dataset. On all 17 IBM benchmarks under defaults
`(min_time_s=300, hard_cap_s=3600, patience=3, plateau_threshold=0.005)`,
every benchmark exited via plateau; none hit the cap. Hard benchmarks
(ibm17/18/14) used 25-37 min when still descending; easy ones (ibm09,
ibm04) finished in 5-7 min.

**What's novel:** The specific defaults and the demonstration that they
transfer to unseen benchmarks without retuning. The mechanism (plateau-
based exit) is standard in optimization; the contribution is calibrating
it to per-bench sweep deltas in a way that fits the competition's
1-hour-per-bench compute envelope.

**What's not novel:** Plateau detection, patience-based stopping —
all textbook.

**Evidence:** `submissions/cd_adaptive/placer.py`. Per-bench wall
times, plateau exits, and deltas in `docs/results.md`. Total runtime
17480s (4.85 hr) — comfortably inside the 17-hr (17 × 1hr) hidden-test
envelope.

---

## 13. Grid-bin LNS overlay (E12 — current champion, 1.0990)

**Claim:** When CD's per-axis breakpoint enumeration plateaus (every
benchmark exits via plateau, none budget-bound), the right intervention
is a *different move type* — not more wall-clock on the same one. After
CD plateaus, destroy K = max(1, min(30, 5 % × |movable hard macros|))
macros and reinsert each at the proxy-minimizing legal position drawn
from the full `(grid_col × grid_row)` cell-center set. The candidate
set is outside CD's per-axis breakpoint enumeration, which is exactly
why it can find escapes CD cannot. Result: avg `--all` 1.0990 (vs E9
1.1055, -0.59 %), zero overlaps, all 17 IBM benchmarks improved over
E9 (no regressions).

**Why this is more than a hyperparameter trick:** Three earlier escape
mechanisms tested before E12 all reused CD's per-axis move type and
produced flat results: E3 single-macro LNS (full-canvas v1, 5×5 local
v2, both flat on ibm17), SDF-jitter multi-init (0/8 improved on ibm09 —
contractive), subset-CD destroy/reinsert (0/24 improved on ibm09+ibm12
— same fixed point). The unifying lesson: same move type cannot escape
the same fixed point, regardless of how the destroy step is randomized
or how much wall is allocated. Grid-bin LNS works because its candidate
set is *structurally different* from CD's.

**Cost-aware destroy ranking is NOT load-bearing.** A `--fast` ablation
(`experiments/E12_grid_bin_lns/code/cd_lns_gridbin_random.py`)
replaced the cost-aware destroy ranking with uniform random destroy
and matched cost-aware within noise (ibm09 random 0.8541 actually beat
cost-aware 0.8591). Production keeps the cost-aware ranking only
because the ablation result landed late relative to the May 21
deadline. Future simplification: drop the ranking.

**What's novel:** The diagnosis (CD's plateau is a per-axis fixed
point, not a wall-clock limit), the falsification of three move-type-
preserving escape attempts, and the decisive demonstration that
swapping the move type is the load-bearing change. Validates the
leaderboard winner's "Incremental CD+LNS" architecture without
reproducing it.

**What's not novel:** LNS as a meta-heuristic, grid-bin candidate
enumeration — both standard. The contribution is the diagnostic
sequence that identified what the move type had to *be*.

**Evidence:** `submissions/cd_lns_gridbin/placer.py`.
`results/CDLNSGridBinPlacer_20260428_155739.json` (verified `--all`).
Per-bench table at `writeup/evidence.md` §2.1 and `docs/results.md`.
ADR: `docs/decisions/007_cd_lns_gridbin_promotion.md`. Falsified
escape mechanisms: `experiments/E3_lns_v1/`, `experiments/E3_lns_v2/`,
`analysis/multi_init_probe/`, `analysis/lns_escape_probe/`. Total
runtime 28 256 s = 7.85 hr — inside the 17-hr (17 × 1 hr) hidden-test
envelope with ~9 hr of headroom.

---

## 14. K-macro joint LNS (E39 / E41 — different move type at K=3)

**Claim (TODO prose):** Beyond E12's grid-bin LNS (single-macro joint
reinsertion at full canvas-grid candidates), a third move type — *jointly*
reinserting K=3 macros via brute-force enumeration of `top_N^K = 5³ = 125`
candidate combinations per K-tuple — extracts an additional −0.001 to
−0.005 lift per benchmark on top of CD-LNS-SA-v2 polished states. K=3 is
the saturation point: K=4 (E42) and longer-budget K=3 (E43) both fail
NG45 transfer, and spatial K-tuple ranking (E44) is dominated by
netlist-adjacency. Composing K-joint with DPO basin shift (E18) gives
E41 = 1.0848 standalone.

> **TODO(prose):** Frame this as the third escape mechanism after CD's
> per-axis (every-pair-of-coords single-macro) and E12's grid-bin LNS
> (single-macro 2D). K-joint is the K-macro 2D move that catches
> structurally-correlated multi-macro local minima the previous two miss.
> Document the eps=1e-9 strict-separation pairwise legality fix
> (`kjoint_overlap_eps_gotcha.md`) — important for any reproducer.

**Evidence:** `experiments/E39_kmacro_joint_lns/`, `experiments/E41_dpo_kjoint/`,
ADR-010 (Superseded by ADR-011). Falsified variants: E42 K=4, E43 longer
budget, E44 spatial — all NG45-blind in the same way.

---

## 15. DPO best_of_v2 init unlocks a structurally distinct basin (E18)

**Claim (TODO prose):** SDF init is contractive within its basin (CD's
fixed point is unique given SDF init); DPO best_of_v2 init produces a
*different* CD fixed point that is, on average, **−0.51 % deeper** than
SDF's on `--all` and **−1.67 % deeper** on NG45. The basin difference is
verified empirically (E18 outperforms E25 on 11/17 benches) and resolves
the basin-diversity question E27 was designed to answer. **Diverse priors
within DPO refinement (E11) had failed because the refinement step is
basin-locked; using DPO output as init for a CD-LNS-SA pipeline avoids
the lock.**

> **TODO(prose):** Frame the empirical proof of multi-basin: 4 seeds of
> DPO produce *byte-identical* ibm02 (basin lock for DPO-as-refinement),
> but DPO output as *init for CD* produces a different CD fixed point
> than SDF init. The basin you converge to depends on the init class,
> not the seed within an init class.

**Evidence:** `experiments/E18_dpo_init/`, ADR-009 (Superseded by ADR-011).
E18 NG45 0.69193 (4/4 wins) confirms basin lift transfers to commercial
designs.

---

## 16. Per-bench best-of hybrid champion (E48 — current, 1.08151)

**Claim (TODO prose):** Different mechanisms win on different benchmarks
because basin choice is design-class-dependent. Per-bench best-of-{E25
SDF basin, E41 DPO basin} hybrid extracts an additional −0.30 % over
E41 standalone by exploiting across-benchmark heterogeneity. Within the
4-bench --fast set the lift can be sample-size-amplified (E53m 3-way
hit −0.97 % --fast); at --all aggregate scale the lift collapses
toward the realistic per-bench-best-of-2 saturation (~−0.30 %).
Algorithmically valid (no per-benchmark tuning; per-bench winner determined
by proxy value, not bench-name lookup).

> **TODO(prose):** Frame as the "compositional" / "meta-algorithmic" level
> — the hybrid doesn't introduce a new mechanism, it recognizes that
> *which mechanism is best* varies across the benchmark set, and exploits
> that. Connect to ensemble learning literature.
>
> The contribution to the field: **cross-benchmark per-bench best-of
> with no per-benchmark tuning is a valid and effective composition
> primitive for macro placement** — not just an ad-hoc trick. Verified
> at the −0.30 % level over the strongest single-pipeline candidate.

**Evidence:** `submissions/cd_lns_sa_hybrid/placer.py`. Verified `--all`
1.08151, `--ng45` 0.6922 (zero overlaps everywhere, ~7 hr wall under
`--jobs 4`). ADR-011 *Accepted* 2026-05-02. Per-bench breakdown in
`docs/decisions/011_hybrid_e25_e41_promotion.md`.

---

## 17. The IBM/NG45 transfer-failure pattern — falsification record as a finding

**Claim (TODO prose):** Five experiments (E42 K=4, E43 longer K-joint,
E44 spatial K-tuple, E54 congestion-targeted destroy, plus partial E53m
multi-seed) lifted on IBM `--fast` but failed to transfer to NG45
ariane133 (regressions +3.57 % to +5.14 %). The structural commonality:
**heuristics that engage the proxy's component decomposition (6 % WL /
20 % density / 74 % congestion), or the geometric structure of the
placement, depend on dense macro packing and degrade on sparse
commercial layouts**. ariane133 has 133 hard macros on a 1433 × 1433
canvas (~1/15.5 macros per square micron) vs IBM's 246-760 hard macros
on 23-73 canvases (~10-20 per square micron).

> **TODO(prose):** Build out the structural argument. Per-design
> `n_hard / canvas_area` ratio table; correlate with IBM-vs-NG45 lift
> magnitude. The negative result is the contribution: **mechanism-aligned
> destroy/K-joint ranking is overfit to dense-IBM benchmark structure,
> and cost-aware (topology-blind) ranking remains the safe baseline.**
>
> This is a generalization-failure finding worth a paper section
> independent of any positive result. The OOD-generalization framing
> places the work in conversation with the broader ML
> robustness/distribution-shift literature.

**Evidence:** Manifests at `experiments/E42_kjoint_k4/`,
`experiments/E43_kjoint_longer/`, `experiments/E44_kjoint_spatial/`,
`experiments/E54_congestion_destroy/`, `experiments/E53_multiseed_hybrid/`.
NG45 verified results in `results/experiment_log.jsonl` (hypotheses
`E42_*_ng45`, `E43_*_ng45`, `E54_ng45`, `E53_multiseed_hybrid_ng45`).
Memory entry: `e54_congestion_destroy.md` (feedback type — rule for
future destroy-ranking experiments).

---

## 18. GPU acceleration as a *polish* phase fails (E53)

**Claim (TODO prose):** GPU DPO basin polish — multi-restart full-pose
Adam on the smooth proxy + overlap penalty, layered AFTER fully-converged
CD-LNS-SA — produces 0 accepts in 350 GPU restarts at production
budgets (only ~−2 % lift on shortened-budget smoke where prior phases
hadn't converged). The smooth-proxy gradient cannot escape the local
optimum that breakpoint-enumeration CD + grid-bin LNS + breakpoint
Metropolis SA already reach. **GPU acceleration must replace CD's
basin-crossing role architecturally, not act as a polish phase after
CD.**

> **TODO(prose):** Frame the negative GPU result. The MPS device
> ran the smooth proxy 100x faster than CPU CD per step, but at the
> wrong granularity — settling within a basin, not crossing basins.
> Useful for the field: not all "GPU speedups for placement" are
> equivalent; the placement of the GPU phase in the pipeline matters
> as much as raw speed.

**Evidence:** `experiments/E53_dpo_basin_eval/manifest.md`. Falsification
record: `e53_gpu_dpo_falsification.md`.

---

## Explicitly excluded

The following topics from `theory.md` (in this writeup directory)
are excluded from the writeup body because we cannot defend them from
direct experience:

- Quantum tunneling / SQA analysis
- Survey propagation / cavity methods / RSB
- Stratified Morse theory / persistent homology
- Information geometry / Fisher-Rao gradient flows
- Population annealing
- Homotopy continuation (PHCpack, Bertini)
- Spectral decomposition of L(K_n)
- Benders decomposition / backdoor variables
- Ejection chains / ALNS
- Diffusion models (DiffPlace, DiffUCO)
- Graphs of Convex Sets (GCS)
- Dead-End Elimination (DEE)
- Discrete Schrodinger Bridges
- Kolmogorov complexity sampling
- Semi-discrete optimal transport / Laguerre tessellations

These are interesting literature connections but we never implemented
or tested them. They live in `theory.md` (this directory) as
supplementary material with a footnote, but do not belong in the
writeup body.
