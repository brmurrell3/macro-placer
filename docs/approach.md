# Approach

The current submission optimizes the canonical proxy directly via
**cascading Hessian saddle escape on top of a multi-lane init + polish
pipeline**, with an optional DREAMPlace lane.

```
SDF init       -> CD + LNS + SA-v2 polish (E25 lane)
DPO init       -> CD + LNS + SA-v2 + K-joint polish (E41 lane)
[Option B only] DREAMPlace -> greedy_macro_legalize -> matched polish (DP lane)
plateau pick: argmin over the 2 or 3 lane outputs
cascading saddle escape:
    while remaining time > 1.2 * avg_iter_wall:
        smooth-proxy Hessian -> Lanczos smallest-algebraic eigvec
        ±ε perturb along soft mode -> CD-adaptive polish
        if no improvement: break
deadline budget: 3000s default (50min/bench, 60-min cap)
validate zero overlaps
```

Both submission options share the cascade base; Option B adds the DP lane.

Last updated: 2026-05-16

---

## 1. Current submission

### 1.1 Tier-1 entry candidates

| Option | Placer | IBM `--all` | NG45 `--ng45` | External deps |
|---|---|---:|---:|---|
| **A** | [`submissions/cd_lns_sa_cascade/placer_adaptive.py`](../submissions/cd_lns_sa_cascade/placer_adaptive.py) | **1.07820** | **0.68102** | none |
| **B** | [`submissions/cd_lns_sa_cascade_dp_lane/placer.py`](../submissions/cd_lns_sa_cascade_dp_lane/placer.py) | **1.06650** | **0.68086** | DREAMPlace (optional; falls back to A) |

Both verified zero overlaps on 17 IBM + 4 NG45 with walls under 60 min.
Option B beats RePlAce 1.4578 by **−26.9 %**; beats public leaderboard
reference 1.1172 by **−4.5 %**. Gap to leaderboard #1 (vmallela
self-reported 1.0109) is **+5.6 %**.

Submission-day decision: ship Option B if eval environment has
DREAMPlace; ship Option A otherwise.

### 1.2 Mechanism (shared cascade base)

- **Multi-lane polish.** Two (or three) initializations are independently
  polished to plateau. Per-bench best wins. This isolates basin-quality
  variance: DPO vs SDF basins polish to different valleys on different
  benches; the DP basin opens a third valley that wins on 3 of 4 hardest
  IBM benches (E91 finding).
- **Cascading Hessian saddle escape** (E84, built on E74). Once the
  plateau is fixed, the cascade phase:
  1. Builds the smooth-proxy Hessian-vector product via
     `torch.autograd.functional.hvp` on `SmoothProxy(plc)`.
  2. Runs scipy `eigsh` with a `LinearOperator` for the smallest-
     algebraic eigenvector (the softest mode).
  3. Perturbs the placement ±ε along that mode and runs a short
     CD-adaptive polish on each branch.
  4. Picks the lower-proxy branch as the new plateau.
  5. Stops when no improvement, or the predicted next-iter cost exceeds
     remaining budget (2026-05-16 budget-management fix uses rolling
     `avg_iter_wall × 1.2` vs the previous fixed 60s safety floor).
- **PATH A delta acceleration** (post-A1). The incremental evaluator
  now exposes a `delta_cost(macro_idx, new_xy)` API that computes the
  smooth-proxy delta *without* mutating state, eliminating the
  move-then-revert pattern. Plus `delta_cost_axis_batch` for vectorized
  K-candidate evaluation. **5.36×** CD speedup on ibm10.

### 1.3 Why DREAMPlace as a third lane

E91 ("DP-full-polish autopsy") falsified the prior PATH B falsification.
The original autopsy compared **DP basins** (no polish) against
**cascade post-polish** — apples to oranges. With matched polish
budgets, DP basins polish to a different valley than SDF/DPO inits —
sometimes deeper:

| Bench | Cascade | DP + matched polish | Δ |
|---|---:|---:|---:|
| ibm12 | 1.3031 | **1.129** | **−13.3 %** |
| ibm17 | 1.4546 | **1.307** | **−10.2 %** |
| ibm14 | 1.2919 | **1.243** | **−3.8 %** |
| ibm10 | 1.0775 | 1.095 | +1.6 % (hint of regression) |

Hard-bench aggregate: cascade 1.2818 → DP+polish 1.1936 = **−6.9 %**.

The DP lane in Option B takes ~12 s for the DP basin (CPU build on
cloud, CUDA build hit nvcc 11.0 vs compute_86 incompatibility) + ~1100 s
for polish (matched to E25/E41 lanes) + ~800 s for cascade saddle on
the picked plateau. Total per-bench wall well within the 60-min cap.

### 1.4 Why this works (basin-level and saddle-level both matter)

The proxy decomposition is **6 % WL / 20 % density / 74 % congestion**
(E8 LP-HPWL diagnostic). On congestion-dominated benches:

- **Basin-level.** Different inits (SDF, DPO, DP) reach different basins
  after polish. No init dominates universally; per-bench best-of is the
  robust strategy (E48 hybrid finding, generalized to 3 lanes in E91).
- **Saddle-level.** Within a basin, the polished plateau is a local-move
  fixed point. Hessian saddle escape via smooth-proxy eigvec + ε perturb
  pushes through the plateau to a lower minimum (E74 finding;
  ariane133 broke from 0.6861 to 0.6641 = **−3.21 %**, ending the
  E42/E43/E44/E54/E62 NG45 failure chain).
- **Cascade.** Iterating saddle escape until no improvement compounds
  per-iteration lifts. Uncapped E84 hit IBM 1.0612 (8/17 walls over
  55-min cap). The wall-safe descendants trade ~0.005 aggregate for
  guaranteed 60-min/bench compliance.

### 1.5 Why this transfers to NG45 + hidden designs

- Plateau-detection thresholds are **per-benchmark, per-run** — adapt
  to the actual descent trajectory rather than IBM-tuned priors.
- Saddle escape operates on the smooth-proxy Hessian, which is built
  from the placement state — no implicit IBM dependence.
- DP-lane uses **auto-adaptive config** (E91 rule-compliance fix):
  `target_density = clip(macro_density × 1.5, 0.40, 0.85)` per
  observable bench geometry; no benchmark-name dispatch. Verified on
  ariane133 NG45 = 0.66167 (auto-config) vs cascade-uncapped 0.6641.

### 1.6 Why this does NOT transfer to Tier-2 ORFS uniformly

Tier-2 ORFS uses real PnR routed metrics (WNS/TNS/Area), not proxy.
Our placer optimizes proxy; ORFS's `rtl_macro_placer` is timing-aware.
Per-design verification (2026-05-15/16, [`handoffs/2026-05-16_tier2_orfs_findings.md`](handoffs/2026-05-16_tier2_orfs_findings.md)):

| Design | Strategy | Why |
|---|---|---|
| ariane133 | Ship **without** `MACRO_PLACEMENT_TCL` | ORFS auto-place beats every cascade variant by 1.2 ns of slack. |
| ariane136 | Ship **with** cascade `MACRO_PLACEMENT_TCL` | Cascade +0.4935 ns vs auto +0.0457 ns. |
| mempool_tile | Untested with the macro-tcl fix | Default: ship cascade pending re-test. |
| nvdla | Auto-place fallback | Cascade triggered PDN failure on pathological E74-class placement. |

This is a *limitation* of the proxy-optimization approach, not a bug.

---

## 2. Lineage (prior approaches — useful for writeup context)

Each generation replaced its predecessor by a **structural change** to
the algorithm, not parameter tuning.

| Era | Method | Best `--all` | Replaced because |
|---|---|---:|---|
| RePlAce baseline | — | 1.4578 | — |
| Polyhedra navigation | LP within disjunctive feasible region | 1.4867 | Congestion barrier structural (E8) |
| DPO | best_of_v2 | 1.3834 | Smooth-proxy gradient cannot cross discrete topology barriers (basin lock byte-identical across 4 seeds) |
| CD-only | fixed 600 s/bench | 1.1193 | Fixed budget undershot hard benches |
| CD-adaptive (E9) | per-bench plateau | 1.1055 | Plateau-bound; couldn't escape per-axis fixed point |
| CD + LNS (E12) | + grid-bin (col × row) escape | 1.0990 | Different move type cleared E9 plateau; ADR-007 |
| E25 (SDF+CD+LNS+SA-v2) | + SA-v2 polish on breakpoints | 1.0954 | Component of E48 hybrid |
| E41 (DPO+CD+LNS+SA-v2+K-joint) | DPO basin variant | 1.0848 | Component of E48 hybrid |
| E48 hybrid | per-bench best-of-{E25, E41} | 1.08151 | ADR-011; superseded by E74 saddle |
| E74 Hessian saddle | smooth-proxy eigvec + ε perturb on E48 plateau | 1.0666 | ADR-012; subsumed by cascade |
| E84 cascade (uncapped) | iterate E74 saddle until no improvement | 1.0612 | 8/17 walls > 55 min |
| **Option A (PATH A post-A1)** | wall-safe E84 + delta-cost CD | **1.07820** | submission floor 2026-05-16 |
| **Option B (cascade + DP-lane)** | + DREAMPlace as 3rd init | **1.06650** | strongest verified 2026-05-16 |

Full per-experiment archive at [`experiment_index.md`](experiment_index.md).

---

## 3. Innovation contributions (for paper)

1. **Empirical falsification of the "DP basin is structurally inferior"
   autopsy** (E91). Stock DP + full polish beats cascade-capped by
   6.9 % on hard-bench aggregate. The original autopsy's basin-only
   comparison was unfair; with matched polish budgets, DP basins
   polish to a different valley than SDF/DPO inits.

2. **Cascading Hessian saddle escape** (E74 → E84). Iterating smooth-
   proxy Lanczos + ε perturb on a CD plateau is a novel application of
   transition-state methods (Henkelman & Jónsson 2000 climbing-image
   NEB; dimer / gentlest-ascent) to combinatorial macro placement.
   The ariane133 breakthrough (−3.21 % vs E48) ended a long chain of
   NG45-blind heuristic failures (E42/E43/E44/E54/E62).

3. **Delta-cost incremental evaluator** (PATH A post-A1). 5.36× CD
   speedup via `delta_cost(macro_idx, new_xy)` returning the smooth-
   proxy delta *without* mutating state, plus `delta_cost_axis_batch`
   for vectorized K-candidate evaluation. Makes the wall-safe cascade
   fit under 60-min/bench.

4. **Multi-lane plateau pick + cascade**. Combining basin-quality
   variance (2-3 inits × per-bench best-of) with saddle-escape
   compounding produces consistent lifts across IBM and NG45 without
   per-benchmark tuning.

5. **Falsification record**: 100+ experiments documented in
   [`experiment_index.md`](experiment_index.md). Key signposts —
   the LP-HPWL disconnect (E8), the basin lock (DPO byte-identical
   across 4 seeds), the infeasibility wall (E65: linear path E25→E41
   has 0/9 legal intermediate placements), and the smooth-RUDY /
   canonical mismatch on hard benches (E92/E95/E98 falsified) — are
   the negative results that motivated the working approach.

---

## 4. See also

- [`problem.md`](problem.md) — formal mathematical formulation
- [`results.md`](results.md) — verified per-benchmark tables
- [`roadmap.md`](roadmap.md) — submission timeline
- [`experiment_index.md`](experiment_index.md) — every experiment, live + falsified
- [`decisions/`](decisions/) — Architecture Decision Records
- [`gotchas.md`](gotchas.md) — codebase footguns
- `writeup/evidence.md` §1, §1.1 — frozen-number archive
- `writeup/paper.md` — innovation prize draft
- `writeup/theory.md` — supplementary mathematical material
