# Roadmap

Last updated: 2026-04-29
Competition deadline: May 21, 2026 (~22 days)

---

## Current Position

| Entry | Avg Proxy (--all) | vs Leaderboard 1.1172 | vs RePlAce 1.4578 |
|-------|-------------------|------------------------|--------------------|
| **CDLNSGridBin (E12, CHAMPION)** | **1.0990** | **-1.63%** | **-24.6%** |
| CDLNSSA (E25, candidate, not promoted) | 1.0954 | -1.95% | -24.9% |
| CDAdaptive (E9, prior, superseded) | 1.1055 | -1.05% | -24.2% |
| CDOnly (prior-prior, superseded) | 1.1193 | +0.18% | -23.2% |
| DPO best-of-v2 (prior, superseded) | 1.3834 | +23.8% | -5.1% |
| RePlAce baseline | 1.4578 | +30.5% | --- |
| SA baseline | 2.1251 | +90.2% | +45.8% |

- **Beats the leaderboard "Incremental CD+LNS" (vmallela 1.1172) by -1.63%** with zero overlaps on all 17 IBM benchmarks.
- Total runtime 28 256 s (7.85 hr) — within the 17-hr competition envelope (17 × 1 hr per-bench cap), with reduced margin vs CDAdaptive's 4.85 hr.
- Champion configuration: full-proxy coordinate descent on incremental evaluator with per-benchmark plateau detection (CD ≤ 3 000 s) followed by grid-bin LNS escape phase (LNS ≤ 600 s). ADR-007.
- **Champion candidate (verified, not promoted):** CDLNSSA (E25) at **1.0954** (−0.33 % vs E12, −1.95 % vs leaderboard). Adds an SA-v2 polish phase on per-axis breakpoints (best-so-far tracking + T₀ = 5e-4). Code at `submissions/cd_lns_sa/placer.py`; ADR-008 *Proposed*; awaiting human decision.

---

## Champion lineage

| Era | Champion | Best (--all) | Date | Replaced because |
|---|---|---|---|---|
| Pre-DPO | RePlAce baseline | 1.4578 | n/a | Target to beat |
| Polyhedra | PolyhedraNavigation | 1.4921 | 2026-04-15 | Hit ceiling — congestion barrier structural |
| DPO v1 | DPO v1 | 1.4255 | 2026-04-23 | First to beat RePlAce; basin lock on hard benchmarks |
| DPO v2/v3/v2-steps | best_of_v2 | 1.3834 | 2026-04-26 | Within-DPO refinements cap at 1-2% |
| **CD-only** | CDOnly | 1.1193 | 2026-04-27 (am) | Fixed 600s budget left hard benchmarks mid-descent |
| **CD-adaptive** | CDAdaptive (E9, superseded) | 1.1055 | 2026-04-27 (pm) | Plateau-bound: every bench exited via plateau, none hit cap. Same move type — couldn't escape per-axis fixed point. |
| **CD + grid-bin LNS** | **CDLNSGridBin (E12)** | **1.0990** | **2026-04-28** | (current — ADR-007) |
| CD + LNS + SA-v2 (candidate) | CDLNSSA (E25) | 1.0954 | 2026-04-29 (verified) | Verified -0.33 % over E12; ADR-008 *Proposed*; not promoted. SA-v2 on per-axis breakpoints (best-so-far + T₀=5e-4) extracts wins LNS-alone misses on easier benches. Hardest benches tie with E12. |

Each transition was structural, not parameter tuning. See `docs/experiment_index.md` for the full catalog including failed attempts.

---

## Completed phases

### Phases 1-5 — Polyhedra navigation (Apr 1 - Apr 15)

Decomposed feasible region as union of convex polyhedra; navigated with SDF init + LP + surrogate-guided search. Hit a 1.49 ceiling. Phase 5's 22-experiment sweep proved the polyhedra surrogate was already well-calibrated — the ceiling was structural (the gradient of relaxed objective lost too much in legalization).

### Phase 6 — DPO (Apr 16 - Apr 26)

Differentiable proxy: gradient through smooth WL+density+congestion with annealed overlap penalty. Reached 1.3834 via best-of-v2. Multi-seed verification showed **basin lock** — `best_of_v2_seed{43,44,45,46}` all within 1% on --all, and ibm02/ibm12 byte-identical across seeds. DPO converges to a deterministic local minimum that gradient steps cannot escape.

Falsified DPO extensions:
- **E5 (batched seeds, 2026-04-26):** B=64 wall = 8.9× B=1; all seeds collapse to same basin under sigma=0.04·canvas perturbation.
- **E10 (congestion-only refinement, 2026-04-27):** Fast set -1.0%; --all -0.33% with ibm02 *worse* (+2.3%). Basin-locked benchmarks regressed.
- **E11 (diverse priors, 2026-04-27):** Fast -0.8%; --all FLAT (+0.04%). ibm02/ibm12 worse. Greedy/random priors land in deeper basins on hard benchmarks.

### Phase 7 — Coordinate descent on full proxy (Apr 26 - Apr 27)

The 2026-04-26 LP-HPWL diagnostic decomposed proxy as **6% WL, 20% density, 74% congestion**. Pure HPWL CD (Miftari-style weighted median) caps at ~5%; *full-proxy* CD captures all three components.

- **E1 (incremental evaluator, validated 2026-04-26):** 4657× speedup per move; bit-for-bit parity. Load-bearing for everything that follows.
- **E2 (CD on ibm10 single-bench, 2026-04-27 early):** 40-min budget hit 1.0632 vs DPO 1.254 (-15%). 10-min budget hit 1.1039 (~85% of value). Generalized to ibm02 (1.1534, broke DPO basin lock from 1.6888) and ibm18 (1.3929).
- **CDOnly --all (2026-04-27 03:50):** Productionized at 600s/bench. avg 1.1193, matched leaderboard within 0.18%.
- **E3 LNS (v1 full-canvas, v2 5×5 local-window, 2026-04-27):** Falsified. Single-macro destroy/reinsert produces flat ibm17 result regardless of speed. Cluster-level joint reinsertion would be needed for real gains.
- **E9 CDAdaptive --all (2026-04-27 17:22):** Per-benchmark plateau detection. avg **1.1055**, beat leaderboard 1.1172 by -1.05%.
- **E12 CDLNSGridBin --all (2026-04-28 15:57):** CD plateau + grid-bin LNS overlay (different move type — escapes CD's per-axis fixed point). avg **1.0990**, beat leaderboard by -1.63% and prior champion by -0.59%. Promoted 2026-04-28 (ADR-007).

---

## What's left

### Submission (required)

| Item | Owner | Status |
|---|---|---|
| Submission package (placer + reproducibility) | this branch | Ready — `submissions/cd_lns_gridbin/placer.py` is the entry |
| Writeup (innovation prize report) | parallel agent (`writeup/`) | In progress — must include process AND failures |
| NG45 hidden-test robustness check | E23 | **DONE 2026-04-28.** Zero overlaps on 4 designs; plateau detection transfers; avg 0.7037, max per-bench wall 1053 s vs 3600 s cap. |

### Optional polish (24 days available)

| Lever | Expected | Effort | Notes |
|---|---|---|---|
| ~~NG45 datapoint for E12~~ | DONE | E23 | **CLOSED 2026-04-28**: avg 0.7037 across 4 designs, zero overlaps, plateau detection transfers. ADR-007 transfer claim now empirically supported. |
| Drop cost-aware destroy ranking | flat (already verified on `--fast`) | 30 min | Saves ~10 % wall per LNS sample at no quality cost; defer until post-deadline |
| Cluster-level LNS (joint reinsertion of K>1 macros) — see E29 | Speculative | 3-4 days | Operationalized as exact K-macro MIQP (Tier 0); composes on top of E25/E12 |
| **GPU-batched CD via conflict-graph coloring (E4)** | 5-20× speedup | 3-4 days | Useful only if E12 wall risks the 1-hr-per-bench cap on NG45; currently 7.85 hr / 17 hr envelope on IBM |
| Multi-seed validation | Confirm robustness | 1 day | E12 is deterministic via SDF init; seed sensitivity TBD |

Decision rule: ship E12 (1.0990) as the safe baseline if no further work lands. E25 (1.0954) candidate awaits human promotion decision (ADR-008 *Proposed*); if accepted, becomes the safe baseline. The post-E25 frontier research below is gated on **E27 persistence-homology diagnostic** — if it shows a single basin, ship E25 + writeup; if multiple, pursue E28-E30 in priority order.

---

## Files of record

| File | Purpose |
|---|---|
| `submissions/cd_lns_gridbin/placer.py` | **CHAMPION submission** (E12, 1.0990) |
| `submissions/cd_adaptive/placer.py` | Prior champion (CDAdaptive E9, 1.1055; superseded 2026-04-28) |
| `submissions/cd_only/placer.py` | Prior-prior champion (CDOnly fixed-budget) |
| `writeup/archive/submissions/lns.py` | E3 LNS module — falsified; deleted from active tree in commit 44efd16, preserved at this archive path |
| `writeup/archive/submissions/cd_lns_placer.py` | E3 placer — falsified; deleted from active tree in commit 44efd16, preserved at this archive path |
| `submissions/dpo/best_of_v2_placer.py` | DPO prior champion (basin-locked) |
| `submissions/dpo/{e10,e11,batched,...}*.py` | Falsified DPO extensions; kept for writeup |
| `submissions/dpo/ablation_*.py` | DPO ablation studies |
| `macro_place/incremental_evaluator.py` | E1 — 4657× speedup, load-bearing |
| `scripts/cd_ibm10_diagnostic.py` | E2 — sparked the CD line |
| `scripts/lp_hpwl_lower_bound.py` | E8 — proxy-decomposition diagnostic |
| `docs/results.md` | Current champion per-bench tables |
| `docs/experiment_index.md` | **Rigorous catalog of every experiment** |
| `analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md` | E8 details |
| `writeup/closing_the_gap.md` | E9 win narrative + E3/E4 design sketches |
| `writeup/cd_ibm10_results.md` | E2 details |
| `writeup/historical_results.md` | DPO/Polyhedra/Overnight per-bench data |
| `results/experiment_log.jsonl` | Source of truth for all numbers |

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hidden NG45 commercial designs differ from IBM (different macro-count regime) | Medium | Plateau detection transfers without per-bench tuning. Verify with `--ng45` run before submission. |
| Float drift between incremental evaluator and `compute_proxy_cost` over thousands of moves | Low (~0.5-1.5% on ibm02 observed) | Final VALID is the canonical value. Internal proxy is biased low ~1% on dense benchmarks but this doesn't change ranking; plateau detection still works. |
| Overlap regression on edge cases | Low | Hard validation in placer raises RuntimeError on overlap. Tested on all 17 IBM with zero overlaps. |
| Submission package missing reproducibility info | Low | `submissions/cd_lns_gridbin/placer.py` self-contained; reuses `IncrementalProxyEvaluator`, SDF init, and the CDAdaptive runner from already-tracked files. |
| Champion wall close to 17-hr envelope (7.85 hr used) | Low | Still ~9 hr headroom on `--all`. Hidden NG45 designs of similar scale should fit within 1 hr/bench. Monitor `--ng45` per-bench wall before submission. |

---

## See also

- [approach.md](approach.md) — current CD-on-incremental-evaluator architecture
- [results.md](results.md) — current champion (CDLNSGridBin) per-benchmark tables
- [experiment_index.md](experiment_index.md) — full catalog including falsified hypotheses
- [lp_hpwl_diagnostic.md](lp_hpwl_diagnostic.md) — the diagnostic that unblocked CD
- `writeup/historical_results.md` — DPO/Polyhedra/Overnight per-bench tables
- `writeup/closing_the_gap.md` — narrative of the leaderboard-beating run
- `writeup/cd_ibm10_results.md` — single-benchmark CD breakthrough
- `writeup/theory.md` — theoretical foundations (polyhedra decomposition, proxy decomposition)

---

## Proposed experiments (from experiments_overnight.md retirement)

Carried forward from the retired `experiments_overnight.md` working doc.
Each has: hypothesis, kill gate (cross-benchmark), generalization check,
expected wall. Path references have been updated to the post-restructure
layout (`experiments/E<NN>_*/code/...`).

### Tier 0 — Frontier escape research (post-E25 floor; PRIORITY)

E25 (1.0954) ties E12 (1.0990) on the hardest benchmarks
(ibm11/13/14/15) — same geometric fixed point. Five mechanisms saturate
at this floor:

- CD per-axis breakpoint
- Grid-bin LNS (single-macro 2D, E12)
- SA-v2 (Metropolis on breakpoints, E25)
- Pair swap (E15 — flat)
- Spatial cluster destroy (E13 — marginal, doesn't generalize)

The signature is a *coupled* fixed point: single-macro and 2-macro
coordinated moves saturate at the same place. Escape requires either
higher-order multi-macro moves, a reparameterized objective, or a
different basin entirely. ~22 days left to the May 21, 2026 deadline —
enough for the diagnostic (E27) plus 2-3 of the high-EV mechanisms below.

> **HIGHEST-EV plan (added 2026-04-29).** E27-E38 below all attack the
> *same* basin from different angles in the *same* parameterization with
> the *same* proxy. **Tier 0a (E40-E43, see below)** reframes the problem
> itself — challenging the load-bearing assumption that placement is
> generic continuous optimization over $\mathbb{R}^{2N}$. After E27's
> verdict, pursue E40-E43 in preference to E28-E38; E42 (symmetry
> quotient) is a force multiplier for E28 (Hessian) regardless of verdict.

**Order of attack (updated 2026-04-29):**
1. **In flight (parallel agent, 2026-04-29):** E27 + E17 + E18 + E32 + E39 —
   basin diagnostic gates everything; the other four are tactical mechanisms
   running concurrently. Decision logic for the joint result set is
   documented in the next subsection.
2. **After E27 verdict — HIGHEST EV:** E40 (BP / tensor networks) + E41
   (multigrid) + E42 (symmetry quotient) + E44 (NEB action paths between
   E27 basins). All four target the *coupled fixed point* signature
   directly via different parameterizations. E42 starts regardless of
   E27; E44 specifically consumes E27's multi-basin endpoints. See
   §"Tier 0a — Structural reframings" below.
3. **Backup if Tier 0a stalls:** E28 (Hessian soft-mode) + E29 (K-MIQP)
   + E43 (diffusion sampling).
4. **Alternative-objective views if (1)–(3) is mixed:** E30 (OT/JKO),
   E31 (parametric), E33 (low-rank reparam).
5. **Last-resort research-grade:** E34-E38.

#### Decision tree on in-flight results (E27 / E17 / E18 / E32 / E39)

The parallel agent is running these concurrently. Each has a clean
yes/no outcome that triggers a downstream action.

**E27 (basin persistence) — the gate.**
- *One dominant H_0 feature below E25 level* → algorithmic escape capped
  near 1.09 on this proxy; ship E25, deprioritize Tier 0a algorithm-side.
  Pivot remaining weeks to the **Tier-2 grand prize** (proxy calibration
  via small ORFS-routed corpus) and writeup. Leaderboard locked at E25.
- *Multiple long-lived H_0 features below E25* → distinct basins exist;
  Tier 0a (E40-E41-E42) is well-posed. Launch E41 first (most tractable),
  E42 second (build-anyway multiplier for E28), E40 third (highest
  theoretical EV but biggest build cost), E43 only if E40/E41 stall.

**E17 (random uniform init + CD).**
- *Random beats SDF on any benchmark* → multi-init best-of-{SDF, random}
  is a free lift; integrate into next champion run. Also signals SDF basin
  is wrong for that benchmark — feed into E40 priors and diversifies E27's
  trajectory set.
- *Random uniformly worse* → SDF is the right basin entry; multi-init
  dead-end stays dead. Reinforces ADR-005.

**E18 (DPO best_of_v2 → E25 polish).**
- *Improves over E25* → DPO basin held topology info CD missed; pipeline
  becomes DPO → CD → LNS → SA. Strongly motivates E33 (low-rank reparam —
  DPO's basin is a low-rank fixed point CD couldn't reach).
- *Worse than E25* → DPO basin lock is real *and* dominated; CD-from-SDF
  is the right ordering. Closes the E11/E18 line.

**E32 (SAM-CD).**
- *IBM `--all` lift* → ship; verify NG45 lift ≥ IBM lift (the SAM
  hypothesis predicts better OOD generalization). If NG45 lift < IBM,
  treat as IBM-overfit and drop.
- *Flat or worse on `--fast`* → CD's fixed point isn't a sharp minimum at
  this proxy resolution; sharpness-aware angle closes.

**E39 (K-macro joint LNS, top-N enumeration).**
- *Improves over E25 on `--fast`* → operationalizes the K-MIQP idea
  cheaply; promote as E25 successor and skip E29 (full MIQP) build cost.
- *Flat* → either K too small (try K=20-30) or proxy-quadratic approx
  inadequate → E29 (full MIQP) is the principled escalation.

**Joint verdict matrix.**
- E27 multi-basin AND any of E17/E18/E32/E39 lifts → ship the lift first
  (cheap), then launch Tier 0a from the new floor.
- E27 single-basin AND all of E17/E18/E32/E39 flat → pivot fully to
  proxy-calibration / writeup; ship E25.
- Mixed (most likely) → run E41 + E42 in parallel (independent code
  paths) while shipping any tactical lift.

#### E27. Persistence-homology basin diagnostic (RUN FIRST)
**Hypothesis:** the E25 floor is either (a) the bottom of one big basin,
in which case 1.05x is unreachable and we ship E25 + writeup, or (b) one
of several deep basins, in which case bridging via E28-E30 is principled.
**Algorithm:** run 50-100 CD trajectories from diverse inits (uniform
random, SDF, SDF+jitter, greedy, DPO best-of-v2). Record proxy at every
sweep. Compute persistence diagram of the level-set filtration over the
union (Ripser/Gudhi). H_0 features = connected components of
`{p : f(p) ≤ c}` as c grows. Long-lived H_0 features below the E25 level
= distinct deep basins.
**Kill gate:** H_0 dominated by one long-lived feature (no separate basin
below E25 level) → kill the entire post-E25 line; ship E25 + writeup.
**Generalization check:** focus on ibm11/13/14/15 — these are where E12
and E25 tie. If the diagnostic shows multiple basins on those specifically,
that's the lift target.
**Wall:** ~1 day (50 trajectories × ~5 min each + persistence computation).
**Status:** proposed.

#### E28. Hessian soft-mode following (dimer / GAD on proxy landscape)
**Hypothesis:** at the E25 fixed point, the proxy gradient is ≈0 in all
per-axis directions, but the joint-Hessian H over the 2N-dim placement has
soft eigenmodes — coordinated multi-macro displacement directions with
near-zero curvature. CD provably cannot follow these (per-axis only). The
dimer method (Henkelman & Jónsson 1999) and gentlest-ascent dynamics
(Weinan E & Zhou 2011) follow soft modes uphill to find saddles, then
descend on the other side.
**Algorithm:** at the E25 fixed point, compute the smallest k Hessian
eigenvectors via Lanczos with finite-difference Hessian-vector products on
the incremental evaluator (~50 Lanczos iters × 2N gradient evaluations
each ≈ 1-2 minutes per benchmark). Take an "uphill" step of magnitude ε
along the softest mode. Resume CD. Repeat on each of the top-k softest
modes. Accept any descent that improves below E25 baseline.
**Kill gate:** zero modes produce a CD-resumed proxy below E25 baseline on
`--fast` after k=10 attempts → CD's fixed point is locally Hessian-stable
in all probed directions → kill.
**Generalization check:** validate on NG45 ariane133 if `--fast` passes.
**Wall:** ~3-4 days build (Lanczos + finite-difference H·v + CD-resume
integration); ~6 hr `--fast` validation; ~12 hr `--all` if generalizable.
**Status:** proposed. Highest theoretical EV — directly attacks the
coupled-fixed-point signature.

#### E29. K-macro joint-MIQP LNS (k-opt for placement)
**Hypothesis:** grid-bin LNS reinserts one macro at a time, and E25 is
preserved under all single-macro reinsertions. Jointly reinserting K=10-20
coupled macros via a small MIQP (continuous (x, y) per destroyed macro,
disjunctive non-overlap as integer constraints, proxy-quadratic
approximation as cost) solves the K-dim sub-problem to global optimality
given the others fixed. Different move type than grid-bin, outside both
CD's and E12's reachable set.
**Algorithm:** at the E25 fixed point, pick K=10-20 coupled macros (top-K
by net adjacency, or all macros sharing a hot congestion cell). Destroy.
Build the K-macro MIQP; solve via Gurobi/CPLEX with ~30 s budget. Splice
back. Iterate over different K-subsets until improvement or budget.
**Kill gate:** zero accepted joint-MIQP placements improve over E25 on
`--fast` after 8 K-subsets per benchmark → kill.
**Generalization check:** NG45 ariane133.
**Wall:** ~3-4 days build (MIQP formulation + proxy-quadratic approx +
incremental evaluator integration).
**Status:** proposed. Operationalizes the "Cluster-level LNS" line in the
Optional polish table as exact MIQP, not heuristic. Prior art: k-opt is
standard in TSP; novelty here is the disjunctive non-overlap MIQP at
small K.

#### E30. Wasserstein-JKO gradient flow (OT-regularized step)
**Hypothesis:** density (20%) is literally an OT problem; congestion
(74%) correlates with mass distribution. Replace gradient steps with the
Jordan-Kinderlehrer-Otto scheme:
`p_{t+1} = argmin_p [ f(p) + (1/2τ) W_2²(p, p_t) ]`. The W_2 regularizer
prefers coordinated multi-macro mass transport over single-macro greedy
moves.
**Algorithm:** alternating CD + JKO. After CD plateau, run K JKO steps
where each step solves the inner argmin via Sinkhorn on the macro-level
transport matrix (~tens of ms per Sinkhorn for N=500). The Sinkhorn output
gives a transport plan; project to a coordinated multi-macro displacement;
apply with overlap repair.
**Kill gate:** zero JKO steps improve over E25 on `--fast` after K=20 → kill.
**Generalization check:** NG45 ariane133.
**Wall:** ~3-4 days build (Sinkhorn + projection + repair).
**Status:** proposed. Lowest mechanistic distance from existing pipeline —
backup if E28/E29 stall.

#### E31. Parametric proxy continuation across pairs
**Hypothesis:** for a soft macro pair (low local Hessian curvature),
continuously rotating one macro around the other while running CD on
everyone else traces a parametric proxy curve. A non-monotone curve signals
a saddle was crossed and a different basin opens. Constraint-space
continuation (vs. the position-space continuation that failed in the
polyhedra-era experiments).
**Algorithm:** identify the K softest pairs by Hessian-block-norm (reuse
E28 infra). Parameterize a continuous rotation θ ∈ [0, 2π] per pair; sweep
proxy at θ ∈ {0, π/8, ..., 2π} with CD on all others at each θ. Accept
any θ whose post-CD proxy beats E25 baseline.
**Kill gate:** zero pairs reveal non-monotone proxy curves → kill.
**Generalization check:** NG45 ariane133.
**Wall:** ~2 days (depends on E28 Hessian infra).
**Status:** proposed.

#### E32. Sharpness-aware proxy (SAM-style)
**Hypothesis:** CD's fixed point may be a sharp local minimum that's an
artifact of the proxy's exact form (smoothing γ, top-5%/10% thresholds).
Optimize `max_{||δ||<ρ} f(p+δ)` instead — penalizes sharp basins, prefers
flat. The flat-minimum optimum should also generalize better to NG45
(slightly different proxy regime).
**Algorithm:** at each CD breakpoint candidate, evaluate proxy at K random
perturbations within radius ρ; use max as effective cost. Foret et al.
(ICLR 2021) SAM in spirit, adapted for breakpoint enumeration.
**Kill gate:** SAM-CD `--fast` proxy ≥ E25 baseline → kill.
**Generalization check:** SAM-CD's NG45 lift relative to E12 should be ≥
its IBM lift (the SAM hypothesis predicts better OOD generalization).
**Wall:** ~2 days build.
**Status:** proposed.

#### E33. Reparameterized DPO via low-rank latent
**Hypothesis:** express `p_i = p_i^(0) + W·φ_i` with low-rank
W ∈ ℝ^{2N×d} and learnable φ_i ∈ ℝ^d, d ≪ N. Optimize (W, φ) via DPO.
Fixed points in (W, φ)-space differ from fixed points in p-space (Mishkin,
Kelly, Erdogdu 2018).
**Algorithm:** initialize W from the top-d Hessian eigenvectors at the E25
fixed point (reuse E28 infra). DPO on (W, φ). Decode to p; legalize;
evaluate.
**Kill gate:** decoded proxy ≥ E25 on `--fast` → kill.
**Generalization check:** NG45 ariane133.
**Wall:** ~2 days build.
**Status:** proposed.

#### E34. RL policy with fast evaluator as simulator
4657× incremental evaluator → millions of trial moves per training run.
PPO on a GNN policy over the macro/net hypergraph. Action = (macro,
target); reward = -proxy delta. Mirhoseini-Goldie 2021 (*Nature*) with
our infrastructure — the previously-fatal cost was eval cost; we've
solved that.
**Wall:** ≥1 week build, ≥1 week training.
**Status:** proposed; deepest investment — pursue only if a 2+ week window
opens.

#### E35. Variational annealing with GMM posterior
Parametrize q(p) as a K-component Gaussian mixture; minimize
`E_q[f] − T·H[q]` with annealed T. Components share gradient signal during
annealing — coordinated multi-basin exploration vs. parallel restart.
**Wall:** ~3 days build.
**Status:** proposed.

#### E36. Tropical HPWL gradient component
HPWL is a tropical polynomial (Allamigeon-Gaubert-Joswig); Talbut-Monod
2024 give tropical gradient descent with classical-rate convergence on
tropically-convex problems. Hybrid: tropical-GD on WL, CD on
density+congestion, alternate. Bounded upside (WL is 6 %) but qualitatively
different breakpoint enumeration could change basin selection.
**Wall:** ~2 days build.
**Status:** proposed.

#### E37. Soft-mode-aware GA crossover
Compute soft-Hessian modes per parent (depends on E28). Crossover swaps
macros whose contributions lie along distinct modes between parents.
Structured GA respecting the manifold of low-curvature directions.
**Wall:** ~3 days build (depends on E28).
**Status:** proposed.

#### E38. Dead-end elimination on grid-bin candidates
DEE from protein side-chain packing (Donald Lab / OSPREY). For each macro,
prove certain grid-bin cells are dominated (no completion of others gives
them lower proxy than some alternative cell). Massive pruning of E12's
candidate set → exhaustive search over survivors. The pairwise interaction
bound is the bottleneck; our incremental evaluator gives those bounds
cheaply.
**Wall:** ~3 days build.
**Status:** proposed.

### Tier 0a — Structural reframings (HIGHEST EV — added 2026-04-29)

E27-E38 above all keep the *same* parameterization (positions in
$\mathbb{R}^{2N}$), the *same* proxy ($f = \text{WL} + 0.5D + 0.5C$), and
the *same* search paradigm (local moves on the proxy landscape). They
explore how to escape the coupled fixed point *within* that frame.

E40-E44 each break one of those three constants. Field analogues are
listed because the breakthrough principle in each case comes from
outside the placement community, where similarly-structured cost
functions have been solved by these methods for decades.

| ID | Breaks | Field analogue |
|---|---|---|
| E40 | Search paradigm (optimization → inference) | DMRG, junction-tree BP, statistical-physics ground states |
| E41 | Parameterization (flat → multi-scale) | V-cycle multigrid, real-space RG, multilevel METIS |
| E42 | Parameterization (full → quotient by automorphism group) | AlphaFold E(3)-equivariance, lattice gauge fixing, crystallographic fundamental domains |
| E43 | Search paradigm (optimize one trajectory → sample many) | RFdiffusion, score-based combopt (Sun-Yang 2023, van Krieken 2024) |
| E44 | Search paradigm (point optimization → least-action path) | Onsager-Machlup, NEB / string method, transition-state theory, instantons |

E40, E43, and E44 are gated on E27's "multiple basins" verdict.
**E41** can launch immediately (its premise is multi-scale structure,
not multi-basin). **E42** is a force multiplier for E28 regardless and
can be built on either E27 verdict. **E44** specifically *consumes*
E27's basin endpoints (it needs path endpoints) — pair them tightly.
**The unifying thread:** stop treating placement as generic continuous
optimization. E40 treats it as inference; E41 as multi-scale; E42 as a
quotient by symmetry; E43 as sampling from the Boltzmann distribution;
E44 as a variational path-finding problem with a least-action principle.
Each maps to decades of cross-field validation, and none is
load-bearing in the placement literature.

#### E40. Belief-propagation / tensor-network inference on the netlist hypergraph
**Hypothesis:** the proxy factorizes — HPWL is a sum over nets, density
and congestion are sums over cells, each touching a small subset of
macros. The macro-interaction graph (edge iff shared net) likely has
small treewidth, or small-treewidth components linked by few cut edges.
On such graphs, junction-tree / DMRG / loopy-BP message passing is the
provably-correct algorithm; local search is provably suboptimal. E12/E25
use the netlist as a *constraint* — never as the *computational
scaffold*. This is the largest unexploited lever in the project.
**Algorithm:** (1) build the macro-interaction graph from the netlist;
estimate treewidth (PACE-class heuristics, ~seconds at N=500). (2)
tree-decompose into bags. (3) discretize each macro's position to grid-
cell centers (~50-500 candidates per macro). (4) junction-tree message
passing: each message is a distribution over bag configurations carrying
proxy lower bounds. For high-treewidth bags, fall back to loopy BP within
bag and exact across bags. (5) decode → legalize → evaluate.
**Kill gate:** zero benchmarks below E25 floor on `--fast` after BP
convergence → discretization too coarse, or high-treewidth bags defeat
the approximation → kill.
**Generalization check:** NG45 ariane133 — the factor structure is
benchmark-agnostic, so a working E40 should transfer immediately.
**Wall:** ~5-7 days build (junction-tree library integration —
NetworkX or libtw — message discretization, decoding/legalization). No
learned components, but substantial engineering.
**Status:** proposed. **Highest theoretical EV among E40-E43** —
directly attacks the "treat netlist as constraint, not computational
scaffold" assumption inherited from RePlAce/DPO.
**Closest existing experiment:** E38 (DEE) — DEE *prunes* the search
space; E40 *factorizes inference* across it. Different paradigm.

#### E41. Hierarchical / multigrid placement (V-cycle)
**Hypothesis:** every approach in the lineage is *flat* over 200-537
macros. Single- and two-macro moves can't move *aggregate mass* — that's
why the coupled fixed point exists. A multigrid V-cycle rearranges
aggregate mass cheaply at the coarse scale and refines at the fine
scale. Multigrid is the standard escape from local-operator saturation
in PDE solvers, materials-science RG, and multilevel partitioning
(METIS / hMETIS) — conspicuously absent from placement.
**Algorithm:** (1) hMETIS clustering of the netlist into ~20-40
super-macros; super-macro size = sum of constituent areas; super-macro
nets = quotient netlist. (2) place super-macros via E12 CD+LNS at the
coarse scale (small problem, ~minutes). (3) uncoarsen one level: for
each super-macro, place its constituents inside the super-macro's
bounding region with surrounding fixed; refine with CD. (4) repeat to
leaf scale. (5) E12 CD+LNS post-smoother on full placement.
**Kill gate:** zero benchmarks below E25 floor on `--fast` after V-cycle
+ post-smoother → multi-scale operator doesn't escape the coupled fixed
point on this proxy → kill.
**Generalization check:** NG45 ariane133, mempool_tile, NVDLA — these
are *more* hierarchical than IBM, predicted larger lift. If NG45 lift
< IBM lift, the multi-scale hypothesis is wrong (commercial designs
should benefit *more*).
**Wall:** ~3-4 days build (hMETIS bindings, quotient-netlist
construction, V-cycle orchestration on top of E12 primitives).
**Status:** proposed. Most tractable Tier 0a entry; principle has the
deepest cross-field track record.

#### E42. Symmetry-quotient placement on the netlist automorphism group
**Hypothesis:** identical-shape macros with isomorphic net signatures
form orbits under a permutation group $G$; the proxy is $G$-invariant.
At the E25 fixed point, the joint Hessian (E28) has *zero eigenvalues*
along orbit-tangent directions — proxy is constant on orbits, so those
soft modes are *trivially* flat. E28's Lanczos wastes iterations on
them. Detecting $G$ a priori (a) tells E28 which modes to follow
(orbit-orthogonal), (b) prevents redundant orbit-element exploration in
LNS/SA, (c) supplies a discrete escape move (orbit-element swap)
outside CD/LNS/SA reach.
**Algorithm:** (1) build the macro/net bipartite graph; run Bliss /
nauty for the automorphism group (seconds at N≈500). (2) compute orbits.
(3) plug into E28: project Hessian Lanczos onto the orbit-orthogonal
complement before soft-mode following. (4) add orbit-swap move type to
E25's pipeline: simultaneously permute positions of an orbit subset by
an automorphism; legalize; check proxy.
**Kill gate:** (a) all 17 IBM benchmarks have $G = \{e\}$ (no
non-trivial automorphism) → no symmetry to exploit → kill cleanly.
(b) E28+orbit-projection finds zero descents below E25 *and* orbit-swap
is flat on `--fast` → kill.
**Generalization check:** NG45 mempool_tile and NVDLA (repeated-tile
structure should give substantially larger $G$ than IBM) → predicted
larger lift on those.
**Wall:** ~2-3 days build (Bliss bindings, orbit-projection in Lanczos,
orbit-swap move in cd_core). **Build-anyway candidate** because of the
E28 multiplier — even if the lift is zero, E28 becomes substantially
more efficient.
**Status:** proposed. **Most novel for the placement community.**
AlphaFold-style equivariance applied to placement; analogue of
fundamental-domain methods in materials science and gauge fixing in
lattice QCD.

#### E43. Score-based diffusion sampling of the proxy Boltzmann distribution
**Hypothesis:** if E27 reveals multi-modal structure, the placement
Boltzmann distribution $p(p) \propto \exp(-\beta f(p))$ is genuinely
multi-modal. Local search collapses to one mode; sampling captures all.
A score network $s_\theta(p) \approx -\nabla f(p)$ + annealed Langevin
samples placements without policy training (distinct from E34 RL — no
episodes, no reward shaping, no exploration policy). The 4657×
incremental evaluator makes long Langevin chains affordable.
**Algorithm:** (1) GNN over the netlist hypergraph; conditioning =
(netlist, current placement); output = $s_\theta(p)$. (2) train via
denoising score matching on a corpus of (E12 / E25 / random-perturbed)
placements: loss $= \|s_\theta(p+\sigma\epsilon) + \epsilon/\sigma\|^2$.
(3) annealed Langevin sampling at decreasing $\beta$; at each step
combine $s_\theta$ with exact $\nabla f$ from the incremental evaluator
(control variate). (4) post-process best sample with E12 CD+LNS polish.
**Kill gate:** annealed-Langevin samples don't beat E25 on `--fast` after
K=20 chains → either score network undertrained or proxy not multi-modal
at this resolution → kill.
**Generalization check:** NG45 ariane133.
**Wall:** ~5-7 days build (GNN architecture, score-matching training
loop, annealed Langevin, polish integration). Most ML-heavy of E40-E44;
pursue only if E27 verdict is multi-basin AND E40/E41 stall.
**Status:** proposed. Most modern; closest to AlphaFold-2 / RFdiffusion
in spirit.

#### E44. Onsager-Machlup least-action paths between E27 basins (NEB / string method)
**Hypothesis:** for the artificial overdamped SDE
$dp = -\nabla f\,dt + \sqrt{2T}\,dW$, the most-likely path between two
basins satisfies the Euler-Lagrange equations of the action functional
$S[p] = \frac{1}{4T}\int_0^\tau \|\dot p + \nabla f(p)\|^2\,dt$. The
minimum-action path crosses a saddle whose unstable Hessian mode
points toward potentially-lower basins on the far side. This is a
*real* variational principle for an artificial dynamics — the
chemistry / materials-science / climate-science workhorse for
transition-state finding, with 30+ years of numerical refinement.
**Algorithm:** (1) take each basin pair $(p_A, p_B)$ E27 returns;
(2) initialize a 30-50 image string $\gamma_0$ linearly interpolating
$p_A \to p_B$; (3) at each interior image, evaluate $-\nabla f$ via the
incremental evaluator and project out the component parallel to the
string tangent; (4) reparameterize to keep image spacing uniform;
(5) iterate to convergence — the highest-energy converged image is the
saddle $p_S$ between A and B; (6) compute the smallest unstable Hessian
eigenvector $v$ at $p_S$ via Lanczos (reuse E28 infra); (7) descend
$\pm \epsilon v$ and run E12 CD+LNS from each side; (8) accept any new
basin below E25 floor.
**Kill gate:** zero new basins below E25 floor on `--fast` after K=10
NEB runs across E27's basin pairs → either the discovered basins are
mutually unreachable in budget (action $S \gg T$) or the new basins on
the far side aren't lower → kill.
**Generalization check:** NG45 ariane133 if E27 multi-basin reproduces
on commercial designs; otherwise the IBM benchmarks where E27 found
multiple basins.
**Wall:** ~2-3 days build (string-image data structure, perpendicular-
gradient projection, reparameterization, integration with cd_core
legalization to keep all images legal). Reuses E27 basin endpoints,
the incremental evaluator, and (optionally) E28's Lanczos for the
saddle-mode follow-up.
**Status:** proposed. **Most physically-principled escape mechanism**
in the catalog — the variational characterization of basin transitions
in stochastic dynamics. Composes with E28: NEB finds saddles *globally*
between known basins; E28 finds saddles *locally* from a single point
via dimer / gentlest-ascent. Complementary, not redundant. Also gives
quantitative Arrhenius escape rates ($\sim e^{-S_\text{action}/T}$) so
you know which inter-basin transitions are budget-reachable.

### Tier 1 — overnight runnable, generalization-safe

#### E16. Tighter plateau threshold + extended cap (cheapest score win)
**Hypothesis:** every hard benchmark exited CDAdaptive with last-3-sweep
deltas 0.001-0.005 — below the 0.005 threshold but each ≈0.001 of proxy.
Tightening to threshold=0.001, cap=2hr captures those tail sweeps.
**Algorithm change:** ONE global hyperparameter swap, applied to ALL
benchmarks. No per-bench tuning.
**Kill gate:** if the new threshold doesn't drop avg --all proxy by ≥0.005,
kill. (NB: applies to all 17 benches, not "hard ones only".)
**Generalization check:** run on the 4 public NG45 designs (ariane133/136,
mempool_tile, nvdla) — proxy on those should also drop or hold steady, not
regress.
**Wall:** ~12 hr (--all run with extended cap).
**Status:** completed; landed at 1.1025 (−0.0030, formally marginal vs the
0.005 kill gate). See `experiments/E16_tight_threshold/code/cd_adaptive_e16.py`
and `writeup/evidence.md` §7.6.

#### E12. Full-canvas grid-bin LNS (the actual LNS recipe we never tested)
**Hypothesis:** prior LNS failures used subset-CD reinsertion, which finds
the *same* per-axis fixed point. The actual recipe (E3 in roadmap) is
grid-bin search: for each destroyed macro, evaluate proxy at every (grid_col
× grid_row) center, pick min. This is a different move type CD can't reach.
**Algorithm:** Adaptive: destroy size = 5% of movable hard macros, capped at
30. Choose destroy via cost-aware ranking. For each destroyed macro,
enumerate (col×row) candidates ~50-2500 per benchmark. Pick globally best
position via incremental evaluator. Iterate until N samples or wall budget.
**Kill gate:** if 0/8 grid-bin LNS samples improve baseline by ≥0.5% on
**the median benchmark** (NOT the worst, NOT the easiest — pick the median
to avoid overfit), kill.
**Generalization check:** if it passes on median, validate on `--fast` and
one NG45 design before --all.
**Wall:** ~2 hr build + 4 hr validation = 6 hr.
**Status:** **GRADUATED 2026-04-28.** `--all` final 1.0990 — promoted to
champion (ADR-007), supersedes E9 CDAdaptive. Code now at
`submissions/cd_lns_gridbin/placer.py`; experiment record at
`experiments/E12_grid_bin_lns/manifest.md`.
**Note on prior art:** vmallela's #1 method is named "Incremental CD+LNS" —
this might be exactly their recipe. TAISPlAce did "ALNS + Thompson Sampling"
and only hit 1.4321, suggesting LNS execution detail matters a lot.

#### E14. SA polish on CDAdaptive output
**Hypothesis:** SA can accept worse moves probabilistically — explicit
tunneling through CD's fixed point. Different mechanism than LNS (which is
structural).
**Algorithm:** After CDAdaptive converges, run SA: single-macro moves drawn
from per-axis breakpoint set (same move space as CD), Metropolis acceptance
with global temperature schedule (T₀ → T_f over N moves). All
hyperparameters global.
**Kill gate:** SA polish on `--fast` shows no improvement over CDAdaptive
baseline → kill.
**Generalization check:** validate on NG45 ariane133.
**Wall:** ~3 hr (build + --fast eval).
**Why higher than E15:** SA is a known-strong method; this is the version
that didn't lock the ibm02 basin in our prior DPO experiments.
**Status:** **FALSIFIED 2026-04-29.** v1 implementation hit the kill gate
hard (avg `--fast` 0.9962 vs baseline 0.9426 = +5.7 % WORSE on every
benchmark). Two compounding bugs: no best-so-far tracking + T₀ = 0.01 too
high (50/50 random walk, not tunneling). Net SA Δ on ibm13 was *positive*
(+0.05578). See `experiments/E14_sa_polish/manifest.md` and
`writeup/evidence.md` §9.B. **E24 (sa_polish_v2)** is the principled retest
with both fixes; manifest at `experiments/E24_sa_polish_v2/manifest.md`.

#### E15. 2-macro pair swap moves
**Hypothesis:** CD plateaus when each macro is at its single-macro fixed
point. Swapping two strongly-coupled macros (sharing nets) is a coordinated
move outside CD's reachable set.
**Algorithm:** Adaptive: candidate pairs = those sharing ≥2 nets (or top-K
by net adjacency, K = 5% of pairs, capped). For each candidate pair, query
proxy with positions swapped via incremental evaluator; accept if improving.
**Kill gate:** zero accepted swaps on `--fast` → CD's fixed point is stable
to swaps too → kill.
**Generalization check:** test on NG45 ariane133.
**Wall:** ~3 hr.
**Status:** v1 was buggy (`min_shared_nets=2` filter found zero candidates —
most macro pairs share exactly 1 net). v2 ran on `--fast` finding 21–53
real swaps per benchmark and landing at 0.9414 (essentially flat vs E16
baseline 0.9425). See `experiments/E15_pair_swap/code/cd_pair_swap.py`.
**Note on prior art:** the leaderboard has "MacroBio (Two-Opt Swap)" as
pending. Either result is data.

#### E23. NG45 sanity test (do this FIRST)
**Hypothesis:** before any new optimization, validate CDAdaptive runs
end-to-end on NG45 designs and produces ORFS-flowable placements. This is
**defensive** — Cezar got DQ-risked from a self-reported / verified
discrepancy. We need to confirm no analogous problem in our pipeline.
**Algorithm:** No change. Just run.
**Kill gate:** N/A — diagnostic only. Failure → fix bugs.
**Wall:** ~30 min.
**Output:** confirmation + any latent bugs surfaced.
**Status:** **VALIDATED 2026-04-28.** Run on E12 champion across 4 NG45
designs (ariane133/136, mempool_tile, nvdla); avg 0.7037, zero overlaps,
max per-bench wall 1053 s vs 3600 s cap. Plateau detection transfers
unchanged. ADR-007's missing NG45 datapoint filled. See
`experiments/E23_ng45_sanity/manifest.md` and `writeup/evidence.md` §9.A.

### Tier 2 — medium-EV, build-if-Tier-1-mixed

#### E13. Congestion-region spatial-cluster LNS
Targeted destroy of macros in high-congestion regions. Adaptive criterion:
cells with congestion > median + 1σ. Different from E12 (which destroys
cost-aware random); E13 destroys spatial neighbors, releasing joint
constraints.
**Wall:** ~3 hr after E12 framework exists.
**Status:** **MARGINAL 2026-04-29.** avg `--fast` 0.9384 (lift 0.45 % over
E16 baseline 0.9426 — below the 0.5 % gen-check threshold). Mechanism works
on ibm01 (single LNS sample lifted 0.76 %) but doesn't generalize to
ibm04/ibm09/ibm13. +0.13 % worse than E12 random-destroy ablation. Did not
queue `--all`. Possible follow-up: destroy nearest-neighbors *within* hot
cells, not just residents (current heuristic). See
`experiments/E13_congestion_lns/manifest.md` and
`writeup/evidence.md` §9.C.

#### E17. Random-uniform init + CD (true diversification)
Replace SDF with random uniform legal placement. Test if SDF basin is
sometimes the wrong basin.
**Generalization check:** if random init is uniformly worse → expected (SDF
is a smart prior). If random init beats SDF on some benchmarks, that's
signal of basin diversity → run best-of-N from random + SDF mixed inits.
**Wall:** ~5 hr.

#### E18. DPO best_of_v2 → CDAdaptive polish (E6 redux)
Use DPO best_of_v2 (1.3134 avg) as init for CDAdaptive instead of SDF. Per
E11: alternative inits sometimes find different basins, but at risk of
worse outcomes on some benches.
**Wall:** ~6 hr full --all if --fast passes.

### Tier 3 — speculative / parking lot

#### E19. RUDY → Steiner-tree congestion
Different proxy modeling. **Risk:** if Steiner is more accurate but the
contest's TILOS evaluator uses RUDY (it does), our proxy diverges from the
scoring proxy. We optimize for a thing the contest doesn't measure. **Skip
unless Tier 2 needs it.**

#### E20. ILP polish on local windows
~3 days to build, uncertain payoff. Only if Tier 1 lacks improvement and
we have time.

#### E21. GPU-batched breakpoint search
Per Proof A, CD is plateau-bound, not budget-bound on IBM. Speed-up
doesn't directly help score on IBM. **Re-live for NG45 if hidden designs
are larger** and budget actually binds.

#### E22. Conflict-graph colored CD
Same caveat as E21. Only useful at scale.

---

## Deprioritized hypotheses

Carried forward from the retired `experiments_overnight.md`.

- **Per-benchmark hyperparameter tuning** — explicitly forbidden by
  competition rules
- **More multi-init CD via SDF jitter** — falsified, 0/8 improved,
  structurally explained (SDF is contractive). See
  `analysis/multi_init_probe/README.md` and ADR-005.
- **More random-destroy LNS via subset-CD** — falsified across 24 samples,
  two benchmarks. See `analysis/lns_escape_probe/README.md`.
- **Within-DPO refinement** (E10/E11 variants) — basin-locked per memory; CD
  already breaks that lock
- **GPU best-of-N seeds** (E5) — already falsified
- **Polyhedra navigation** — superseded
- **Reinforcement learning / GNN priors** — too long to build given May 21
  deadline; could be revisited if we want to chase Innovation Award narrative

---

## Open follow-ups

Carried forward from the retired `findings.md` and `experiments_overnight.md`.

- ~~**NG45 sanity test**~~ — **CLOSED 2026-04-28 by E23.** Public NG45 designs
  are present locally at `benchmarks/processed/public/{ariane133,ariane136,
  mempool_tile,nvdla}_ng45.pt` (the prior "we don't have them locally" note
  was stale). E12 ran clean on all 4: avg 0.7037, zero overlaps, max
  per-bench wall 1053 s vs 3600 s cap.
- **ORFS local integration** — multi-day project; the actual path to the
  $20K Tier-2 grand prize. Should be its own multi-day track.
- **Verify zero overlaps preserved through legalization on E12 final
  placement** — should be enforced by `compute_overlap_metrics` raise in
  placer; verify against the full --all result.
- ~~**Tier 2 robustness check on E12**~~ — **CLOSED 2026-04-28 by E23.** E12
  generalizes to NG45 commercial designs without modification.
- **Multi-seed variance report** — produce a writeup figure showing
  CDAdaptive's variance across seeds on each benchmark (innovation-award
  material).
- **Adversarial / hidden-design stress test** — synthesize random
  "NG45-like" benchmarks (random sizes, densities, topologies) and verify
  CDAdaptive doesn't catastrophically fail on edge cases.
- **Tier 2 OpenROAD flow integration** — running ORFS locally to iterate on
  WNS/TNS/Area instead of proxy.
