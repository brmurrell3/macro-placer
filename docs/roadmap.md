# Roadmap

Last updated: 2026-04-29
Competition deadline: May 21, 2026 (~22 days)

## TL;DR

- **Champion:** E12 CDLNSGridBin (1.0990, ADR-007). Submission-ready.
- **Candidate awaiting promotion:** E25 CDLNSSA (1.0954, ADR-008 *Proposed*).
- **In flight (parallel agent, 2026-04-29):** E17, E18, E27, E32, E39, E40.
- **Gate:** E27 basin-persistence diagnostic. Single-basin verdict → ship E25 + writeup. Multi-basin verdict → pursue Tier 0a structural reframings (E41–E44 / E45 — see §6.2 for the unresolved E40 numbering collision).
- **Submission:** E12 is locked in if no further work lands. NG45 transfer already verified (E23). Safety margin 9 hr on the 17-hr cap.

---

## 1. Current Position

| Entry | Avg Proxy (--all) | vs Leaderboard 1.1172 | vs RePlAce 1.4578 |
|-------|-------------------|------------------------|--------------------|
| **CDLNSGridBin (E12, CHAMPION)** | **1.0990** | **−1.63 %** | **−24.6 %** |
| CDLNSSA (E25, candidate, not promoted) | 1.0954 | −1.95 % | −24.9 % |
| CDAdaptive (E9, prior, superseded) | 1.1055 | −1.05 % | −24.2 % |
| CDOnly (prior-prior, superseded) | 1.1193 | +0.18 % | −23.2 % |
| DPO best-of-v2 (prior, superseded) | 1.3834 | +23.8 % | −5.1 % |
| RePlAce baseline | 1.4578 | +30.5 % | — |
| SA baseline | 2.1251 | +90.2 % | +45.8 % |

- E12 beats the leaderboard's "Incremental CD+LNS" (vmallela) entry by −1.63 % with zero overlaps on all 17 IBM benchmarks.
- E12 wall: 28 256 s (7.85 hr) on `--all` — within the 17-hr competition envelope (17 × 1 hr cap), reduced margin vs CDAdaptive's 4.85 hr.
- E25 wall: 37 189 s (10.33 hr) on `--all` — still within envelope, +2.5 hr vs E12.
- Champion config: full-proxy CD on the incremental evaluator with per-benchmark plateau detection (CD ≤ 3000 s) followed by grid-bin LNS escape (LNS ≤ 600 s). ADR-007.
- Candidate config: same as E12 plus an SA-v2 polish phase on per-axis breakpoints with best-so-far tracking and T₀ = 5e-4 (≤ 600 s). ADR-008 *Proposed*.

### Champion lineage

| Era | Champion | Best (--all) | Date | Replaced because |
|---|---|---|---|---|
| Pre-DPO | RePlAce baseline | 1.4578 | n/a | Target to beat |
| Polyhedra | PolyhedraNavigation | 1.4921 | 2026-04-15 | 1.49 ceiling — congestion barrier structural |
| DPO v1 | DPO v1 | 1.4255 | 2026-04-23 | First to beat RePlAce; basin lock on hard benchmarks |
| DPO v2/v3/v2-steps | best_of_v2 | 1.3834 | 2026-04-26 | Within-DPO refinements cap at 1–2 % |
| **CD-only** | CDOnly | 1.1193 | 2026-04-27 (am) | Fixed 600 s budget left hard benchmarks mid-descent |
| **CD-adaptive** | CDAdaptive (E9) | 1.1055 | 2026-04-27 (pm) | Plateau-bound: every bench exited via plateau, none hit cap |
| **CD + grid-bin LNS** | **CDLNSGridBin (E12)** | **1.0990** | **2026-04-28** | (current — ADR-007) |
| CD + LNS + SA-v2 *(candidate)* | CDLNSSA (E25) | 1.0954 | 2026-04-29 | Verified −0.33 % over E12; ADR-008 *Proposed*; not promoted |

Each transition was structural, not parameter tuning. See `docs/experiment_index.md` for the full catalog including failures.

---

## 2. Completed phases

### Phases 1–5 — Polyhedra navigation (2026-04-01 → 2026-04-15)
Decomposed feasible region as union of convex polyhedra; navigated with SDF init + LP + surrogate-guided search. Hit a 1.49 ceiling. Phase 5's 22-experiment sweep proved the polyhedra surrogate was already well-calibrated — the ceiling was structural (the gradient of the relaxed objective lost too much in legalization).

### Phase 6 — DPO (2026-04-16 → 2026-04-26)
Differentiable proxy: gradient through smooth WL+density+congestion with annealed overlap penalty. Reached 1.3834 via best-of-v2. Multi-seed verification showed **basin lock** — `best_of_v2_seed{43,44,45,46}` all within 1 % on `--all`, ibm02/ibm12 byte-identical across seeds. DPO converges to a deterministic local minimum that gradient steps cannot escape. Falsified extensions:
- **E5** (batched seeds, 2026-04-26): B=64 wall = 8.9× B=1; all seeds collapse to same basin under σ=0.04·canvas perturbation.
- **E10** (congestion-only refinement, 2026-04-27): Fast set −1.0 %; `--all` −0.33 % with ibm02 *worse* (+2.3 %).
- **E11** (diverse priors, 2026-04-27): Fast −0.8 %; `--all` FLAT (+0.04 %). Greedy/random priors land in deeper basins on hard benchmarks.

### Phase 7 — Coordinate descent on full proxy (2026-04-26 → 2026-04-28)
The 2026-04-26 LP-HPWL diagnostic decomposed proxy as **6 % WL, 20 % density, 74 % congestion**. Pure HPWL CD (Miftari-style weighted median) caps at ~5 %; *full-proxy* CD captures all three components.
- **E1** (incremental evaluator, 2026-04-26): 4657× speedup per move; bit-for-bit parity. Load-bearing for everything below.
- **E2** (CD on ibm10 single-bench, 2026-04-27 early): 40 min budget hit 1.0632 vs DPO 1.254 (−15 %). 10 min hit 1.1039 (~85 % of value). Generalized to ibm02 (1.1534, broke DPO basin lock from 1.6888) and ibm18 (1.3929).
- **CDOnly --all** (2026-04-27 03:50): Productionized at 600 s/bench. avg 1.1193, matched leaderboard within 0.18 %.
- **E3 LNS v1/v2** (2026-04-27): Falsified. Single-macro destroy/reinsert produces flat ibm17 result regardless of speed.
- **E9 CDAdaptive --all** (2026-04-27 17:22): Per-benchmark plateau detection. avg **1.1055**, beat leaderboard by −1.05 %.
- **E12 CDLNSGridBin --all** (2026-04-28 15:57): CD plateau + grid-bin LNS overlay (different move type — escapes CD's per-axis fixed point). avg **1.0990**, beat leaderboard by −1.63 % and prior champion by −0.59 %. Promoted (ADR-007).

### Phase 8 — Compositional polish (2026-04-28 → 2026-04-29)
- **E14** (SA polish v1, 2026-04-29): **FALSIFIED** (+5.7 % worse). No best-tracking; T₀=0.01 too high.
- **E16** (tighter plateau threshold, 2026-04-28): Landed 1.1025 (−0.0030, formally marginal vs 0.005 kill gate). Used as ablation baseline.
- **E15** (pair swap, 2026-04-28): v2 found real swaps (21–53/bench) but Δ vs baseline below noise.
- **E13** (congestion-region cluster LNS, 2026-04-29): Marginal — passed kill gate but failed gen-check (lift 0.45 % < 0.5 % threshold).
- **E23** (NG45 sanity, 2026-04-28): **VALIDATED** — E12 zero-overlaps on 4 NG45 designs; avg 0.7037; max wall 1053 s vs 3600 s cap.
- **E24** (SA polish v2 = E14 with both fixes, 2026-04-29): Marginal — passes gen-check; ties E12-random-destroy ablation; ibm13 zero SA lift.
- **E25** (CD + LNS + SA-v2 compositional, 2026-04-29): **CHAMPION CANDIDATE** at 1.0954 on `--all`. Wins 11/17, ties 4/17, sub-noise regression on 2/17. ADR-008 *Proposed*.
- **E26** (longer SA budget, 2026-04-29): **FALSIFIED** — extending SA from 600 s → 900 s produced only float-drift fluctuation. 600 s SA saturation is real.

---

## 3. Submission status

| Item | Status | Notes |
|---|---|---|
| Submission package (placer + reproducibility) | **Ready** | `submissions/cd_lns_gridbin/placer.py` is the entry. Self-contained; reuses `IncrementalProxyEvaluator`, SDF init, CDAdaptive runner. |
| NG45 hidden-test robustness check | **DONE** (E23, 2026-04-28) | Zero overlaps on 4 designs; plateau detection transfers; avg 0.7037, max per-bench wall 1053 s vs 3600 s cap. |
| Writeup (innovation prize report) | **In progress** | Parallel agent, `writeup/`. Must include process AND failures. |
| E25 promotion decision (ADR-008) | **Awaiting human** | Adopt → 1.0954 becomes the safe baseline. Defer → 1.0990 (E12) ships. |

**Decision rule:** ship E12 (1.0990) if no further work lands. If ADR-008 is accepted, ship E25 (1.0954) instead.

---

## 4. Currently in flight (parallel agent, 2026-04-29)

Six experiments running concurrently. Each has a clean yes/no outcome that triggers a downstream action.

| ID | Hypothesis | Decision logic |
|---|---|---|
| **E27 basin_persistence** *(THE GATE)* | Persistence-homology diagnostic on 50–100 CD trajectories from 11 diverse inits across ibm11/13/14/15. H₀ filtration measures whether the E25 floor is one big basin or several. | Single dominant H₀ feature below E25 → kill the post-E25 algorithmic line; ship E25 + writeup; pivot remaining weeks to Tier-2 grand prize (ORFS proxy calibration). Multiple long-lived H₀ features below E25 → distinct basins exist; launch Tier 0a (E41 multigrid first, E42 symmetry second, E40-or-renumbered BP third). |
| **E17 random_init** | Replace SDF init with uniform-random legal placement; rerun E25 pipeline unchanged. Tests if SDF is contractive. | Random beats SDF on any benchmark → multi-init best-of-{SDF, random} is a free lift; integrate into next champion run; also signals SDF basin is wrong for that bench, feeds into E40-BP priors and diversifies E27's trajectory set. Random uniformly worse → SDF is the right basin entry; multi-init dead-end stays dead; reinforces ADR-005. |
| **E18 dpo_init** | DPO best_of_v2 → CD → LNS → SA-v2 (replaces SDF init with DPO output). Tests if DPO basin holds topology info CD missed. | Improves over E25 → pipeline becomes DPO → CD → LNS → SA; strongly motivates E33 (low-rank reparam — DPO's basin is a low-rank fixed point CD couldn't reach). Worse → DPO basin lock is real *and* dominated; CD-from-SDF is the right ordering; closes the E11/E18 line. |
| **E32 sam_cd** | SAM-style worst-case-perturbation breakpoint scoring inside CD. Tests if CD's fixed point is sharp (proxy artifact) or flat (genuine). Predicts better OOD generalization to NG45. | IBM `--all` lift → ship; verify NG45 lift ≥ IBM lift. NG45 lift < IBM → IBM-overfit; drop. Flat-or-worse on `--fast` → CD's fixed point isn't a sharp minimum at this proxy resolution; sharpness-aware angle closes. |
| **E39 kmacro_joint_lns** | After E25 pipeline, run 600 s K-macro joint LNS (top-N enumeration; cartesian product N^K=125, pairwise non-overlap check, brute-force best). Cheap operationalization of K-MIQP idea. | Improves over E25 on `--fast` → operationalizes K-MIQP cheaply; promote as E25 successor and skip E29 (full MIQP) build cost. Flat → either K too small (try K=20–30) or proxy-quadratic approx inadequate → E29 is the principled escalation. |
| **E40 multi_sa_seed** | Snapshot post-LNS state, fork 4 SA seeds {42,1,2,3} from the same state, keep best. Tests SA-v2 seed sensitivity. (CD+LNS shared across forks → cheap.) | Lift over E25 on benches where E25 SA already won (ibm01/04/08/09/10) → multi-seed SA is a free lift; integrate. No lift on hardest benches (ibm12–18) → confirms plateau is robust to move set, not seed. |

### Joint verdict matrix (when all six return)

- **E27 multi-basin AND any tactical lift (E17/E18/E32/E39/E40)** → ship the lift first (cheap), then launch Tier 0a from the new floor.
- **E27 single-basin AND all tactical experiments flat** → pivot fully to ORFS / writeup; ship E25.
- **Mixed (most likely)** → run E41 (multigrid) + E42 (symmetry quotient) in parallel — independent code paths — while shipping any tactical lift.

---

## 5. Open hypothesis queue

Hypotheses are sorted by expected value × tractability. Ones already in flight are linked back to §4. None of the items below has been started unless the status column says otherwise.

### 5.1 Tier 0 — Frontier escape research (post-E25 floor)

E25 (1.0954) ties E12 (1.0990) on the hardest benchmarks (ibm11/13/14/15) under five mechanisms — CD per-axis breakpoint, grid-bin LNS, SA-v2, pair swap (E15), spatial cluster destroy (E13). The signature is a *coupled* fixed point: single-macro and 2-macro coordinated moves saturate at the same place. Escape requires higher-order multi-macro moves, a reparameterized objective, or a different basin entirely.

**Order of attack (post E27 verdict):**

1. **Tactical mechanisms running now (§4):** E17, E18, E27, E32, E39, E40.
2. **Highest-EV next:** Tier 0a structural reframings (§5.2). E42 starts regardless of E27; E44 specifically consumes E27's multi-basin endpoints.
3. **Backup if Tier 0a stalls:** E28 (Hessian soft-mode) + E29 (K-MIQP) + E43 (diffusion sampling).
4. **Alternative-objective views if (1)–(3) is mixed:** E30 (OT/JKO), E31 (parametric), E33 (low-rank reparam).
5. **Last-resort research-grade:** E34, E35, E36, E37, E38.

| ID | Hypothesis | Wall | Status |
|---|---|---|---|
| **E27 basin_persistence** | Persistence-homology basin diagnostic over diverse-init CD trajectories. **THE GATE** for everything else in this tier. Multiple long-lived H₀ features below E25 → multi-basin → bridging via E28/E29/E30 is principled. One dominant H₀ → ship E25 + writeup. | ~1 day | **In flight** (§4) |
| **E28 hessian_softmode** | Lanczos-based smallest-k Hessian eigenvectors at the E25 fixed point; "uphill" step ε along softest mode; resume CD. Dimer / gentlest-ascent. **Highest theoretical EV** — directly attacks the coupled-fixed-point signature. | 3–4 days build + 6 hr `--fast` | proposed |
| **E29 kmiqp** | K=10–20 macro joint MIQP (continuous (x,y) per destroyed macro, disjunctive non-overlap, proxy-quadratic cost) via Gurobi/CPLEX. Operationalizes "Cluster-level LNS" line as exact MIQP. **E39 (§4) is the cheap top-N proxy of this.** | 3–4 days build | proposed (E39 may obviate) |
| **E30 wasserstein_jko** | JKO scheme: `p_{t+1} = argmin_p [f(p) + (1/2τ)·W_2²(p, p_t)]`. Sinkhorn on macro-level transport matrix. Lowest mechanistic distance from existing pipeline. | 3–4 days build | proposed |
| **E31 parametric_continuation** | Continuous rotation θ of soft pairs; CD on others; sweep θ ∈ {0, π/8, …, 2π}. Constraint-space continuation. Reuses E28 Hessian infra. | 2 days | proposed (depends on E28) |
| **E32 sam_cd** | SAM-style worst-case perturbation in breakpoint scoring. Predicts better OOD generalization. | 2 days | **In flight** (§4) |
| **E33 lowrank_dpo** | `p_i = p_i^(0) + W·φ_i`, low-rank W ∈ ℝ^{2N×d}, learnable φ_i ∈ ℝ^d. DPO on (W, φ). Initialize W from E28 top-d Hessian eigenvectors. | 2 days | proposed (depends on E28) |
| **E34 rl_policy** | PPO on a GNN policy over the macro/net hypergraph. Action = (macro, target); reward = −proxy delta. Mirhoseini-Goldie 2021 with our 4657× evaluator solving the previously-fatal eval cost. | ≥1 wk build + ≥1 wk train | proposed (deepest investment) |
| **E35 variational_gmm** | q(p) as K-component GMM; minimize E_q[f] − T·H[q] with annealed T. Components share gradient signal during annealing. | 3 days | proposed |
| **E36 tropical_hpwl** | Tropical-GD (Talbut-Monod 2024) on WL component, CD on density+congestion, alternate. WL is only 6 % so bounded upside. | 2 days | proposed |
| **E37 ga_softmode_crossover** | GA crossover that swaps macros along distinct soft-Hessian modes between parents. Depends on E28. | 3 days | proposed (depends on E28) |
| **E38 dee_pruning** | Dead-end elimination (Donald Lab / OSPREY) on grid-bin candidates. Pairwise interaction bound from incremental evaluator → exhaustive search over survivors. | 3 days | proposed |
| **E39 kmacro_joint_lns** | Cheap top-N enumeration version of K-MIQP. | ~1 day | **In flight** (§4) |
| **E40 multi_sa_seed** | 4 SA seeds forked from shared post-LNS state. | ~1 day | **In flight** (§4) — see §6.2 numbering note |

### 5.2 Tier 0a — Structural reframings (HIGHEST EV)

E27–E39 above all keep the *same* parameterization (positions in ℝ^{2N}), the *same* proxy (f = WL + 0.5D + 0.5C), and the *same* search paradigm (local moves on the proxy landscape). Tier 0a breaks one of those three.

**The unifying thread:** stop treating placement as generic continuous optimization. Each entry maps to decades of cross-field validation; none is load-bearing in the placement literature.

| ID | Breaks | Field analogue |
|---|---|---|
| E40-BP † | Search paradigm (optimization → inference) | DMRG, junction-tree BP, statistical-physics ground states |
| E41 | Parameterization (flat → multi-scale) | V-cycle multigrid, real-space RG, multilevel METIS |
| E42 | Parameterization (full → quotient by automorphism group) | AlphaFold E(3)-equivariance, lattice gauge fixing, crystallographic fundamental domains |
| E43 | Search paradigm (optimize one trajectory → sample many) | RFdiffusion, score-based combopt (Sun-Yang 2023, van Krieken 2024) |
| E44 | Search paradigm (point optimization → least-action path) | Onsager-Machlup, NEB / string method, transition-state theory, instantons |

† **Numbering collision** — see §6.2. The label "E40-BP" below is provisional pending a renumber.

#### E40-BP (provisional). Belief-propagation / tensor-network inference on the netlist hypergraph

**Hypothesis:** the proxy factorizes — HPWL is a sum over nets, density and congestion are sums over cells, each touching a small subset of macros. The macro-interaction graph likely has small treewidth, or small-treewidth components linked by few cut edges. On such graphs, junction-tree / DMRG / loopy-BP message passing is the provably-correct algorithm; local search is provably suboptimal. E12/E25 use the netlist as a *constraint* — never as the *computational scaffold*. **Largest unexploited lever in the project.**

**Algorithm:** (1) build macro-interaction graph; estimate treewidth (PACE-class heuristics, ~seconds at N=500). (2) tree-decompose into bags. (3) discretize each macro position to grid-cell centers (~50–500 candidates per macro). (4) junction-tree message passing carrying proxy lower bounds; loopy BP within high-treewidth bags. (5) decode → legalize → evaluate.

**Kill gate:** zero benchmarks below E25 on `--fast` after BP convergence → discretization too coarse, or high-treewidth bags defeat the approximation → kill.

**Generalization check:** NG45 ariane133 — factor structure is benchmark-agnostic, so a working E40-BP should transfer immediately.

**Wall:** 5–7 days build (junction-tree library — NetworkX or libtw — message discretization, decoding/legalization). No learned components, but substantial engineering.

**Status:** proposed. **Highest theoretical EV among Tier 0a** — directly attacks the "treat netlist as constraint, not scaffold" assumption inherited from RePlAce/DPO. **Closest prior experiment:** E38 (DEE) prunes search space; E40-BP factorizes inference across it. Different paradigm.

#### E41. Hierarchical / multigrid placement (V-cycle)

**Hypothesis:** every approach in the lineage is *flat* over 200–537 macros. Single- and two-macro moves can't move *aggregate mass* — that's why the coupled fixed point exists. A multigrid V-cycle rearranges aggregate mass cheaply at the coarse scale and refines at the fine scale. Standard escape from local-operator saturation in PDE solvers, materials-science RG, and multilevel partitioning (METIS / hMETIS) — conspicuously absent from placement.

**Algorithm:** (1) hMETIS clustering of the netlist into ~20–40 super-macros; super-macro size = sum of constituent areas; super-macro nets = quotient netlist. (2) place super-macros via E12 CD+LNS at coarse scale (~minutes). (3) uncoarsen one level: place each super-macro's constituents inside its bounding region with surroundings fixed; refine with CD. (4) repeat to leaf scale. (5) E12 CD+LNS post-smoother on full placement.

**Kill gate:** zero benchmarks below E25 on `--fast` after V-cycle + post-smoother → kill.

**Generalization check:** NG45 ariane133, mempool_tile, NVDLA — these are *more* hierarchical than IBM, predicted larger lift. NG45 lift < IBM lift → multi-scale hypothesis is wrong (commercial designs should benefit *more*).

**Wall:** 3–4 days build (hMETIS bindings, quotient-netlist construction, V-cycle orchestration on E12 primitives).

**Status:** proposed. **Most tractable Tier 0a entry; deepest cross-field track record.** Launch first if E27 returns multi-basin.

#### E42. Symmetry-quotient placement on the netlist automorphism group

**Hypothesis:** identical-shape macros with isomorphic net signatures form orbits under a permutation group G; the proxy is G-invariant. At the E25 fixed point, the joint Hessian has *zero eigenvalues* along orbit-tangent directions — proxy is constant on orbits, so those soft modes are *trivially* flat. E28's Lanczos wastes iterations on them. Detecting G a priori (a) tells E28 which modes to follow (orbit-orthogonal), (b) prevents redundant orbit-element exploration in LNS/SA, (c) supplies a discrete escape move (orbit-element swap) outside CD/LNS/SA reach.

**Algorithm:** (1) build macro/net bipartite graph; run Bliss / nauty for the automorphism group (seconds at N≈500). (2) compute orbits. (3) plug into E28: project Hessian Lanczos onto orbit-orthogonal complement before soft-mode following. (4) add orbit-swap move type to E25's pipeline.

**Kill gate:** (a) all 17 IBM benchmarks have G = {e} (trivial) → no symmetry to exploit → kill cleanly. (b) E28+orbit-projection finds zero descents below E25 *and* orbit-swap is flat on `--fast` → kill.

**Generalization check:** NG45 mempool_tile and NVDLA (repeated-tile structure should give substantially larger G than IBM) → predicted larger lift.

**Wall:** 2–3 days build (Bliss bindings, orbit-projection in Lanczos, orbit-swap in cd_core). **Build-anyway** because of the E28 multiplier — even if lift is zero, E28 becomes substantially more efficient.

**Status:** proposed. **Most novel for the placement community.** AlphaFold-style equivariance. Run regardless of E27 verdict.

#### E43. Score-based diffusion sampling of the proxy Boltzmann distribution

**Hypothesis:** if E27 reveals multi-modal structure, p(p) ∝ exp(−β·f(p)) is genuinely multi-modal. Local search collapses to one mode; sampling captures all. Score network s_θ(p) ≈ −∇f(p) + annealed Langevin samples placements without policy training (distinct from E34: no episodes, no reward shaping). The 4657× incremental evaluator makes long Langevin chains affordable.

**Algorithm:** (1) GNN over netlist hypergraph; conditioning = (netlist, current placement); output = s_θ(p). (2) train via denoising score matching on a corpus of (E12 / E25 / random-perturbed) placements. (3) annealed Langevin sampling at decreasing β; combine s_θ with exact ∇f from incremental evaluator (control variate). (4) post-process best sample with E12 CD+LNS polish.

**Kill gate:** annealed-Langevin samples don't beat E25 on `--fast` after K=20 chains → kill.

**Generalization check:** NG45 ariane133.

**Wall:** 5–7 days build (GNN architecture, score-matching training loop, annealed Langevin). **Most ML-heavy of Tier 0a; pursue only if E27 returns multi-basin AND E40-BP/E41 stall.**

**Status:** proposed.

#### E44. Onsager-Machlup least-action paths between E27 basins (NEB / string method)

**Hypothesis:** for the artificial overdamped SDE dp = −∇f·dt + √(2T)·dW, the most-likely path between two basins satisfies Euler-Lagrange equations of S[p] = (1/4T)·∫₀ᵀ ‖ṗ + ∇f(p)‖²·dt. The minimum-action path crosses a saddle whose unstable Hessian mode points toward potentially-lower basins on the far side. **Real variational principle for an artificial dynamics** — chemistry/materials/climate workhorse with 30+ years of numerical refinement.

**Algorithm:** (1) take each basin pair (p_A, p_B) E27 returns; (2) initialize 30–50 image string γ₀ linearly interpolating; (3) at each interior image, evaluate −∇f via incremental evaluator and project out component parallel to string tangent; (4) reparameterize to keep image spacing uniform; (5) iterate to convergence — highest-energy converged image is the saddle p_S between A and B; (6) compute smallest unstable Hessian eigenvector v at p_S via Lanczos (reuse E28); (7) descend ±ε·v and run E12 CD+LNS from each side; (8) accept any new basin below E25 floor.

**Kill gate:** zero new basins below E25 on `--fast` after K=10 NEB runs across E27's basin pairs → kill.

**Generalization check:** NG45 ariane133 if E27 multi-basin reproduces on commercial designs.

**Wall:** 2–3 days build (string-image data structure, perpendicular-gradient projection, reparameterization, integration with cd_core legalization). **Reuses E27 basin endpoints, incremental evaluator, optionally E28 Lanczos.** Pair tightly with E27.

**Status:** proposed. **Most physically-principled escape mechanism in the catalog.** Composes with E28: NEB finds saddles *globally* between known basins; E28 finds saddles *locally* via dimer / gentlest-ascent. Complementary, not redundant. Also gives quantitative Arrhenius escape rates (∼ exp(−S_action / T)) so you know which inter-basin transitions are budget-reachable.

### 5.3 Tier 1 — overnight runnable, generalization-safe

| ID | Hypothesis | Status |
|---|---|---|
| **E12 grid-bin LNS** | The actual LNS recipe (grid-bin search, not subset-CD). Different move type CD can't reach. Adaptive destroy=5 % capped at 30; cost-aware ranking; iterate until budget. | **GRADUATED 2026-04-28** → champion. Code at `submissions/cd_lns_gridbin/placer.py`. ADR-007. *Note on prior art:* leaderboard's "Incremental CD+LNS" might be exactly this recipe; TAISPlAce's "ALNS + Thompson" hit only 1.4321, suggesting LNS execution detail matters a lot. |
| **E16 tighter plateau threshold** | Threshold 0.005 → 0.001, cap 1 hr → 2 hr. Single global hyperparameter swap, no per-bench tuning. | **Completed.** 1.1025 (−0.0030 vs E9; formally marginal vs 0.005 kill gate). See `experiments/E16_tight_threshold/` and `writeup/evidence.md` §7.6. |
| **E14 SA polish v1** | Metropolis acceptance over per-axis breakpoints. T₀ → T_f over N moves. | **FALSIFIED 2026-04-29.** +5.7 % worse on every `--fast` bench. Two compounding bugs: no best-so-far tracking + T₀=0.01 too high (50/50 random walk). See `experiments/E14_sa_polish/`. **E24 is the principled retest with both fixes.** |
| **E15 pair swap** | Swap two macros sharing ≥1 net. Coordinated 2-macro move outside CD's reachable set. | **FALSIFIED 2026-04-28.** v1 buggy (`min_shared_nets=2` filter found zero candidates). v2 found 21–53 real swaps/bench but Δ vs baseline below noise. CD's plateau is robust to pair swaps too. See `experiments/E15_pair_swap/`. *Prior art:* leaderboard "MacroBio (Two-Opt Swap)" pending. |
| **E23 NG45 sanity** | Defensive — validate E12 runs end-to-end on NG45 designs and produces ORFS-flowable placements. | **VALIDATED 2026-04-28.** avg 0.7037 across 4 designs; zero overlaps; max wall 1053 s vs 3600 s cap. Plateau detection transfers unchanged. Fills ADR-007's missing NG45 datapoint. |

### 5.4 Tier 2 — medium-EV, build-if-Tier-1-mixed

| ID | Hypothesis | Status |
|---|---|---|
| **E13 congestion_lns** | Targeted destroy of macros in high-congestion regions (cells > median + 1σ). Spatial-cluster destroy releases joint constraints. | **MARGINAL 2026-04-29.** avg `--fast` 0.9384 (lift 0.45 % over E16 baseline, below 0.5 % gen-check threshold). Mechanism works on ibm01 (0.94 % lift) but doesn't generalize. +0.13 % worse than E12-random-destroy ablation. Did not queue `--all`. Possible follow-up: destroy nearest-neighbors *within* hot cells, not just residents. |
| **E17 random_init** | Replace SDF with random-uniform legal init. Test if SDF basin is wrong on some benches. | **In flight** (§4) |
| **E18 dpo_init** | DPO best_of_v2 → CD+LNS+SA-v2 polish (E6 redux on E25). Per E11, alternative inits sometimes find different basins. | **In flight** (§4) |
| **E24 SA polish v2** | E14 with best-so-far tracking + T₀=5e-4. Fair retest of "Metropolis escapes CD basin". | **MARGINAL 2026-04-29.** Passes gen-check; ties E12-random-destroy ablation 0.9372; ibm13 zero SA lift. Compositional test E25 launched from this. |
| **E26 longer SA budget** | LNS 300 s + SA 900 s; tests if SA's "best at t≈599 s" had real headroom. | **FALSIFIED 2026-04-29.** Fluctuation float-drift only. +20 % wall for zero quality gain. 600 s SA saturation is real. |

### 5.5 Tier 3 — speculative / parking lot

| ID | Hypothesis | Notes |
|---|---|---|
| **E19 RUDY → Steiner-tree congestion** | Different proxy modeling. **Risk:** if Steiner is more accurate but TILOS evaluator uses RUDY (it does), we optimize for a thing the contest doesn't measure. **Skip unless Tier 2 needs it.** |
| **E20 ILP polish on local windows** | ~3 days build, uncertain payoff. Only if Tier 1 lacks improvement and we have time. |
| **E21 GPU-batched breakpoint search** | Per Proof A, CD is plateau-bound, not budget-bound on IBM. Speed-up doesn't directly help score on IBM. **Re-live for NG45 if hidden designs are larger** and budget actually binds. |
| **E22 Conflict-graph colored CD** | Same caveat as E21. Only useful at scale. |

### 5.6 Optional polish (24 days available)

| Lever | Expected | Effort | Notes |
|---|---|---|---|
| Drop cost-aware destroy ranking | flat (already verified on `--fast`) | 30 min | Saves ~10 % wall per LNS sample at no quality cost; defer until post-deadline |
| Cluster-level LNS (joint reinsertion of K>1 macros) | speculative | 3–4 days | Operationalized as exact K-macro MIQP (E29) or top-N enumeration (E39 — in flight); composes on top of E25/E12 |
| **GPU-batched CD via conflict-graph coloring (E4)** | 5–20× speedup | 3–4 days | Useful only if E12 wall risks the 1-hr-per-bench cap on NG45; currently 7.85 hr / 17 hr envelope on IBM |
| Multi-seed validation | Confirm robustness | 1 day | E12 is deterministic via SDF init; seed sensitivity TBD. Subsumed by E40 multi_sa_seed for the SA phase. |

---

## 6. Bookkeeping

### 6.1 Files of record

| File | Purpose |
|---|---|
| `submissions/cd_lns_gridbin/placer.py` | **CHAMPION submission** (E12, 1.0990) |
| `submissions/cd_lns_sa/placer.py` | **Champion candidate** (E25, 1.0954, ADR-008 *Proposed*) |
| `submissions/cd_adaptive/placer.py` | Prior champion (CDAdaptive E9, 1.1055; superseded 2026-04-28) |
| `submissions/cd_only/placer.py` | Prior-prior champion (CDOnly fixed-budget) |
| `submissions/dpo/best_of_v2_placer.py` | DPO prior champion (basin-locked) |
| `submissions/dpo/{e10,e11,batched,...}*.py` | Falsified DPO extensions; kept for writeup |
| `submissions/dpo/ablation_*.py` | DPO ablation studies |
| `writeup/archive/submissions/lns.py` | E3 LNS module — falsified; deleted from active tree (commit 44efd16); preserved at archive path |
| `writeup/archive/submissions/cd_lns_placer.py` | E3 placer — falsified; same provenance |
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

### 6.2 Numbering note — E40 collision

**As of 2026-04-29 there are two experiments labelled E40:**

1. `experiments/E40_multi_sa_seed/` — has manifest + `code/` + run logs. Currently in flight (§4). Listed in §5.1 Tier 0.
2. **E40-BP / tensor-networks** — Tier 0a structural reframing in §5.2. Proposed only; no code yet. Documented in `MEMORY.md` and the prior roadmap as part of "E40–E44". **Provisionally labelled "E40-BP" pending a renumber.**

Recommended resolution: renumber the proposed structural reframing to **E45** (next free slot). The current Tier 0a numbering would then become E45 (BP/tensor-networks), E41 (multigrid), E42 (symmetry quotient), E43 (diffusion sampling), E44 (NEB). Both alternatives — renumbering `multi_sa_seed` instead, or keeping the dual-E40 label — preserve all existing names but introduce different downstream churn. **No renumber has been applied; the project_overview memory still records E40-E44 for Tier 0a.**

### 6.3 Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hidden NG45 commercial designs differ from IBM (different macro-count regime) | Medium | Plateau detection transfers without per-bench tuning. E23 already verified 4 NG45 designs. |
| Float drift between incremental evaluator and `compute_proxy_cost` over thousands of moves | Low (~0.5–1.5 % on ibm02 observed) | Final VALID is the canonical value. Internal proxy biased low ~1 % on dense benchmarks but doesn't change ranking; plateau detection still works. |
| Overlap regression on edge cases | Low | Hard validation in placer raises RuntimeError on overlap. Tested on all 17 IBM with zero overlaps. |
| Submission package missing reproducibility info | Low | `submissions/cd_lns_gridbin/placer.py` self-contained; reuses tracked files. |
| Champion wall close to 17-hr envelope (7.85 hr used) | Low | ~9 hr headroom on `--all`. Hidden NG45 designs of similar scale should fit within 1 hr/bench. Verified by E23 (max per-bench 1053 s vs 3600 s cap). |
| E25 wall (10.33 hr) eats more headroom if promoted | Low | Still within 17-hr cap; +2.5 hr vs E12. |

### 6.4 Deprioritized hypotheses

Carried forward from the retired `experiments_overnight.md`.

- **Per-benchmark hyperparameter tuning** — explicitly forbidden by competition rules
- **More multi-init CD via SDF jitter** — falsified, 0/8 improved, structurally explained (SDF is contractive). See `analysis/multi_init_probe/README.md` and ADR-005.
- **More random-destroy LNS via subset-CD** — falsified across 24 samples, two benchmarks. See `analysis/lns_escape_probe/README.md`.
- **Within-DPO refinement** (E10/E11 variants) — basin-locked per memory; CD already breaks that lock
- **GPU best-of-N seeds** (E5) — already falsified
- **Polyhedra navigation** — superseded
- **Reinforcement learning / GNN priors** (subsumed by E34) — too long to build given May 21 deadline; could be revisited if we want to chase Innovation Award narrative

### 6.5 Open follow-ups

Carried forward from the retired `findings.md` and `experiments_overnight.md`.

- ~~**NG45 sanity test**~~ — **CLOSED 2026-04-28 by E23.** Public NG45 designs at `benchmarks/processed/public/{ariane133,ariane136,mempool_tile,nvdla}_ng45.pt`.
- ~~**Tier 2 robustness check on E12**~~ — **CLOSED 2026-04-28 by E23.**
- **ORFS local integration** — multi-day project; the actual path to the **$20K Tier-2 grand prize**. Should be its own multi-day track. *If E27 returns single-basin, this becomes the highest-EV remaining work.*
- **Verify zero overlaps preserved through legalization on E12 final placement** — should be enforced by `compute_overlap_metrics` raise in placer; verify against the full `--all` result.
- **Multi-seed variance report** — produce a writeup figure showing CDAdaptive's variance across seeds on each benchmark (innovation-award material). Subsumed in part by E40 multi_sa_seed for the SA phase.
- **Adversarial / hidden-design stress test** — synthesize random "NG45-like" benchmarks (random sizes, densities, topologies) and verify CDAdaptive doesn't catastrophically fail on edge cases.
- **Tier 2 OpenROAD flow integration** — running ORFS locally to iterate on WNS/TNS/Area instead of proxy.

---

## See also

- [approach.md](approach.md) — current CD-on-incremental-evaluator architecture
- [results.md](results.md) — current champion (CDLNSGridBin) per-benchmark tables
- [experiment_index.md](experiment_index.md) — full catalog including falsified hypotheses
- [lp_hpwl_diagnostic.md](../analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md) — the diagnostic that unblocked CD
- [decisions/README.md](decisions/README.md) — ADRs (ADR-005 SDF init, ADR-007 E12 promotion, ADR-008 E25 candidate *Proposed*)
- `writeup/historical_results.md` — DPO/Polyhedra/Overnight per-bench tables
- `writeup/closing_the_gap.md` — narrative of the leaderboard-beating run
- `writeup/cd_ibm10_results.md` — single-benchmark CD breakthrough
- `writeup/theory.md` — theoretical foundations (polyhedra decomposition, proxy decomposition)
