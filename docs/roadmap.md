# Roadmap

> ⚠️ **HISTORICAL — last updated 2026-05-05/06.** Current plan is in repo-root `TODO.md` (two-path structure: Path A cascade speedup, Path B DREAMPlace exploration). Submission target as of 2026-05-11: `submissions/cd_lns_sa_cascade/placer_adaptive.py`.

Last updated: 2026-05-05 (post-E74 promotion)
Competition deadline: May 21, 2026 (~16 days)

## TL;DR

- **CHAMPION:** **E74 CDLNSSAHessian at 1.0666 --all + 0.6813 --ng45** (**−1.38 % vs E48 1.08151**, **−4.53 % vs leaderboard 1.1172**, **−26.8 % vs RePlAce 1.4578**). NG45 ariane133 = 0.6641 (**−3.21 % vs E48 — BREAKS the failure point that killed E42/E43/E44/E54/E62**). Beats every VERIFIED leaderboard entry by ≥17 %. Mechanism: smooth-proxy Hessian via `torch.autograd.functional.hvp` → Lanczos smallest-algebraic eigenvectors → ±ε saddle perturbation → CD polish. Implements roadmap E28 (proposed since 2026-04-29) and vmallela's leaderboard #2 mechanism class. Wall ~50 min/bench; --all ~14 hr serial / ~4 hr `--jobs 4`. **PROMOTED 2026-05-05 (ADR-012 *Accepted*)**; supersedes ADR-011. Entry: `submissions/cd_lns_sa_hessian/placer.py`.
- **Prior champion (kept as fallback):** E48 CDLNSSAHybrid (1.08151, ADR-011). Code remains at `submissions/cd_lns_sa_hybrid/placer.py`.
- **Components called by the champion (load-bearing):** E25 (`submissions/cd_lns_sa/placer.py`) and E41 (`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`); both run BEFORE the Hessian saddle phase. DPO smooth-proxy primitives at `writeup/archive/submissions/dpo/ablation_v2_steps.py` are imported for the Hessian autograd cost function.
- **Submissions folder cleaned 2026-05-05:** archived superseded prior champions (cd_only, cd_adaptive, cd_lns_gridbin, will_seed) under `submissions/_archive/`. Active: `cd_lns_sa_hessian/` (champion), `cd_lns_sa_hybrid/` (fallback), `cd_lns_sa/` (component), `examples/`.

## 0. What's next (post-E74)

**Position vs leaderboard (May 4 refresh):**
- Cezar 1.037 (unverified, +2.9 % above E74; previous variant +14 % drift on verification)
- vmallela 1.1 (unverified, **−3.0 % below E74**; same mechanism class — Hessian saddle escape)
- Hoop Dreams 1.2206 (DREAMPlace+Optuna, unverified)
- MTK 1.2818 (best **VERIFIED** — we beat by −16.8 %)

E74 is the **strongest VERIFIED submission on the board.** Cezar 1.037 is the only entry numerically lower; whether it survives verification is the open question.

**Highest-EV next moves (in priority order):**

### 0.1 Sharper E74: more eigvecs + finer ε sweep on hard benches

E74 wave used `n_eigvecs=2, eps_values=(0.3, 1.0, 3.0)`. Some benches lifted only marginally (ibm09 −0.11 %, ibm17 −0.29 %, ibm18 −0.38 %). With **k=5 eigvecs** and **eps_values=(0.1, 0.3, 0.7, 1.5, 3.0, 5.0)** on the hardest 6 benches (ibm12-18), the search is ~6× larger and may find deeper saddles. Wall: ~6 hr per bench × 6 = ~36 hr serial, ~9 hr `--jobs 4`. **Expected lift: 0.2–0.5 %** on aggregate. Cheap relative to potential gain.

### 0.2 Layer E61V2-fresh + E74 on remaining tied benches

Tied benches (E25/E41 gap < 1 %): ibm08, ibm12 (done), ibm14, ibm15 (done), ibm16, ibm17, ibm18. The pattern from ibm12/ibm15: E61V2-fresh produces a third basin → E74 on top compounds. Lift on ibm12 was −0.61 % (vs −0.40 % from E74-only); ibm15 −1.73 % (vs −0.77 %). Per bench wall: E61V2 fresh ~5 hr + E74 ~50 min. For 5 remaining tied benches: ~30 hr. **Expected lift: 0.1–0.3 %** on aggregate.

### 0.3 DREAMPlace integration as 3rd basin source

Six of nine top leaderboard entries use DREAMPlace (Cezar / Hoop Dreams / Shoom / MTK / UTAustin AS / Mike Gao [DQ]). E76 manifest has scoping. Multi-day dev with macOS / CUDA install risk. **Expected lift: 0.3–1.5 %** if it works. Highest variance.

### 0.4 Hardware portability check

Champion runs on M3 Max; judges run on AMD EPYC 9655P (slower per-core clock, faster overall via 16 cores). Wall caps in champion (CD 2400 s, polish 240 s) may fire before plateau on slower hardware → quality drops. Defensive action: **work-bound termination** (terminate on plateau metric, not wall time). E68 (other Claude) was working on this; check progress and adopt.

### 0.5 Submit + track verification

The user-facing decision: **submit E74** (Tier 1 ranks by proxy; verified score is what matters). After submission:
- Track Cezar / vmallela / etc. verification on partcl hardware.
- If Cezar drifts at historical rates (+14 %), E74 lands at #1 verified.
- If Cezar holds at 1.037, we're #2 (still well in top-7 for Tier 2 Grand Prize).

### 0.6 Tier 2 Grand Prize prep

E74 NG45 = 0.6813 with ariane133 0.6641 — strongest NG45 result we've seen. Tier 2 evaluates top-7 by proxy through full OpenROAD flow on NG45 (incl. 1-2 hidden designs). Need to verify our placements pass the OpenROAD feasibility gate (WNS / TNS / Area not regressing below SA + RePlAce baselines on any design). **Action**: read `SCORING.md` carefully, verify our NG45 placements are ORFS-flowable.

### 0.7 Innovation Award ($4k) writeup

Hessian saddle escape on a CD plateau via smooth-proxy autograd Hessian + Lanczos is a novel contribution to the placement field. Henkelman & Jónsson 2000 climbing-image NEB has not been applied to placement before. Writeup at `writeup/paper.md` (TODO markers) — needs E74 mechanism, eigenvalue analysis, NG45 transfer story.

## Recommended sequence

| Day | Work | Expected outcome |
|-----|------|------------------|
| 1 | Submit E74 + write submission form | Locks in #1 verified rank |
| 1–2 | Sharper E74 wave (k=5, more ε) on ibm12-18 | +0.2–0.5 % aggregate |
| 3–4 | E61V2 fresh + E74 layered on ibm08/16/17/18 | +0.1–0.3 % aggregate |
| 5–7 | DREAMPlace integration if time | +0.3–1.5 % (variance) |
| 5–7 | Hardware portability fixes | Defensive |
| 8–10 | Tier 2 ORFS verification | Grand Prize gate |
| 11–14 | Writeup + Innovation Award prep | $4k prize |
| 15–16 | Buffer + final verification | — |

The ABSOLUTE BEST realistic outcome is ~1.04 if all of 0.1–0.4 work — same neighborhood as Cezar 1.037, where verification drift would decide #1 vs #2.
- **Overnight 2026-05-01 → 02 — three follow-ups, NONE lifted past E48:**
  - **E53 GPU DPO basin polish** (multi-restart full-pose Adam on smooth proxy after CD-LNS-SA): `--fast` 0.9254, **0/350 GPU restarts accepted** at full budgets. Smooth-proxy gradient cannot escape fully-converged CD-LNS-SA. **FALSIFIED.** Lesson: GPU must replace CD's basin-crossing role, not polish after it.
  - **E53m multi-seed hybrid** (3-way: E25 + E41 seed=42 + E41 seed=1): `--fast` 0.91128 (−0.97 % vs E48 — sample-size outlier on 4 small benches); `--all` 1.08128 (essentially tied with E48 by −0.02 %); `--ng45` 0.6938 (+0.23 %). **MARGINAL.** Multi-seed within DPO is dead-end at --all aggregate scale.
  - **E54 congestion-targeted destroy** (engages E8 6/20/74 % proxy decomposition; rank LNS-destroy by abu-top-5 % cell contribution): `--fast` 0.9222 tied; `--all` 1.08568 (+0.39 %); **`--ng45` 0.7022 with ariane133 +5.14 % catastrophic regression**. **FALSIFIED.** Joins E42/E43/E44 in the IBM-aware/NG45-blind failure class.
- **The structural finding from the night:** Five experiments (E42 K=4, E43 longer K-joint, E44 spatial K-tuple, E54 congestion-destroy, plus the partially-IBM-aware E53m) all show the same pattern — IBM-fast lift that doesn't transfer to NG45 commercial designs. **ariane133 is the consistent failure point**; its sparser packing (133 hard macros on 1433 × 1433 canvas vs IBM 246-760 hard macros on 23-73 canvas) breaks structurally-aligned destroy/K-tuple heuristics that depend on dense macro packing.
- **Submission status:** E48 hybrid locked in; submission-ready as of 2026-05-02. 3.4 hr safety margin on the 17-hr `--all` cap. NG45 transfer verified at 0.6922 (zero overlaps).

---

## 1. Current Position

| Entry | Avg Proxy (--all) | vs Leaderboard 1.1172 | vs RePlAce 1.4578 |
|-------|-------------------|------------------------|--------------------|
| **CDLNSSAHybrid (E48, CHAMPION)** | **1.08151** | **−3.21 %** | **−25.8 %** |
| CDLNSSAMultiseedHybrid (E53m, marginal) | 1.08128 | −3.24 % | −25.8 % |
| CDLNSSADPOKJoint (E41, hybrid component) | 1.0848 | −2.90 % | −25.6 % |
| CDLNSSADPOKJointCongDestroy (E54, falsified NG45) | 1.08568 | −2.83 % | −25.5 % |
| CDLNSSADPOInit (E18, ADR-009 superseded) | 1.08979 | −2.45 % | −25.2 % |
| CDLNSSA (E25, hybrid component) | 1.0954 | −1.95 % | −24.9 % |
| CDLNSGridBin (E12, prior champion) | 1.0990 | −1.63 % | −24.6 % |
| CDAdaptive (E9, prior, superseded) | 1.1055 | −1.05 % | −24.2 % |
| CDOnly (prior-prior, superseded) | 1.1193 | +0.18 % | −23.2 % |
| DPO best-of-v2 (prior, superseded) | 1.3834 | +23.8 % | −5.1 % |
| RePlAce baseline | 1.4578 | +30.5 % | — |
| SA baseline | 2.1251 | +90.2 % | +45.8 % |

- **E48 beats the leaderboard's "Incremental CD+LNS" (vmallela) entry by −3.21 %** with zero overlaps on all 17 IBM benchmarks.
- E48 wall: ~7 hr `--jobs 4` (88 452 s aggregate CPU-time across workers); each bench runs E25 then E41 sequentially. Within the 17-hr competition envelope.
- Champion config: per-bench `min(E25_pipeline_output, E41_pipeline_output)`. E25 = SDF init → CD plateau → grid-bin LNS → SA-v2. E41 = DPO best_of_v2 init → CD plateau → grid-bin LNS → SA-v2 → K-joint K=3 top_N=5. **No per-benchmark hardcoded logic** — winner determined by proxy value. ADR-011 *Accepted* 2026-05-02.

### Champion lineage

| Era | Champion | Best (--all) | Date | Replaced because |
|---|---|---|---|---|
| Pre-DPO | RePlAce baseline | 1.4578 | n/a | Target to beat |
| Polyhedra | PolyhedraNavigation | 1.4921 | 2026-04-15 | 1.49 ceiling — congestion barrier structural |
| DPO v1 | DPO v1 | 1.4255 | 2026-04-23 | First to beat RePlAce; basin lock on hard benchmarks |
| DPO v2/v3/v2-steps | best_of_v2 | 1.3834 | 2026-04-26 | Within-DPO refinements cap at 1–2 % |
| **CD-only** | CDOnly | 1.1193 | 2026-04-27 (am) | Fixed 600 s budget left hard benchmarks mid-descent |
| **CD-adaptive** | CDAdaptive (E9) | 1.1055 | 2026-04-27 (pm) | Plateau-bound: every bench exited via plateau, none hit cap |
| CD + grid-bin LNS | CDLNSGridBin (E12) | 1.0990 | 2026-04-28 | Superseded 2026-05-02 by E48 hybrid (ADR-007 superseded by ADR-011) |
| CD + LNS + SA-v2 *(component)* | CDLNSSA (E25) | 1.0954 | 2026-04-29 | Now lane 1 of E48 hybrid; standalone score did not promote |
| DPO init + CD + LNS + SA-v2 + K-joint *(component)* | CDLNSSADPOKJoint (E41) | 1.0848 | 2026-04-30 | Now lane 2 of E48 hybrid; standalone score did not promote (ADR-010 superseded by ADR-011) |
| **Per-bench best-of-{E25, E41} hybrid** | **CDLNSSAHybrid (E48)** | **1.08151** | **2026-05-02** | (**current — ADR-011**) |

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

## 4. Overnight wave 2026-04-29 → 30 outcomes

Six tactical experiments + one diagnostic + one combo decided overnight. Summary table:

| ID | Result | --fast vs E25 0.9336 | --all vs E25 1.0954 | NG45 vs E12 0.7037 | Status |
|---|---|---|---|---|---|
| E17 random_init | Random init lands in separate, much worse basin | far worse | n/a | n/a | **falsified** |
| **E18 dpo_init** | DPO basin → E25 polish: 11/17 IBM wins, 4/4 NG45 wins | 0.92542 (-0.91 %) | **1.08979 (-0.51 %)** | **0.69193 (-1.67 %)** | **CHAMPION CANDIDATE — awaiting human promotion** |
| E27 basin_persistence | 16/44 trajectories (DPO/jitter inits crashed); SDF basin tight, off-init basins separate | n/a | n/a | n/a | **marginal** (empirical signal from E18 supplies the diagnostic instead) |
| E32 sam_cd | K=4 perturbations multiply CD eval ~5× → CD never plateaus | 0.98501 (+5.5 %) | n/a | n/a | **falsified** (kill gate fired) |
| E39 kmacro_joint_lns | K-joint phase commits 12-58 K-tuples/bench post-CD-LNS-SA | 0.93070 (-0.31 %) | crash on ibm07 (overlap-validation bug; **fixed 2026-04-30 04:35**) | 0.70126 (-0.35 % vs E12) | **marginal — bug fixed, partial --all valid** |
| E40 multi_sa_seed | Multi-seed forks don't compound; fork 1 (seed=42) usually best | 0.93295 (-0.07 %) | skipped (~14 hr for ≤0.1 % gain) | n/a | **marginal** |
| **E41 dpo_kjoint** | E18 ⊕ E39: composes DPO basin shift with K-joint escape | **0.92178 (-1.27 %)** | stalled at 6/17 (multiproc + K-joint bug) | **0.69022 (-1.91 % vs E12)** | **strong candidate — --all rerun pending K-joint fix verification** |

### Verdict synthesis

- **Multi-basin ACROSS init classes:** E18's --all win (-0.51 %) and 4/4 NG45 wins prove the DPO basin is structurally distinct from the SDF basin AND deeper. E27 captured this empirically through E18, not through its own incomplete persistence-homology output. The basin-diagnostic line is closed: the answer is multi-basin in the productive direction (different inits), single-basin in the unproductive direction (within SDF init class).
- **Productive next moves:** *different inits* (E18 is the proof of concept) and *different move types* (E39 K-joint adds 0.001-0.005 lift on top of post-CD-LNS-SA on every bench). *Combinations* (E41) compose multiplicatively on --fast and --ng45.
- **Unproductive next moves:** within-basin ensemble methods (E40 confirms zero compounding), random init (E17 confirms the SDF basin is contractive in a useful way for legal sub-regions; uniform basin is structurally worse), sharpness-aware probes (E32 fails on the eval-cost multiplier).

### Pending actions

1. **Human promotion decision on E18.** ADR draft needed at `docs/decisions/009_e18_dpo_init_promotion.md` if accepted.
2. **E41 --all rerun** after K-joint overlap-fix verification on ibm07 (verification in progress 2026-04-30 04:36; ~50-65 min single-bench wall). Re-run --all sequentially or with --jobs 4 to avoid the multiproc.Queue stall observed earlier (occurred under 6+ concurrent --all jobs).
3. **DPO trajectory-init bug fix.** `BestOfV2Placer` references missing `writeup/archive/submissions/polyhedra/init/sdf.py`. Fix unblocks any future E27 rerun and any DPO-basin diagnostic on harder benches.
4. **K-joint mechanism extension.** E39 K=3 already extracts wins; K=4-5 is the next probe. E29 (MIQP) is the principled escalation if K=5 saturates.

---

## 4.5 Post-E48 wave 2026-05-01 → 02 outcomes (3 falsifications, 1 marginal)

After E48 hybrid (1.08151) was verified, three orthogonal escape directions
were tested overnight. **None lifted past E48.** The pattern across all three is
itself the load-bearing finding for the path forward.

| ID | Hypothesis | --fast | --all | NG45 | Status |
|---|---|---|---|---|---|
| ~~E53~~ | GPU DPO basin polish (multi-restart full-pose Adam on smooth proxy after CD-LNS-SA) | 0.9254 (+0.36 %) | — | — | **FALSIFIED** — 0/350 GPU restarts accepted at full budgets. Smooth-proxy gradient cannot escape fully-converged CD-LNS-SA local optimum. |
| E53m | Multi-seed hybrid: 3-way best-of-{E25, E41 seed=42, E41 seed=1} | 0.91128 (−0.97 %) | 1.08128 (−0.02 %) | 0.6938 (+0.23 %) | **MARGINAL** — `--fast` lift was sample-size outlier on 4 small benches; --all aggregates dampen DPO seed-noise to noise-floor. |
| ~~E54~~ | Congestion-targeted destroy ranking (engages E8 6/20/74 % proxy decomposition; rank LNS-destroy by abu-top-5 % cell contribution) | 0.9222 (tied) | 1.08568 (+0.39 %) | **0.7022 (+1.45 %)** | **FALSIFIED** — ariane133 +5.14 % catastrophic regression. Joins E42/E43/E44 in the IBM-aware/NG45-blind failure class. |

### The structural finding: IBM-aware mechanisms break on ariane133

Five experiments now show the same pattern: lift on IBM (often --fast)
that doesn't transfer to NG45 commercial designs, with **ariane133 as the
consistent failure point** (regression magnitudes +3.57 % to +5.14 %).

| ID | Mechanism | --fast Δ vs E41 | NG45 ariane133 Δ vs E41 |
|---|---|---:|---:|
| E42 | K-joint K=4 | −0.35 % | **+3.57 %** |
| E43 | Longer K-joint (1200 s) | −0.33 % | +4.10 % |
| E44 | Spatial K-tuple selection | +0.63 % (kill gate) | n/a (skipped) |
| E54 | Congestion-targeted destroy | tied | **+5.14 %** (vs E48) |

ariane133 has 133 hard macros on a 1433 × 1433 canvas (sparse: ~1/15.5 macros/sq-micron),
vs IBM's 246-760 hard macros on 23-73 canvas (dense: ~10-20 macros/sq-micron).
**Conjecture (not yet formally verified):** structurally-aligned destroy and
K-tuple heuristics depend on dense macro packing; they degrade on sparse
commercial layouts because the cells they target (top-congestion, spatial
neighbors) don't match the structural-coupling clusters that K-joint exploits.

### Verdict synthesis

- **Cost-aware destroy + netlist-adjacency K-tuple ranking** are the safe
  baselines. Engaging proxy structure (E54) or geometric structure (E44)
  in the destroy/K-joint phases consistently breaks NG45 transfer.
- **Multi-seed within DPO is dead-end at --all aggregate scale.** Seed
  variation captures real per-bench variance (E52 confirmed) but
  averages out across 17 benches.
- **GPU acceleration as an *additive* polish phase on top of fully-converged
  CPU CD-LNS-SA is dead-end.** The basin local optimum smooth-proxy
  gradient finds is at-or-above the breakpoint-enumeration optimum CD-LNS
  finds.

---

## 4.6 Path forward — global topology navigation (REFRAMED 2026-05-03 07:35)

### Overnight 2026-05-02 → 03 hardened the framing

E65 NEB cross-section (graduated 2026-05-03) confirmed the structural
finding: **the SDF (E25) and DPO (E41) basins are separated by an
INFEASIBILITY WALL, not a proxy barrier.** Two cross-sections (ibm01
and ibm12) both showed 0/9 intermediate k-values legalize on a linear
interpolation between the two basins. Wall residuals ranged 88-155 hard
overlaps; on ibm12 the endpoints differ by only 0.2 % in proxy yet are
separated by a 233-residual feasibility gap. **The wall is a function
of spatial-configuration distance, not proxy distance.** This rules out
any mechanism that bridges basins at the *solution* level
(interpolation, blending, macro-level crossover).

E61_v2 spatial-block GA crossover (graduated marginal 2026-05-03)
threaded through the wall by operating at the BLOCK level:
2×2-quadrant chunks of macros from each parent are internally feasible,
so block-level recombination stays close to the feasible manifold. Got
**verified --all 1.08083 (−0.07 % vs E48 — marginal)** and **--ng45
0.6908 (−0.20 %; ariane133 0.6760, −1.47 % — first break of the
ariane133 failure point** that killed E42/E43/E44/E54/E62). Best-of-2
{E48, E61_v2} = 1.08025 (−0.12 % vs E48); best-of-3 with E53m = 1.07995
(−0.14 %). Strong hybrid contribution; standalone champion margin too
small to promote.

E63 spectral init V2/V3 (still blocked, 2026-05-03): Hungarian-on-slot-grid
solves in 0.02 s but ibm01 fits only 28 max-size slots vs 246 hard
movables; row-packing is correct but doesn't avoid fixed macros
pre-placed on canvas (111 residuals). **Both algorithms work for
movable-only sub-problems**; what's needed is row-packing AROUND fixed
macros (subtract fixed regions from row x-spans).

**The bottleneck remains global topology navigation.** Updated
direction map below reflects the post-overnight reality.

To break out, we need an algorithm that explicitly NAVIGATES across
many topologies, exploiting the structure of the topology landscape
rather than randomly sampling it. The infeasibility wall finding shapes
which mechanisms are even possible:

### 4.6.A — LP-bounded beam K-joint (E64 — multi-day dev)

K=10 macros forming a structurally-coupled cluster, beam search over
top-3 candidates per macro (3¹⁰=59049 raw combos), pruned by
**LP-relaxation lower bound** on each branch. The LP-HPWL component
gives a globally-aware lower bound that prevents the IBM-overfit
failure mode of E42 K=4 / E43 longer-K-joint / E44 spatial.

*Why this matters:* All current K-joint failures share a structural
flaw — the heuristic that picks K-tuples and prunes branches uses
**local benchmark structure** (adjacency density, congestion peaks,
spatial neighbors), and that local structure differs between IBM
and NG45 ariane133. An LP relaxation is **benchmark-blind** — the
same LP works on any netlist; only the bound's tightness varies.

*Implementation notes:* Requires reconstructing the polyhedra LP
infrastructure (HiGHS solver + assignment extraction + dual
extraction + cascade overlap repair) deleted in cleanup commit
44efd16. See `writeup/archive/submissions/cd_lns_placer.py` for
historic reference. Estimated 2-3 days dev.

*Status:* **scaffolded as E64 manifest; not yet implemented.**

### 4.6.B — Spectral / quadratic init as orthogonal basin seed (E63 — tonight)

The netlist hypergraph's graph Laplacian L = D − A has eigenvectors
encoding global connectivity modes. Use top-k smallest non-trivial
eigenvectors as (x, y) coordinates for macros (Gordian-style quadratic
placement). Provably orthogonal to SDF (analytical density spread) and
DPO (gradient-descent topology) basins.

*Why this matters:* Spectral coords encode the netlist's global
graph structure, which is invariant under any benchmark (IBM or NG45).
The basin landed by CD on a spectral init reflects netlist topology,
not local density patterns. NG45 ariane133's sparser layout would
produce a different spectral embedding but the *mechanism* (eigenvector
init) is benchmark-blind.

*Implementation notes:* `scipy.sparse.linalg.eigsh` for top-k
eigenvectors of the netlist Laplacian. Construction: each net of
≥2 pins contributes 1/(|pins|-1) edge weight to each pin pair (clique
expansion with normalization). Estimated 4 hours dev + 2-3 hr --fast.

*Status:* **scaffolding + launching tonight as E63.**

### 4.6.C — Population-based search with topology-distance diversity penalty (future)

Maintain N placements (population), evolve via crossover + mutation,
add a **topology-distance penalty** that pushes population apart in
L/R/A/B-assignment space. Topology distance = Hamming distance over
pairwise relations.

*Why this matters:* Forces explicit exploration of MANY topologies,
not local polish of one. Crossover with proper LP feasibility
projection (E61's missing piece) ensures legal offspring.

*Implementation notes:* Multi-day dev. Reuses E61 GA crossover
infrastructure but adds (a) topology-distance metric, (b) diversity
penalty in selection, (c) LP-projection repair for crossover.

*Status:* **roadmap-only; future research direction.**

### 4.6.D — Score-based diffusion sampling of basins (future, paper)

Train a small score network on the ~60 converged placements from this
project's experiment_log. Sample from the learned distribution to get
diverse globally-consistent placements; refine each via CD-LNS-SA.

*Why this matters:* The model learns the project's empirical basin
landscape directly, no hand-designed mechanism. Multi-week dev,
high-variance, worth a paper independent of competition lift.

*Status:* **roadmap-only; post-deadline research.**

### 4.6 Pre-deadline execution plan (REVISED 2026-05-03 07:35)

1. **DECIDE on E61_v2 promotion** (today, ~30 min review):
   - Standalone --all 1.08083 vs E48 1.08151 = −0.07 % (marginal,
     within noise — fails the −0.30 % promotion threshold).
   - --ng45 0.6908 (−0.20 %), **ariane133 0.6760 (−1.47 % LIFT)** —
     first NG45-positive mechanism since E18.
   - best-of-{E48, E61_v2} = 1.08025 (−0.12 % over E48 standalone);
     best-of-3 with E53m = 1.07995 (−0.14 %).
   - Recommendation: ADR-012 *Proposed* either way; promotion question
     is whether NG45 lift + per-bench hybrid contribution outweighs
     the marginal --all standalone delta.
   - If promote: extend `cd_lns_sa_hybrid/` to 3-lane
     {E25 SDF, E41 DPO, E61_v2 spatial-block crossover}, sharing the
     E25/E41 internal placers between lanes (no redundant work).
2. **Finish E63 legalization** (~2-4 hr): row-pack with fixed-region
   subtraction. The V3 row-packing algorithm is correct for movable
   macros; need to (a) compute per-row x-span available given fixed
   macros' bboxes intersecting that row, (b) skip x-positions inside
   fixed bboxes during pack, OR (c) post-pack greedy displacement of
   movables overlapping fixed macros to nearest free spot. Manifest
   at `experiments/E63_spectral_init/manifest.md` has full V2/V3
   diagnosis.
3. **E64 LP-bounded beam K-joint** — still scaffolded at
   `experiments/E64_lp_beam_kjoint/manifest.md`; implementation gates
   unchanged (reconstruct polyhedra LP from
   `writeup/archive/submissions/cd_lns_placer.py`; ~2-3 days dev).
   Now lower priority than E61_v2 promotion + E63 legalization fix
   given E65's wall finding suggests local-move escapes are blocked.
4. **POST-DEADLINE research:** Tier 0a structural reframings (BP /
   tensor-networks, multigrid, symmetry quotient, diffusion sampling)
   per `memory/tier_0a_structural_reframings.md`; specifically the
   directions E65 manifest suggests still work — (i) constructing
   structurally new third basin from scratch via BP / diffusion;
   (ii) K=50 Hungarian re-pack non-local moves; (iii) constrained
   NEB on the feasible manifold (active-set / barrier methods on
   the no-overlap region).

**Overnight lessons (2026-05-02 → 03):**
- **The infeasibility wall is the dominant structural feature** of
  the SDF↔DPO basin pair. Linear interpolation between basins doesn't
  legalize, regardless of proxy gap. Bridging mechanisms must operate
  at granularity coarser than per-macro (E61_v2 spatial blocks work)
  or use non-local moves that don't pass through the wall (NEB
  constrained to feasible manifold, K=50 Hungarian re-pack).
- **Spatial-block crossover threads the needle** but only delivers
  marginal --all lift (sample-size effect on small benches collapses
  at scale). Real value is on tied-parent benches (ibm12/14/15)
  where neither parent dominates and selective swap finds new basin.
- **Spectral init has correct algorithm but stumbles on legalization**.
  The 1980s-era spectral placement papers used custom legalizers we
  haven't ported. V3 row-pack with fixed-region awareness is the
  shortest path to a working spectral basin lane.
- **Placement legalization is harder than placement optimization** —
  this is the meta-lesson from E61 V1, E62 WillSeed, E63 V2/V3, and
  E65 NEB. Any new mechanism producing invalid intermediate
  placements needs first-class legalization design, not
  `project_overlaps` afterthought.

**Hard constraint (still):** every promotion candidate MUST verify on
NG45 ariane133. E61_v2 is the FIRST mechanism since E18 to clear that
bar with positive lift; E42/E43/E44/E54/E62 all failed it.

---

## 4.6-OLD Path forward — escape the IBM/NG45 transfer-failure class (PRE-2026-05-02 16:10)

[Content below preserved for reference; superseded by §4.6 above.]

### 4.6.1 Tier A — non-DPO inits (PARTIALLY FALSIFIED 2026-05-02)

Original hypothesis: a third independent basin (Will's seed, RePlAce
output, greedy) would extend E48's best-of-2 monotonically. The first
test refuted the hypothesis as written.

| Init | Standalone | Polished --fast | Status |
|---|---|---|---|
| SDF | ~1.50 | 1.0954 (E25) | E48 lane 1 |
| DPO best_of_v2 | 1.3834 | 1.0848 (E41) | E48 lane 2 |
| **WillSeed v4** | 1.5338 | **0.9335 (E62, +1.44 % vs E48)** | **FALSIFIED 2026-05-02** — loses every fast bench |
| RePlAce | 1.4578 | not tested | requires RePlAce setup to extract placement |
| Greedy | ~2.20 | not tested (low EV given WillSeed failure) | deprioritized |

**Structural lesson from E62**: standalone init quality does not predict
polished basin quality. WillSeed's GPU-legalized starting state lands
CD in a structurally inferior basin to SDF or DPO inits, even though
WillSeed init proxy (1.5338) is between SDF and DPO. The basin
geometry that CD walks toward depends on the init's local-density /
pin-cluster structure, not just its global proxy.

**Remaining Tier A action item**: RePlAce-as-init is the only candidate
plausibly distinct from both SDF (analytical density spread) and DPO
(gradient-descent topology). It requires running RePlAce on each
benchmark to extract placement positions (not just proxy baselines that
are already cached in `REPLACE_BASELINES`). Setup work: 1-2 days of
RePlAce build + per-bench runs + position extraction. EV per WillSeed
result: ≤30 %. **Deprioritize Tier A; redirect to Tier B/C below.**

### 4.6.2 Tier B — GPU replacing CD entirely (architectural)

E53 falsified GPU as a *polish* phase. The complement is GPU *replacing*
CD entirely:

- **GPU-batched CD via conflict-graph coloring (E4 in old roadmap).**
  CD is sequential; macros that don't share a net can update in parallel.
  Color the macro adjacency graph; macros in the same color update
  jointly each sweep. Could 5-20× speedup CD itself, freeing wall budget
  for more LNS / K-joint cycles or more diverse inits.
- **GPU-DPO replaces CD with explicit topology-jumps.** Augment the
  smooth-proxy gradient with discrete topology-flip moves (e.g.,
  net-bbox edge-flip moves). Hybrid gradient-plus-discrete-jumps could
  escape the basin lock that pure smooth-proxy gradient hits.

EV: high variance. Architectural changes risk breaking basin properties
that current pipeline depends on. 1-2 weeks dev each.

### 4.6.3 Tier C — OOD-aware mechanisms (structural defenses)

Since IBM-aware mechanisms break on ariane133, what would a mechanism
that's robust on BOTH look like?

- **Use both IBM and NG45 in development**, not just IBM-fast for
  iteration speed. Each `--fast` test would also gate-check ariane133
  before promotion.
- **Density-targeted destroy** (analog of E54 but targeting the 20 %
  density component instead of 74 % congestion). Density depends less
  on net structure and may be more transferable. Speculative.
- **Adaptive destroy strategy** that switches between cost-aware and
  congestion-aware based on benchmark-local congestion concentration.
  Also speculative; would be the first per-benchmark-adaptive mechanism
  in the codebase (CLAUDE.md says no per-bench tuning, but adaptive on
  measurable benchmark properties is allowed).

### 4.6.4 Recommended order — REVISED 2026-05-02 16:05

1. ~~**NOW (one shift, ~7 hr wall):** add Will's seed as a 3rd basin lane.~~
   **Done 2026-05-02 — FALSIFIED**: E62 --fast 0.9335 (+1.44 % vs E48,
   loses on every fast bench).
2. ~~**NEXT (overnight, ~12-14 hr wall):** add RePlAce output as a 4th
   basin lane.~~ **Deprioritized**: requires RePlAce setup (1-2 days)
   and EV is reduced after E62 failure pattern.
3. **PIVOT — wait for other Claude's E61 GA crossover result.** GA
   crossover *mixes* the SDF and DPO basins from E25/E41 outputs at
   the placement level rather than introducing a third independent
   basin. Different mechanism than Tier A; may succeed where
   independent-basin extension failed. ETA: in flight 2026-05-02 14:24.
4. **IF E61 also fails:** Tier B (GPU-replaces-CD architectural) becomes
   the highest-EV next move. ~1-2 weeks dev; high variance.
5. **PARALLEL** (anytime, low compute): writeup. Falsification record
   now includes E62; structural lesson "standalone init quality doesn't
   predict polished basin quality" added to §8.5.2 (DPO basin) /
   contributions §15.

**Hard constraint:** every promotion candidate MUST verify on NG45
ariane133. The six-experiment pattern (E42/E43/E44/E54/E53m/E62) is the
strongest signal in the project.

---

## 5. Open hypothesis queue (DEPRECATED — pre-E48)

[The hypotheses below were drafted during the E25 (1.0954) era, before
E48 hybrid (1.08151) was verified. They are kept for historical reference
and writeup material but are SUPERSEDED as a forward-looking plan by §4.6
above. Most assume the "E25 frontier" framing that E48 has now closed.]

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
