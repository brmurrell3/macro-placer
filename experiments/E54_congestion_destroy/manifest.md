---
id: E54
name: congestion_destroy
status: falsified
parent: E41
created: 2026-05-01
decided: 2026-05-02
champion_at_time: 1.0990 (E12 verified champion); 1.08151 (E48 hybrid strongest verified candidate); 1.0848 (E41 prior-strongest verified)
fast_outcome: 0.9222 (--fast); tied with E41 0.92178 and E48 0.9202 within run noise. Compositional best-of-{E25,E41,E54} = 0.91708 (-0.31% vs E48), but best-of-{E48, E52 E41-seed1, E53_multiseed, E54} only -0.04% better than E53_multiseed alone — E54 contributes near-zero on top of multiseed.
ng45_outcome: **0.7022 (--ng45); +1.45 % REGRESSION vs E48 NG45 0.6922; ariane133 +5.14 % catastrophic loss (0.7214 vs E48 0.6861, E41 0.6733)**. Same NG45 failure pattern as E42 K=4 and E43 longer-K-joint. ariane136/mempool_tile/nvdla tied or near-tied; ariane133 alone dominates the regression. Congestion-targeted destroy is IBM-aware and NG45-blind on ariane-class designs.
all_outcome: **1.08568 (--all); +0.39 % vs E48 1.08151; +0.40 % vs E53_multiseed 1.08128.** Wall 52334 s = 14.5 hr (under heavy contention). E48 wins per-bench on 7 (ibm01, ibm03, ibm04, ibm06, ibm07, ibm10, ibm17 — biggest gap ibm01 −2.04 %); E54 wins on 3 (ibm02, ibm12, ibm16 by −0.10 to −0.13 %); 7 tied. Best-of-{E48, E54} = 1.08126 (only −0.023 % vs E48 — trivial). Confirms standalone E54 --all is slightly worse than E48 and adds nearly zero as a 3rd hybrid lane. Combined with NG45 +1.45 % regression, E54 is comprehensively falsified.
outcome: **falsified** — IBM-fast lift (ibm04 −1.12 %, ibm13 −0.15 %) does not transfer to NG45 commercial designs. Hybrid extension dead because E54 loses to E48's lane on every NG45 design.
champion_delta: NG45 +1.45 % vs E48; --fast +0.20 % vs E48 (within noise)
graduated_to: null
superseded_by: null
---

# E54: congestion_destroy — congestion-targeted LNS destroy ranking

## Hypothesis
The E8 LP-HPWL diagnostic decomposed proxy as **6 % WL / 20 % density
/ 74 % congestion**. Yet none of our destroy heuristics explicitly
target congestion: LNS-gridbin uses cost-aware destroy (move-to-center
total Δproxy, often dominated by density/WL deltas), K-joint uses
adjacency ranking (pin-pair count, structurally orthogonal to
congestion), SA samples uniformly.

If the dominant component (74 %) is going untargeted by destroy
heuristics, there's slack: a destroy ranking that explicitly identifies
the highest-congestion grid cells and rips out the macros most
contributing to them should find lifts the existing rankings can't
reach — *because* it targets the dominant cost component head-on.

The current pipeline saturates at K-joint's exit (E53 confirmed:
DPO-basin's 350 GPU restarts produced 0 accepts on top of full E41
backbone). The slack must therefore come from a different SEARCH
DIRECTION, not a different basin or a different optimizer. Targeting
congestion-driving macros is the most direct way to inject that
direction.

## Method
Pipeline per benchmark identical to E41 *except* step 5 (LNS-gridbin)
swaps the destroy ranking:

  1. DPO best_of_v2 init (E18, unchanged).
  2. project_overlaps (unchanged).
  3. Build IncrementalProxyEvaluator (unchanged).
  4. CD adaptive (≤ 2400 s, unchanged).
  5. **Grid-bin LNS with CONGESTION-AWARE destroy (≤ 600 s).**
     - Replicate the abu top-5 % cell calculation that drives the
       proxy's `congestion_cost` term.
     - Score each hard movable macro by its total contribution to those
       top cells: direct `macro_cong_contrib` routing + per-net share
       of `net_cong_contrib` (1 / n_pins per net). Both are already
       cached on `IncrementalProxyEvaluator` as part of E1.
     - Destroy the K highest-scoring macros. Reinsert via existing
       grid-bin (col × row) enumeration.
  6. SA-v2 (≤ 600 s, unchanged).
  7. K-joint LNS K=3 top_N=5 (≤ 600 s, unchanged — keeps the existing
     joint-move escape mechanism).
  8. Validate, preserve fixed macros, return.

The destroy step itself is also **dramatically cheaper** than
cost-aware: cost-aware does N proxy evaluations (~50 ms each = 5 s/N),
while congestion-aware reads cached state in O(N × avg_nets_per_macro
× avg_contrib_size) — sub-second for any benchmark. That savings
flows back into more LNS samples within the same 600 s budget, which
itself could be the lift if cost-aware was wall-bound.

## Kill gate
- **--fast > 0.929:** E41 fast 0.92178 + 0.8 % regression. Worse than
  E41 by more than measurement noise → kill.
- **--fast < E41 fast – 0.3 %:** strong signal. Promote to --all
  + --ng45.

## Generalization check
Run --ng45 if --fast passes. The hypothesis applies equally to
commercial designs (whose congestion concentration is, if anything,
*higher* per E48's NG45 numbers). Hold up under NG45 = mechanism
generalizes; regression on NG45 = IBM-specific tuning, falsified.

## Outcome (filled when decided)

### --fast (2026-05-02 00:45, jobs=4)
- **avg --fast: 0.9222** (E41 fast 0.92178; E48 fast 0.9202).
- All VALID, zero overlaps. Total wall 8764 s.
- Per-bench: ibm01=0.9109, ibm04=0.9876, ibm09=0.8419, ibm13=0.9483.

**Standalone vs E48 fast:** essentially tied (+0.20 %), within run-to-run
noise. NOT a clear standalone win.

**Compositional (best-of-{E25, E41, E54} per-bench):**
| Bench | Pick | Value |
|---|---|---|
| ibm01 | E25 | 0.8909 |
| ibm04 | **E54** | **0.9876** (vs E48 0.9988, **−1.12 %**) |
| ibm09 | E41 | 0.8415 |
| ibm13 | **E54** | **0.9483** (vs E48 0.9497, −0.15 %) |
| **avg** | | **0.91708** (vs E48 0.9202, **−0.31 %**) |

E54 wins per-bench on ibm04 and ibm13. Same magnitude lift over E48
(−0.31 %) as E48's lift over E41 (−0.30 %). The mechanism contributes
**orthogonal information** to the existing hybrid lanes.

### Mechanism observation: K-joint compounds with congestion-destroy
K-joint commit counts in E54 are unusually high — 50 commits on
ibm09 (Δ=−0.00289), 41 on ibm13 (Δ=−0.00180), 24 on ibm04 (Δ=−0.00142).
Hypothesis: congestion-aware destroy in step 5 leaves placements where
K-joint finds more correlated multi-macro moves. The two mechanisms are
complementary.

### Mechanism cost
Destroy step itself measured at **0.01–0.03 s wall** in production
(vs ~5 s for `_cost_aware_destroy` per K=12 call). The savings flow
into more LNS samples within 600 s, but in practice reinsert was
already the bottleneck (samples=2-4 per bench, same as cost-aware).

### --ng45 (2026-05-02 02:53, jobs=4) — FALSIFICATION SIGNAL

| Design | E54 | E48 | E41 | Δ vs E48 |
|---|---|---|---|---|
| ariane133 | **0.7214** | 0.6861 | 0.6733 | **+5.14 %** |
| ariane136 | 0.6730 | 0.6685 | 0.6728 | +0.67 % |
| mempool_tile | 0.7375 | 0.7375 | 0.7375 | tied |
| nvdla | 0.6771 | 0.6767 | 0.6767 | tied |
| **avg** | **0.7022** | 0.6922 | 0.69022 | **+1.45 %** |

ariane133 catastrophic regression. Same NG45 transfer failure as E42
K=4 (+3.57 % ariane133) and E43 longer-K-joint (+4.10 % ariane133).
**Congestion-targeted destroy ranking is IBM-tuned: helps on IBM
ICCAD04 layouts but hurts on commercial ariane-class designs where
the dominant congestion cells aren't the right destroy targets.**

K-joint phase on NG45 contributed 11 commits / Δ=−0.0018 on nvdla
(captured the post-LNS-cong-output's slack). On ariane133, the
congestion-aware destroy directed K-joint toward sub-optimal
high-congestion regions where joint moves couldn't recover.

### Decision
- **Hybrid lane status: dead.** E54 loses to E48 on every NG45 design.
  Adding E54 as a 4th lane never helps NG45 (best-of always picks E48
  there) and contributes ~0 on IBM-fast (per best-of-N analysis vs
  E53_multiseed_hybrid). Falsified.
- **E55 ablation status: dead too** — no point testing
  `include_net_share=False` since the parent mechanism is NG45-blind.
- **`--all` status: completing for the falsification record.** The
  result will land but is now low-value; expected ~tied with E48 on
  IBM with same NG45-style regression on benches that resemble
  commercial designs (ariane133-like dense-pin patterns).

### Lesson for the project (paper-relevant)
**Mechanism-aligned destroy ranking is double-edged.** Engaging the
proxy decomposition (E8: 6/20/74 %) to target the dominant component
(congestion) lifted on IBM ICCAD04 layouts but FAILED to generalize
to commercial NG45 designs. The congestion-cell hot-spots on ariane133
are not where the K-joint mechanism can find correlated joint moves;
destroying macros there starves K-joint of structural-coupling
candidates and leaves the placement worse than cost-aware destroy
would have left it.

Joins the falsification record alongside E42 (K=4), E43 (longer
K-joint), E44 (spatial K-tuple), E45 (multigrid), E53 (GPU DPO basin
polish). All are IBM-aware mechanisms that fail NG45 transfer in
the same direction.

## Pointers
- Code: `code/cd_lns_sa_dpo_kjoint_congdestroy.py` (defines
  `CDLNSSADPOKJointCongDestroyPlacer`).
- Parents: E41 (pipeline backbone), E39 (LNS-gridbin reinsert + K-joint
  primitives), E18 (DPO init).
- Diagnostic: `analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md` (E8;
  6/20/74 % decomposition).
- Discussion: motivated 2026-05-01 by E53 negative result (GPU DPO
  basin polish contributed 0 accepts on full E41 backbone) — slack must
  come from search direction not optimizer.
