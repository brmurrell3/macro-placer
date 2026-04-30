---
id: E42
name: kjoint_k4
status: falsified
parent: E41
created: 2026-04-30
decided: 2026-04-30
champion_at_time: 1.0990 (E12; E41 1.0848 strongest verified candidate; ADR-010 *Proposed*)
fast_outcome: 0.91859 (--fast); -0.35 % vs E41 fast 0.92178; 3/4 per-bench wins (ibm01 -0.51 %, ibm04 -0.46 %, ibm09 -1.34 %); 1 loss (ibm13 +0.83 %); zero overlaps. K-joint K=4 phase commits 3-48 K-tuples per bench (vs K=3's 9-61) with Δ -0.0006 to -0.0046. Passes regression gate (< E41 +0.5 %) and gen-check threshold (≥ 0.3 % lift over E41) — but barely; the 0.35 % lift is at the noise floor of the DPO-init stochasticity (~1-2 % per bench).
ng45_outcome: 0.6960 (--ng45); **+0.84 % REGRESSION vs E41 ng45 0.69022**. Per-design: ariane133 0.6973 (vs E41 0.6733 = **+3.57 % LOSS**), ariane136 0.6725 (tied), mempool_tile 0.7375 (tied), nvdla 0.6767 (-0.09 %). Catastrophic ariane133 regression KILLS the IBM-marginal lift. K=4 is IBM-fast-overfit; the per-K-tuple cost (5×) starves the budget on benches like ariane133 where K=3 found genuine joint structure but K=4 trades enumeration depth for breadth at the wrong margin.
all_outcome: skipped (falsified by NG45 gate)
outcome: falsified — IBM-overfit; NG45 ariane133 regression +3.57 % vs E41
champion_delta: NG45 +0.84 % regression; --fast +0.35 % marginal lift
graduated_to: null
superseded_by: null
---

# E42: kjoint_k4

## Hypothesis
E39 K=3 K-joint extracts 12-58 commits/bench post-CD-LNS-SA with Δ
−0.0013 to −0.0047 on --fast. E41 (DPO + K=3 K-joint) lifts this further
on benches where the DPO basin opens new K-tuple structure (ibm11 −4.06 %
on the in-flight --all run). The hardest plateau benches (ibm14, ibm15,
ibm17) still tie or near-tie E25 even under E41's joint-move escape.

If those plateaus are *4-coupled* multi-basin floors (4 macros must move
*simultaneously* to escape, no 3-macro joint move improves), then K=4
joint reinsertion is the right next move type. Standard signature: K=3
extracts wins on ibm11 but ties on ibm14/15 → 3-coupled bound is reached
on the easier hard benches but the harder ones need K≥4.

If E42 still ties on ibm14/15/17, K=4 isn't enough either; the right
escalation is K=5 (E43, see backup plan), then full MIQP joint
reinsertion (E29).

## Method
Same pipeline as E41 (DPO best_of_v2 init -> CD plateau -> grid-bin LNS
-> SA-v2 polish -> K-joint LNS -> validate) with **K=4 instead of K=3**
in the K-joint phase:

1-7. Identical to E41 phases 1-6 (DPO init, project_overlaps, CD,
   grid-bin LNS, SA-v2 polish).
8. K-macro joint LNS phase, K=4, top_N=5, kjoint_budget_s=600,
   kjoint_seed=42. Brute-force inner loop is now N^K = 5^4 = 625 combos
   per K-tuple (vs K=3's 125). Per-K-tuple wall scales ~5×; total
   K-tuples-tried in 600 s budget drops ~5× but each K-tuple covers a
   larger reachable set.
9. Validate (zero overlaps), preserve fixed macros, return.

K-joint legality / pairwise-non-overlap eps direction is the post-fix
strict-separation version (eps = 1e-9 in the `dx < min + eps` direction)
plus the post-commit `compute_overlap_metrics` defensive revert
(committed in `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py`,
imported by E41). E42's placer constructs `CDLNSSADPOKJointPlacer` with
`kjoint_K=4` — single-line change relative to E41.

## Kill gate
- **Regression on --fast:** if avg --fast > E41 fast 0.92178 + 0.5 %
  (i.e., > 0.9264) → kill, status=falsified.
- **No incremental lift over E41:** if avg --fast within ±0.05 % of
  E41 fast 0.92178 (sub-noise tie), mark marginal — K=4 doesn't extract
  additional structure on top of E41's K=3 reachable set.

## Generalization check
- If --fast passes the regression gate AND shows ≥ 0.3 % lift over E41
  fast 0.92178 → run --ng45 to verify OOD transfer.
- If --ng45 lift ≥ E41 ng45 0.69022 (no regression on commercial
  designs) → queue --all.
- If --ng45 regression > 0.5 %, K=4 is IBM-overfit; do not queue --all.

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/cd_lns_sa_dpo_kjoint_k4.py` (defines `CDLNSSADPOKJointK4Placer`).
- Parents: E41 (DPO + K=3 K-joint), E39 (K-joint primitive).
- Related: E29 (MIQP joint reinsertion — the principled escalation if
  K=4 also flat).
- Discussion: `docs/experiment_index.md`; `writeup/evidence.md` §9.M
  if graduated.
