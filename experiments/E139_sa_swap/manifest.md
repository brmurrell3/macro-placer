---
id: E139
name: sa_swap
status: falsified
parent: thinkorplace-v2
created: 2026-05-21
decided: 2026-05-21
champion_at_time: 0.984    # thinkorplace-v2 M3 --all
outcome: ibm04 smoke: SA-swap zero net lift. Init=best=0.91578 throughout; 38780 better-accepts found but each immediately escaped by subsequent worse-accept. Best-found at t=0.0s. chain-final=0.94982, restored to best=init.
champion_delta: 0.0
graduated_to: null
superseded_by: null
---

# E139: sa_swap

## Hypothesis
CD's plateau is single-axis line-search-stable: every macro is at its
fixed point under axis-aligned moves. Pair-swapping two macros is a
*joint* move type that CD never considers. Adding it as a post-polish
phase opens a strict superset of the reachable neighborhood — and SA
acceptance lets the search climb out of local-swap-pessimum saddles
that the greedy E15 variant got stuck in (E15 falsified at Δ=−0.0011
because greedy swap can't tunnel through swap-cost-ramps).

Mechanism analogous to E12 LNS adding grid-bin destroy to CD: a new
move type. SA acceptance is essential because most pair swaps are
proxy-worse (the basin is already CD-converged) so a greedy filter
would essentially reject everything.

## Method
Pipeline = thinkorplace-v2 (V4+Gaussian descent → CD polish) +
post-polish SA on pair-swap moves. Per swap:
  1. Sample i, j uniformly from movable hard macros
  2. Legality check: would the swap create overlap with any third macro?
     (Bounding-box test excluding i, j themselves.) If illegal, skip.
  3. 2-step `evaluator.move(i, pos_j)` then `evaluator.move(j, pos_i)`.
     The intermediate state has overlap but the evaluator just tracks
     proxy — it doesn't enforce legality. Final state has no overlap
     between i and j because they're at each other's old positions.
  4. Δ = new_proxy − cur_proxy. Accept if Δ < 0 OR uniform < exp(−Δ/T).
  5. Reject: manual 2-step revert (`evaluator.move(j, pos_j)` then
     `evaluator.move(i, pos_i)`) — single-step revert is single-step only,
     so we use move-back, which is bit-identical at this granularity.

Schedule: T_start = 0.01 × current proxy, T_end = 0.0001 × current proxy,
exponential decay over an estimated `max_iters` derived from a brief
"pace" measurement (first 10s tells us swaps/sec, then we scale).

Budget: 300s per bench beyond v2's existing budget. Best-so-far tracking
restores the lowest-proxy placement at exit.

## Kill gate
ibm04 SA-swap final proxy ≥ v2 baseline 0.929 (no lift). If the smoke
test shows lift signal (≤ 0.925), expand to --fast then --all.

## Generalization check
If --fast shows lift, validate on --all with focus on (a) hard benches
(ibm12/14/17/18) where v2 is budget-limited, (b) NG45 ariane133 where
historically structural changes have failed (E54).

## Outcome (filled when decided)
**Falsified 2026-05-21 on ibm04 smoke.**

ibm04 result (M3 MPS, single-bench):
- v2 baseline (CD polish): 0.91578 (proxy_cost reported as 0.92042 because
  final f32 cast and re-evaluation through canonical proxy adds float noise;
  the SA delta is what matters)
- SA-swap best-so-far: **0.91578** (never improved past init)
- Final restored: 0.91578 → final proxy 0.92042 (canonical eval)
- Kill gate: SA final proxy ≥ v2 baseline 0.929. Met (no lift).

SA-swap stats:
- proposed=268331, better=38780 (14.4%), worse=38583, rejProxy=36048,
  rejIlleg=154920 (57.7%)
- 1053 proposals/s on ibm04 M3
- chain-final=0.94982 (above init): the SA never converged to a local
  min; T_end fraction of 0.0001 still admits worse moves and the
  trajectory was diffusing UP from the CD basin
- best_found_at_t=0.0s → not a single accepted move ever pushed below
  the CD-converged proxy

**Diagnosis.** The CD polish basin is *swap-lower-bounded*: any pair swap
strictly raises proxy. With ~268K proposals over 290s and a temperature
that never falls below ~2.6e-4 × init = noise floor, we still see
38780 improvements over CUR — but each improvement is undone by a later
worse-accept before becoming best-so-far. Effectively SA is doing random
walk on the swap manifold above the CD basin; best-so-far never moves.

This is a much harder result than E15's flat Δ=−0.0011: even with the
move type "expansion" Henkelman-style mechanism, after a strong basin
(V4-Gaussian-CD, ~16 % below E15's CD-only basin) the swap topology is
**lower-bounded by CD's per-axis fixed point**. The basin is robust to
joint 2-macro repositioning.

**Implication for the active plan.** Don't try multi-macro joint moves on
the post-v2 basin without a structurally different scoring/acceptance
mechanism. Coordinated jitter alone doesn't escape — the CD fixed point
is tight to pair-swap perturbations as well.

## Pointers
- Code: `code/sa_swap.py`, `code/placer.py`
- Smoke log: `/tmp/e139_smoke.log`
- Prior pair-swap: `experiments/_archive/E15_pair_swap/` (greedy, falsified)
