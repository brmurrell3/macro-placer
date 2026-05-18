---
id: E110
name: lp_dual_congestion
status: in_progress
parent: E84
created: 2026-05-18
decided: null
champion_at_time: 1.05750
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E110: lp_dual_congestion — LP-dual prices as canonical-aligned descent

## Hypothesis

Smooth-RUDY congestion descent has been falsified three times (E88, E95,
E98) by the same failure: smooth RUDY (bbox-uniform demand spread) has
opposite gradient direction from canonical RUDY (per-net trace-route) on
hard benches.

The LP dual of a multi-commodity flow relaxation of the routing problem
on the gcell grid — using the same per-net trace-route topology canonical
proxy uses (`IncrementalProxyEvaluator._net_cong_contrib_flat`) — yields
per-gcell capacity prices π that are valid first-order subgradients of
canonical congestion cost. Plugging π-weighted scoring into LNS destroy
ranking (replacing the move-to-center heuristic `_cost_aware_destroy`)
attacks congestion through a gradient signal that's *aligned* with
canonical by construction, not in spite of it.

## Method

Subset multi-commodity flow LP on top-20% utilized gcells + top-K=2000
nets crossing them. Per-net pattern choice: L1 (H-then-V via sink-col
corner) vs L2 (V-then-H via source-col corner). Constants from canonical
proxy.

Variables (sparse LP):
- `f_n_p ∈ [0, 1]` for each subset net n, pattern p ∈ {L1, L2}
- `s_e ≥ 0` slack/overflow per binding gcell

Constraints:
- ∀n: f_n_L1 + f_n_L2 = 1                                   (dual λ_n)
- ∀ binding gcell e: Σ_n Σ_p f_n_p · w_n_p_e + c_fixed_e ≤ cap_e + s_e   (dual π_e ≥ 0)

Objective: minimize Σ_e s_e.

Solve via `scipy.optimize.linprog(method='highs-ds')`. Cap wall at 5s;
on timeout fall through to `_cost_aware_destroy` (graceful degradation).
Score each hard movable macro by `Σ_net∈nets(m) Σ_e∈route(net) π_e`.
Destroy top-K.

Pipeline integration: monkey-patch `_cost_aware_destroy` in
`submissions/cd_lns_sa/placer.py` from the v2 composer
(`submissions/cd_lns_sa_cascade_v2/placer.py`) when env `MPC_V2_H1=1`.

## Kill gate

1. **Calibration (May 18 EOD)**: on ibm01 cascade plateau, Spearman
   correlation between π-weighted macro scores and measured canonical
   Δproxy across 64 random destroy candidates must be ≥ 0.4.
   - < 0.2 → abandon H1 same day.
   - 0.2–0.4 → tune `top_k_nets` ∈ {1000, 2000, 4000}, `top_edge_frac`
     ∈ {0.10, 0.20, 0.30}; re-test before committing May 19.

2. **`--fast` lift (May 19 EOD)**: `MPC_V2_H1=1` `--fast` aggregate must
   beat re-measured Option C `--fast` baseline by ≥ 0.5%. Below 0.5% =
   drop H1 from v2.

3. **Overlap**: zero overlaps required on all `--fast` benches.

4. **NG45 sanity**: ariane133 must not regress > +1% vs Option C 0.6755.

## Generalization check

- IBM `--all` ≤ 1.046 (-1% vs Option C 1.0575).
- NG45 `--ng45` ≤ 0.690 (within noise of Option C 0.6893).
- ariane133 ≤ 0.685 strict.

## Outcome

[Empty until decided.]

## Pointers

- Code: `code/mcf_lp.py`, `code/lp_destroy_rank.py`, `code/calibrate_duals.py`
- Integration: `submissions/cd_lns_sa_cascade_v2/placer.py`
- Reused: `macro_place/incremental_evaluator.py::_net_cong_contrib_flat` (line 403)
- Parent: E84 cascade saddle
