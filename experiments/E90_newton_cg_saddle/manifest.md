---
id: E90
name: multi_direction_saddle
status: in_progress
parent: E74
created: 2026-05-12
decided: null
champion_at_time: 1.0612          # canonical cascade --all uncapped
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E90: multi-direction saddle escape on cascade plateau (PATH C3)

## Hypothesis
At the cascade-converged plateau, the smooth-proxy Hessian has **multiple
negative-curvature eigenvectors** (PATH A's diagnostic on ibm01:
`λ = [-0.14, -0.08, -0.07, -0.06]` for k=4 — 4 negative directions).
E74's saddle escape perturbs along **one eigenvector at a time** (`±ε · v_k`
for k=1..K). It cannot reach configurations that require **simultaneous
displacement along multiple soft modes** — e.g., `+ε·v_1 + ε·v_2 − ε·v_3`.

Iterated cascading saddle escape (E84) compounds single-direction escapes
but is greedy: it only follows the most-negative direction at each step and
never combines them. C2's ILP falsification (lift saturates at +0.009 %)
confirms small joint moves in real space don't recover this gap — the
geometry of the multiple soft modes must be jointly exploited in *Hessian*
coordinates.

C3 hypothesis: simultaneously stepping along combinations of negative-curvature eigvecs (sign-permuted) finds basins that single-direction escape misses.

## Method
1. From the cached cascade plateau (e.g. `experiments/E84_cascading_saddle/
   results/cascade_ibm01.pt`, canon 0.84528, 0 ovl), build the E74
   `SmoothProxy`.
2. Find the top K (= 4) softest eigenvectors via E74's
   `find_softest_eigenvectors`. (Reuses existing infrastructure — no edits
   to E74 code.)
3. For each non-zero sign permutation `s ∈ {-1, 0, +1}^K \ {0}^K`, build
   `v_combo = Σ_k s_k · v_k / ||Σ_k s_k · v_k||`. Skip duplicates
   (s and -s produce equivalent search via E74's sign sweep).
4. For each ε in `epsilon_values`, perturb cascade plateau by ε · v_combo,
   project overlaps via `project_overlaps`, run CD polish, eval canonical.
5. Track best-result; compare to E74's single-direction-only baseline.

**Spike gate:** any multi-direction (rank > 1) combination finds a polished
proxy lower than the best single-direction polished proxy on ibm01 cached.
If yes, C3 has value; scale to --fast then --all.

## Kill gate
**Kill C3 if the best multi-direction polished proxy on ibm01 cached is
NOT lower than the best single-direction polished proxy.** This is the
cleanest possible test: same infrastructure, same eigvecs, same
project+polish pipeline — the only difference is whether we combine
eigvecs. If combining doesn't help, the single-direction approach E74
already does is sufficient and the multi-direction extension is dead.

## Generalization check
If ibm01 passes the spike, repeat on ibm04 (mid-hard) and ibm10 (cascade
hit wall on 8/17 benches — ibm10's cached output is less converged and
may have more saddle headroom).

If both pass, the next step is a wall-safe integration: replace E84's
single-direction inner loop with multi-direction (capped attempt count
under the per-bench time budget).

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/multi_saddle.py`, `code/spike_ibm01.py`
- E74 SmoothProxy + find_softest_eigenvectors:
  `experiments/E74_hessian_saddle/code/hessian_saddle.py` (read only —
   no edits).
- E84 cascading driver: `experiments/E84_cascading_saddle/code/cascading_saddle.py`
- Cached input: `experiments/E84_cascading_saddle/results/cascade_ibm01.pt`
  (canon 0.84528, 0 overlaps).
- PATH A diagnostic: ibm01 plateau k=4 eigvals
  `[-0.1406, -0.0758, -0.0657, -0.0605]` (memory: `path_a_session_status_2026_05_12.md`).
- TODO.md §PATH C C3 (lines 150-163).
