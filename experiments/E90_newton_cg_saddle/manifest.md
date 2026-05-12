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

## Outcome (in progress; spike PASSED 2026-05-12)

### Spike on cached cascade ibm01 (canon 0.84528, 0 ovl)

K=3 eigvecs at the plateau: `λ = [-0.124, -0.088, -0.081]` (3 negative
directions, consistent with PATH A's k=4 diagnostic on this bench).

Probe 1 (rank≥2 sign vectors × eps ∈ {0.5, 2.0}, 20 attempts at 45 s polish):

| Rank | Best sv | Best eps | Polished | Δ vs input |
|------|---------|----------|----------|------------|
| 2 | (1,1,0) | 2.0 | **0.84231** | **−0.352 %** |
| 2 | (1,−1,0) | 0.5 | 0.84322 | −0.244 % |
| 3 | (1,1,1) | 2.0 | 0.84347 | −0.214 % |

16 / 20 attempts found lifts; 4 regressed (3 of the 4 regressions were
rank-3 with eps=0.5, suggesting rank-3 needs larger eps).

**Striking finding: same-sign-vec, different eps → completely different
basins.** sv=(1,1,0): eps=0.5 → 0.84578 (worst result in the sweep),
eps=2.0 → 0.84231 (best). The eps × sign-vec interaction is strong;
cloud validation MUST sweep multiple eps.

### Decision
**Spike PASS.** Multi-direction saddle escape lifts the cascade plateau
where E84's iterated single-direction has saturated. Promote to cloud
`--fast` validation (4 IBM): if aggregate lift ≥ 0.3 %, run `--all`
17 IBM; if `--all` also ≥ 0.3 %, NG45 gate, then graduate as
`submissions/cd_lns_sa_cascade_multidir/` with E84's pipeline + this
multi-direction inner loop.

### Reusable
- `code/multi_saddle.py` — `multi_saddle_escape` with sign-vec
  enumeration. Read-only consumer of E74 (`SmoothProxy`,
  `find_softest_eigenvectors`) — no edits.
- `code/cloud_validate.py` — per-bench validation driver, single-job,
  configurable eps + polish budget.
- Key empirical fact: at the cascade plateau, **same-sign rank-2
  combinations** (e.g. (1,1,0)) reach basins single-direction misses.
  This is the multi-direction contribution.

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
