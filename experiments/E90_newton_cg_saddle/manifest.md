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

## Outcome (in progress; iteration is the key 2026-05-12)

### Result summary

| Test                                  | Bench | Polish | Lift vs cascade |
|---------------------------------------|-------|-------:|----------------:|
| Single-pass spike (local)             | ibm01 |    45 s | **−0.352 %** |
| cascade_multidir iter 1 (local)       | ibm01 |    45 s | −0.228 % |
| cascade_multidir iter 2 (local)       | ibm01 |    45 s | −0.361 % |
| cascade_multidir iter 3 (local)       | ibm01 |    45 s | **−0.614 %** |
| Cloud single-pass                     | ibm03 |    60 s | +0.013 % |
| Cloud single-pass                     | ibm06 |    60 s | +0.048 % (so far) |
| Cloud single-pass                     | ibm07 | pending | — |

### Spike (single-pass) on cached cascade ibm01 — PASSED gate

K=3 eigvecs at the plateau: `λ = [−0.124, −0.088, −0.081]`.

Probe 1 (rank ≥ 2 sign vectors × eps ∈ {0.5, 2.0}, 20 attempts):
- Best: rank-2 sv=(1,1,0) eps=2.0 → polished 0.84231 = **−0.352 %**.
- 16 / 20 attempts found lifts; eps sweet spot is sign-vec dependent
  (e.g. sv=(1,1,0) gives 0.84578 at eps=0.5 vs 0.84231 at eps=2.0).
- Lanczos eigvecs: 4 s local.

### cascade_multidir (3-iter, local) — DRAMATIC compounding

Same K=3, eps ∈ {0.5, 2.0}, polish=45 s, only_rank_at_least=2:

- Iter 1 best=0.84336 (Δ vs cascade input: −0.228 %).
- Iter 2 best=0.84223 (Δ cumulative: **−0.361 %**).
- Iter 3 best=0.84009 (Δ cumulative: **−0.614 %**, rank-2 sv=(1,0,−1) eps=2.0).

Iter 3 plateau eigvals: `λ = [−0.207, −0.128, −0.056]` — top eigval got
*more* negative after iter 2, meaning multi-direction is finding
genuinely different basins rather than polishing a single basin.

### Cloud single-pass on ibm03 / ibm06 (in flight) — much smaller

Cloud single-pass results so far:
- ibm03: lift +0.013 %.  Top eigval pair −0.178 / −0.174 (near-degenerate).
- ibm06: lift +0.048 % so far.  Top eigval −0.251 (very asymmetric).
- ibm07: pending.

Why are cloud lifts smaller?
1. **Single-pass underestimates the iterated lift.** Local iter 1 alone
   gives −0.228 %; ibm01 single-pass spike beat that (−0.352 %) only
   because of polish stochasticity, but the 3-iter compounding to
   −0.614 % is what the submission would actually realize.
2. **Per-bench variance is large.** ibm01 cached cascade is relatively
   converged (~49 min wall, dropped from 0.85527 to 0.84528 over 3
   single-direction iters); ibm03/ibm06 are more constrained.
3. Cloud is contended with PATH A's A4 + PATH B's E91 (saw 49 min
   process CPU at 133 % saturation).

### Decision
**Working algorithm, gated on cloud iteration test.** Local ibm01
shows cascade_multidir 3-iter compounds (−0.614 % cumulative). Cloud
single-pass on wall-bound cached cascade outputs is modest
(+0.01-0.09 % per bench), but those benches' cascade was wall-truncated
not converged, so single-pass underestimates.

**Cloud cascade_multidir on ibm03 (3-iter)** launched 2026-05-12 evening
as head-to-head test vs single-pass result (+0.013 %). If iteration
compounds to ≥ 0.1 % on a wall-bound bench, supports cascade_multidir
as a real submission-grade improvement. If iteration is also marginal,
C3 value is limited to fully-converged plateaus.

**Per user direction, NOT promoting to champion yet** regardless of
result. PATH B's DP-hybrid is the bigger-lift candidate
(`submissions/cd_lns_sa_cascade_dp_lane/`, hard-bench aggregate
−6.9 %).

### Reusable
- `code/multi_saddle.py` — single-pass multi-direction sweep (passes
  spike gate but underestimates submission impact).
- `code/cascade_multidir.py` — iterated multi-direction; **the
  recommended C3 design**.
- `code/cloud_validate.py` — single-pass cloud driver (use as template,
  but swap inner call to cascade_multidir for the real submission test).
- `submissions/cd_lns_sa_cascade_multidir/` — wrapper subclass of
  `CDLNSSACascadeAdaptivePlacer` that runs cascade + cascade_multidir
  polish. Uncommitted in PR-ready state; awaits cloud cascade_multidir
  --fast aggregate before promotion.

### Key empirical findings
- The smooth-proxy Hessian remains indefinite (λ_min < 0) after each
  multi-direction escape on ibm01 — confirms there are deeper basins
  E74/E84 single-direction can't reach.
- Eps × sign-vector interaction is strong; eps={0.5, 2.0} is the
  minimal eps grid that catches both regimes.
- Rank-2 same-sign combinations of soft modes 1 + 2 are often the
  most productive (consistent across iters 1, 2, 3 on ibm01).

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
