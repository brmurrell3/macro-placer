---
id: E99
name: tabu_cascade
status: in_progress
parent: E97
created: 2026-05-13
decided: null
champion_at_time: 1.078
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E99: tabu_cascade — Tabu eigvec selection + Lévy magnitudes

## Hypothesis
The cascade saddle escape revisits the same soft mode after a small CD polish
swing, wasting iterations. Maintain a small LRU list of recently-used eigvecs
(default tabu_size=2). When picking the next perturbation direction, score
each of the top-k=tabu_size+1 smallest-algebraic eigvecs by
`eigval + λ·max_overlap_with_tabu` and choose the minimum. This forces the
cascade to explore higher-rank soft modes that single-vec cascade misses.

Composable with E97 Lévy: tabu picks the DIRECTION, Lévy picks the MAGNITUDES.

## Method
Modify the cascade saddle inner loop:
1. Request k=tabu_size+1 smallest-algebraic eigvecs from `find_softest_eigenvectors`.
2. Score by `eigval + overlap_lambda * max(|<v_j, tabu_t>|)`.
3. Pick lowest score → soft AND orthogonal to history.
4. Append chosen direction to tabu list (LRU).
5. Sample ε grid via Lévy (heavy-tailed), iterate over ±signs as in E84/E97.

Defaults: tabu_size=2, overlap_lambda=10.0, K_eps=3, eps_scale=1.0.

## Kill gate
Spike on cached cascade_ibm03.pt (post-E84 plateau, proxy ~0.946). If
Tabu+Lévy gives strictly worse best_proxy than pure Lévy at same wall, kill.
Spike + ibm01/ibm10 H2H needed before integration.

## Generalization check
After spike validation, integrate as
`submissions/cd_lns_sa_cascade_tabu_levy/placer.py`. Compare aggregate on
EPYC --all vs cascade_levy (E97 production).

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Spike code: `code/tabu_cascade.py`
- Parent E97 Lévy: `experiments/E97_levy_saddle/`
- Plateau caches: `experiments/E84_cascading_saddle/results/cascade_ibm0{1,3,10}.pt`
