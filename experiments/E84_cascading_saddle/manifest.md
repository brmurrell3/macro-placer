---
id: E84
name: cascading_saddle
status: in_progress
parent: E74 (single-shot Hessian saddle escape)
created: 2026-05-10
decided: null
champion_at_time: 1.0666 (E74 CDLNSSAHessian, ADR-012)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E84: cascading_saddle — iterate Hessian saddle escape

## Hypothesis

E74 found a smooth-proxy saddle at the E48 plateau and stepped ε across
it to land in a deeper basin. **That deeper basin may itself be a saddle
of the smooth proxy** — i.e., it might have its OWN negative smallest-
algebraic eigenvalue. Iterating the saddle escape (compute Hessian on
the new state, find soft mode, step ε, polish, repeat) may compound
lift until we reach a true local minimum (all eigenvalues positive).

E74 single-shot lift on hard benches was small (ibm17 −0.29 %, ibm18
−0.38 %, ibm09 −0.11 %). If those benches have multiple stacked saddles,
cascading should extract the missed depth.

## Method

```
state = E48 hybrid output (initial plateau)
for iter in range(N_max):
    smooth = SmoothProxy(benchmark, plc)
    eigvals, eigvecs = lanczos_smallest(smooth, state, k=1)
    if eigvals[0] >= -eps_min_eigval:
        break  # true local min; no more saddles
    best_proxy_this_iter = proxy(state)
    for sign in [+1, -1]:
        for eps in eps_values:
            candidate = state + sign * eps * eigvec
            project_overlaps(candidate)
            polished = cd_polish(candidate, budget=180s)
            if proxy(polished) < best_proxy_this_iter:
                state = polished
                best_proxy_this_iter = proxy(polished)
    if best_proxy_this_iter >= prev_iter_proxy - 1e-5:
        break  # cascading converged
```

Stop criteria:
- Smallest eigenvalue >= 0 (true local min reached).
- No improvement over previous cascade iteration (saturation).
- Wall budget exhausted (default 60 min total).
- N_max iterations reached (default 5).

Wall per cascade iteration: Lanczos ~10 s + saddle trials ~10 min on
medium bench → ~10–15 min/iter on a 600s polish budget. 3–5 iterations
fits ~45–60 min total.

## Kill gate

- Smoke ibm01 (E74 baseline 0.85527): if cascading lift over E74 < 0.5 % → kill mechanism.
- ibm17 / ibm18 (E74 lifts smallest): if cascading delivers no extra
  lift on benches where E74 was weak → cascading doesn't extract
  stacked saddles on the binding type.
- --fast aggregate: must lift ≥ 0.5 % over E74 best per bench.

## Generalization check

NG45 — must not regress below E74 0.6813.

## Outcome (filled when decided)

[Empty until decided.]

## Pointers

- Code: `code/cascading_saddle.py`, `code/run_cascading.py`.
- Parent: E74 (`submissions/cd_lns_sa_hessian/placer.py`,
  `experiments/E74_hessian_saddle/code/hessian_saddle.py`).
