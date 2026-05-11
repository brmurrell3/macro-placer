---
id: E85
name: multi_init_ensemble
status: in_progress
parent: E74 (single-init Hessian saddle escape)
created: 2026-05-10
decided: null
champion_at_time: 1.0666 (E74 CDLNSSAHessian, ADR-012)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E85: multi_init_ensemble — Hessian saddle escape from N distinct inits

## Hypothesis

E74 starts from a single E48-equivalent plateau (min of fresh E25 + E41).
Different inits land CD-LNS-SA in different basins; saddle escape from
each basin reaches a different deeper minimum. Taking the per-bench
best across an ENSEMBLE of (init → polish → saddle escape) trajectories
should compound over the single-trajectory E74.

Init diversity is the lever. Candidate inits:
- **SDF baseline** (seed=42) — what E25 starts from.
- **SDF jittered** (seeds 1, 7, 1337) — same algorithm, different RNG.
- **DPO best_of_v2** (seed=42) — what E41 uses.
- **DPO jittered** (seeds 1, 7, 1337).
- **Random uniform** — feasible random placement.
- **E61V2 spatial-block crossover output** (for tied-basin benches; we
  have cached for ibm12 / ibm14 / ibm15).

For each init: brief CD polish (300 s budget — establishes a basin) →
Hessian saddle escape with our standard params (k=2, ε={0.3, 1.0, 3.0},
polish 180 s/trial). Take per-bench min over N trajectories.

## Method

```
for bench in IBM_17:
    candidates = []
    inits = [
        ("E25_seed42", run_e25(bench, seed=42)),
        ("E25_seed1", run_e25(bench, seed=1)),
        ("E41_seed42", run_e41(bench, seed=42)),
        ("E41_seed1", run_e41(bench, seed=1)),
        # ...
    ]
    for label, init_placement in inits:
        plateau = cd_polish(init_placement, budget=300s)
        saddle_result = saddle_escape(plateau, ...)
        candidates.append(saddle_result)
    bench_winner = min(candidates, key=proxy)
```

Ensemble size N: 4–8 (budget-constrained). Wall ~4 hr/bench at N=4 on
M3 Max. --all aggregate ~70 hr serial; ~17 hr `--jobs 4`.

## Kill gate

- N=4 smoke on ibm01/04/09/12: ensemble best must lift ≥ 0.5 % over E74
  on at least 2/4 benches.
- --fast aggregate must lift ≥ 0.5 % over E74's per-bench best.
- If ensemble best is consistently the E25_seed42 trajectory (which IS
  E74), the diversity hypothesis is false → kill.

## Generalization check

NG45 — must not regress below E74 0.6813.

## Outcome (filled when decided)

[Empty until decided.]

## Pointers

- Code: `code/multi_init_ensemble.py`, `code/run_ensemble.py`.
- Parent: E74 (`submissions/cd_lns_sa_hessian/placer.py`,
  `experiments/E74_hessian_saddle/code/hessian_saddle.py`).
- Init helpers: `macro_place/sdf_init.py` (SDF),
  `experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py` (DPO).
