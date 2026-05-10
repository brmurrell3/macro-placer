---
id: E77
name: sharper_hessian
status: marginal
parent: E74
created: 2026-05-05
decided: null
champion_at_time: 1.0666
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E77: sharper_hessian

## Hypothesis

E74 used k=2 smallest-algebraic eigenvectors of the smooth-proxy Hessian
and a coarse 3-point ε grid {0.3, 1.0, 3.0}. On ibm01 the lift came mostly
from `eig0 sign=-1 eps=1.0` and `eig1 sign=+1 eps=3.0` — the right ε
varied per eigvec/sign. On harder benches (ibm12-18) the proxy plateau is
deeper and more directions in the Hessian's null/negative spectrum may
encode meaningful saddle-escape directions that k=2 misses.

E77 sweeps **k=5 eigvecs** and a **finer ε grid** {0.1, 0.3, 1.0, 3.0, 10.0}
on the larger IBM benchmarks (ibm12-18). Expected lift +0.2–0.5 % per
bench where E74 already saw partial wins.

## Method

- Reuse E74's `_saddle_escape` (load from submissions/cd_lns_sa_hessian/placer.py).
- Override hyperparameters: `n_eigvecs=5`, `eps_values=(0.1, 0.3, 1.0, 3.0, 10.0)`,
  `polish_budget=180.0` (kept short to bound 5×2×5 = 50 polish trials).
- Reuse the same parallel E25 ⊥ E41 plumbing as E79.
- Loader monkey-patch (Windows path normalization).
- Per-bench: parallel E25/E41 (~30 min) + Hessian (50 trials × 180s = 150 min)
  → ~3 hr/bench. Total `--all` = 17 × 3 = ~51 hr serial; ~13 hr `--jobs 4`.
  TODO.md projection of 9 hr at `--jobs 4` assumes only running on ibm12-18
  (7 benches × 3 hr / 4 = 5.25 hr — comfortable inside 9 hr).

## Kill gate

- Per-bench proxy on ibm12-18 NOT lower than E74 by ≥ 0.0005 → no signal.
  If 7/7 benches show no improvement → falsified.
- Wall > 24 hr `--jobs 4` on `--all` → falsified (signals oversize budget).

## Generalization check

- For benches that improve, validate the lift is reproducible (re-run once).
- If E77 wins on 3+ ibm12-18 benches, layered with E79 / E74 → integrate
  per-bench best into champion (best-of strategy).

## Outcome (partial — first run, ibm12 only, 2026-05-05)

| Bench | E77 proxy | E74 ref | Wall | Verdict |
|---|---:|---:|---:|---|
| ibm12 | 1.20980 | ~1.211 (TODO −0.61 %) | 161 min | marginal (-0.1 % within noise) |

### Saddle behavior on ibm12

Lanczos found 5 negative eigenvalues: -0.357, -0.208, -0.184, -0.098, -0.089.
**Only 2/50 polish trials improved proxy**, both on eig0 at ε=0.1.  Higher-
order eigvecs (eig1-eig4) and larger ε on eig0 yielded zero NEW BEST polishes.

### Falsification (partial)

The hypothesis "k=5 eigvecs + finer ε grid lifts +0.2-0.5 % on ibm12-18"
is **not supported** on ibm12.  The dominant saddle direction is eig0 with
small ε; extra eigvecs are inert.  k=2 (E74) was already capturing essentially
all the available saddle lift.

Status: **marginal**.  Will not extend to ibm13-18 unless ibm12's −0.1 % is
later judged below noise threshold (likely is).  Compute resources better
spent on E80 (work-bounded streaks) which addresses the wall-cap kill gate.

### Decision

Mark E77 as `marginal` and do not pursue the k=5 / finer-ε direction on
ibm13-18.  The ε=0.1 finding (smaller perturbations than E74's 0.3) may be
worth integrating into E79's eps_values default, but TODO lift is below
the +0.2-0.5 % threshold the manifest set.

## Pointers

- Code: `code/cd_lns_sa_hessian_sharper.py` (reuses E74 saddle escape +
  E79 parallel plumbing).
- Worker: shared with E79 (`experiments/E79_hardware_portability/code/_worker.py`).
- Results: `results/`.
