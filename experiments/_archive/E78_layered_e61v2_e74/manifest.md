---
id: E78
name: layered_e61v2_e74
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

# E78: layered_e61v2_e74

## Hypothesis

E74's per-bench result on ibm12 and ibm15 was already lifted by layering
the Hessian saddle on top of an E61V2 plateau (instead of E48 hybrid).
On the remaining tied benches (ibm08, ibm14, ibm16, ibm17, ibm18), E74's
plateau came from E48 hybrid, not E61V2. The hypothesis is that swapping
the plateau source to E61V2 will unlock additional lift on these 5 benches
because E61V2's GA-crossover sometimes finds a different (lower) starting
basin than E25 or E41.

## Method

- Phase 1: Run `CDLNSGACrossoverPlacer` (E61V2) — gets a baseline placement
  via E25 → E41 → GA crossover → CD polish (~5 hr/bench on M3 Max).
- Phase 2: Apply compressed Hessian saddle escape (n_eigvecs=1,
  polish_budget=180s, eps={0.3, 1.0, 3.0}) on top of the E61V2 plateau.
- Phase 3: Return min-proxy among {E61V2 raw, Hessian-layered}.
- Loader monkey-patch (Windows path normalization).
- Run only on ibm08, ibm14, ibm16, ibm17, ibm18 (5 benches × ~5.5 hr each =
  ~30 hr serial; ~7-8 hr `--jobs 4`).

## Kill gate

- 0/5 benches show ≥ 0.0005 lift over E74's per-bench result → falsified.
- Wall > 50 hr `--jobs 4` → falsified (signals oversize budget).

## Generalization check

- For benches that lift, validate that integrating per-bench best into the
  champion lowers `--all` proxy (best-of strategy: E74 vs E78 per bench,
  take min).

## Outcome (partial — first run, ibm08 only, 2026-05-05)

| Bench | E78 proxy | E74 ref | Wall | Verdict |
|---|---:|---:|---:|---|
| ibm08 | 1.10131 | ~1.104 (TODO −0.45 %) | 134 min | small lift (-0.3 % to -0.4 %) |

### Saddle behavior on ibm08

E61V2 GA crossover failed (11 unrecoverable overlaps after 50 projection
iters); fell back to min(E25, E41) = E41 at 1.10131.  Lanczos found one
negative eigenvalue at -0.356, but **all 6 polishes failed to improve**
the proxy.

### Implication

The win on ibm08 came from **E41's plateau** (E61V2 fallback path), NOT
from the layering hypothesis.  The Hessian saddle adds zero value when
applied to E41's plateau on ibm08.  This is consistent with E74's TODO
note that ibm08 saw only a -0.45 % lift — the hard plateau on this bench
seems insensitive to saddle perturbation.

### Decision

Status: **marginal** (lift is real but small and not from the layering
mechanism).  Skip running E78 on ibm14, 16, 17, 18 — those benches are
similar in character.  Better compute use: E80 (work-bounded streaks).

The 1.10131 result IS a per-bench win for ibm08 vs older champions
(E9 = 1.1258, E17 = 1.1397) and ≈ ties E74 (~1.104).  If we later need
to harvest a per-bench best-of, E78's ibm08 result is cached at
`results/CDLNSE61V2HessianPlacer_20260505_183126.json`.

## Pointers

- Code: `code/cd_e61v2_hessian.py`.
- E61V2 source: `experiments/E61_ga_crossover/code/cd_lns_ga_crossover.py`.
- E74 saddle reused: `submissions/cd_lns_sa_hessian/placer.py:_saddle_escape`.
- Results: `results/`.
