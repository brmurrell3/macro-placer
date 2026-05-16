# E97 Lévy-flight cascade saddle — plan & spike notes

## Spike result (2026-05-13 ibm01)

| Variant | init | best | Δ | candidates | wins | wall |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian (eps=0.3,1,3 × 2 signs) | 0.9135 | 0.9115 | -0.22% | 6 | 2/6 | 195s |
| **Lévy K=6 Cauchy × sign+ only** (budget cut early) | 0.9136 | **0.9105** | **-0.35%** | 3 | 3/3 | 290s |

Lévy's K=6 draws from `|Cauchy(0, σ=1.0)|`:
  [0.149, 0.824, **1.943**, 2.697, 4.428, 26.102]

Key win: eps=1.943 (between Gaussian's fixed 1.0 and 3.0) gave the deepest
saddle. Gaussian's grid misses this magnitude. Hit rate 3/3 with denser
sampling around the typical scale.

## Confounders to verify
- Lévy used K=6 vs Gaussian K=3 → fair K=3 H2H pending (ibm03 in flight).
- 1 bench only → ibm03 ibm10 ibm14 needed.
- Different init seeds → multi-seed verification.

## Production integration plan (if H2H confirms)

Two integration points:

**1. Inline replacement in `experiments/E84_cascading_saddle/code/cascading_saddle.py`:**
Replace fixed `eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0)` arg with
`eps_strategy: str = "levy"` (default keeps gaussian for backcompat). When
`levy`: draw K=3 magnitudes from half-Cauchy(σ=1.0) per iter.

**2. Drop-in via `levy_saddle_escape` function:**
Hybrid placer (`submissions/cd_lns_sa_cascade_dp_multi_lane/placer.py`) imports
`cascading_saddle_escape` directly. Add a sibling `submissions/.../placer_levy.py`
that imports `levy_saddle_escape` instead. Validate side-by-side on `--all`.

## Expected lift

- Per-bench in cascade saddle phase: +0.1 to +0.3% additional
- On hard 4 (cascade-adaptive 1.200): expected new value ~1.193-1.196
- On `--all` 17 (cascade-adaptive 1.078): expected new value ~1.075-1.077

This is a SMALL lift but DROP-IN (no other changes). Should ship.

## Risks
- Cauchy outliers (eps > 10) may cause overlap-projection failures more often.
  Verify with overlap-count tracking per candidate; cap at 50*σ already.
- K=6 doubles cascade saddle wall vs K=3. For 60-min cap, may need K=4 in
  production (eps draws + signs = 8 attempts per iter, vs Gaussian's 6).
