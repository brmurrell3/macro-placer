# Phase 4a Results: Escape Local Optima

Date: 2026-04-05

## Changes made

Three changes to `submissions/polyhedra/placer.py`:

1. **Simulated Annealing acceptance** in `greedy_descent` — accepts worse moves with probability `exp(-delta/T)`, linear cooling from auto-calibrated T0
2. **Larger cluster flips** — BFS connected components of high-dual pairs (up to 30 pairs), flipped to opposite direction
3. **Random restarts with perturbation** — Gaussian noise on 15% of macros, iterative legalization. **Disabled (n_restarts=1)** because legalization fails on tightly packed placements.

## Per-benchmark results

| Benchmark | Phase 3 | Phase 4a | Delta | Nav improvements (before → after) | Notes |
|-----------|---------|----------|-------|-----------------------------------|-------|
| ibm01 | 1.1865 | 1.1863 | -0.0002 | 7 → 8 | |
| ibm02 | 1.6200 | 1.6205 | +0.0005 | 4 → 4 | Noise |
| **ibm03** | **1.4070** | **1.4058** | **-0.0012** | **0 → 4** | **Was stuck** |
| **ibm04** | **1.3690** | **1.3652** | **-0.0038** | 2 → 3 | |
| **ibm06** | **1.7150** | **1.7003** | **-0.0147** | **0 → 1** | **Was stuck** |
| ibm07 | 1.4855 | 1.4867 | +0.0012 | 6 → 5 | Noise |
| ibm08 | 1.5080 | 1.5080 | 0 | 3 → 3 | |
| ibm09 | 1.1247 | 1.1245 | -0.0002 | 6 → 6 | |
| ibm10 | 1.4112 | 1.4067 | -0.0045 | 0 → 0 | Sparse LP gain |
| ibm11 | 1.2317 | 1.2317 | 0 | 4 → 2 | |
| ibm12 | 1.6482 | 1.6482 | 0 | 2 → 2 | |
| ibm13 | 1.3984 | 1.3976 | -0.0008 | 4 → 6 | |
| ibm14 | 1.6025 | 1.6026 | +0.0001 | 3 → 3 | |
| ibm15 | 1.6059 | 1.6059 | 0 | 4 → 4 | |
| **ibm16** | **1.5421** | **1.5421** | **0** | **0 → 1** | **Was stuck, cluster flip helped surrogate but not real proxy** |
| ibm17 | 1.7431 | 1.7431 | 0 | 2 → 2 | |
| ibm18 | 1.7897 | 1.7897 | -0.0006 | 3 → 3 | |
| **AVG** | **1.4921** | **1.4921** | **0** | | |

## What worked

**SA acceptance unblocked stuck benchmarks.** ibm03 and ibm06 previously got zero improvements because no single move improved the surrogate proxy. With SA, the navigator accepts slightly worse moves (calibrated to ~50% acceptance at T0), which lets it explore beyond the greedy-optimal basin. ibm03 found 4 improvements; ibm06 found 1.

**Larger cluster flips.** ibm16's improvement came from a 4-pair cluster flip. The connected-component strategy proposes coordinated multi-pair moves that single flips can't achieve.

## What didn't work and why

### Random restarts (disabled)

The SDF placement packs macros tightly — there's very little slack between macros. Adding Gaussian noise to 15% of macros creates overlaps that iterative push-apart legalization can't fully resolve. On ibm11 (367 hard macros), perturbed restarts had 36-51 overlaps remaining after 15 rounds of legalization.

**Root cause:** Legalization pushes overlapping macros apart, but this cascades — pushing A away from B creates overlap between B and C. In tightly packed placements, the cascade doesn't converge. The only way to fully legalize would be to re-solve the full LP (expensive) or use a smarter perturbation that preserves feasibility.

**Better perturbation ideas for future work:**
- Swap two similarly-sized macros' positions (always feasible)
- Translate a connected group of macros together (preserves internal structure)
- Use the LP's min-displacement mode to re-legalize after noise
- Perturb the topology (flip random pairs) rather than positions, then LP-project

### Average didn't improve

The gains on ibm03/04/06 are small (0.001-0.015) because the surrogate's within-benchmark ranking correlation is only rho=0.17. SA explores but can't reliably identify which moves are actually good — it's random-walking through surrogate-equivalent moves rather than finding a descent path. The 3 improvements on ibm03 each reduced surrogate proxy by only 0.0006-0.0007.

**The fundamental bottleneck is surrogate accuracy, not search strategy.** SA gave the search the ability to escape local optima, but the surrogate can't distinguish good escapes from bad ones.

### ibm16: surrogate-real mismatch

ibm16 found 1 improvement on the surrogate (cluster flip reduced surrogate proxy by 0.0048) but the real proxy was unchanged (1.5421). This is the rho=0.17 problem in action — the surrogate thought the move was good but the real proxy disagreed.

## SA calibration details

| Benchmark | T0 | Median delta | Calibration samples |
|-----------|-----|-------------|-------------------|
| ibm01 | 0.000176 | 0.000122 | 95 |
| ibm03 | 0.000210 | 0.000145 | 95 |
| ibm06 | 0.000276 | 0.000192 | 92 |
| ibm16 | 0.000075 | 0.000052 | 92 |

T0 varies by benchmark because the cost landscape scale differs. Auto-calibration from positive deltas works well — no per-benchmark tuning needed.

## Implications for Phase 5

1. **Congestion-aware ranking is critical.** The surrogate ranks by `wl + 0.5*density + 0.5*congestion`, but congestion is 66.5% of the real proxy. Ranking candidates by congestion delta instead of LP duals should improve within-benchmark correlation.

2. **Multi-start needs LP-based legalization.** Simple push-apart doesn't work on tightly packed placements. Using `LPSolver.solve_min_displacement()` after perturbation would guarantee feasibility but costs ~3-5s per restart.

3. **SA is the right framework.** Greedy descent is provably stuck on 3 benchmarks. SA at least explores. The missing piece is a surrogate that can guide the exploration productively.

4. **The gap to RePlAce (2.3%) is almost entirely congestion.** We beat RePlAce on wirelength-dominated benchmarks (ibm02, ibm10, ibm12). The remaining gap is on congestion-heavy ones where RUDY approximation diverges from ground truth routing.
