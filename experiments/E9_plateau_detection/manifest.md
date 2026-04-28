---
id: E9
name: plateau_detection
status: graduated
parent: E2
created: 2026-04-27
decided: 2026-04-27
champion_at_time: 1.1193
outcome: 1.1055
champion_delta: -0.0138
graduated_to: submissions/cd_adaptive/placer.py
superseded_by: E12
---

# E9: plateau_detection

## Hypothesis
CDOnly used a uniform 600 s / benchmark budget. Hard benchmarks (ibm17/18/14/12)
still had Δ ≈ 0.001–0.003 at the 600 s mark — CD wasn't done, the budget
was. Easy benchmarks (ibm04/09/11) plateau within 5–6 minutes, wasting 2–4
of the allotted minutes. A *per-benchmark, per-run* plateau detector
should reallocate that wasted time to the hard benchmarks while still being
a single global policy (no per-bench hyperparameters).

## Method
After every CD sweep, compute the absolute proxy delta. Exit when:

```
wall_clock >= hard_cap_s, OR
(wall_clock >= min_time_s AND last `patience` deltas < plateau_threshold)
```

Defaults that survived NG45-transfer reasoning: `min_time_s = 300`,
`hard_cap_s = 3 600` (matches the contest's per-benchmark cap),
`patience = 3`, `plateau_threshold = 0.005`.

## Kill gate
Avg `--all` must improve over CDOnly (1.1193) and must beat the
leaderboard target (1.1172) to graduate.

## Generalization check
Verify all 17 benchmarks exit cleanly via plateau (none hits the cap) on
the IBM set, so the hard cap has headroom for the (potentially harder)
NG45 designs in Tier 2.

## Outcome (filled when decided)
**Graduated 2026-04-27 to `submissions/cd_adaptive/placer.py`. Was the
champion 2026-04-27 to 2026-04-28; superseded 2026-04-28 by E12 grid-bin
LNS overlay (1.0990) — see `experiments/E12_grid_bin_lns/manifest.md` and
ADR-007. The CDAdaptive runner remains in use as the CD-phase implementation
inside the E12 production placer.**

| Metric | CDOnly | **CDAdaptive (E9)** |
|---|---:|---:|
| Avg `--all` | 1.1193 | **1.1055** |
| Δ vs RePlAce | +23.2 % | **+24.2 %** |
| Δ vs leaderboard 1.1172 | +0.18 % | **−1.05 %** |
| Total wall | 10 316 s | 17 480 s (4.85 hr) |
| Plateau exits | n/a | **17 / 17** |
| Overlaps | 0 / 17 | 0 / 17 |

Hard benchmarks (ibm10/12–18) gained 1.7–3.6 % from extra time. Easy
benchmarks (ibm01–04, 09, 11) tied within ±0.4 %. **All 17 exit via
plateau, none hits the 3 600 s cap** — substantial headroom for NG45.

Beats the public leaderboard (vmallela 1.1172 unverified) by −1.05 %.
Verified score; effectively top-1 on Tier 1.

ADR-003 captures the structural argument: CDAdaptive is plateau-bound, not
budget-bound, on IBM. E16 (tighter threshold) is the natural sensitivity
follow-up; it landed at 1.1025, formally marginal at the strict gate, and
champion was kept at the conservative defaults.

## Pointers
- Code: graduated to `submissions/cd_adaptive/placer.py`.
- Result: `results/CDAdaptivePlacer_20260427_132214.json`.
- Discussion: `writeup/evidence.md` §2.1, §7.5; `docs/experiment_index.md`
  (E9 row); ADR-003.
- Parent: `experiments/E2_cd_ibm10_breakthrough/manifest.md`.
- Sensitivity follow-up: `experiments/E16_tight_threshold/manifest.md`.
