---
id: E10
name: cong_only_refine
status: falsified
parent: null
created: 2026-04-27
decided: 2026-04-27
champion_at_time: 1.3834
outcome: 1.3788
champion_delta: -0.0046
graduated_to: null
superseded_by: null
---

# E10: cong_only_refine

## Hypothesis
Per E8, congestion accounts for 74.9 % of proxy. After DPO converges,
freezing WL and density and continuing gradient descent on the congestion
term alone should improve proxy by attacking the dominant component
without disturbing the already-good WL/density.

## Method
After DPO best-of-v2 converges, freeze WL and density gradients and run
additional gradient descent against the congestion term only.
Implementations: `submissions/dpo/congestion_refine_placer.py` plus mild
and aggressive sweep variants `_e10_sweep_mild.py` / `_e10_sweep_aggressive.py`.

## Kill gate
Avg `--all` must improve by ≥ 1 % vs DPO best-of-v2 (1.3834 → ≤ 1.3696)
without regressing any benchmark.

## Generalization check
Avg `--fast` improvement must replicate on `--all`. Per-benchmark
regressions on the basin-locked benchmarks (ibm02, ibm12) are veto
conditions.

## Outcome (filled when decided)
**Marginal / falsified.**

- `mild` config: 1.0 % gain on `--fast` (1.1636), only **−0.33 %** on
  `--all` (1.3788).
- `aggressive` configs were worse on `--all`.
- **ibm02 got WORSE +2.3 %** — basin-locked benchmarks regressed.

*Lesson: within-DPO congestion optimization can't overcome RUDY's
directional error (E8 §4.2 / §5.1 confirmed RUDY top-cells overlap with
real top-cells at only 10.9 %, Jaccard 0.057). Amplifying a noisy
gradient on a congestion-dominated objective makes basin-locked
benchmarks worse, not better.*

## Pointers
- Code: removed in post-CD cleanup (commit `44efd16`). Originally at
  `submissions/dpo/congestion_refine_placer.py`,
  `submissions/dpo/_e10_sweep_mild.py`,
  `submissions/dpo/_e10_sweep_aggressive.py`.
- Discussion: `writeup/evidence.md` §9.2; `docs/experiment_index.md`
  (E10 row).
