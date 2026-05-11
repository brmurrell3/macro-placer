---
id: E18
name: dpo_init
status: champion_candidate
parent: E25
created: 2026-04-29
decided: 2026-04-30
champion_at_time: 1.0990 (E12)
fast_outcome: 0.92542 (--fast); -0.88 % vs E25 fast 0.9336; 3/4 per-bench wins; zero overlaps
ng45_outcome: 0.69193 (--ng45 avg of 4 commercial designs); -1.67 % vs E12 NG45 0.7037; 4/4 per-bench wins (ariane133 -3.75 %, ariane136 -1.64 %, mempool_tile -0.85 %, nvdla -0.42 %); zero overlaps. DPO basin transfers strongly to OOD.
outcome: 1.08979 (--all); -0.51 % vs E25 candidate 1.0954; -0.84 % vs E12 champion 1.0990; 11 wins / 6 losses; zero overlaps; awaiting human promotion decision (also see E41 which adds K-joint lift on top of E18).
champion_delta: -0.0092 (-0.84 %) vs E12; -0.0056 (-0.51 %) vs E25
graduated_to: null
superseded_by: null
---

# E18: dpo_init

## Hypothesis
DPO best_of_v2 converges to a basin different from SDF's. Use DPO placement
as init for E25 (CD + LNS + SA-v2). If DPO-init beats E25 on any bench →
multi-basin signal. Per E11, alternative inits sometimes find different
basins, but at risk of worse outcomes on some benches.

## Method
Same as E25 but step 1 SDF init replaced by DPO best_of_v2 placer. After DPO
finishes, run project_overlaps → CD → LNS → SA-v2.

## Kill gate
avg --fast > E25 fast 0.9336 by ≥ 5 % → kill.

## Generalization check
If any --fast bench beats E25 → queue --all.

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/cd_lns_sa_dpo_init.py` (defines `CDLNSSADPOInitPlacer`).
- Parents: E25 (pipeline), E11 (alternative init basin notes).
- Wall: ~7-8 hr --all (DPO ~30 min/bench + E25 pipeline).
