---
id: E165
name: v4_sdf_dpo_multi_cascade
status: in_progress
parent: E164
created: 2026-05-21
decided: null
champion_at_time: 0.984
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E165: v4_sdf_dpo_multi_cascade — add DPO init lane to E164's multi-seed+cascade

## Hypothesis

E164's multi-seed search (seeds {42, 123, 999}, init="sdf") explores RNG
diversity within the SDF-init basin family. Real basin diversity comes
from initialization, not just RNG. E18 / E41 history shows DPO init
finds structurally different basins than SDF — sometimes much better on
hard / commercial benches (E18: 4/4 NG45 wins as init for E25).

V4-Gaussian descent has never been run from DPO init alongside SDF init
as a basin-pick lane. Adding one DPO seed to the multi-seed phase gives
us SDF×3 + DPO×1 = 4 candidate basins. The DPO lane should dominate
on benches where SDF basin is locally bad.

## Method

Multi-init multi-seed:
- Lane A: V4-Gaussian init="sdf" seed=42
- Lane B: V4-Gaussian init="sdf" seed=123
- Lane C: V4-Gaussian init="sdf" seed=999
- Lane D: V4-Gaussian init="dpo" seed=42 (E18 `_best_of_v2_init`)

Each lane: descent + greedy legalize (no CD polish). Pick lowest-proxy
basin. CD polish picked basin. Cascade saddle escape on polished result.

Budget (2400s default — extended from E164's 1500s since 4 lanes):
- 4 × V4 descent + legalize: ~1000s (250s each on hard bench)
- CD polish on winner: 700s
- Cascade saddle escape: 650s (max_iters=3, eps=(0.5, 1.5, 3.0), polish_budget=130s)
- 50s safety margin

Wall fits under 60-min/bench cap (2400s = 40 min).

Skip remaining lanes if budget runs tight; falls back to whatever
basins succeeded.

## Kill gate

ibm03 smoke: final proxy >= E164 ibm03 final (0.88874) - 0.001 (no
compounding lift over E164) → kill the DPO-lane hypothesis. Lane D was
wasted compute.

If smoke shows compounding (≤ 0.888), promote to --fast.

## Generalization check

--fast aggregate: must beat E164 --fast 0.8384 by ≥ 0.3% AND no
single-bench regression > 0.5%. If passes, run --all + --ng45.

## Outcome (filled when decided)

**Smoke ibm03 FALSIFIED on DPO lane** (2026-05-21):
- Per-lane basin (pre-CD): sdf123:1.06336 (WIN), sdf999:1.07209, sdf42:1.07315, dpo42:1.09836
- PICK: sdf123 (DPO lost by 3.3%)
- CD polish 700s: proxy=0.89649
- Cascade 3 iters: 0.89280 (-0.41%)
- **Final: 0.89280** (ovl=0), wall=1495s

vs E164 ibm03 final: 0.88874 → E165 is +0.46% WORSE.
Kill gate triggered (E164 final - 0.001 = 0.88774; E165 0.89280 ≥ 0.88774).

**DPO lane killed on ibm03**. The DPO basin's higher canonical (1.098 vs
1.063) translated to a worse polished basin. DPO init for V4-Gaussian
does not produce the basin diversity seen in E25/E41 (where DPO basin
was structurally different from SDF in unweighted-WL terms). V4's
gradient descent washes out the init difference.

CAVEAT: M3 MPS contention nondeterminism — same seed=123 SDF lane gave
basin 1.07484 in E164 vs 1.06336 here. Some of the gap is run-variance.
Re-run under no parallel load might narrow gap.

Status: falsified on ibm03 for V4-basin context. DPO lane should be
dropped from E166/E143/E144/E145 to save budget for cascade/portfolio
phases.


## Pointers

- Code: `code/placer.py`
- Parent: E164 (`experiments/E164_v4_multiseed_cascade/`)
- DPO init basis: E18 (`experiments/E18_dpo_init/`)
- V4 production: `submissions/thinkorplace-v2/placer.py` (read-only)
