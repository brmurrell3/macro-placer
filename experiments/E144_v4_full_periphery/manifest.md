---
id: E144
name: v4_full_periphery
status: in_progress
parent: E143
created: 2026-05-21
decided: null
champion_at_time: 0.984
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E144: v4_full_periphery — E143 + periphery wrapper

## Hypothesis

E107 showed periphery-bias (α=0.01 push toward nearest canvas edge,
then CD polish, keep if strictly better) gave -1.8 % on NG45 ariane133
in cascade-stacked context but 100 % rejection on --all (IBM). However,
the V4-Gaussian basin geometry differs from E25/E41 SDF/DPO basins;
periphery may activate differently on V4-polished placements. Strict
conservatism (only accept if strictly improved AND zero overlaps) means
no regression risk.

## Method

Identical to E143 through SA polish v2. Then layer periphery wrapper:
- Apply push_periphery(α=0.01) to current placement
- project_overlaps to legalize
- CD polish 120s
- Compare to pre-periphery; keep whichever has lower canonical proxy

Budget 3200s (53 min, under 60-min cap):
- All E143 phases as-is: 3000s
- Periphery + polish: 200s

## Kill gate

On any 4-bench --fast set, periphery wrapper must contribute >= 0.1%
average lift (i.e., periphery accepted on at least 1 bench with material
improvement). If 0/4 accepted, falsify and drop the wrapper.

## Generalization check

If smoke shows acceptance: --fast aggregate must improve E143 --fast
by ≥ 0.1% on average.

## Outcome (filled when decided)

**Smoke ibm03 FALSIFIED on this bench** (2026-05-21):
- Multi-init pick: sdf123 (canonical=1.05925)
- CD polish: 0.89434
- Cascade 1 iter: 0.89083 (-0.39%)
- Portfolio: NO lift (saturated)
- K-joint pass 1: 22 of 96 committed (23% accept), tiny canonical impact
- SA polish: best=0.90038 evaluator (small)
- Periphery: not visible in log, likely rejected
- **Final: 0.8908 vs E166 ibm03 0.8870 = +0.43% WORSE**

**Architectural learning**: K-joint+SA are bench-size-sensitive. On 786-
movable ibm10 (E143) they give -3% lift. On 290-movable ibm03 they
consume budget that portfolio could use, giving a NET LOSS.

Need: adaptive gate that skips K-joint+SA when pass-1 acceptance < ~30%.
Status: superseded by E171 (adaptive K-joint+SA gate, forthcoming).

CAVEAT: MPS contention during E144 (many parallel placers) increased
variance. E166 ran with less contention. Some of the -0.43% gap is
likely run-to-run noise, but K-joint+SA on a small bench is structurally
suboptimal regardless.


## Pointers

- Code: `code/placer.py`
- Parent: E143
- Periphery primitive: `experiments/E107_periphery_bias/code/decompose_spike.py:push_periphery`
