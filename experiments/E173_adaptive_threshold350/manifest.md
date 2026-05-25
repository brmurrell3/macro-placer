---
id: E173
name: adaptive_threshold350
status: in_progress
parent: E171
created: 2026-05-21
decided: null
champion_at_time: 0.984
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E173: adaptive_threshold350 — E171 with threshold lowered to 350

## Hypothesis

E171 uses threshold=400 to route benches. Bench size data:
- ibm15 (393): K-joint gave -0.42% (E145) — meaningful lift
- ibm11 (373): untested K-joint
- ibm03 (290): K-joint gave 0% — wasted
- ibm04 (295): untested K-joint

Threshold=400 misses ibm15. Lowering to 350 routes ibm15 → K-joint stack.
ibm11(373) also moves to K-joint stack. Cost: 2 more benches eat
K-joint+SA budget; benefit: capture ibm15's -0.4% lift.

## Method

Identical to E171 except large_bench_threshold=350.

Bench routing change vs E171:
- ibm11 (373): E166 → E143 (gain K-joint, lose portfolio depth)
- ibm15 (393): E166 → E143 (gain K-joint -0.4%, lose portfolio depth)

## Kill gate

ibm15 smoke: final ≤ E145 ibm15 (1.04406) - 0.001 → K-joint adds value
on this bench in E173 context. If final ≥ E145 → portfolio depth was
worth more than K-joint.

## Generalization check

--fast vs E171 --fast for head-to-head.

## Outcome (filled when decided)

(pending smoke)

## Pointers

- Code: `code/placer.py`
- Parent: E171
