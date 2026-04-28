---
id: E<NN>
name: <short_snake_case>
status: proposed       # proposed | in_progress | graduated | falsified | superseded | marginal
parent: null           # null or E<NN> of the experiment this descends from
created: 2026-MM-DD
decided: null          # date status changed to a terminal state
champion_at_time: <avg> # avg_proxy of the current champion when this was created
outcome: null          # avg_proxy on --all when this completed (or partial result)
champion_delta: null   # negative = win
graduated_to: null     # path if status=graduated
superseded_by: null    # E<NN> if status=superseded
---

# E<NN>: <short_name>

## Hypothesis
[What we believe; one paragraph.]

## Method
[What changes from the parent / current champion. One paragraph.]

## Kill gate
[Quantitative criterion that triggers status=falsified. Be specific.]

## Generalization check
[Cross-benchmark or NG45 gate before declaring promotion.]

## Outcome (filled when decided)
[Empty until decided. Then: result, decision, rationale.]

## Pointers
- Code: `code/<file>.py` (if local) or removed (`<git ref>`).
- Results: `results/...` (the JSON or partial snapshots).
- Discussion: `writeup/evidence.md` §X.Y; `docs/experiment_index.md`; etc.
