---
id: E48
name: hybrid_e25_e41
status: in_progress
parent: E25, E41
created: 2026-04-30
decided: null
champion_at_time: 1.0990 (E12; E41 1.0848 strongest verified candidate; ADR-010 *Proposed*)
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E48: hybrid_e25_e41

## Hypothesis
Per-bench analysis of E25 (1.0954) and E41 (1.0848) on `--all` shows
E25 wins on 5/17 benches (ibm01, ibm06, ibm07, ibm17, ibm18) while
E41 wins on 12/17 (the hard plateau benches and most others). E18
(1.08979) is dominated per-bench by max(E25, E41) — never wins.

The math says **best-of-{E25, E41} per-bench gives avg 1.08121** —
a **−0.33 % improvement over E41** and **−1.62 % vs E12**, just by
picking the right pipeline per bench.

The wins concentrate by structural type:
- E41 wins on benches where DPO basin opens K-tuple structure.
- E25 wins on benches where SDF basin's SA-v2 phase finds wins
  (ibm01 −1.58 % vs E12 in E25 baseline; DPO disturbs that lift).
- ibm17, ibm18 are E41's biggest losses (DPO basin is GLOBALLY
  worse on those).

A hybrid placer that runs both pipelines and returns the lower-cost
output realizes this bound. It's not per-benchmark TUNING (forbidden)
because the SAME algorithm runs on every benchmark — it just always
runs both pipelines and picks per-bench best. Algorithmically valid.

## Method
Per-benchmark pipeline:

1. Run **E25 pipeline** (CDLNSSAPlacer): SDF init → CD plateau → grid-bin
   LNS → SA-v2 polish → validate. ~30-50 min/bench.
2. Run **E41 pipeline** (CDLNSSADPOKJointPlacer): DPO best_of_v2 init →
   project_overlaps → CD plateau → grid-bin LNS → SA-v2 polish →
   K-joint LNS → validate. ~50-70 min/bench.
3. Compare `compute_proxy_cost` of both outputs.
4. Return the lower-cost placement (with overlap == 0 verified).

All hyperparameters from each pipeline preserved unchanged. No
per-benchmark tuning — the per-bench winner is determined by the
PROXY VALUE, not by hardcoded bench-name logic.

Wall budget: per-bench wall = E25 + E41 walls = ~80-120 min/bench.
Total `--all` wall: ~22-26 hr serial. **Exceeds 17-hr competition cap
on serial execution.** Mitigation: run two pipelines in PARALLEL
(2 cores per bench), reducing wall to max(E25, E41) ≈ E41 wall ≈
50-60 min/bench × 17 = 14-17 hr. Tight but within envelope.

## Kill gate
- **Regression on --fast:** if avg `--fast` > E41 fast 0.92178 → kill
  (something broke; theoretical bound says 0.91652 should be reachable).
- **No realization of bound:** if avg `--fast` > E41 fast − 0.3 %
  (i.e., > 0.9190), the hybrid composition isn't producing the
  per-bench best — investigate (likely a bug in proxy comparison or
  placement reconstruction).

## Generalization check
Run --ng45. The hypothesis is that hybrid > E41 on NG45 too because
ariane133 favors E41 and other designs may favor E25. If hybrid lifts
≥ E41 NG45 0.69022, queue --all.

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/cd_lns_sa_hybrid.py` (defines `CDLNSSAHybridPlacer`).
- Parents: E25 (`submissions/cd_lns_sa/placer.py`), E41
  (`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`).
- Discussion: theoretical bound computation in commit message.
