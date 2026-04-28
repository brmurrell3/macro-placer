# ADR-004: Ship plateau_threshold=0.005, not the tighter 0.001 from E16

**Status:** Accepted
**Date:** 2026-04-28
**Deciders:** project owner

## Context

ADR-003 established plateau detection with `plateau_threshold=0.005` as
the CDAdaptive default, scoring 1.1055 avg on `--all`. A natural follow-
on was: does a tighter plateau threshold buy more proxy?

E16 (`experiments/E16_tight_threshold/code/cd_adaptive_e16.py`) ran with
`plateau_threshold=0.001` and `hard_cap_s=7200`, producing 1.1025 avg
(-0.0030 vs the 1.1055 champion, -0.27 %) at 25503 s = 7.1 hr total,
+46 % wall vs CDAdaptive. 14 wins, 1 tie (ibm15), 1 small regression
(ibm13 +0.0013). The result is a 0.0030 absolute drop.

The project's strict graduate gate requires >= 0.005 avg drop on `--all`
to promote a champion. E16's drop of 0.0030 is formally marginal — it
does not clear the bar.

The question is whether to ship the better-on-IBM number (1.1025) or
keep the conservative-defaults configuration (1.1055).

## Decision

Champion remains CDAdaptive at `plateau_threshold=0.005`. E16 stays as
a one-datapoint sensitivity result, not a new champion. The writeup
narrative is "we chose the looser threshold to avoid overfitting compute
to IBM," and the wall-budget headroom (4.85 hr of 17 hr available) is
preserved for NG45's potentially-larger designs.

## Consequences

Positive: conservative defaults narrative for the writeup. Wall budget
stays at 4.85 hr for 17 IBM benches, leaving ~12 hr of headroom in the
competition's 17-hr envelope; if NG45 designs are harder than IBM, this
margin matters. Avoids a single-datapoint promotion based on a
formally-marginal drop.

Negative: leaves ~0.27 % proxy on the table on IBM. The decision is
explicitly supersedable: if a future sensitivity sweep at 0.002, 0.005,
0.01 (ideally with NG45 evidence) shows that 0.005 is genuinely too
loose, write a follow-on ADR and promote the tighter setting. The TODO
in evidence.md §7.6 calls this out.

Alternatives ruled out: shipping E16 directly (one datapoint, marginal
drop, +46 % wall — does not meet the graduate gate); raising the gate
to absorb E16 (gate exists for a reason; relaxing it post-hoc to
accommodate one result is the wrong direction); running a full
sensitivity sweep before deciding (would delay other work; the
conservative default is safe in the meantime).

## Evidence

- `writeup/evidence.md` §7.6 (E16: 1.1025 at +46 % wall, marginal under
  graduate gate)
- `experiments/E16_tight_threshold/code/cd_adaptive_e16.py`
