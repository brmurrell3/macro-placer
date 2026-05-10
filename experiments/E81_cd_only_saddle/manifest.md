---
id: E81
name: cd_only_saddle
status: falsified
parent: E74
created: 2026-05-05
decided: null
champion_at_time: 1.0666
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E81: cd_only_saddle

## Hypothesis

E79/E80 demonstrate that the §Derisk wall target (~50 min/bench) is unreachable
when E25/E41's full pipeline (CD + LNS + SA [+ K-joint]) is the plateau source —
each lane consumes ~50–70 min on hard benches even with parallelism, streaks,
and dropped K-joint.

**Hypothesis: the proxy lift in E74 comes primarily from the Hessian negative-
eigenvalue saddle escape, NOT from the deep LNS/SA polish on the plateau.**

If true, we can replace the expensive E25/E41 pipeline with a much cheaper
"SDF init + project_overlaps + CD adaptive" plateau (~20–30 min) and apply the
saddle escape on top (~18 min). Total wall ~40–50 min/bench on M3 baseline,
~60–75 min on AMD EPYC — **fits the 60-min cap on most benches**.

The risk: a CD-only plateau may be too shallow for the saddle escape to find
useful negative eigenvalues, or may lie in a different basin where the saddle
direction doesn't lift.

## Method

Pipeline per benchmark:
  1. SDF init (≈ 10 s).
  2. project_overlaps (≈ 5 s).
  3. CD adaptive (cap 1800 s = 30 min, plateau threshold 0.001).
  4. Hessian saddle escape (k=1, ε ∈ {0.3, 1.0, 3.0}, polish 180 s).
  5. Validate zero overlaps; preserve fixed macros.

Hardware probe scales CD cap and polish budget; floor at 1.0 (per E79 v2 fix).

Reuses:
  * `_saddle_escape` from `submissions/cd_lns_sa_hessian/placer.py` (E74).
  * Loader monkey-patch for Windows path normalization.

NO LNS, NO SA, NO K-joint, NO E41/DPO. Just CD + saddle.

## Kill gate

- proxy on ibm01 worse than E48 ibm01 (≈ 0.91 baseline) → falsified
  (means saddle didn't help on the cheap plateau).
- proxy on ibm01 worse than E79 v1 ibm01 (0.882) → marginal: saddle helps
  but plateau cost matters.
- proxy on ibm01 ≤ 0.88 → success: saddle delivers most of the lift even
  on a cheap plateau.

## Generalization check

If smoke succeeds:
- Run --fast (4 benches): avg ≤ 0.95 (vs E79 v2 0.9131) is success.
- Validate wall on ibm13 ≤ 60 min on Windows (probe ~1.0).
- If both, schedule --all and --ng45 runs.

If smoke fails (saddle doesn't lift cheap plateau):
- Status → falsified.
- Conclusion: E74's lift requires the deep E41 plateau.
- Pivot: try DREAMPlace plateau (E76 wave) instead.

## Outcome (2026-05-05)

Smoke on ibm01: proxy **0.91065**, wall 34 min.

* CD plateau (cheap) ended at 0.91189
* Hessian saddle lifted to 0.91065 (Δ = -0.00124)
* Saddle eigvalue: -0.060 (small but real negative)

### Verdict

**Falsified at the kill gate** (proxy 0.91065 ≈ E48 ibm01 baseline 0.91).
The saddle escape does work on a cheap CD-only plateau, but the lift is
~25× smaller than on E25/E41's deep plateau (-0.00124 vs ~-0.030 in E79).

The plateau depth materially affects the saddle's negative eigenvalues
and thus the achievable lift.  The hypothesis "saddle is the lift mechanism
independent of plateau quality" is **not supported**.

### Implication

Cheap CD-only plateau forfeits saddle's value entirely; effectively reduces
to E48 quality.  E81 path is dead as a standalone strategy.  The deep
E25/E41 pipeline is essential for E74-class lift.

## Pointers

- Code: `code/cd_saddle.py`.
- Reference: `submissions/cd_lns_sa_hessian/placer.py:_saddle_escape`.
- Results: `results/`.
