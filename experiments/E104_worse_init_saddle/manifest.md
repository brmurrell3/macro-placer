---
id: E104
name: Worse-init saddle probe
status: partial
hypothesis: cascade variant exploration
date_proposed: 2026-05-14
date_decided: 2026-05-16
---

# E104 — Worse-init saddle probe

## Hypothesis

Cascade saddle escape (E84) finds saddles starting from a *good*
plateau (E25 + E41 + best-of plateau pick). The conjecture: a *worse*
init exposes more saddle directions because the plateau lies further
from the local-min landscape, so the smallest Hessian eigenvalue is
more negative (sharper saddle) and the cascade walks further before
re-plateauing.

If true, deliberately initializing from a *random jitter of the
plateau* would expose more saddle escape opportunities than starting
from the plateau itself — a "look around from a worse vantage point"
trick.

## Method

`code/worse_init_saddle.py`:
- Start from cascade's plateau placement.
- Apply Gaussian jitter at σ ∈ {0.02, 0.05, 0.1} × canvas to all
  movable macros.
- Legalize via `extended_legalize` (E71 jitter+project utility).
- Run cascade saddle escape on the jittered placement.
- Compare final proxy to cascade-on-plateau directly.

## Kill gate

Cascade on jittered init lands ≥ +1 % vs cascade on plateau on ibm01.

## Generalization check

If ibm01 lifts, also verify ibm10/12/17 (hard benches where saddle
escape matters most).

## Outcome

**Partial 2026-05-14/15.** Tested σ ∈ {0.02, 0.05} on ibm01:
- σ=0.02: lifts +0.4% (worse, but close to plateau)
- σ=0.05: lifts +1.2% (kill gate hit)
- σ=0.10: legalization fails on 3 of 10 trials.

Direction of effect is wrong: jitter degrades cascade's quality,
doesn't expose more saddles. The plateau placement already sits
near-tangent to multiple saddles per E84; pushing off-plateau just
moves further from all of them.

**Falsified for the original hypothesis.** Did not run on hard
benches because the ibm01 result already failed the kill gate.

**Reusable**: the `extended_legalize` jitter+project utility from
E71 was confirmed to work on cascade outputs. The σ=0.02 placements
(at +0.4% vs plateau) are within the "fragile-third-basin" envelope
documented in E71/E72; these could be reused as seeds for a future
multi-basin ensemble (would compose with E107's periphery-bias init).
