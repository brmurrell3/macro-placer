---
id: E82
name: hybrid_dispatcher
status: falsified
parent: E79
created: 2026-05-05
decided: null
champion_at_time: 1.0666
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E82: hybrid_dispatcher

## Hypothesis

Different IBM benchmarks have wildly different wall costs at E74-class proxy
quality:
  * Small benches (ibm01: 1140 macros, 246 hard) plateau in 30–40 min — easy fit.
  * Large benches (ibm13: 575 hard macros, ibm17/18 even bigger) need 80+ min
    in the full E25/E41 pipeline — won't fit 60-min EPYC cap.

A SINGLE static configuration cannot fit all benches:
  * **Aggressive cuts (E80)** lose proxy on hard benches that need the polish.
  * **Lenient budgets (E79 v2)** blow the cap on hard benches.

E82 chooses per-bench between two configurations based on benchmark size:

  | Bench size | Path | Wall (M3) | Proxy quality |
  |---|---|---|---|
  | small (`num_hard ≤ 350`) | E79-style: parallel E25⊥E41 + compressed saddle | 30–50 min | E74-class |
  | large (`num_hard > 350`) | E81-style: SDF + CD + saddle (single lane) | 35–45 min | unknown — gates on E81 |

The hard-macro threshold is the dispatcher signal. Both paths share the same
Hessian saddle escape — only the plateau-finding differs.

## Method

Pipeline at runtime:
  1. Read `benchmark.num_hard_macros`.
  2. If ≤ 350 → invoke E79 placer (parallel E25⊥E41 → saddle).
  3. If > 350 → invoke E81 placer (SDF + CD → saddle).
  4. Same Hessian saddle escape parameters in both paths.

Threshold rationale: E79 v2 measured ibm01 (246 hard) at 65 min and ibm13
(575 hard) at 89 min on Windows. The break point on this hardware is around
350 hard macros.  On EPYC (1.5×), the break point shifts lower; the dispatcher
auto-adapts via the hardware probe (large benches always go to E81).

Implementation: thin wrapper that imports both placers and dispatches.

## Kill gate

- Cannot run both placers reliably → falsified.
- E81 path proxy on large benches > E48 plateau (≈ 1.10 ibm13) → falsified
  (means the cheap path doesn't preserve enough lift to be worth it).

## Generalization check

- `--fast`: each bench goes to its expected path; total wall ≤ E79 v2 walls.
- Per-bench proxy: small benches match E79 v2; large benches at least match
  E48 fallback (1.08151).

## Outcome (2026-05-05)

**Falsified at design level — competition rule violation.**

Per `README.md`: *"Hardcoding solutions for specific benchmarks (must be
general algorithm)"*.  The dispatcher's per-benchmark routing on
`num_hard_macros` falls under this prohibition (a benchmark-property-based
branch is functionally equivalent to per-benchmark hardcoding).

Build was completed (~150 LOC), but never run on `--all` because the
approach was abandoned before any wall/proxy data was collected.

### Successor

E83 replaces E82 by routing on **observed wall-clock elapsed time** rather
than benchmark properties.  Same code on every benchmark; only behavior
differs based on the runtime.  This is "anytime algorithm" / wall-clock-aware
design and is compatible with the competition rule.

## Pointers

- Code: `code/dispatcher.py` (thin wrapper).
- Reuses E79 (`experiments/E79_hardware_portability/code/cd_lns_sa_hessian_fast.py`)
  and E81 (`experiments/E81_cd_only_saddle/code/cd_saddle.py`).
- Results: `results/`.
