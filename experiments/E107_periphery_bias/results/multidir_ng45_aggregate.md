# Multidir --ng45 final aggregate — 2026-05-17 04:15 UTC

**Verdict: NEW BEST on NG45.** Multidir 0.6779 beats Lévy, E74, and PATH B.

## 4-bench NG45 results (all 0 overlaps, all VALID)

| Bench | Multidir | M3 Lévy | E74 | PATH B (dp_lane) |
|---|---|---|---|---|
| ariane133 | **0.6512** | 0.65212 | 0.6641 | — |
| ariane136 | **0.6493** | 0.64980 | 0.6518 | — |
| mempool_tile | **0.7353** | 0.73750 | 0.7376 | — |
| nvdla | 0.6758 | 0.67646 | 0.6716 | — |
| **AVG** | **0.6779** | 0.67897 | 0.68128 | 0.68086 |

## Improvements

- vs M3 Lévy: -0.16% (3 wins, 0 losses, 1 close-tie)
- vs E74 champion: -0.50% (3 wins, 1 regression on nvdla)
- vs PATH B verified: -0.43% (aggregate)

nvdla regression vs E74 (+0.62%) is the only weakness, but offset
by the other 3 wins.

## Combined 21-bench picture

| Placer | IBM 17 | NG45 4 | 21-bench unweighted avg |
|---|---|---|---|
| PATH A (cascade-adaptive) | 1.07564 | 0.68102 | 1.00048 |
| **PATH B (dp_lane)** | **1.06650** | 0.68086 | **0.99304** ⭐ |
| Multidir | 1.07415 | **0.6779** | 0.99868 |

PATH B remains the best aggregate Tier-1 entry. But multidir is the
best NG45-only entry and is CPU-only (no GPU dependency unlike PATH B).

## Submission options

**Option 1 (best aggregate, CURRENT TIER-1 PICK):**
- Tier-1: PATH B (1.06650 IBM / 0.68086 NG45 / 0.99304 21-bench)
- Risk: GPU dependency — falls back to PATH A if no DP

**Option 2 (best NG45, CPU-only safety):**
- Tier-1: Multidir (1.07415 IBM / 0.6779 NG45 / 0.99868 21-bench)
- No GPU required; slightly worse IBM but beats PATH B on NG45 by -0.43%

**Option 3 (Tier-2 angle):**
- For OpenROAD WNS/TNS validation, multidir's NG45 -0.43% could matter
- Per TIER2_FINDINGS, ariane133 wants no-tcl regardless; ariane136 wants cascade-tcl
- Multidir's ariane136 0.6493 (best ever) is the right pick for that

## Recommendation

Ship **PATH B** as Tier-1 entry — best aggregate, GPU-conditional fallback.

Use **Multidir output** for Tier-2 NG45 submissions (especially ariane136).

CPU-only Multidir is also worth submitting as Option C if rules allow
multiple placer entries (one with GPU, one without).
