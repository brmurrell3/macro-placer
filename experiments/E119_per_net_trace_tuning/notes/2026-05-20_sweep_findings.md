# E119 sweep findings — 2026-05-20

## Sweep result (144 cfgs × {ibm10, ibm12, ibm17}, 45s grid wall)

| | sr | σ | β_mm | β_rg | ibm10 | ibm12 | ibm17 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Default** | 2 | 0.5 | 6 | 4 | +18.9% | +23.7% | +14.4% | 23.7% | 19.0% |
| **Best (3-bench)** | 2 | 0.2 | 16 | 10 | +16.2% | +17.3% | +12.2% | **17.3%** | 15.2% |
| **Best on ibm17** | 2 | 0.2 | 16 | 4 | +15.9% | +17.7% | **+11.6%** | 17.7% | 15.1% |

## Directional learnings

- **σ_cell_frac** is the most-impactful axis. 0.5 → 0.2 reduces gap by
  ~2-4pp on all 3 hard benches. Sharper pin-to-cell soft assignment
  removes mass that the canonical's floor() does not deposit on
  neighbor cells.
- **β_minmax** matters at the second order. 6 → 16 reduces gap by
  ~1-2pp by recovering more of the exact L-route extent. Higher β
  (more saturated sigmoid) approaches the canonical's hard endpoints.
- **β_range_per_cell** is third-order. Default 4 already mostly fine;
  10 adds ~0.3pp lift on the ibm12/ibm17 worst case.
- **smooth_range** = 2 is canonical; 1 sharpens too much, 3 blurs.
  No lift outside that range.

## Status

The kill gate (max abs gap < 15%) is missed by 2.3pp. The lift is
real (6.4pp absolute reduction in worst-case gap) but the structural
limit of the L-route star-routing approximation appears to be reached.

## Downstream test BLOCKED

Two ibm17 V3+CD pipeline runs (`eval_ibm17.py` and
`eval_ibm17_fast.py`) were launched but hit severe M3 CPU contention
(10-17 concurrent python experiments by other agents on the same box).
Both got into V3 attempt 1 but did not finish descent in the time
budget. Final ibm17 lift number deferred.

## Next-step thoughts (if revisited)

1. **3-pin Steiner correction.** Current proxy approximates 3-pin nets
   as star (driver → each sink). Canonical does L/T Steiner. ~10-20%
   of nets are 3-pin in IBM benches; correcting these may close
   another 3-5pp of gap.
2. **Macro footprint partial-overlap edge fix.** Per the proxy
   comment, "Canonical has a partial-overlap fix-up at the edges; for
   the smooth proxy we omit that." This is another structural source
   of mismatch that can't be reached via the 4 sweep axes.
3. **DiffProxyV3 weight on cong.** Currently `cost = wl + 0.5*density
   + 0.5*cong`. A reweight on the cong term may help reach a more
   canonical-aligned basin even with the +12-17% gap.
4. **Re-test under low-contention conditions.** The downstream V3+CD
   eval is the missing data point. When M3 is idle, the 720s pipeline
   completes in ~12 min. Run with the best trace_kwargs and compare.
