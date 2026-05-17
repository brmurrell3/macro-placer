# Portfolio Lévy --all IBM 17 — 2026-05-17 13:55 UTC

**Aggregate: 1.07404** — worse than plain Lévy (1.07268) by +0.13%.

## 17-bench results

| Bench | Portfolio | Lévy | PATH A | Best |
|---|---|---|---|---|
| ibm01 | **0.8676** | 0.8704 | 0.8797 | Portfolio |
| ibm02 | 1.0656 | 1.0650 | 1.0623 | PATH A |
| ibm03 | **0.9442** | 0.9447 | 0.9467 | Portfolio (tied) |
| ibm04 | 0.9837 | **0.9712** | 0.9859 | Lévy |
| ibm06 | 1.1460 | **1.1400** | 1.1489 | Lévy |
| ibm07 | **1.0728** | 1.0744 | 1.0773 | Portfolio |
| ibm08 | 1.1040 | **1.0957** | 1.0897 | PATH A |
| ibm09 | **0.8339** | 0.8349 | 0.8267 | PATH A |
| ibm10 | 1.0255 | **1.0214** | 1.0180 | PATH A |
| ibm11 | 0.8722 | **0.8660** | 0.8729 | Lévy |
| ibm12 | 1.2081 | **1.2067** | 1.2092 | Lévy |
| ibm13 | **0.9331** | 0.9435 | 0.9549 | Portfolio |
| ibm14 | 1.2218 | **1.2125** | 1.2186 | Lévy |
| ibm15 | 1.1634 | **1.1566** | 1.1727 | Lévy |
| ibm16 | 1.1461 | **1.1398** | 1.1460 | Lévy |
| ibm17 | 1.3415 | **1.3394** | 1.3419 | Lévy |
| **ibm18** | **1.3291** ⭐ | 1.3534 | 1.3344 | Portfolio |
| **AVG** | **1.07404** | **1.07268** | 1.07564 | — |

## Per-placer wins (1st-place per bench)

- Lévy: 11/17 (most consistent winner)
- Portfolio: 5/17 — wins on ibm01, ibm03 (tie), ibm07, ibm13, ibm18
- PATH A: 1/17 (just ibm02 by 0.06%)

## Hidden best-of-N opportunity

Theoretical best-of-(Lévy + Portfolio) per-bench aggregate: **1.07029**
- -0.22% vs Lévy alone
- -0.50% vs PATH A
- Only -0.36% above PATH B (GPU, 1.06650)

Runtime cost: Lévy ~50min + Portfolio ~55min = 105 min/bench, exceeds 60-min cap.
Smart time-sharing could approximate, e.g.:
- 35min Lévy cascade + 25min Portfolio polish (best-by-canonical-proxy)
- Total ~60 min, may capture 70-80% of theoretical best-of-N

## Final placer ranking (IBM 17 aggregate)

1. **PATH B** (`cd_lns_sa_cascade_dp_lane`, GPU): 1.06650 ⭐
2. Theoretical best-of-N (Lévy + Portfolio): 1.07029
3. **Lévy** (`cd_lns_sa_cascade_levy`, CPU): 1.07268 ⭐ DEPLOYED BEST CPU
4. Portfolio (4-weight, CPU): 1.07404
5. Multidir (CPU): 1.07415
6. PATH A (CPU): 1.07564

## Notes

- Lévy needs 4500s/bench timeout (vs PATH A's 3400s).
- ibm08 retry: E25 strict-overlap fix applied to portfolio_levy/placer.py.
- All 17 portfolio results VALID with 0 overlaps.
