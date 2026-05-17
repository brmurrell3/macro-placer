# Multidir --all final aggregate — 2026-05-17 01:04 UTC

**Verdict: REJECTED for promotion** (gate: --all aggregate ≥0.3%, achieved -0.14%)

## 17-bench results

| Bench | Multidir | PATH A | Δ |
|---|---|---|---|
| ibm01 | 0.8792 | 0.8797 | -0.06% |
| ibm02 | 1.0421 | 1.0623 | **-1.90%** ⭐ |
| ibm03 | 0.9476 | 0.9467 | +0.10% |
| ibm04 | 1.0027 | 0.9859 | +1.70% |
| ibm06 | 1.1392 | 1.1489 | -0.84% |
| ibm07 | 1.0746 | 1.0773 | -0.25% |
| ibm08 | 1.0952 | 1.0897 | +0.50% |
| ibm09 | 0.8182 | 0.8267 | -1.03% |
| ibm10 | 1.0256 | 1.0180 | +0.75% |
| ibm11 | 0.8681 | 0.8729 | -0.55% |
| ibm12 | 1.2146 | 1.2092 | +0.45% |
| ibm13 | 0.9434 | 0.9549 | -1.20% |
| ibm14 | 1.2201 | 1.2186 | +0.12% |
| ibm15 | 1.1562 | 1.1727 | -1.41% |
| ibm16 | 1.1439 | 1.1460 | -0.18% |
| ibm17 | 1.3465 | 1.3419 | +0.34% |
| ibm18 | 1.3433 | 1.3344 | +0.67% |
| **AVG** | **1.07415** | **1.07564** | **-0.14%** |

## Why it failed promo gate

The --fast spike showed -0.63% aggregate (4 benches: ibm01, ibm04, ibm09, ibm13).
The --all aggregate is only -0.14%. The --fast was unrepresentative — those
4 benches happened to be among the strongest multidir wins.

Multidir helps on a subset (ibm02, ibm13, ibm15 strong) but hurts on
others (ibm04, ibm10, ibm18 regressions). Net is within RNG variance.

## Decision

**Stick with PATH B** (cd_lns_sa_cascade_dp_lane) as Tier-1 entry:
- Verified IBM 1.06650 / NG45 0.68086 (Lambda GPU)
- Falls back to PATH A (1.07820) when no GPU

PATH A (cd_lns_sa_cascade/placer_adaptive.py) verified 1.07564 fresh AWS-cpu.

Multidir's value: a 4th lane that could be combined in a runtime
best-of-N placer, but its standalone aggregate doesn't justify replacing
the current Tier-1 candidates.
