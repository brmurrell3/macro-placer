# Lévy cascade --all IBM 17 — 2026-05-17 09:49 UTC

**NEW BEST CPU-only IBM placer: 1.07268** (-0.28% vs PATH A)

## 17-bench results

| Bench | Lévy | PATH A | Δ |
|---|---|---|---|
| ibm01 | 0.8704 | 0.8797 | **-1.06%** ⭐ |
| ibm02 | 1.0650 | 1.0623 | +0.25% |
| ibm03 | 0.9447 | 0.9467 | -0.21% |
| ibm04 | 0.9712 | 0.9859 | **-1.49%** ⭐ |
| ibm06 | 1.1400 | 1.1489 | -0.78% |
| ibm07 | 1.0744 | 1.0773 | -0.27% |
| ibm08 | 1.0957 | 1.0897 | +0.55% |
| ibm09 | 0.8349 | 0.8267 | +0.99% |
| ibm10 | 1.0214 | 1.0180 | +0.33% |
| ibm11 | 0.8660 | 0.8729 | -0.79% |
| ibm12 | 1.2067 | 1.2092 | -0.21% |
| ibm13 | 0.9435 | 0.9549 | **-1.20%** ⭐ |
| ibm14 | 1.2125 | 1.2186 | -0.50% |
| ibm15 | 1.1566 | 1.1727 | **-1.37%** ⭐ |
| ibm16 | 1.1398 | 1.1460 | -0.54% |
| ibm17 | 1.3394 | 1.3419 | -0.19% |
| ibm18 | 1.3534 | 1.3344 | +1.42% ❌ |
| **AVG** | **1.07268** | 1.07564 | **-0.28%** |

## Aggregate ranking (IBM 17)

1. **PATH B** (`cd_lns_sa_cascade_dp_lane`, GPU): **1.06650** ⭐
2. **Lévy** (`cd_lns_sa_cascade_levy`, CPU): **1.07268** ⭐ NEW BEST CPU
3. Multidir (`cd_lns_sa_cascade_multidir`, CPU): 1.07415
4. PATH A (`cd_lns_sa_cascade/placer_adaptive`, CPU): 1.07564

## Caveats

- Lévy needs 4500s timeout per bench (vs PATH A's 3400s) due to extra
  heavy-tail cascade time. run_parallel.sh template should be updated
  for this placer.
- ibm18 regression (+1.42%) is the weakest. Multidir wins ibm18 (1.3433).
- ibm01, ibm04, ibm13, ibm15 are Lévy's strongest wins (>1.0%).

## Combined 21-bench

| Placer | IBM 17 | NG45 4 | 21-bench |
|---|---|---|---|
| PATH B (GPU) | 1.06650 | 0.68086 | 0.99304 |
| **Lévy** | **1.07268** | 0.67897* | 0.99956 |
| Multidir | 1.07415 | **0.6779** | 0.99868 |
| PATH A | 1.07564 | 0.68102 | 1.00048 |

*M3 fresh Lévy NG45 (same placer, different hardware)

PATH B remains best aggregate.

## Recommendation

For partcl submission:
- **Tier-1 (GPU available)**: PATH B
- **Tier-1 (CPU-only)**: Lévy (`cd_lns_sa_cascade_levy/placer.py`)
- For OpenROAD Tier-2: use Multidir's NG45 placements (best at 0.6779)
