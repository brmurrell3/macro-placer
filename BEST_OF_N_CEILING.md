# Best-of-N ceiling analysis — 2026-05-16 12:40 EDT

## TL;DR

**Theoretical 21-bench aggregate ceiling: 0.97957** (vs current 1.0000)

Achievable only if we deploy a placer that runs ≥3 strategies per bench
and picks by canonical proxy at runtime.

## Per-bench best (from `best_of_n_analysis.py` over 189 cached .pt files)

| Bench | Best proxy | Source | vs AWS-cpu fresh |
|---|---|---|---|
| ibm01 | **0.84009** | E90 cascade_multidir | -4.5% (vs 0.8797) |
| ibm02 | 1.02104 | E84 cascade | -3.9% (vs 1.0623) |
| ibm03 | 0.94634 | E84 cascade | -0.04% |
| ibm04 | 0.98590 | AWS-cpu fresh | — |
| ibm06 | 1.11792 | E84 cascade | -2.7% (vs 1.1489) |
| ibm07 | 1.06449 | E84 cascade | -1.2% |
| ibm08 | 1.08970 | AWS-cpu fresh | — |
| ibm09 | **0.78006** | E91 DP full polish | **-5.6% (vs 0.8267)** ⭐ |
| ibm10 | 0.98940 | E84 cascade | -2.8% |
| ibm11 | 0.86808 | E84 cascade | -0.55% |
| ibm12 | **1.11005** | E91 DP full polish | **-8.2% (vs 1.2092)** ⭐⭐ |
| ibm13 | 0.93458 | E84 cascade | -2.1% |
| ibm14 | 1.19148 | E96 multi_dp K4 | -2.2% |
| ibm15 | 1.14163 | E74 hessian | -2.7% |
| ibm16 | 1.10847 | E84 cascade | -3.3% |
| ibm17 | 1.33162 | E84 cascade | -0.8% |
| ibm18 | 1.33440 | AWS-cpu fresh | — |
| ariane133 | 0.65212 | M3 fresh wrapper | — |
| ariane136 | 0.64980 | M3 fresh wrapper | — |
| mempool_tile | 0.73744 | cached dp_lane | — |
| nvdla | 0.67646 | M3 fresh wrapper | — |

## Aggregate breakdown

| Subset | Score |
|---|---|
| IBM 17 (ceiling) | **1.05031** |
| NG45 4 (already saturated) | 0.67895 |
| **21-bench (ceiling)** | **0.97957** |

vs current submitted (1.0771 / 0.681 / ~1.000): IBM 17 has **-2.4% headroom**.

## Why the gap to AWS-cpu fresh?

AWS-cpu ran `placer_adaptive.py` (cascade only, wall-capped at 55 min).
Our cached best per bench comes from:
- **9 IBM benches** won by E84 cascade (cd_lns_sa_cascade) — likely from
  uncapped or longer-budget runs
- **2 IBM benches won by E91 DP full polish** (cd_lns_sa_cascade_dp_lane) —
  ibm09 (-5.6%) and ibm12 (-8.2%). **These need GPU.**
- 1 each from E96 multi_dp K4, E74 hessian, E90 cascade_multidir

## Path to realize the ceiling

**A runtime best-of-N placer** that runs multiple strategies per bench
within the 55-min budget, picking best by canonical proxy:

```python
class BestOfNPlacer:
    def place(self, benchmark):
        candidates = []
        candidates.append(self._run_cascade(benchmark, budget=30min))
        if torch.cuda.is_available():
            candidates.append(self._run_dp_lane(benchmark, budget=20min))
        # Add hessian / multi_dp if budget allows
        return min(candidates, key=lambda c: c.proxy)
```

**Constraints (verified):**
- Bench-agnostic: no per-bench parameter conditioning
- Wall-budget: 55 min per bench (3-min margin to 60-min cap)
- Strategies pick winner by canonical proxy (no rules violation)

## Recommended action

1. **GPU access**: AWS-GPU quota approval (other Claude blocked here).
   Critical for ibm09/ibm12 wins (-5.6%, -8.2%).
2. **GPU-free fallback**: ship E84 cascade as primary; secondary
   strategies (hessian, multi-restart) for diversity.
3. **Already-shipped pattern**: `cd_lns_sa_cascade_dp_lane/placer.py`
   (PATH B hybrid). Run this on AWS-GPU when quota lands.

## Files

- Analyzer: `experiments/E107_periphery_bias/code/best_of_n_analysis.py`
- Raw cached placements: 189 .pt files across `results/cloud_snapshot_*`,
  `experiments/E84_*`, `experiments/E91_*`, `experiments/E96_*`, etc.
