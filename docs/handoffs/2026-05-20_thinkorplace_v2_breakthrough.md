# thinkorplace-v2 breakthrough — 2026-05-19 → 2026-05-20

## Summary

Built and verified a new placer (`thinkorplace-v2`) that lifts our
public-leaderboard projection from **rank 9** (current 1.0771) to
**rank 3** (projected 1.008 on EPYC).

| Metric | v1 (submitted 5/13) | v2 (2026-05-20) | Δ |
|---|---:|---:|---:|
| IBM avg (M3) | 1.0575 | **1.00279** | **−5.17 %** |
| IBM avg (AWS EPYC g5.2xlarge) | 1.08 (proj) | **1.00835** | **−4.65 %** verified |
| NG45 avg (M3) | 0.6893 | 0.67861 | −1.55 % |
| Per-bench wall | ~55 min | ~12 min | 5× faster |
| Overlaps | 0 | 0 | — |

EPYC variance vs M3: **+0.55 %** (much tighter than v1's ~2.4 %).
Per-bench biggest EPYC delta: +3.51 % on ibm17.

## What was the breakthrough

**The smooth congestion model was wrong on hard benches.** The existing
`_rudy_congestion` in `writeup/archive/.../ablation_v2_steps.py` spreads
each net's routing demand UNIFORMLY over the net's bounding box. The
canonical `PlacementCost.get_routing()` instead traces each net's actual
routing pattern (L-route / Steiner-tree for ≤3 pins, star-from-source
for ≥4). On hard benches (ibm10, ibm12, ibm17), smooth bbox-uniform
diverges from canonical by 3–4×.

The fix: replace bbox-uniform with a differentiable per-net-trace
approximation that matches canonical within 15–25 %. Implementation in
`experiments/E111_per_net_trace_congestion/code/per_net_trace_proxy.py`.

For each (source, sink) pin pair: soft cell-assignment (Gaussian σ=0.5×cell),
smooth row/col endpoints (logsumexp β=6), sigmoid range indicator
matching canonical's half-open `[lo, hi)` interval. H demand accumulated
on source-row × col-range; V demand on sink-col × row-range. Plus
hard-macro footprint contribution and ±smooth_range box smoothing with
edge-active-cell-count normalization (matches canonical's boundary
clipping).

## Architecture

```
thinkorplace-v2 (12 min/bench):
  ┌─────────────────────────────────────────────────────┐
  │ SmoothGlobalPlacerV3                                │
  │   ├─ SDF init                                       │
  │   ├─ Adam descent on DiffProxyV3 (E88/E95 + E111)  │
  │   │   ├─ LSE-HPWL                                  │
  │   │   ├─ Grid density                              │
  │   │   └─ PerNetTraceCongestion ← NEW              │
  │   ├─ γ-anneal 5e-3 → 5e-5                          │
  │   ├─ overlap_lambda ramp 0 → 10                    │
  │   └─ greedy_macro_legalize                         │
  ├─ project_overlaps (safety net)                     │
  └─ run_cd_adaptive (CD polish to canonical plateau)  │
```

Defensive retry chain in `submissions/thinkorplace-v2/placer.py`:
- Primary: ovl_lambda_end=10
- Fallback 1: ovl_lambda_end=50 (more aggressive overlap penalty)
- Fallback 2: ovl_lambda_end=100, num_steps=300
- Hard fallback: SDF init + project_overlaps + CD polish

Tested via Docker (x86 image on ARM Mac): V3 produced residual overlaps
under QEMU emulation; defensive chain caught it and produced a valid
placement (proxy 1.127 on ibm01, vs 0.847 native). On native x86 EPYC,
V3 works directly with no fallback needed.

## Iteration log (~12 hours of work)

| Time | Variant | M3 IBM | Δ vs prev |
|---|---|---:|---:|
| Day 0 baseline | Option C (thinkorplace v1) | 1.0575 | — |
| 19:00 | Subagent built E111 per-net-trace | — | — |
| 21:36 | V3Min default 4-min | 1.0302 | −2.58 % |
| 22:19 | V3Min ovl10 4-min | 1.0213 | −0.83 % |
| 22:28 | V3Min default 8-min | 1.0195 | −0.18 % |
| 22:54 | V3Min ovl10 8-min | 1.0039 | −1.53 % |
| 23:43 | **V3Min ovl10 12-min** | **1.00279** | **−0.11 %** |
| 00:41 | V3Min ovl10 margin 8-min | 1.0026 | TIE |
| 01:20 | V3Min ovl10 DPO 12-min | 1.0102 | +0.74 % |
| 02:16 | V3Min ovl10 multiseed 12-min | 1.0051 | +0.23 % |

Champion locked at V3Min ovl10 12-min. Further iteration showed
diminishing returns (DPO init, multi-seed, margin trick all marginal or
worse than the simple baseline).

## What didn't work

- **DPO init**: V3 prefers a fresh SDF basin to DPO's pre-optimized layout
- **Multi-seed ensemble**: splitting budget across seeds underperforms single full-budget run
- **Margin overlap (V5's trick)**: helps when ovl_lambda is high but conflicts with ovl=10
- **ovl5 / ovl20**: too weak (ovl5) leaves residual overlaps; too strong (ovl20) over-constrains
- **DREAMPlace patching** (Path B v1): congestion gradient too expensive to reimplement
- **V5 Nesterov-BB optimizer**: ePlace argument doesn't transfer to our grid-bin density

## Validations

- M3 --all 17 IBM: 1.00279, zero overlaps, qualified
- M3 --ng45 4 commercial: 0.67861, zero overlaps, qualified
- AWS EPYC g5.2xlarge --all 17 IBM: 1.00835, zero overlaps, qualified
- Docker (x86 image on ARM): defensive fallback verified

## Submission state

- Root `placer.py` launcher → `submissions/thinkorplace-v2/placer.py`
- `submissions/thinkorplace/` = v1 (5/13 submission, kept as fallback)
- `submissions/_archive/` = 26 experimental variants archived
- AWS instance stopped (cost saved)

## Open questions for further optimization (24 hr left)

1. **Closing the gap to Carrotato 0.967**: we're at 1.008, gap +4.2 %.
   Requires structural changes:
   - Better density model (electrostatic / FFT-Poisson like DREAMPlace)
   - Hierarchical refinement (cluster then refine)
   - Triton-fast gradient kernels for 100× more iterations
2. **NG45 EPYC validation**: not yet run; assumed similar variance
3. **Per-bench hyperparameter adaptation**: smooth_range, num_steps could
   scale with bench size (but risks per-bench tuning concern)
4. **Cascade-on-V3 hybrid**: V3 basin → cascade saddle (~50 min/bench
   total, too slow but might lift to 0.99)
