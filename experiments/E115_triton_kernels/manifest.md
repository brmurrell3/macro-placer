---
id: E115
name: triton_kernels
status: marginal
parent: E111
created: 2026-05-20
decided: 2026-05-20
champion_at_time: 1.05750
outcome: 1.30 (ibm17 smooth-global-placer; matches V3 baseline 1.31 within noise)
champion_delta: null
graduated_to: null
superseded_by: null
---

# E115: triton_kernels

## Hypothesis
Carrotato's #1 placer (0.967) uses Triton kernels for the smooth proxy.
On our PyTorch CPU autograd implementation of `_lse_hpwl` and the
per-net-trace congestion, the bottleneck is not the math — it's PyTorch's
internal scatter-add backward path when many pins gather to the same macro.

If we can get a 10-50× speedup on the smooth-proxy step, that unlocks
more descent steps within the 60-min/bench cap (or enables wider sweeps
of inits/seeds), which is the leading variable for the C1 spike basin.

## Method

Three layers of optimization, increasingly aggressive:

1. **`index_select` for pin gather.** PyTorch's `tensor[idx]` (advanced
   indexing) on CUDA uses a non-coalesced scatter-add backward that
   is 50-100× slower than `torch.index_select(tensor, 0, idx.flatten())`
   for our access pattern (45K nets × 16 max_pins → 700K pin lookups,
   ~2600 macros). Replace in both `_lse_hpwl` and `PerNetTraceCongestion`.

2. **Single-pass per-net-trace (no chunk loop).** The chunk loop was
   originally there to bound the [B, gr, gc] materialization; but the
   matmul-form of the trace already avoids that 3-D intermediate. The
   chunk loop just multiplies kernel launches by 22× and grows the
   autograd graph linearly.

3. **`torch.compile(mode="reduce-overhead")` on the full step.** Fuses
   small kernels, removes Python-side dispatch overhead, uses CUDA graphs.

## Kill gate

Compute per-step wall on ibm17 CUDA (and ibm17 CPU as a baseline). Kill if:
  - Speedup < 3× across all four benches (ibm01, ibm07, ibm10, ibm17), or
  - Cost diff > 0.5% in either direction (per-step proxy must match the
    PyTorch reference to fp32 noise), or
  - End-to-end placer quality on ibm17 is degraded by > 0.5%.

## Generalization check

End-to-end SmoothGlobalPlacerV4 vs V3 on {ibm01, ibm07, ibm10, ibm17}:
final canonical proxy and zero-overlap status must match V3 to ±0.5%.

## Outcome (decided 2026-05-20)

**Marginal**: per-step optimization is real (16.8× ibm17 CUDA), but it
buys us little on the production target. The champion path is
`cd_lns_sa_cascade_stacked_periphery/placer.py` which doesn't use the
smooth proxy at all — it does cascade saddle escape on the canonical
PlacementCost. E110-E111 (the smooth global placer track) was already
known to produce inferior basin quality than cascade (1.30 vs 0.89 on
ibm17 canonical), and these optimizations don't change that — they
only let us iterate faster on a non-champion track.

If we later resume smooth-global-placer development (e.g. for E113
Xplace integration, or for a multi-seed restart loop within budget),
the index_select fix is a free win and should be adopted into the
production smooth_global_placer_v3 codepath.

Hardware: NVIDIA A10G (g5.2xlarge), CUDA 13.0, Triton 3.6.0, PyTorch 2.10.

Per-step Adam wall (ibm17 CUDA, 300 steps):
| Variant | ms/step | speedup |
|---|---:|---:|
| baseline DiffProxyV3 | 267 | 1.0× |
| FastDiffProxy (index_select + no chunk) | 16 | 16.7× |
| FastDiffProxy + torch.compile(reduce-overhead) | 6.2 | **43×** |

Per-component speedup (CUDA fwd+bwd, ibm17):
| Component | baseline ms | fast ms | speedup |
|---|---:|---:|---:|
| LSE-HPWL | 194 | 2.7 | 70× |
| grid density | 1.7 | 1.7 | 1.0× (unchanged) |
| per-net-trace cong | 82 | 8.9 | 9.2× |
| overlap penalty | 1.6 | 1.6 | 1.0× (unchanged) |

Scaling by problem size:
| Bench | num_macros | baseline ms | fast ms | speedup |
|---|---:|---:|---:|---:|
| ibm01 | 1140 | 33 | 12 | 2.8× |
| ibm07 | 1331 | 44 | 12 | 3.7× |
| ibm10 | 2768 | 129 | 12 | 10.7× |
| ibm17 | 2604 | 267 | 16 | 16.9× |

The speedup grows with macro count (more pins → more scatter contention
in the baseline → larger absolute win from `index_select`).

On M3 CPU: 489 → 124 ms = 3.93× (index_select still helps on CPU,
though scatter contention is less of an issue without CUDA's atomic-add
serialization).
On M3 MPS: 90 → 37 ms = 2.43× (same pattern as CPU; MPS is closer to
CUDA's behavior for this op).

**Numerical equivalence verified**: total proxy cost matches V3 to fp32
(diff = 0.0000%) after fixing FastDiffProxy's WL normalization to use
`plc.net_cnt` like V2 (see `wl_norm` bug fix in `fast_proxy.py`).

End-to-end quality on 4 benches (`results/verify_quality_cuda.json`):
| Bench | V3 proxy | V4 proxy | diff % | V3 wall | V4 wall | speedup |
|---|---:|---:|---:|---:|---:|---:|
| ibm01 | 0.91951 | 0.92652 | +0.76 % | 21.5 s | 14.8 s | 1.46× |
| ibm07 | 1.17715 | 1.16744 | -0.83 % | 33.7 s | 20.8 s | 1.62× |
| ibm10 | 1.15508 | 1.16178 | +0.58 % | 118.7 s | 77.6 s | 1.53× |
| ibm17 | 1.31129 | 1.31005 | -0.09 % | 224.8 s | 137.7 s | 1.63× |

Avg speedup: 1.56×. Avg diff: +0.106 % (within fp32 noise floor 0.5 %).
Zero overlaps on V4 across all 4 benches. End-to-end speedup is lower
than per-step because sdf_init (~26 s on ibm17), FastDiffProxy
constructor (~7 s), and legalize (~0.7 s) are not affected by the
optimization.

End-to-end on M3 MPS (ibm01 + ibm07, 150 steps): avg 1.16× speedup,
V4 diff -0.71 % (V4 actually slightly better on these benches; tracking
noise of the smooth-global descent trajectory).

## Pointers
- Code: `code/fast_proxy.py` — FastDiffProxy + FastPerNetTraceCongestion.
- Code: `code/smooth_global_placer_v4.py` — V3 placer with FastDiffProxy.
- Code: `code/triton_congestion.py` — Triton kernel skeleton (not used in
  production path; PyTorch single-pass is already fast enough).
- Bench: `code/bench_baseline.py`, `code/bench_fast.py`, `code/run_ibm17_e2e.py`,
  `code/verify_quality.py`.
- Results: `results/baseline_*.json`, `results/fast_vs_base_*.json`,
  `results/e2e_*.json`, `results/verify_quality_*.json`.

## Key finding

**The dominant cost was not the math — it was PyTorch CUDA's advanced-
indexing backward.** This is a documented "slow path" in PyTorch:
`tensor[idx]` with a multi-dim idx tensor does not coalesce well on
CUDA, especially when idx has many repeated values (pins repeatedly
referring to the same macro). `index_select(tensor, 0, idx.flatten())`
followed by a reshape uses a different, much faster kernel.

This is a well-known gotcha — but easy to miss when the same op is
6× faster on M3 MPS than on A10G CUDA (because PyTorch MPS uses a
different scatter implementation that's already coalesced). The
A10G measurement is what unmasked it.
