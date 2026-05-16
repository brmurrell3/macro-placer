---
id: E103
name: Differentiable trace-route RUDY (rebuild)
status: blocked
hypothesis: PATH B / cascade variant exploration
date_proposed: 2026-05-14
date_decided: 2026-05-16
---

# E103 — Differentiable trace-route RUDY rebuild

## Hypothesis

Same gap as E92 (`E92_dp_tilos_rudy`): the smooth RUDY surrogate used in
DPO and E95 is **bbox-uniform-spread** while the canonical
`get_routing` is **per-net trace-route**. On hard benches (ibm10/12/17)
the smooth surrogate diverges 3-4× from canonical, killing C1-class
spike attempts and making E92's diff-RUDY-as-DP-loss term useless.

E103 attempts to rebuild a differentiable trace-route RUDY that
matches canonical at ρ > 0.95 on hard benches. If successful,
unblocks both the E92 DP integration and the E95 calibrated-proxy
spike retry.

## Method

Four iteration files:
- `code/calibrate_trace.py` — compute ρ between candidate smooth
  formulations and canonical `get_routing` on a per-bench basis.
- `code/canon_pin_extract.py` — extract pin positions from canonical
  PLC, since smooth proxies were previously using DP's pin model
  (which diverges from canonical).
- `code/diff_trace_rudy.py` — first rebuild (per-net trace, vectorized
  with `torch.searchsorted` for breakpoints).
- `code/diff_trace_rudy_v2.py` — second rebuild after `diff_trace_rudy`
  hit a memory issue on ibm17 (one large net, ~400 pins).

## Kill gate

ρ < 0.90 on ibm10/12/17 vs canonical `get_routing` (same gate as E92).

## Generalization check

If ρ > 0.95 on hard IBM, must also verify NG45 ariane133/ariane136
ρ > 0.95 (commercial designs have different routing topology, and the
E54 falsification pattern is "IBM-aware, NG45-blind").

## Outcome

**Blocked 2026-05-16.** Same gap as E92, ~2-3 days of dev work to
finish vectorizing the per-net trace loop and rebuild DP's PlaceObj
with the new term. Time-budget pressure pulled effort to PATH A
post-A1 cascade speedup, which delivered the 1.07820 floor without
needing this work. Subsumed by E91/DP-lane: stock DP + full polish
beat the trace-route-RUDY gap empirically without needing a custom
loss term.

Reactivate if leaderboard reach requires another -2 to -3 % aggregate
lift (e.g., to chase vmallela #1 at 1.011), or if a future
differentiable-proxy spike (C1-class) wants accurate congestion in
the loss.

Code preserved for future use. `diff_trace_rudy_v2.py` has the
not-yet-finished per-net vectorization that would be the starting
point.
