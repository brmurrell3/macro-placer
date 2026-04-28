# RUDY fidelity probe

## Question

How accurately does RUDY — the differentiable congestion approximation
used inside the DPO placer — match the real ICCAD-evaluator congestion
grid, cell by cell? The DPO gradient points wherever RUDY says
congestion lives. If RUDY disagrees with the actual evaluator on
*which* cells are hot, the gradient is pointing at the wrong cells and
no amount of tuning the congestion weight can fix it.

## How to run

The script runs DPO on ibm01, then reads back the RUDY grid (recomputed
from the optimized positions) alongside `plc.H_routing_cong` /
`plc.V_routing_cong` from the same placement, and compares them in
several ways (per-cell ratios, top-K overlap, spatial quadrants, H vs V
decomposition, macro-blockage isolation).

```bash
uv run python analysis/rudy_fidelity/rudy_analysis.py
```

No flags. Subject is hard-coded to ibm01 (the worst RUDY mismatch
benchmark; see `writeup/evidence.md` §5).

## Headline finding

From `writeup/evidence.md` §5.1:

- **Real/RUDY ABU-5% ratio = 3.1×** on ibm01 — *not* the ~2× implied by
  aggregate proxy metrics. Real exceeds RUDY on 1843/1845 cells.
- **Top-5% hotspot overlap is 10.9%** (10/92 cells agree); Jaccard
  similarity 0.057 — essentially random.
- **77% of real top-5% cells are not even in RUDY's top-10%** — the two
  models disagree on *where* congestion is, not just how much.
- **Three structural divergence sources:** L-routing vs uniform bbox
  (~2.74×), macro blockage (RUDY ignores it; 28.3% of real congestion),
  spatial smoothing (smooth_range = 2 redistributes peaks).
- **Vertical worse than horizontal:** V correlation 0.327, H 0.635.

## Implication

DPO's congestion gradient points at the wrong cells. A weight scaling
cannot fix this — see the killed congestion-weight sweep in
`writeup/evidence.md` §5.2: scaling 0.5 → 1.5 produced monotonic
worsening (1.2081 → 1.2382). Amplifying a noisy signal makes things
worse. This is the diagnostic that motivated the second pivot
("bypass don't fix" — abandon RUDY-based gradient methods, build the
incremental real-proxy evaluator and do CD on the actual cost
function).

## Re-run policy

Static / one-shot. Re-run only if the RUDY model in
`submissions/dpo/placer.py` is modified (different smoothing, different
net decomposition, macro blockage added). The DPO placer is currently
archived; this probe exists as evidence for the writeup, not as a live
diagnostic.

## Frozen output

`TODO(data): writeup/data/rudy_ibm01.txt` — the stdout of the script
on ibm01 has not yet been captured. Capture once and commit.
