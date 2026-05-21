# E123 Sinkhorn calibration findings — 2026-05-20

## Key result: ε=0.5 matches canonical congestion within 0.26%

Eps sweep on ibm17 (cached cascade placement; canonical cong = 1.89497):

| ε     | iters | sinkhorn cong | rel% vs canonical | n_movers (out of 2604) | top10_grad |
|------:|------:|--------------:|------------------:|-----------------------:|-----------:|
| —     | —     | 2.16710 (topk)| **+14.36 %**      | 469                    | 5.71e-3    |
| 1.000 | 50    | 1.73162       | -8.62 %           | 740                    | 1.94e-3    |
| **0.500** | **100**   | **1.89998**   | **+0.26 %** (best fidelity!) | **582**          | 3.38e-3    |
| 0.300 | 166   | 2.00331       | +5.72 %           | 504                    | 4.74e-3    |
| 0.200 | 200   | 2.06609       | +9.03 %           | 481                    | 5.60e-3    |
| 0.100 | 200   | 2.13182       | +12.50 %          | 479                    | 5.97e-3    |
| 0.050 | 200   | 2.15706       | +13.83 %          | 480                    | 5.84e-3    |
| 0.020 | 200   | 2.16540       | +14.27 %          | 463                    | 5.83e-3    |
| 0.010 | 200   | 2.16597       | +14.30 %          | 479                    | 5.54e-3    |

(Run command: `uv run python experiments/E123_sinkhorn_topk/code/eps_sweep.py ibm17`)

## Interpretation

1. **The naive "ε → 0 limit = canonical" claim from the task description
   is WRONG.** ε → 0 recovers `torch.topk` (which is itself +14 % off
   canonical because of the E111 trace's L-route, σ smoothing, etc.).
   Canonical fidelity does NOT improve at small ε.

2. **The SMOOTHING from ε > 0 happens to compensate for the E111 over-
   estimation bias.** torch.topk picks K=5 % of cells; Sinkhorn at ε=0.5
   spreads mass over ~582 cells (≈13 % of N), and that broader
   averaging happens to lower the mean toward canonical's value. This
   is *accidental fidelity* — useful but not what Cuturi proved.

3. **Gradient flow** (n_movers above 5 % threshold): ε=0.5 gives 582 vs
   topk's 469 — 24 % more cells receiving gradient signal. The top-10
   gradient magnitude is lower (~3.4e-3 vs ~5.7e-3) because gradient
   mass is *distributed*, not lost.

4. **Recommended config for placement**: ε=0.5 with iters=100. ε=0.3
   trades off some canonical fidelity (+5.7 %) for sharper top-K (504
   movers) — might be better for finishing a basin.

## Convergence note

Auto-iters bump (added to `sinkhorn_topk.py`): at ε < 0.05 we
auto-bump iters because Sinkhorn convergence rate ∝ 1/ε. Verified
self-test: ε=0.005 + 200 iters → 0.00 % rel error to torch.topk.

## Cost

- Forward+backward at ε=0.5, iters=100 on ibm17: 504 ms.
- 500 Adam steps ≈ 252 s (4.2 min) of descent. Same order as topk path.
- Memory: O(N + 2) = ~36 KB for log-potentials. Negligible.
