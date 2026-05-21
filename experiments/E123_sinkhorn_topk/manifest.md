---
id: E123
name: sinkhorn_topk
status: in_progress
parent: E111
created: 2026-05-20
decided: null
champion_at_time: 1.05750
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E123: Sinkhorn-relaxed top-K (cross-domain ML→placement)

## Hypothesis

The E111 per-net-trace congestion uses `torch.topk(combined, K).mean()`
for the ABU-5 % scalar (canonical's congestion metric). `torch.topk` is
differentiable through the values of the K selected cells, but
**its gradient is zero on the N − K non-selected cells**. When the K-th
and (K+1)-th cells are close — which happens on smooth basins where Adam
has flattened the V+H map — the selection set "flickers" between steps
and Adam sees a non-smooth signal.

**Sinkhorn-relaxed top-K** (Cuturi NeurIPS 2019; Xie et al. NeurIPS
2020) treats the top-K assignment as a regularized optimal-transport
problem. With ε > 0, the relaxed assignment T*[i] ∈ [0, 1] gives the
soft "membership" of cell i in the top-K set. The differentiable mean
becomes:

    sinkhorn_topk(x, K, ε) = Σ_i x[i] · T*[i] / K

with gradient flow proportional to each cell's PROXIMITY TO THE TOP-K
BOUNDARY. As ε → 0, the operator recovers exact `torch.topk`; ε > 0
provides graceful smoothing. This is novel for placement — no public
placer uses optimal-transport relaxations of the ABU scalar.

## Method

1. **`code/sinkhorn_topk.py`** — pure-PyTorch Sinkhorn-Knopp fixed-point
   iteration in log-space (numerically stable). 2-anchor formulation
   (selected / non-selected) avoids O(N²) cost. Self-test verifies
   ε → 0 limit matches `torch.topk` within 0.1 % at 100 iters.

2. **`code/sinkhorn_per_net_trace.py`** — `SinkhornPerNetTraceCongestion`
   subclasses E111 `PerNetTraceCongestion` and overrides
   `compute_congestion()` to use `sinkhorn_topk_mean` instead of
   `torch.topk`. New hparams: `sinkhorn_eps`, `sinkhorn_iters`,
   `use_sinkhorn` (ablation toggle).

3. **`code/smooth_global_placer_v3_sinkhorn.py`** — `SmoothGlobalPlacerV3Sinkhorn`
   subclasses E111 `SmoothGlobalPlacerV3` to thread `sinkhorn_*` through
   to the proxy. Otherwise identical (same Adam descent, γ-anneal,
   overlap λ ramp, legalize).

4. **`code/test_ibm17.py`** — single-bench A/B test:
   - V3Sinkhorn (use_sinkhorn=False) → exactly equivalent to E111 V3
     reference (ablation control).
   - Sinkhorn ε=0.1, 0.05, 0.3, and anneal ε=0.3→0.05.
   Each variant runs V3 descent + greedy legalize + 600s CD polish.

## Kill gate

- ibm17 polish_proxy ≥ 1.20 (no lift over V3Min ovl10 720s baseline) →
  Sinkhorn doesn't help at our basin scale; mark falsified.
- If Sinkhorn-calibrate scalar mismatch (vs canonical) > +30 % at all
  ε settings → ε can't be tuned to recover ABU-5 %.

## Generalization check

Single-bench ibm17 win is necessary but NOT sufficient. If ibm17 lifts:
1. Validate on ibm10 + ibm12 (the other "hard" benches).
2. Run --fast (4 benches) end-to-end.
3. Run NG45 ariane133 to gate against the E54/E62 IBM-overfit pattern.
Only after all three pass do we consider --all.

## Outcome (filled when decided)

[Empty until decided. Then: result, decision, rationale.]

## Pointers

- `code/sinkhorn_topk.py` — core Sinkhorn fixed-point + self-test.
- `code/sinkhorn_per_net_trace.py` — proxy subclass + calibration utility.
- `code/smooth_global_placer_v3_sinkhorn.py` — V3 placer subclass.
- `code/test_ibm17.py` — end-to-end smoke (5 configs).
- `results/calibration_ibm17.json` — Sinkhorn-vs-canonical scalar + Δcong corr.
- `results/ibm17_smoke.json` — V3 + CD600s for each variant.
- Literature: Cuturi 2019 (arXiv:1905.11885), Xie 2020 (NeurIPS).
- Survey context: `docs/research/2026-05-20_congestion_model_survey.md`
  §3 IMPROVEMENT 4 and §8 Sinkhorn-relaxed top-K formula.
