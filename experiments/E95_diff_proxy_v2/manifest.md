---
id: E95
name: diff_proxy_v2
status: in_progress
parent: E88
created: 2026-05-13
decided: null
champion_at_time: 1.0612         # canonical cascade --all uncapped; submission floor 1.137
outcome: null
champion_delta: null
graduated_to: null
superseded_by: null
---

# E95: differentiable canonical proxy — spike v2 (revised PATH C1)

## Hypothesis
E88's spike failed not because surrogate primitives were structurally misaligned
with canonical (top-K density and ABU-5% congestion in `_grid_density`/
`_rudy_congestion` are already matched), but because two narrower issues
killed descent:

1. **LSE-HPWL bias.** γ = 0.0005 × canvas leaves a per-net additive bias
   ≈ γ·log(n_pins). At cascade's canonical optimum, the smooth gradient is
   non-zero in a direction that the canonical doesn't see. Annealing γ → 0
   during descent should recover canonical-aligned gradients.
2. **AdamW first-step geometry.** At t=1, bias-corrected m_hat = g and
   v_hat = g², so the update reduces to −lr·sign(g) regardless of |g|.
   With lr=0.5 and ibm01 canvas ≈ 23, every macro shifts ~2% of canvas
   on step 1 in a correlated direction — instant 224 overlaps even from
   a zero-overlap cascade init. **L-BFGS with strong Wolfe line search**
   adapts step size and rejects basin-blowing updates by construction.

Together these explain E88's data quantitatively: smooth(cascade) − canonical
= +2.1% (matches γ·log(npins) bias), and the descent log shows ovl=224 at
step 0 (post-first-AdamW-step) regardless of initial cascade legality.

## Method
Reuse E88's `diff_proxy.py` primitives unchanged. Build a new driver:

1. **γ-annealed LSE-HPWL.** Rebuild DiffProxy with shrinking γ each window
   (γ_start = 0.005·canvas → γ_end = 1e-4·canvas, 4 stages of 100 steps).
2. **L-BFGS-B with strong Wolfe.** torch.optim.LBFGS(history_size=20,
   line_search_fn='strong_wolfe'). Closure recomputes smooth proxy +
   overlap Lagrangian + boundary penalty at each line-search trial.
3. **Aggressive overlap λ schedule.** λ_start = 500 (vs E88's 100),
   λ_end = 10000. Boundary λ also high (500) — boundary escapes were a
   secondary E88 failure mode.
4. **Honest best-canon tracker.** Update best only on 0-overlap canonical
   evaluations (E88 had this bug fixed in iter 2 already).
5. **Inits:** {cascade-cached ibm01, SDF init}. Two of each.

## Kill gate
**Spike fails if best canonical ibm01 proxy > 0.898 after L-BFGS + legalize
(cascade input × 1.05).** Same gate as E88.

If a variant clears the gate, run on ibm10/12/14/17 (B-R0' hard-bench set,
60-min budget each on cloud A100) for hard-bench validation.

If all variants fail, the deeper structural finding becomes: **even with
matched primitives + line-search optimization, smooth-gradient descent
cannot preserve cascade basin.** That ends PATH C1 standalone and the
PATH B Claude can stop chasing B-R3.

## Generalization check
Phase 2 hard-bench (ibm10/12/14/17). If 3 of 4 land ≤ B-R0' (i.e., the
DP+full-polish numbers), promote C1 as a third basin lane alongside SDF
(E25) and DPO (E41) in the hybrid placer.

## Outcome (interim, 2026-05-13 ~01:30 UTC)

### Calibration fix (load-bearing)
E88's "smooth-vs-canonical gap" was misattributed to LSE-HPWL bias. The
real cause was a **WL normalization mismatch**: E88's `diff_proxy.py`
used `total_net_count = len(plc.nets)` in `wl_norm`, but canonical
`PlacementCost.get_cost()` normalizes by `plc.net_cnt` (which is the
**sum** of net weights, not the count). On ibm01: `plc.net_cnt = 7269`
vs `len(plc.nets) = 5993`, ratio 1.213. This **exactly** explained
E88's smooth_wl 0.1045 vs canonical_wl 0.0861 → smooth_wl 1.214 ×
canonical.

After patching `wl_norm = (cw + ch) * plc.net_cnt` in `DiffProxyV2`,
the smooth-vs-canonical gap at the cascade-cached ibm01 placement
dropped from **+2.1 % to +0.05 %** (γ = 5e-4 fixed) and to **−0.14 %**
with γ → 1e-5. Smooth gradient is essentially zero at the cascade
optimum.

This is the actual root cause E88 documented as "structural mismatch
that even matched primitives might not bridge" — except matched
primitives were never the issue; the normalization was.

### ibm01 spike v2 results (4-variant sweep)
| Variant | Init | Optimizer | Post-leg canon | Δ vs init | Notes |
|---------|------|-----------|---------------:|----------:|-------|
| cascade_lbfgs | cascade-cached | L-BFGS strong Wolfe | 0.8461 | +0.10 % | basin held |
| cascade_sgd   | cascade-cached | projected SGD       | 0.8459 | +0.07 % | basin held |
| sdf_lbfgs     | SDFPlacer init | L-BFGS strong Wolfe | 1.1452 | −4.2 %  | descends but far from cascade quality |
| sdf_sgd       | SDFPlacer init | projected SGD       | 1.1924 | −0.3 %  | SGD too slow from SDF basin |

All 4 variants pass the gate (0.898), but **only by virtue of
basin preservation** — neither cascade variant improves cascade input.

### Cascade-cliff finding (perturb seeds)
4 seeds of 1 %-of-canvas Gaussian perturbation on the cascade-cached
ibm01 placement, each followed by L-BFGS descent, landed at canonical
**1.21–1.24** (vs cascade input 0.85). The cascade basin is a narrow
local minimum on the smooth-proxy landscape — even 1 % perturbation
drops descent into a substantially worse basin. **C1 cannot escape
cascade's basin via perturb-and-redescend.**

### Smooth RUDY ≠ canonical congestion on hard benches (new finding)
The WL normalization fix made smooth and canonical agree to **0.05 % on
ibm01**. But on bigger benches the smooth/canonical decomposition
diverges sharply in congestion:

| Bench | canon_proxy | smooth_proxy | WL gap | Dens gap | Cong gap |
|-------|------------:|-------------:|-------:|---------:|---------:|
| ibm01 | 0.845 | 0.846 | +1.4 % | 0.0 % | −0.2 % |
| ibm10 | 0.989 | 2.328 | +2.9 % | 0.0 % | **+205 %** |
| ibm12 | 1.198 | 3.134 | +2.4 % | 0.0 % | **+234 %** |
| ibm17 | 1.332 | 3.845 | +2.0 % | 0.0 % | **+265 %** |

Density matches exactly across all benches (canonical's grid-cell
overlap computation is essentially identical to smooth's). WL gap is
within 3 % (LSE smoothing at γ=0.005). **Congestion diverges 3-4× on
hard benches.**

Root cause: DPO-era `_rudy_congestion` distributes net wire demand
uniformly across each net's bounding box (Rectangular Uniform Wire
DensitY); canonical `get_routing` traces actual route paths per
two/three/N-pin net topology through grid cells via
`__two_pin_net_routing` / `__three_pin_net_routing` / `__split_net`,
then applies `__smooth_routing_cong` (per-axis ±2-cell box smoothing).
The two methods produce comparable values on small benches with mostly
few-pin nets, diverge sharply on bigger netlists where bbox spread
over-counts demand on most cells.

Implication: smooth gradient on hard benches is dominated by a 3-4×
over-counted congestion contribution. Direction of smooth gradient is
**not aligned** with canonical's gradient direction. AdamW / L-BFGS
descent moves toward smooth-optimum, not canonical-optimum.

### ibm10 sweep result (consistent with diagnosis)
| Variant | post-leg canon | wall (s) | Notes |
|---------|---------------:|---------:|-------|
| lbfgs   | 0.9894         | 173      | basin held exactly — L-BFGS line search rejected every move into smooth-improving region (because canonical eval shows regression) |
| sgd     | 0.9894         | 203      | tiny-lr warmup → cosine decay → bounced back to baseline |
| perturb | (skipped, 2768 macros > 2000 limit) | — | — |

Lift on ibm10: **+0.000 %.** Smooth proxy can't navigate; either
optimizer holds the input or any motion would worsen canonical.

[Updated when sweep completes ibm12/14/17, walltrunc finishes.]

## Pointers
- Code: `code/diff_proxy_v2.py`, `code/spike_v2.py`, `code/calibrate_v2.py`
- Cached input: `experiments/E84_cascading_saddle/results/cascade_ibm01.pt`
- Legalizer: `experiments/E76_dreamplace_integration/code/macro_legalizer.py`
- Primitives reused unchanged: `experiments/E88_diff_proxy/code/diff_proxy.py`
- Hard-bench comparison (B-R0' table): TODO.md PATH B section
- E88 falsification (the failed cheap spike): `experiments/E88_diff_proxy/manifest.md`
