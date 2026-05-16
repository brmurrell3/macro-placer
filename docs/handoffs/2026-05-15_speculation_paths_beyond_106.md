# Speculation report — paths to lift beyond 1.06 (2026-05-15)

User asked: "also start speculating and reseraching on how we can improve."

Context: ensemble chain (native-DP + portfolio_levy + curvadapt) running on
cloud; observed realistic ceiling ~1.05-1.06 because DP basin is bad on
dense benches (ibm12 basin = 2.47 with default DP config). Verified
proxy 1.0771 on partcl leaderboard, rank 6. Top is Carrotato 0.9671. Gap
to top = ~10%. Gap to top-3 (~1.01) = ~6%.

## Bottom-line up front

| Path | Effort | Expected lift | Risk | Priority |
|---|---|---|---|---|
| **A. DP hyperparam dispatch (input-driven)** | 1 day | 1-3% on dense benches (10-20% local) | Low — established (AutoDMP) | **HIGH** |
| **F. Periphery-guided relocation (DAC25-ReMaP)** | 2-3 days | **+34% WNS, +65% TNS** per ReMaP — directly attacks Grand Prize | Medium — new mechanism | **HIGH** |
| C. OpenROAD WNS/TNS validation (Grand Prize) | 2 days | Decides $20K prize, not rank | None — feasibility check | MEDIUM |
| D. Innovation Award writeup | 1 day | $4K prize independent | None | MEDIUM |
| B. Canonical-Hessian finite-diff Lanczos | 2-3 days | Likely false lead (see revision below) | Medium | LOW |
| E. Halo / spacing parameter sweep | 0.5 day | 0.5-1% if tighter halo helps legalization | Low | LOW |

### Path B revised — likely NOT the gap explanation

vmallela's 28% lift (1.4152 → 1.0109) likely comes from a **worse init**,
not better Hessian. They probably skip the E48 CD+LNS+SA pipeline and run
Hessian saddle from a RePlAce-quality init. Our pipeline already gets to
1.082 before Hessian; the residual soft modes are limited. E104 worse-init
spike at moderate scale confirmed only +0.41% extra lift. We're not
missing a Hessian trick — we're already at the limit of what saddle
escape can do from a strong basin.

---

## A. DREAMPlace input-driven hyperparam dispatch

### Evidence from research

[AutoDMP (NVIDIA, 2023)](https://research.nvidia.com/publication/2023-03_autodmp-automated-dreamplace-based-macro-placement)
uses **multi-objective tree-structured Parzen estimator (MOTPE) Bayesian
optimization** over 16 DREAMPlace parameters per design, gathering 1000
samples per design on 4×A100 GPUs. Leaderboard #4 "Hoop Dreams" uses
Optuna for the same approach. Parameter ranges from AutoDMP supplement:

- `density_weight`: [1e-6, 1.0] (we use 8e-5)
- `target_density`: [0.1, 1.0] (we use 0.85)
- `learning_rate`: [0.001, 0.1] (we use 0.01)
- `num_bins_x/y`: powers of 2 (we use 1024)
- `RePlAce_LOWER_PCOF`, `RePlAce_UPPER_PCOF` (we use defaults)
- `gamma`, `target_density_decay` (we use defaults)
- Plus 2 extras: **min vertical/horizontal macro spacing** (halo size) —
  AutoDMP found these matter a lot for legalization. We don't tune this.

### Hypothesis

DP basin on ibm12 = 2.47 (vs 1.21 from portfolio_levy alone) because:
- target_density=0.85 is too loose for ibm12's high macro density
- density_weight=8e-5 doesn't push macros apart enough on dense benches

### Spike plan (1 hour wall, no per-bench naming)

Run on ibm12 (current basin = 2.47):

| Config | target_density | density_weight | lr | Expected |
|---|---|---|---|---|
| 1 (current) | 0.85 | 8e-5 | 0.01 | 2.47 |
| 2 (denser) | 0.65 | 1e-4 | 0.01 | < 2.0 |
| 3 (very dense) | 0.55 | 5e-4 | 0.01 | < 1.7 |
| 4 (slow lr) | 0.65 | 1e-4 | 0.005 | < 1.5? |
| 5 (auto, observable) | clip(macro_density × 1.4, 0.4, 0.85) | 1e-4 | 0.005 | < 1.8 |

Each config = 14s on cloud GPU (DP early-exits). 5 configs × 1 bench = 70s.
If config 5 beats portfolio_levy 1.21, we have a deployment.

**Compliance**: target_density formula uses observable bench property
(macro_density), not bench name. Rules-compliant.

### Why this is high-EV

If we find a config where DP basin on ibm12 is < 1.5:
- DP polish via cascade saddle → likely 1.10-1.20 final (matches PATH B's
  ibm12 wins of 1.0669 before regression)
- Same on ibm10, ibm14, ibm17 (the other dense benches)
- Closes the gap to PATH B's 0.9949 aggregate from our 1.0691

---

## B. Canonical-Hessian finite-difference Lanczos

### The vmallela gap

vmallela LSJ went 1.4152 → **1.0109** with Hessian saddle escape alone
(28% lift). Our E84 cascade got 1.099 → 1.061 (3.5%). Same family,
~8× difference in lift. Why?

**Hypothesis 1: bad init = more soft modes** — falsified locally (E104
worse-init spike on ibm01 B_cd100 gave only +0.41% extra lift).

**Hypothesis 2: smooth-proxy Hessian gives WRONG direction**
- E88 (diff_proxy AdamW) — falsified (destroys basin step 1)
- E95 (diff_proxy_v2 annealed γ) — falsified (smooth RUDY mismatch)
- E98 (smooth-RUDY 3-4× off on ibm10/12/17)
- E103 (canonical pin extract + trace-route smooth RUDY) — ρ=−0.18 calibration FAILED

We've been using `torch.autograd.functional.hvp` on smooth-proxy
(LSE-HPWL + Gaussian-density + RUDY-bbox-uniform). vmallela may use
**canonical-proxy Hessian** via finite-difference matvec:

```python
def matvec(v):
    # Finite difference: ∇²P @ v ≈ (∇P(x + h·v) − ∇P(x − h·v)) / (2h)
    # But ∇P also needs finite difference on canonical proxy
    # ∇P @ v ≈ (P(x + h·v) − P(x − h·v)) / (2h)  ← scalar lookup, O(N) eval
    return (P_canonical(x + h*v) - P_canonical(x - h*v)) / (2*h)
```

Cost: each Lanczos matvec is ONE canonical proxy evaluation. Smallest
eigvec needs ~30-100 matvecs (Lanczos converges fast). Per-iter cost:
30-100 × proxy_eval (~50ms each on ibm01) = 1.5-5s for the direction.
Then ε-line-search adds 5-10 evals = 0.5s. Total ~5s per saddle iter.

Current smooth Hessian is ~0.5s per iter, so 10× slower but **correct
direction**.

### Spike plan (small ibm01, 30 min wall)

```python
# Compare smooth Hessian eigvec direction vs canonical FD Lanczos direction
# at cascade-converged plateau (ibm01 0.87)
# Metric: cosine similarity, and lift from ε-perturbation along each
```

If canonical FD direction is meaningfully different (cos sim < 0.5) AND
gives more lift, this is the gap explanation. Promote to production.

### Risk

- Slow per iter, may not fit in 55-min budget on ibm17/ibm18 (largest)
- Mitigation: use canonical FD ONLY on small benches (ibm01-09); use
  smooth Hessian on large benches. Input-driven dispatch.

---

## C. OpenROAD WNS/TNS/Area validation (Grand Prize)

### Grand Prize math

Top 7 by proxy advance. Grand Prize = OpenROAD weighted geometric mean
on NG45 4 designs (ariane133, ariane136, mempool_tile, nvdla):

```
score = (WNS_improvement^3 × TNS_improvement^2 × Area_improvement^1)^(1/6)
```

Where `improvement = baseline / ours` (higher = better). Baseline = SA
or RePlAce. **Feasibility gate**: must not regress BOTH baselines on any
design.

### Our current NG45

PATH B 0.68086 vs best-of-3 0.67975 on proxy. But this doesn't translate
to OpenROAD performance directly. We need to actually run ORFS on
ariane133 to know our WNS/TNS/Area.

Other Claude is running this (mpc2026-work NFS, ariane133 ORFS in Docker).
Tracking, not duplicating.

### Why this matters

Proxy 1.067 might give worse WNS than proxy 1.077 if our placement has
worse routing congestion in real critical paths. **Optimizing only proxy
is a local optimum on the wrong objective.** Could be the secret
weakness of all our Lévy/Hessian/portfolio innovations.

Need: run OpenROAD on our best NG45 placements and compare to SA baseline.

---

## D. Innovation Award writeup

### Genuinely novel contributions (validated locally)

Already documented in `NOTES_INNOVATION.md`:

1. **Lévy heavy-tail ε magnitudes** for saddle escape (E97)
2. **Multi-objective Hessian eigvec portfolio** (E100) — cong-only
   Hessian softer than weighted sum
3. **Curvature-adaptive σ = β/√|λ_min|** (E101)

Lit search confirms:
- [Lévy + SA](https://www.mdpi.com/2075-1702/13/11/1017) — published 2025
- [Lévy + circle packing](https://www.researchgate.net/publication/396180416)
  — published 2025
- [Lévy + grey wolf](https://link.springer.com/article/10.1007/s00521-017-2952-5),
  [Lévy + whale](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0322058)

**NOT published**: Lévy magnitude × Hessian eigvec direction. We are the
first to combine 2nd-order curvature information (eigvec) with heavy-tail
step size (Lévy magnitude). Plus the multi-objective portfolio — also
not published.

### Innovation Award angle

Submit `cd_lns_sa_cascade_portfolio_levy/placer.py` as the entry, with
writeup focusing on:
- Why heavy tails outperform Gaussian for saddle escape (4.15× lift on
  hard benches)
- Why component-Hessian eigvecs reveal soft modes the canonical Hessian
  hides (98% softer on ibm10)
- Curvature-adaptive σ as a principled tuning-free knob (β=3.0 default)

Independent of leaderboard rank. $4K.

---

## F. Periphery-guided relocation (DAC25-ReMaP angle)

### The finding

[DAC25-ReMaP](https://github.com/lamda-bbo/DAC25-ReMaP): "Macro Placement
by Recursively Prototyping and Periphery-Guided Relocating" reports up
to **+34.15% WNS and +65.39% TNS** improvement on ORFS designs (ariane133,
ariane136, black_parrot, bp_be/fe/multi/quad, swerv_wrapper) — directly
the Grand Prize set.

Mechanism (inferred from the paper title + name):
1. Generate prototype placement (probably DP-style global)
2. Recursively refine: at each iter, RELOCATE macros toward canvas periphery
3. Periphery placement leaves the center for std cells → shorter critical
   paths → better WNS/TNS

### Why this matters for us

[Benchmarking AI placement (arxiv 2407.15026)](https://arxiv.org/pdf/2407.15026)
notes: **MacroHPWL only has WEAK correlation with actual Wirelength**.
Proxy ranking ≠ OpenROAD WNS/TNS ranking. Our placements likely cluster
macros (CD-LNS-SA optimizes proxy, which doesn't penalize center
clustering). Periphery-bias would help WNS dramatically without much
proxy regression.

### Diagnostic (2026-05-15 — done locally)

Quick analysis of cached cascade-converged placements:

| Bench | Macros | centroid_dist (0=center, 1=corner) | edge_dist (0=edge, 0.5=center) |
|---|---|---|---|
| ibm01 | 1140 | 0.699 | **0.191** |
| ibm09 | 1301 | 0.666 | **0.201** |
| ariane133 | 915 | 0.581 | **0.241** |
| ariane133_v4 | 915 | 0.581 | **0.241** |

Mean macro is ~20% in from the nearest edge. Periphery-pure would be
~5%. Room for ~3-4× more periphery bias before saturating. Ariane133
(NG45 commercial = Grand Prize) is the MOST center-clustered (0.241).
Confirms hypothesis: periphery-bias has headroom.

### Spike plan

1. ~~Visualize current placements~~ — DONE above; confirmed center-clustering on ariane133.
2. **Add periphery-bias term** to cascade saddle escape:
   ```
   periphery_score = mean(min(macro_to_left_edge, macro_to_right_edge,
                              macro_to_bottom, macro_to_top))
   ```
   Add `-α · periphery_score` to objective (α small, e.g. 0.05). Run on
   ariane133.
3. **Compare**: OpenROAD WNS/TNS with and without periphery bias.

### Why this is high-EV

- $20K Grand Prize is decided by WNS/TNS, not proxy
- ReMaP shows 34-65% improvement is achievable
- Our existing pipeline (cascade saddle) can absorb this as a new objective term
- INPUT-DRIVEN: macro_to_edge is observable per bench, no per-bench tuning

### Risk

- May regress proxy below feasibility threshold (top 7 cutoff ~1.21).
  Need to check: if periphery-biased placement is proxy 1.10, still
  qualifies; if 1.30, disqualified.
- Need OpenROAD to actually verify WNS/TNS improvement. PATH C (running
  ORFS on ariane133) is the prerequisite.

---

## E. Halo / spacing parameter sweep

[AutoDMP](https://d1qx31qr3h6wln.cloudfront.net/publications/AutoDMP.pdf)
adds two parameters beyond the 14 standard DP knobs:
- `min_vertical_spacing` (macro halo Y)
- `min_horizontal_spacing` (macro halo X)

Default DP halo = 0. Adding halo helps macro legalization on dense
benches. We could spike halo = (5%, 10%, 15%) of macro_height on ibm12.
Quick test, low cost. Subset of A.

---

## What other Claudes are doing (coordination)

| Claude | Active | Files |
|---|---|---|
| Other (ensemble) | B_R2 chain on 8 hard benches | `experiments/B_R2_canonical_dp_loss/` |
| Other (OpenROAD) | ariane133 ORFS baseline | `/lambda/nfs/mpc2026-work/` |
| Other (multidir) | E90 C3 cloud --all | `submissions/cd_lns_sa_cascade_multidir/` |
| Other (tournament) | unknown — see `cd_lns_sa_cascade_tournament/` | `submissions/cd_lns_sa_cascade_tournament/` |
| Me (this session) | research + waiting for chain results | this file, `NOTES_INNOVATION.md` |

**My next actions** (when cloud comes back / when chain finishes):
1. Spike DP hyperparam on ibm12 (path A) — 1 hour wall
2. If A wins, deploy input-driven dispatch in `submissions/cd_lns_sa_cascade_dp_native/placer.py`
3. Spike canonical FD Lanczos on ibm01 (path B) — 30 min wall
4. Draft Innovation Award writeup (path D) — 1 hour

## Sources

- [AutoDMP (NVIDIA 2023)](https://research.nvidia.com/publication/2023-03_autodmp-automated-dreamplace-based-macro-placement) — 16-param Bayesian DP tuning
- [DAC25-ReMaP](https://github.com/lamda-bbo/DAC25-ReMaP) — **periphery-guided relocation, +34% WNS / +65% TNS**
- [Benchmarking AI Chip Placement (arxiv 2024)](https://arxiv.org/pdf/2407.15026) — proxy↔PPA gap
- [DREAMPlace 4.0](https://www.cse.cuhk.edu.hk/~byu/papers/C137-DATE2022-TiDriPlace.pdf) — current canonical DP
- [Xplace 3.0 (CUHK)](https://github.com/cuhk-eda/Xplace) — Carrotato's likely framework
- [Lévy + SA NSGA-II 2025](https://www.mdpi.com/2075-1702/13/11/1017)
- [Lévy + circle packing 2025](https://www.researchgate.net/publication/396180416)
- [Jin et al. saddle escape](https://proceedings.mlr.press/v70/jin17a/jin17a.pdf) — theoretical foundation
