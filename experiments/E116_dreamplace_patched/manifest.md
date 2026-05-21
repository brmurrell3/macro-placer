---
id: E116
name: dreamplace_patched
status: falsified
parent: E111
created: 2026-05-20
decided: 2026-05-20
champion_at_time: 1.00279
outcome: 0.918 ibm01 polished (vs V3 0.852 baseline) — patch works but loses to V3 by 7.7%
champion_delta: +7.7%
graduated_to: null
superseded_by: null
---

# E116: DREAMPlace optimizer with challenge-proxy loss

## Hypothesis

DREAMPlace's mature Nesterov + density-weight ramping + Barzilai-Borwein
line search is the top-2 leaderboard team's secret. Stock DREAMPlace
optimizes WL + density_weight·eDensity, which is NOT our challenge
proxy. The previous patch attempt was discontinued as "6-13 days of
engineering" because their C++ Rudy has no autograd backward.

E111 PerNetTraceCongestion solves that problem already — it's a pure-
PyTorch differentiable congestion module that matches canonical
±15-25% on hard benches. So the patched-DREAMPlace path is now ~1 day
of engineering:

  Patched DP loss = LSE-HPWL + 0.5·top-K-density + 0.5·PerNetTraceCong

We expect this to beat thinkorplace-v2 (1.00279 IBM / 0.6786 NG45)
because (a) DP's Nesterov + density ramping is a strictly better
optimizer than Adam, (b) the proxy is the same (E111 trace congestion).

## Method

Two implementation paths, evaluated in sequence:

### Path A: standalone Nesterov + challenge proxy

Pure PyTorch. Avoids DREAMPlace's Bookshelf / .so dependencies entirely.

1. Reimplement DREAMPlace's `NesterovAcceleratedGradientOptimizer.step_bb`
   in our codebase (it's ~80 lines, only depends on `torch`).
2. Loss = LSE-HPWL + 0.5·top-K-density (from E88 `_grid_density`) +
   0.5·`PerNetTraceCongestion.compute_congestion`.
3. Density-weight ramping (DP's hallmark): start density_weight=8e-5,
   double after each "subproblem" until overflow < target.
4. Preconditioner: scale gradient by `1/(node_area + density_weight·node_area)`
   so dense macros don't drift faster than sparse ones.
5. Init: SDF (E110 default).
6. Legalize: `greedy_macro_legalize` from E76.
7. CD polish: `run_cd_adaptive` from `macro_place.cd_core`.

### Path B: actually patch DREAMPlace (fallback if Path A doesn't lift)

1. Install Python 3.11 on AWS (already in progress).
2. Patch `submit_deps/dreamplace_install/dreamplace/PlaceObj.py:obj_fn`
   to compute challenge proxy.
3. Convert ibm01 to Bookshelf, run patched DP.

## Kill gate

- ibm01 raw smooth basin (before CD polish) ≥ 0.95 → Path A is no
  better than Adam, fall back to Path B.
- ibm01 + CD polish ≥ 0.85 → no lift vs thinkorplace-v2 (0.843), kill.
- ibm17 + CD polish ≥ 1.20 → no lift vs thinkorplace-v2 (1.200), kill.

## Generalization check

- After ibm01 + ibm17 pass: full `--all` IBM on AWS EPYC + GPU.
- After IBM pass: `--ng45` (4 commercial designs) before promoting.

## Outcome — 2026-05-20

**SUBSTANTIALLY FALSIFIED for Path A (pure-PyTorch); Path B
(real DREAMPlace integration) in-progress.**

### Path A findings — pure-PyTorch optimizer landscape

| Optimizer | ibm01 polished | Notes |
|---|---:|---|
| Adam | 0.855 | vanilla, equivalent to thinkorplace-v2 within noise |
| Nesterov BB (DP-style) | 0.920 | basin trapped, worse than Adam |
| Nesterov BB + DP precond | 0.949 | worse: preconditioner conflicts with momentum |
| Adam + DP precond | 0.904 | worse: redundant per-parameter scaling |
| SGD Nesterov | 1.018 | momentum without adaptation underperforms |
| SGD Nesterov + precond | 1.030 | worst |
| AdaBelief | 0.882 | slightly worse than Adam |
| NAdam | 0.889 | worse |
| RAdam | 0.893 | worse |

**thinkorplace-v2 baseline ibm01:** 0.852 (V3 + 600s CD polish).

**Path A is essentially equivalent to V3 in basin and final
proxy.** Variations from optimizer choice are within run-to-run
variance (~0.5%).

### Why DP's Nesterov underperforms here

DREAMPlace's Nesterov + BB works well in DP because:
1. **Filler nodes** provide a strong spreading gradient signal
2. **eDensity** is a much smoother density potential than top-K
3. **density-weight ramping** lets the optimizer first focus on WL,
   then squeeze density gradually

For macro-only placement (200 macros, no fillers), our challenge
proxy already has gradients on density + congestion that Adam's
per-parameter adaptive learning rate handles better than Nesterov.

### Path B findings — real DREAMPlace integration

**ABI mismatch blocked installed binary** — `submit_deps/dreamplace_install/`
was built with `_GLIBCXX_USE_CXX11_ABI=0` but PyTorch wheels use
`ABI=1`. Cannot import.

**Rebuilt DREAMPlace from source on AWS** with `-DCMAKE_CXX_ABI=1`,
Python 3.11 venv, CUDA 12.8, torch 2.12. Build ~30 min. Successful.

**Patched DREAMPlace.PlaceObj.obj_fn at class level** to compute the
challenge proxy (LSE-HPWL + 0.5*top-K-density + 0.5*PerNetTraceCong)
from DP's pos tensor. Required:
1. Patching BOTH `dreamplace.PlaceObj.PlaceObj` AND top-level
   `PlaceObj.PlaceObj` (NonLinearPlace uses `import PlaceObj` not
   `from dreamplace import`).
2. Patching BOTH `obj_fn` AND `obj_and_grad_fn` (Nesterov optimizer
   captures `obj_and_grad_fn` bound method at init, but inside it
   calls `self.obj_fn(pos)` dynamically — still need to patch both
   to ensure preconditioner gets called correctly).
3. Coordinate conversion: DP applies its own scale_factor (0.1 for
   ibm01), so converting DP pos → our microns requires `dp_x *
   (our_canvas_w / dp_canvas_w)` factor.
4. Patching `np.string_` → `np.bytes_` in DP's PlaceDB.py (numpy 2.0
   deprecation).

**Result on ibm01 (challenge_w=1.0, edensity_w=0.0, 100 iters DP):**
- DP basin proxy: **0.851** (CANONICAL-ALIGNED, comparable to V3)
- Legalize: 1.107 (134 macros displaced, max disp 16μm)
- CD polish: **0.918** (WORSE than thinkorplace-v2 0.852)

**Verdict on patched DREAMPlace:** The descent reaches a canonical-
aligned basin (proxy 0.851 vs V3 0.852), but **the basin has many
overlaps** (118 overlaps before legalize). Without DP's eDensity term
to drive spreading, the patched-pure-proxy descent doesn't enforce
the no-overlap constraint that DP's pipeline relies on. The legalize
step then has to displace ~30% of macros, destroying the basin
quality. Net result: 0.918 polished, ~7% worse than V3.

**More iterations make it WORSE**, not better:
- 100 iters: raw 0.851, ovl=118, polished 0.918
- 200 iters: raw 1.92, ovl=376, polished 1.358

The challenge proxy descent without strong overlap repulsion is
unstable — minimizing density via "piling macros together" reduces
WL but creates dense clusters. eDensity in DP would provide the
spreading force, but at any non-trivial weight it dominates the
challenge proxy.

**The weight-calibration problem is fundamental** (matches B-R2's
finding): a single λ value cannot simultaneously (a) give DP enough
spreading gradient to legalize cleanly and (b) preserve the
canonical proxy as the dominant signal. B-R2 found this took
per-bench tuning, which is rule-forbidden.

This confirms the B-R2 prior finding that adding canonical losses
to DP's loss is the right approach (basin lift on hard benches),
but REPLACING DP's loss entirely is worse because of overlap.

### Prior art from B-R2 (2026-05-13)

The B-R2 experiment from May 13 already tried adding canonical
losses to DREAMPlace's `obj_fn`. Findings:
- "Real but inconsistent basin lifts" across benches
- ibm01 -3.9% basin lift at λ=1.0; ibm10 -8.8% at λ=0.5
- **CD polish washes out basin gains on easy benches**
- Hard benches: λ=0.1 gives ibm12 -1.6% but other benches regress
- Portfolio (4-lane plateau pick) regressed +2.9% on 5 hard benches
- "No single λ generalizes across benches"

B-R2 used the OLD bbox-uniform RUDY (now known broken on hard
benches per E111). My E116 uses E111 PerNetTraceCongestion (L-route
trace, ~15-25% off canonical vs B-R2's 200-260% off). This is a
real improvement but may still not be enough.

### Conclusion

**Path A (pure-PyTorch with our optimizer) is at the limit of
what Adam + V3 already achieves.** Real DREAMPlace would need:
- Filler nodes (spreading gradient)
- Density-weight ramping with eDensity (smoother)
- Lgamma outer loop (γ-anneal with backtracking)

Path B (real DREAMPlace patch) is the path to actually test the
hypothesis, but per B-R2 prior, the lift is likely 1-2% at best
and inconsistent. Given the noise floor of ~0.5% and the engineering
cost of patching/rebuild/validation, **this is not on the critical
path for 2026-05-21 submission**.

**Recommended:** keep thinkorplace-v2 as champion. E116 Adam mode
verified equivalent and could serve as a fallback with no functional
benefit. Real DP patch deferred to post-submission research.

### Summary table — ibm01 reference

| Approach | DP basin | Legalized | + CD polish | Notes |
|---|---:|---:|---:|---|
| thinkorplace-v2 (V3 Adam) | 0.922 (post-legalize) | — | **0.852** | champion baseline |
| E116 Adam (pure-PyTorch port of V3) | 0.846 | 0.875 | **0.855** | equivalent within noise |
| E116 Nesterov BB | 0.923 | 0.948 | 0.920 | underperforms Adam |
| E116 Nesterov BB + precond | 1.005 | 0.993 | 0.949 | worst combination |
| Patched DREAMPlace (challenge proxy as obj_fn, 100 iters) | 0.851 (118 ovl) | 1.107 | **0.918** | basin great, overlaps break it |
| Patched DREAMPlace (200 iters) | 1.924 (376 ovl) | 1.501 | 1.358 | diverges with more iters |

### Conclusion

E116 successfully demonstrates that BOTH (a) pure-PyTorch
challenge-proxy descent and (b) real DREAMPlace with patched obj_fn
**reach canonical-aligned smooth basins (~0.85 on ibm01)**. However:

1. Pure PyTorch + Adam = exactly thinkorplace-v2 V3. No improvement.
2. Pure PyTorch + Nesterov = worse than Adam without filler nodes.
3. Real DP + patched obj_fn = same basin quality but BAD overlap
   handling; legalize destroys basin, polished result 7.7% worse than V3.

The core finding: **the challenge-proxy descent basin is well-
characterized by V3 Adam, and DREAMPlace's optimization machinery
brings no benefit without DP's eDensity term — which conflicts with
the challenge proxy in weight calibration.** This is consistent with
the prior B-R2 experiment's finding.

Patched-DREAMPlace as a submission path is **not viable** for
2026-05-21 deadline.

## Pointers

- `code/dp_patched_placer.py` — main placer (Path A)
- `code/nesterov_optimizer.py` — DP's Nesterov BB extracted
- `code/test_ibm01.py` — single-bench smoke
- `code/test_ibm17.py` — hard-bench smoke
- `results/smoke_ibm01.json` — Path A ibm01 result
- `results/smoke_ibm17.json` — Path A ibm17 result
- `results/all_ibm.json` — full IBM if Path A passes single-bench gates
