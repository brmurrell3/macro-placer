---
id: E132
name: polish_relax
status: marginal
parent: E127
created: 2026-05-21
decided: 2026-05-21
champion_at_time: 0.987  # combined 21-bench (Option C stacked_periphery)
outcome: 0.92211         # ibm04 smoke; warmstart REVERTED, equivalent to CD-only
champion_delta: null     # not comparable on single bench; warmstart didn't fire
graduated_to: null
superseded_by: null
---

# E132: polish_relax (V4Gauss + CD + warmstart re-descent + CD)

## Hypothesis

After CD polish reaches a single-axis plateau, a small-step Adam
warm-restart (γ tiny, λ_overlap strong) from the CD-polished basin
can move groups of macros simultaneously and escape the CD plateau.
The second CD polish then locks in the new lower basin. This
differs from E128 (Hessian saddle) by skipping the eigenvector
machinery; pure warm-restart of descent.

## Method

Pipeline (vs thinkorplace-v2 which is V4Gauss → CD only):

1. V4Gaussian descent (500 steps, init="sdf", normal params)
2. greedy_macro_legalize + project_overlaps
3. run_cd_adaptive (~200s)  → pos_polished
4. **NEW**: V4Gaussian warm-restart from pos_polished
   - 80 steps, lr_frac=0.0008 (tiny), γ flat at 5e-5,
     overlap_lambda_end=50 (strong), init="warmstart"
5. project_overlaps
6. run_cd_adaptive (~200s) → final

Implementation: subclass `SmoothGlobalPlacerV4Gaussian` to add a
`init="warmstart"` branch using a `warmstart_pos` ctor kwarg.
No modification of shipped files.

## Kill gate

ibm04 smoke: final proxy > 0.93 (i.e. no improvement vs v2 0.929)
OR wall > 900s (over budget) → falsified.

## Generalization check

If smoke passes (proxy < 0.92 or comparable to E128 0.9165),
proceed to --all. Need combined avg < 0.987 to graduate over Option C.

## Outcome (filled when decided)

**Smoke ibm04 (M3 MPS, 4 E129 evaluators running concurrently):**

| Stage | Proxy | Ovl | Wall |
|---|---:|---:|---:|
| V4Gauss basin | 1.08619 | 0 | 21s |
| CD1 (200s budget) | 0.92395 | 0 | ~190s |
| Warmstart raw (80 steps Adam) | 1.01642 | **6** | 5.6s |
| Warmstart projected | (failed; still 4 ovl) | 4→revert | — |
| Used for CD2 | 0.92395 (CD1 pos) | 0 | — |
| CD2 (200s budget) | 0.92211 | 0 | ~110s |
| **Final** | **0.92211** | **0** | **330s** |

Reference: v2 ibm04 0.929, E128 ibm04 0.9165.

**Warm-restart didn't fire as designed.** The 80-step Adam re-descent
from a CD-polished basin with lr_frac=0.0008, γ=5e-5, λ_ovl_end=50
ramped over 40 steps **diverged by +10.0 % proxy and created 6
overlaps**. `project_overlaps` could not legalize (4 residual overlaps),
so the 5 %-divergence guard reverted to the CD1 position. The CD2 polish
ran from the same CD1 position as CD1 started but with different sweep
RNG and found a marginal -0.18 % improvement (within the documented
M3 run-to-run variance of 0.3-0.5 %).

**Diagnosis.** lr_frac=0.0008 × canvas_width (ibm04: 700 µm) gives a
0.56 µm per-step displacement, which Adam compounds over 80 steps with
momentum into multi-cell motion. The macros are already in legal
non-overlapping positions after CD1, so even a small "refinement" push
under the high overlap penalty (50) makes some macros bounce out of
their tight nearest-neighbor packings into infeasible territory.

The warmstart-relax hypothesis as parameterized here is FALSIFIED on
ibm04 (no escape signal, just destruction). Possible re-tunes
(decimal-order smaller lr, much shorter step count) are independent
experiments; cost would still be ~30s overhead with no clear evidence
of basin escape mechanism. **Do not run --all** at these settings.

Smoke wall 330s is well under the 720s budget and well under the kill
gate (900s).

**Status decision.** `marginal`, not `falsified`, because the smoke
also showed nothing structurally broken in the pipeline — the guard
correctly reverted on overlap divergence, CD2 ran cleanly, final is
0-overlap. If a future experiment wants to revisit warm-restart with
smaller lr (e.g. 1e-4) or fewer steps (e.g. 20), the infrastructure
is in place.

## Pointers
- Code: `code/placer.py` + `code/smooth_global_placer_v4_gaussian_warmstart.py`.
- Smoke log: `/tmp/e132_smoke.log`.
