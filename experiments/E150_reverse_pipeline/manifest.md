---
id: E150
name: reverse_pipeline
status: falsified
parent: E132
created: 2026-05-21
decided: 2026-05-21
champion_at_time: 0.984  # v2-extCD EPYC --all
outcome: 1.01312        # ibm04 smoke; midAdam reverted, CD2 = CD1 + noise
champion_delta: null    # single-bench, not comparable to combined avg
graduated_to: null
superseded_by: null
---

# E150: reverse_pipeline (SDF → CD → Adam → CD)

## Hypothesis

Our entire v2 / E127 / E128 lineage starts with Adam descent on a smooth
proxy from an SDF init, then polishes with CD. E132 confirmed that
warm-starting Adam from a CD-polished V4Gauss basin diverges and gets
discarded — i.e. running Adam *after* its own descent + CD destroys the
basin. But maybe the ordering itself was wrong. If we **skip the initial
Adam descent** and instead run CD directly on the SDF init, the
CD-polished position is a canonical (sharp) local optimum unbiased by
the smooth-proxy mismatch. Adam started from *that* canonical anchor
descends a smooth landscape from a *different* starting basin than
Adam-from-SDF; the second CD lock-in finds either the same basin or
a structurally different one.

Pipeline B (E150):
  SDF → project_overlaps → CD₁ (300s)
       → Adam descent (smooth proxy, warmstart from CD-polished pos,
         tiny lr, tight γ, strong λ_ovl)
       → legalize + project_overlaps
       → CD₂ (400s)

vs Pipeline A (v2/E127/E132 lineage):
  SDF → Adam → legalize → CD (single polish, possibly + warmstart Adam
       like E132 → CD₂).

## Method

- Re-use `experiments/E132_polish_relax/code/smooth_global_placer_v4_gaussian_warmstart.py`
  (already supports `init="warmstart"` + `warmstart_pos=` ctor kwarg).
  Do **not** modify shipped files.
- Stage 1: `sdf_init` + `project_overlaps`, then `run_cd_adaptive` 300s.
- Stage 2: Adam descent via the warmstart class. lr_frac=0.001 (very
  small — half of E132's lr_frac=0.0008 in displacement terms, because
  here we don't have the V4Gauss-basin starting point), γ flat 5e-5,
  λ_ovl_end=20 (medium, not 50 — E132 saw 50 push macros out of legal
  packings).
- Stage 3: project_overlaps, then `run_cd_adaptive` 400s.
- Total wall budget: 1500s.

## Kill gate

- ibm04 smoke: middle-Adam wrecks CD1 basin AND CD2 cannot recover
  (final > CD1 by >2%) → falsified.
- ibm04 smoke final proxy > 0.93 (worse than v2-extCD 0.92 by >1%) AND
  middle-Adam shows no escape signal → falsified.
- Wall > 1800s (over budget) → falsified.

## Generalization check

If smoke proxy < 0.92 (matches or beats v2-extCD) → run --all.
Need EPYC --all combined avg < 0.984 to graduate over v2-extCD.

## Outcome (filled when decided)

**Smoke ibm04 (M3 MPS):**

| Stage | Proxy | Ovl | Wall |
|---|---:|---:|---:|
| SDF init + project | 1.38256 | 0 | 6.2s |
| CD1 (300s budget) | **1.01552** | 0 | ~290s |
| midAdam raw (300 steps Adam) | 1.08094 | **3** | 8.8s |
| midAdam projected | (failed; 1 ovl) | revert | — |
| midAdam used (= CD1 pos) | 1.01552 | 0 | 15.3s |
| CD2 (400s budget) | 1.01312 | 0 | ~73s |
| **Final** | **1.01312** | 0 | **389s** |

Reference: v2-extCD ibm04 0.92, E128 ibm04 0.9165.

**Hypothesis falsified — same failure mode as E132.** midAdam (Adam
descent from CD-polished pos, lr_frac=0.001, γ=5e-5, λ_ovl_end=20)
diverged from CD1 by +6.4 % and created 3 overlaps in 8.8s.
`project_overlaps` could not legalize (1 residual overlap), so the
divergence guard reverted midAdam to CD1. CD2 ran from the same CD1
position with different sweep RNG and found Δ=−0.24 % (well within the
M3 run-to-run variance band 0.3–0.5 %).

**More importantly: SDF→CD1 alone yields 1.01552 on ibm04, vs v2-extCD's
SDF→V4Gauss→CD ≈ 0.92.** The CD-only basin is structurally inferior to
the V4Gauss-then-CD basin by ~10 %. CD alone, even with 300s budget,
cannot escape SDF's HPWL-driven greedy packing into a low-congestion
configuration; only Adam's smooth-proxy descent (with congestion in the
gradient) can drag macros into the low-cong basin that CD then locks in.

So the reverse-pipeline framing was the wrong intuition: V4Gauss isn't
biasing CD into a worse basin (which the hypothesis assumed) — V4Gauss
is the *only* mechanism that finds the low-cong basin at all. CD-first
starts at a much higher canonical local optimum, and midAdam from there
cannot reliably step (Adam destroys the legal packing for the same reason
E132 hit).

**Status: falsified.** ibm04 final 1.01312 ≫ v2-extCD's 0.92 (+10.1 %).
This is not a marginal call — the pipeline is missing the V4Gauss
descent that creates the working basin. Do NOT run --all; even if midAdam
fired cleanly on some other bench, the CD1 starting basin is the binding
constraint.

**Implication for E132's interpretation.** E132 ascribed warmstart's
divergence to the small-but-compounding Adam step destroying the tight
CD basin. E150 reproduces the exact same divergence (1.08094 vs 1.01552
= +6.4 %, vs E132's 1.01642 vs 0.92395 = +10 %) at lr_frac=0.001 vs
E132's 0.0008. Basin-destruction is robust across lr scale and starting
basin, which suggests the failure is intrinsic to "Adam from a
CD-polished canonical optimum" rather than a hyperparameter choice.
Future work should NOT pursue warm-restart Adam after CD without a
different mechanism (e.g. lock high-net-degree macros, project gradient
onto tangent space of feasible cones, etc.).

## Pointers
- Code: `code/placer.py` (imports E132's warmstart class).
- Smoke log: `/tmp/e150_smoke.log`.
- Per-stage trace: SDF 1.38256 → CD1 1.01552 → midAdam 1.08094
  (3 ovl, reverted) → CD2 1.01312. Wall 389s, well under 1500s budget.
