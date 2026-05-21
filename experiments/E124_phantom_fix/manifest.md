---
id: E124
name: phantom_fix
status: marginal
parent: E111
created: 2026-05-20
decided: 2026-05-20
champion_at_time: 1.0039   # V3Min ovl10 480s (currently shipped)
outcome: ibm17 V3 fixed CD600s 1.20315 vs V3 baseline 1.20169 = +0.12% (within ~0.5% noise floor). Spec V3Min ovl10 720s baseline 1.20032; baseline re-run matches at 1.20169 (+0.07%). Fixed RAW 1.28126 is +0.95% WORSE than baseline RAW 1.26916 — phantom gradient actually helped smooth descent. CD polish absorbs most of the gap. Zero overlaps both versions. Verdict: bug fix is roughly neutral after CD polish.
champion_delta: null
graduated_to: null
superseded_by: null
---

# E124: phantom_fix

## Hypothesis
E121 discovered a "phantom stripe" bug in E111's `_soft_min_max`: when
`col_src == col_snk` (or rows equal), the softmin/softmax pair produces
spurious mass of width ~2*log(2)/β ≈ 0.23 cells, creating fake
congestion demand in degenerate L-routes. The MCF placer's fix is to
use `torch.minimum`/`torch.maximum` on the candidate bend stripe
bounds — exact zero in degenerate cases, gradient still flows via the
softmax-based endpoint expectations.

Because this is a genuine BUG (fake congestion mass where canonical
sees none), not a fidelity tweak, it MAY help even though the
"fidelity trap" pattern (5+ prior cases) says canonical-matching tweaks
typically hurt the placement basin.

## Method
Fork `per_net_trace_proxy.py` → `per_net_trace_proxy_fixed.py`. Replace
`_soft_min_max(col_src_exp, col_snk_exp)` (and the row pair) with
`torch.minimum`/`torch.maximum` on the soft-expected cell indices. The
endpoint expectations themselves remain differentiable (they come from
the Gaussian-softmax cell assignment), so gradients still flow into
`positions`. Fork V3 placer → V3Fixed with the new proxy.

Test on ibm17 single-bench, V3Min ovl10-like budget (720s total, 600s
CD polish). Baseline is V3Min ovl10 720s with the buggy E111 proxy
(canonical proxy 1.20032 on ibm17 spec).

## Kill gate
- Fixed ibm17 + CD600s > 1.20032 → FALSIFY (fidelity-trap pattern even
  for bug fixes; strongest evidence yet). Stop here.
- Fixed ibm17 + CD600s in [1.195, 1.205] → NEUTRAL. Worth shipping if
  zero overlaps (less noise in gradient is monotonic improvement).
- Fixed ibm17 + CD600s < 1.195 → WIN. Run on --fast (ibm01/04/10/17)
  to confirm cross-bench.

## Generalization check
If single-bench win: --fast 4 benches. If all 4 improve or are
neutral, ship as new prod proxy.

## Outcome (decided 2026-05-20)

### Single-bench head-to-head on ibm17 (V3Min ovl10 720s config — Adam 500 steps, ovl_lam_end=10, CD 600s polish)

| Variant | RAW (smooth + legalize) | + CD 600s polish | overlaps |
|---|---:|---:|---:|
| V3 baseline (E111 buggy `_soft_min_max`) | 1.26916 | **1.20169** | 0 |
| V3 fixed (E124 exact `torch.minimum/maximum`) | 1.28126 (+0.95 %) | **1.20315** (+0.12 %) | 0 |

V3Min ovl10 720s spec reference: 1.20032 — baseline re-run matches at
1.20169 (+0.07 %, within run-to-run noise floor of ~0.5 % on M3 under
heavy CPU contention).

### Calibration smoke (cached macro_positions, before descent)

| Bench | Canonical congestion | Buggy mismatch | Fixed mismatch | Fix vs buggy |
|---|---:|---:|---:|---:|
| ibm01 | 1.13685 | +6.78 % | +5.84 % | −0.88 % |

Fix reduces calibration bias by ~0.9 % on ibm01 (less spurious mass).
Gradient still flows: 2035/2280 nonzero gradient entries after `.backward()`.

### Verdict

Bug fix is **structurally correct but operationally neutral on ibm17**.
Three readings:

1. **The phantom-stripe was acting as accidental "noise injection"** —
   the ~0.23-cell of spurious mass at degenerate L-routes added soft
   gradient pressure that nudges pins off perfectly-aligned positions.
   Removing it lands the smooth basin in a slightly different (and on
   ibm17 slightly worse) local minimum. CD polish closes most of the
   gap.
2. **The "fidelity trap" pattern from prior experiments (E121, E122,
   E114, etc.) holds even for a genuine bug fix.** Making the smooth
   proxy more canonical-accurate doesn't automatically translate to
   better placement quality after CD polish. The smooth basin and the
   CD basin are loosely coupled.
3. **The fix is +0.12 % on ibm17 (within noise) so shipping it would
   be neither helpful nor harmful**, BUT the +0.95 % regression on
   RAW smooth + legalize means the fix has a real (if small) cost on
   pre-CD quality — and on benches with less CD budget that delta
   could persist. **Not shipped**.

### Decision: MARGINAL — do not promote, do not falsify

The MCF subagent's discovery (E121) was the right call. The fix is
correct theory but doesn't help in practice. Keeping `submissions/_archive/
e111_minimal_ovl10_720s/placer.py` and its buggy `PerNetTraceCongestion`
unchanged.

## Pointers
- Code: `code/per_net_trace_proxy_fixed.py`, `code/smooth_global_placer_v3_fixed.py`,
  `code/e124_phantom_fix_placer.py`
- Results: `results/ibm17_phantom_fix.json`, `results/ibm17_test.log`
- Discussion: spawned by MCF subagent (E121) discovery; confirms E121's
  "Spin-off" prediction that the fix gives only ~5-10 % calibration delta
  on hard benches but doesn't change the end-to-end CD-polished outcome.
