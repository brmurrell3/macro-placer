---
id: E55
name: congdestroy_ablation
status: dropped
parent: E54
created: 2026-05-02
decided: 2026-05-02
champion_at_time: 1.0990 (E12); 1.08151 (E48 hybrid); 1.0848 (E41); E54 --fast 0.9222 (compositional 0.91708)
outcome: dropped before launch — parent E54 falsified on NG45 (ariane133 +5.14 %) before this ablation became worth running. No point testing the net-share term when the destroy direction itself is NG45-blind.
champion_delta: n/a
graduated_to: null
superseded_by: null
---

# E55: congdestroy_ablation — load-bearing test of net-share weighting

## Hypothesis
E54's `_congestion_aware_destroy` scores each macro as
`direct_macro_routing + sum_over_nets(net_routing × 1/n_pins)`. The
1/n_pins pin-share weighting is a defensible default but not the only
choice. If `include_net_share=False` (i.e., direct macro routing only)
gives equivalent or better lift, the simpler ranking wins (Occam's
razor + smaller wall) and we can drop the net-share computation.

If the simpler version regresses, we know the net-share contribution
is load-bearing — the indirect routing through connected nets is what
drives the lift, not direct macro contribution.

This is a clean ablation: same pipeline as E54, only change is the
boolean flag.

## Method
Subclass `CDLNSSADPOKJointCongDestroyPlacer` from E54 with
`lns_include_net_share=False`. Everything else identical.

## Kill gate
- **--fast > E54 fast 0.9222 + 0.5 % = 0.927:** kill — net-share is
  load-bearing, can't drop.
- **--fast within ±0.2 % of E54:** simpler ranking is sufficient;
  promote as E55 to displace E54's net-share branch.
- **--fast < E54 by ≥ 0.3 %:** unexpected — simpler version is BETTER.
  Investigate (would be surprising; net-share is theoretically
  beneficial).

## Generalization check
If --fast is within range, run --ng45. NG45 ariane133 is the
sensitive design; net-share might matter MORE there since its
high-pin-count nets contribute disproportionately.

## Outcome (filled when decided)
[Empty until decided.]

## Pointers
- Code: `code/cd_lns_sa_dpo_kjoint_congdestroy_no_netshare.py` (thin
  subclass of E54's placer).
- Parent: E54 (`experiments/E54_congestion_destroy/code/cd_lns_sa_dpo_kjoint_congdestroy.py`).
- Discussion: motivated 2026-05-02 by E54 --fast success (0.91708
  compositional). Whether the lift comes from `top_pct` cell selection
  or the per-net pin-share weighting determines if the simpler version
  is sufficient.
