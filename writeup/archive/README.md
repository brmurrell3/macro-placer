# Archive

April–May 2026 material that later records still cite. Technical report 1.0
supersedes the numbers and framings here.

- `2026-05-drafts/evidence.md`: the May 2026 experimental data log (per-design
  tables, ablations, dead ends), cited by experiment manifests, ADRs and the
  `analysis/` probes. The May paper drafts beside it were removed in October 2026;
  read them at commit `b7de373` with `git show b7de373:<path>`.
- `submissions/`: superseded placer code. `dpo/ablation_v2_steps.py` and
  `dpo/best_of_v2_placer.py` are the original v1 smooth-placement sources, which the
  focused tests, the frozen curvature diagnostics and `thinkorplace/v1/dpo_init.py`
  name by path; `cd_lns_placer.py` and `lns.py` are the E3 neighborhood-search code.

The solver package imports nothing from here.
