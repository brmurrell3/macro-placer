---
id: E68
name: workbounded_refactor
status: marginal
parent: E48 (champion)
created: 2026-05-03
decided: 2026-05-05
champion_at_time: 1.08151 (E48 hybrid, ADR-011 *Accepted* 2026-05-02)
outcome: **ABANDONED 2026-05-05** (parallel agent ended). Implementation complete: `code/cd_lns_sa_workbounded.py` mirrors E25 + E41 + E48 with §4.5 work-bounded termination (CD plateau-3-streak/3000 s, LNS 5-streak/900 s, SA 1000-no-improve/900 s, K-joint 30-no-commit/900 s). `--fast` evaluated showed +0.11 % regression; `--ng45` showed +0.19 % regression (per E70 cross-reference notes). The variance was attributed to **lane variance**, not the work-bounded change itself — same regression pattern in E70's lanes-only baseline. Never resolved. **Hardware-portability rationale still valid as a future submission-hardening step**; defensive against partcl AMD EPYC slower-clock evaluation. Status frozen at last known state.
champion_delta: +0.0012 (+0.11%) --fast lane variance, no clean signal
graduated_to: null
superseded_by: null
---

# E68: workbounded_refactor — terminate on work, fall back to wall

## Hypothesis

The §4.5 termination changes (work-bounded saturation as primary,
wall caps as soft secondary) preserve quality on the M3 Max baseline
**and** defend against slower-clock hardware (partcl AMD EPYC 9655P)
where the current wall caps would fire before plateau / saturation
detection, leaving each phase mid-descent.

The probe in E66 (2026-05-03) showed the cap-hit pattern observed in
existing --jobs 4 logs (5 / 17 E41 CD benches hit cap; 14 / 17 E41
K-joint phases hit cap) was largely a parallel-contention artefact:
under thread-limit (`OMP_NUM_THREADS=1` etc.) ibm10 plateau-exited CD
at 1829 s instead of hitting the 2400 s cap, with the same proxy.
E68 is the *complementary* defence — making the termination criterion
itself work-based so that slower per-core clock translates to "fewer
sweeps in same wall" without tripping the cap.

## Method

Refactor the four phases inside each lane (E25 SDF lane, E41 DPO+K-joint
lane) per §4.5:

| Phase    | Old primary           | New primary                              | Old wall cap | New soft cap |
|----------|-----------------------|------------------------------------------|--------------|--------------|
| CD       | plateau OR cap        | plateau (delta < 0.001 × 3 sweeps)       | 2400 s       | 3000 s       |
| LNS      | 1 non-improving sample (+ cap) | 5 consecutive non-improving samples | 600 s | 900 s |
| SA-v2    | wall cap only         | no improvement in last 1000 moves        | 600 s        | 900 s        |
| K-joint  | full pass with 0 commits (+ cap) | no commits in last 30 K-tuples | 600 s        | 900 s        |

CD primary is unchanged in semantics (existing `run_cd_adaptive`
plateau check matches §4.5 exactly); only the soft cap loosens.
LNS / SA / K-joint receive new saturation counters wired into the
inner loops.

Total per-lane budget: 3000 + 900 + 900 + 900 = 5700 s upper bound
(was 4200 s). On M3 Max the plateau / saturation should fire well
within budget on most benches — the soft caps are catastrophic-case
floors. Total wall on --all is what the harness measures (the
1-hr-per-bench legal cap is per the rules; each lane runs separately
inside that envelope and on M3 Max the plateau-exiting reality is
~30-60 min/lane regardless of soft cap).

The actual code lives in
`code/cd_lns_sa_workbounded.py`. To stay honest about *what changed*,
the file inlines two flat copies of the lane bodies (E25 + E41) with
side-by-side termination edits and a hybrid wrapper analogous to
`submissions/cd_lns_sa_hybrid/placer.py`. **No** champion files are
modified.

Hyperparameters fixed:

- `cd_hard_cap_s = 3000.0` (was 2400.0)
- `cd_plateau_threshold = 0.001`, `cd_patience = 3` (unchanged; matches §4.5 directive)
- `lns_budget_s = 900.0`, `lns_patience = 5` (was 600 / 1)
- `sa_budget_s = 900.0`, `sa_no_improve_moves = 1000` (was 600 / none)
- `kjoint_budget_s = 900.0`, `kjoint_no_commit_tuples = 30` (was 600 / pass-bound)

No per-benchmark tuning. No adaptive caps. No hardware probe at startup.
Strictly the §4.5 prescription.

## Kill gate

Either of:

- **--fast** avg proxy > 0.92024 + 0.5 % = 0.92484. (E48 baseline
  --fast 0.92024 from `experiments/E48_hybrid_e25_e41/manifest.md`.)
- Any single-bench overlap_count > 0 (zero-overlap is non-negotiable
  per the challenge rules).

## Generalization check

NG45 ariane133 must not regress vs E48 0.7214. (E54 was killed for
+5.14 % regression here; this is the canonical NG45 sentinel bench.)

If --fast and ariane133 both clear, run --all and compare to E48
1.08151. A tied or lifted --all is a candidate; a regression > 0.5 %
on --all kills the variant. Promotion requires --ng45 ≤ E48 0.6922
plus --all ≤ E48 1.08151.

## Outcome (filled when decided)

Empty. The variant has been written but **not yet evaluated** —
another probe is using compute. Static review only this session.

## Pointers

- Code: `code/cd_lns_sa_workbounded.py` (lane copies + hybrid wrapper)
- Notes: `notes.md` (per-phase before/after audit trail)
- Parent placer: `submissions/cd_lns_sa_hybrid/placer.py` (E48, untouched)
- Parent lanes:
  - E25: `submissions/cd_lns_sa/placer.py`
  - E41: `experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`
  - E39 inner LNS / SA / K-joint: `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py`
- Mitigation rationale: `docs/research_principles_for_walls.md` §4.5
- Hardware probe context: `experiments/E66_hardware_probe/manifest.md`
