# CLAUDE.md — Macro Placement Challenge 2026

## What this project is

Partcl/HRT Macro Placement Challenge. Place 200-537 rectangular macros on a 2D
chip canvas to minimize proxy cost (wirelength + density + congestion) with zero
overlaps. Prize: $29K+. Deadline: May 21, 2026.

## TWO-PATH WORK PLAN

See `TODO.md` for the active plan. Two parallel paths:

- **PATH A — speed up cascade pipeline (HIGHEST EV)**. Cascade saddle escape
  verified canonical **1.0612 uncapped** on 17 IBM; wall-safe variant
  plateaus at 1.137 because 96% of CD time is single-threaded Python (cProfile
  on ibm04 confirms `IncrementalProxyEvaluator.move()/revert()` = 32s of 33s).
  10-30× speedup unlocks cached-quality basin under the 60-min cap.
- **PATH B — DREAMPlace exploration (parallel, lower EV)**. Mature install
  on cloud A100, full integration pipeline. Honest data says DP basin
  polishes 10-23% worse than cascade on our objective; B1 (50-config sweep)
  is a fair test before killing.

Anything outside these two paths is in `submissions/_archive/` or
`experiments/_archive/` — do not work on it without explicit redirection.

## Current submission floor (Tier-1 proxy)

> Three verified placers as of 2026-05-17. **Submission-day pick is
> Option C** — beats both A and B on combined 21-bench avg and has
> zero external dependencies. Keep A and B as fallback if C has issues
> on the partcl judge box.

**Option C (NEW CHAMPION 2026-05-17) — `submissions/cd_lns_sa_cascade_stacked_periphery/placer.py`** —
IBM **1.05750** / NG45 **0.68930** / combined **0.987**. Cascade saddle
(canonical eigvec) → portfolio saddle (3 non-canonical Hessian weights:
cong-focus, density-focus, non-WL) → periphery wrapper (strict-
conservative). No external dependencies. Beats Option B by −0.85 % IBM
and −0.6 % combined. See
[`docs/handoffs/2026-05-17_morning_champion.md`](docs/handoffs/2026-05-17_morning_champion.md)
for per-bench numbers, architecture details, and failed parallel attempts.

**Option B — `submissions/cd_lns_sa_cascade_dp_lane/placer.py`** —
IBM 1.06650 / NG45 0.68086 / combined 0.993. Adds DREAMPlace as a third
init lane alongside SDF/DPO; plateau picks best of {E25, E41, DP-
polished}. Falls back to Option A if `DREAMPLACE_ROOT` is not set.

**Option A — `submissions/cd_lns_sa_cascade/placer_adaptive.py`** —
IBM 1.07820 / NG45 0.68102 / combined 1.003. PATH A post-A1 wall-safe
cascade. No external dependencies. Safest fallback.

**Option D (LIVE SUBMISSION 2026-05-21 — root placer.py routes here) — `experiments/E171_adaptive_kjoint_gate/code/placer.py`** —
adaptive size-gated stack on V4-Gaussian basin. Movables < 400: E166 lane
(V4 multi-init + 3-seed multi-seed + cascade + portfolio saddle with
`portfolio_max_iters=3` per E169). Movables ≥ 400: E143 lane (E166 +
K-joint LNS K=3 + SA-v2 polish). Per-bench dispatch by
`benchmark.num_hard_macros` — algorithmically legal (input-dimension gate).
**Verified so far:** ibm03 0.8870 (E166), --fast 0.8384 (E166), ibm10
0.95671 (E143, −3.0 % vs Option C baseline on hard bench); E169 ibm03
tracking ≤0.880 in portfolio iter 2/3. **--all in flight**; target ≤ 0.970
IBM combined (parity with Carrotato #2 at 0.967). ADR-014 *Accepted*
with auto-fallback safety (root `placer.py` falls back to thinkorplace-v2
on any exception or overlap-positive return → worst-case 0.984 combined,
cannot regress past V4 floor). Components live in `experiments/E166_v4_full_stack/`,
`experiments/E143_v4_full_kjoint_sa/`, `experiments/E169_v4_portfolio_3iter/`,
`experiments/E171_adaptive_kjoint_gate/`.

Leaderboard reference: vmallela #1 at 1.011 (gap +4.6 % to Option C),
Carrotato 0.967 via Xplace+Triton at 3.8 min/bench (gap +9.4 %). Xplace
integration attempted 2026-05-16/17 and FALSIFIED — see handoff doc; our
Xplace produces 1.41 polish on ibm01 vs SDF-cascade 0.85, basin
structurally inferior, must require patched Xplace internals we don't have.

**Tier-2 ORFS** ($20K, separate objective: WNS/TNS/Area from full
PnR): per-design strategy verified 2026-05-15/16. ariane133 — **ship
without** `MACRO_PLACEMENT_TCL` (auto-place beats our cascade by 1.2 ns
of slack). ariane136 — **ship with** cascade (`+0.4935` ns vs
`+0.0457` ns auto). mempool_tile + nvdla untested. Full report:
[`docs/handoffs/2026-05-16_tier2_orfs_findings.md`](docs/handoffs/2026-05-16_tier2_orfs_findings.md).

## Prior champion lineage (now superseded by cascade)

CDLNSSAHessian (E74), avg proxy **1.0666** on --all (17 IBM benchmarks),
**NG45 0.6813** (4 designs, including critical ariane133 at **0.6641** —
−3.21 % vs E48 0.6861, BREAKING the failure point that killed
E42/E43/E44/E54/E62). Beats public leaderboard 1.1172 by **−4.53 %**,
beats E48 1.08151 by **−1.38 %**, beats RePlAce 1.4578 by **−26.8 %**,
zero overlaps on all 17 IBM + 4 NG45. Entry:
`submissions/cd_lns_sa_hessian/placer.py`. **Now subsumed by cascade variant**;
kept active because cascade imports its saddle escape primitives.

NG45 per-design (all ZERO overlaps):
| Design | E48 ref | **E74** | Lift |
|---|---:|---:|---:|
| ariane133 | 0.6861 | **0.6641** | **−3.21 %** |
| ariane136 | 0.6685 | **0.6518** | −2.50 % |
| mempool_tile | 0.7375 | 0.7376 | tied |
| nvdla | 0.6767 | **0.6716** | −0.75 % |
| **avg** | 0.6922 | **0.6813** | **−1.57 %** |

Mechanism: E48 hybrid (E25 + E41 best-of) → smooth-proxy Hessian via
`torch.autograd.functional.hvp` → Lanczos smallest-algebraic eigenvectors
(scipy `eigsh` with LinearOperator) → ±ε perturbation along soft modes →
CD-adaptive polish. Applies transition-state methods (Henkelman & Jónsson
2000 climbing-image NEB, dimer / gentlest-ascent) to the local-move
plateau in combinatorial macro placement. Implements roadmap E28 (proposed
2026-04-29).

Lifts per bench (vs my fresh E48-equivalent): ibm01 −3.86 %, ibm02 −7.13 %,
ibm03 −0.66 %, ibm04 −0.93 %, ibm06 −1.86 %, ibm07 −1.67 %, ibm08 −0.45 %,
ibm09 −0.11 %, ibm10 −0.90 %, ibm11 −0.61 %, ibm12 −0.61 % (E61V2-fresh +
Hessian layered), ibm13 −0.69 %, ibm14 −0.55 %, ibm15 −1.73 % (layered),
ibm16 −0.91 %, ibm17 −0.29 %, ibm18 −0.38 %. Wall ~50 min/bench (E25 ~25 +
E41 ~30 + Hessian ~20 + polish per ε); --all ~14 hr serial, ~4 hr `--jobs 4`.

**Champion lineage:**
- E12 CDLNSGridBin (1.0990) — ADR-007 *Accepted* 2026-04-28.
- E48 CDLNSSAHybrid (1.08151) — ADR-011 *Accepted* 2026-05-02 (superseded by ADR-012).
- E74 CDLNSSAHessian (1.0666) — ADR-012 *Accepted* 2026-05-05. Supersedes ADR-011.
- E84 cascade saddle (uncapped 1.0612) → wall-safe `cd_lns_sa_cascade/placer_adaptive.py` 1.07820 → DP-lane `cd_lns_sa_cascade_dp_lane/placer.py` 1.06650. **ADR-013 forthcoming** for the cascade-DP-lane promotion; pending submission-day decision.

**E84 cascading saddle escape (uncapped)** — canonical **1.0612 across 17
IBM, zero overlaps** (−1.88 % vs E48, −0.51 % vs E74, gap to RePlAce
+27.2 %). 8/17 unbudgeted walls over 55-min cap; the wall-safe
descendant `submissions/cd_lns_sa_cascade/placer_adaptive.py` (PATH A
post-A1 acceleration) verified at **1.07820** under 60-min/bench cap.
DP-lane variant `submissions/cd_lns_sa_cascade_dp_lane/placer.py` adds
DREAMPlace as a third init lane and verified **1.06650** (−1.07 % vs
adaptive) on the same 17 IBM + 4 NG45 with zero overlaps.

**Wall-safe variants** (all default budget_seconds=3300s = 55 min, fits
the 1-hr/bench partcl cap):
- `submissions/cd_lns_sa_hessian/placer.py` — E74 + deadline (smoke ibm03
  b=600: 1.00480, wall 598s compliant).
- `submissions/cd_lns_sa_cascade/placer.py` — E48 → E41 → cascading
  saddle with deadline.
- `submissions/cd_lns_sa_hessian_dp/placer.py` — E74 + optional
  DREAMPlace lane (subprocess to `$DREAMPLACE_ROOT/dreamplace/Placer.py`
  on cloud GPU; skipped if DREAMPLACE_ROOT not set or insufficient time).
  Cloud: OCI A100-SXM4-40GB at 132.145.135.39 (ssh alias `mpc-cloud`).
  Built native at /opt/DREAMPlace/install with _GLIBCXX_USE_CXX11_ABI=1
  (matches system torch 2.7).

**Prior champion (kept as fallback):** CDLNSSAHybrid (E48), avg **1.08151**
on --all. Entry: `submissions/cd_lns_sa_hybrid/placer.py`. Promoted
2026-05-02 ADR-011 *Accepted*. NG45 commercial 0.6922.

**Component placers (called by the E48 hybrid; not separate champions):**
- E25 CDLNSSA (`submissions/cd_lns_sa/placer.py`) — SDF init pipeline used by E48 lane 1.
- E41 CDLNSSADPOKJoint (`experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py`) — DPO init + K-joint pipeline used by E48 lane 2.

**Overnight 2026-05-01 → 02 — three follow-ups, none lifted past E48:**
- E53 GPU DPO basin polish — falsified (0/350 GPU restarts accepted at full budgets).
- E53m multi-seed hybrid (E25 + E41 s42 + E41 s1) — marginal (--all tied at 1.08128, --ng45 +0.23 %).
- E54 congestion-targeted destroy — falsified (--ng45 0.7022, ariane133 +5.14 %).

Reference baselines: RePlAce 1.4578, Will's pre-fork seed 1.5338, leaderboard
target 1.1172.

## Your role

You autonomously implement, evaluate, iterate, and kill hypothesis variants.
The human's role is (1) sourcing hypotheses and (2) judging graduated results.
**Never ask "should I continue?" for routine decisions.** The thresholds answer that.

## Commands

```bash
# Fast iteration (4 benchmarks, ~10s, structured output)
uv run evaluate <placer.py> --fast --json --hypothesis <name>

# Full validation (17 benchmarks, ~40s)
uv run evaluate <placer.py> --all --json --hypothesis <name>

# NG45 commercial designs (top-7 submissions only)
uv run evaluate <placer.py> --ng45 --json

# Single benchmark for debugging
uv run evaluate <placer.py> -b ibm01
```

## Feedback loop

1. **Set up the experiment.** Copy `experiments/_template/` to
   `experiments/E<NN>_<short_name>/`. Fill in the manifest's `Hypothesis`,
   `Method`, `Kill gate`, and `Generalization check`. Put placer code in
   `code/`. Find the next free `E<NN>` with `ls experiments/ | grep '^E'`.

2. **Run.** Always pass `--hypothesis E<NN>` so the row in
   `results/experiment_log.jsonl` is attributable:
   ```
   uv run evaluate experiments/E<NN>_*/code/<placer>.py --fast --json --hypothesis E<NN>
   ```

3. **Decide from the JSON.**
   - **fast_gate fail** → tweak parameters, try next variant.
   - **fast_gate pass** → run `--all --json`.
   - **avg < 1.12** → likely noise; verify carefully.
   - **avg < 1.082 (at-or-below prior champion E48 1.08151)** → STOP. Surface to human.
   - **avg < 1.068 (at-or-below current Option B cd_lns_sa_cascade_dp_lane 1.06650)** → STOP. New floor territory.
   - **avg < 1.05** → STOP immediately. New champion territory.

4. **If killed:** Set the manifest's `status: falsified`, fill `decided`
   and `outcome`, write the killing data into `Outcome`. Falsified
   experiments stay forever — they're the signpost that stops the next
   person re-running the dead end.

5. **If graduated:** Surface to human first. Promotion = move code to
   `submissions/<name>/placer.py`, set `status: graduated, graduated_to: ...`,
   write an ADR in `docs/decisions/NNN_short_name.md` if the decision is
   load-bearing.

If best score hasn't improved by >2 % after N variants → kill the hypothesis.

## On a result, update these

| When | Update |
|------|--------|
| Any terminal decision | Manifest's `status` / `decided` / `outcome` / `champion_delta` |
| Falsified | Manifest's `Outcome` section with the killing data |
| Graduated | Move code to `submissions/<name>/placer.py`; set `graduated_to:` |
| New champion | CLAUDE.md champion line; `docs/results.md`; `submissions/README.md`; `writeup/evidence.md` §1 |
| Structural decision | New ADR in `docs/decisions/NNN_short_name.md` (immutable once accepted) |
| Paper-relevant finding | `writeup/evidence.md` (the experimental archive) |
| Always | `docs/experiment_index.md` (status table) |

**Manifest vs ADR.** A manifest records *what you tried and how it went*
(per-experiment, many per generation). An ADR records *a decision about
how the project works going forward* (rare, structural, immutable). Most
experiments only need a manifest. ADRs are reserved for things like
"we abandon LP-HPWL ranking" or "we promote E12 as champion."

## Key files

### Where to put new work

| Path | Role |
|------|------|
| `experiments/_template/` | Boilerplate for a new experiment manifest |
| `experiments/E<NN>_*/` | Per-experiment manifest + `code/` + `notes.md` |
| `analysis/<name>/` | Diagnostic probes (one-shot, not score-improvement experiments) |
| `submissions/<name>/placer.py` | Graduated placer (post-promotion only) |

### Read at session start

| Path | Role |
|------|------|
| `results/experiment_log.jsonl` | Append-only log of all runs |
| `docs/roadmap.md` | Phased action plan, champion lineage, risk register |
| `docs/results.md` | Current champion per-benchmark tables |
| `docs/approach.md` | Current architecture + prior approaches |
| `docs/experiment_index.md` | Catalog of every experiment (live + falsified) |
| `experiments/*/manifest.md` | Per-experiment hypothesis, status, kill gate, outcome |

### Decisions, gotchas, formal spec

| Path | Role |
|------|------|
| `docs/decisions/` | ADRs — structural decisions, immutable once accepted |
| `docs/gotchas.md` | Codebase footguns (net_nodes empty bug, single-step revert, etc.) |
| `docs/problem.md` | Formal mathematical problem statement |
| `analysis/lp_hpwl_diagnostic/lp_hpwl_diagnostic.md` | E8 — 6 % WL / 20 % density / 74 % congestion |

### Library and submissions

| Path | Role |
|------|------|
| `macro_place/evaluate.py` | Evaluation harness (don't modify unless infra work) |
| `macro_place/objective.py` | Proxy cost computation |
| `macro_place/incremental_evaluator.py` | E1 — 4657× speedup; load-bearing for CD |
| `macro_place/benchmark.py` | Benchmark dataclass (PyTorch tensors) |
| `macro_place/sdf_init.py` | SDF initialization (used by every champion) |
| `submissions/cd_lns_sa_cascade_dp_lane/placer.py` | **TIER-1 OPTION B (verified, awaiting submission-day pick)** — cascade + DREAMPlace third lane, IBM 1.06650 / NG45 0.68086 (2026-05-14) |
| `submissions/cd_lns_sa_cascade/placer_adaptive.py` | **TIER-1 OPTION A (verified, awaiting submission-day pick)** — PATH A post-A1 wall-safe cascade, IBM 1.07820 / NG45 0.68102 (2026-05-16) |
| `submissions/cd_lns_sa_cascade/placer.py` | Wall-safe E84 cascade (E25 → E41 → cascading saddle, deadline-bound). Component of Option A. |
| `submissions/cd_lns_sa/placer.py` | Component of E48 hybrid / cascade (lane 1, SDF basin) — E25 CDLNSSA, 1.0954 standalone |
| `submissions/_archive/cd_lns_sa_hessian/placer.py` | Prior champion — E74 Hessian saddle, 1.0666 (ADR-012). Logic absorbed by cascade. |
| `submissions/_archive/cd_lns_sa_hybrid/placer.py` | Prior champion — E48 best-of-{E25, E41}, 1.08151 (ADR-011, superseded by ADR-012). |
| `submissions/_archive/cd_lns_gridbin/placer.py` | Prior champion — E12 CD + grid-bin LNS, 1.0990 (ADR-007). |
| `submissions/_archive/cd_adaptive/placer.py` | Prior champion — E9 CDAdaptive, 1.1055. |
| `submissions/_archive/cd_only/placer.py` | Prior-prior — CDOnly fixed-budget, 1.1193. |
| `submissions/examples/` | Reference placers (greedy, random). |

### Writeup (innovation prize)

| Path | Role |
|------|------|
| `writeup/paper.md` | Canonical draft with TODO markers |
| `writeup/evidence.md` | Experimental archive — every paper.md datapoint lives here |
| `writeup/contributions.md` | Claim / novelty / evidence registry |
| `writeup/theory.md` | Supplementary mathematical material |

## At session start

1. Read `results/experiment_log.jsonl` to know what's been tried
2. Read `docs/roadmap.md` for current phase and next actions
3. Read `docs/results.md` for latest results
4. Read `docs/approach.md` for current strategy
5. Find live experiments: `grep -l "status: in_progress" experiments/*/manifest.md`
6. Ask what to work on, or continue the most promising alive hypothesis

## Writing a placer

A placer is a Python file with a class that has a `place(self, benchmark) -> Tensor` method.
The returned tensor is `[num_macros, 2]` with (x, y) center coordinates.
Fixed macros (`benchmark.macro_fixed`) must not be moved.
Hard macros must not overlap. Soft macros can overlap.

See `submissions/examples/greedy_row_placer.py` for reference.
See `SETUP.md` for the full API (Benchmark fields, compute_proxy_cost, validation).

## Hard constraints

- **Zero overlaps.** Any overlap on any benchmark = disqualified.
- **Fixed macros stay fixed.** Don't move them.
- **Canvas bounds.** All macros must be fully within the canvas.

## Hardware

M3 Max (36GB unified, MPS/PyTorch). Fast subset: ~10s. Full --all: ~40s.
Champion (`submissions/cd_lns_sa_hybrid/placer.py`, E48) takes ~7 hr on --all under `--jobs 4` parallel (88 452 s aggregate CPU-time across workers); each bench runs E25 then E41 sequentially within a worker. Prior champion E12 (`submissions/cd_lns_gridbin/placer.py`) was ~7.85 hr.
Cloud not needed for development. The bottleneck is thinking, not compute.
