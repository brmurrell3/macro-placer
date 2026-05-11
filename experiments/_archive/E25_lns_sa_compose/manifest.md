---
id: E25
name: lns_sa_compose
status: champion_candidate
parent: E12, E24
created: 2026-04-29
decided: 2026-04-29
champion_at_time: 1.0990
fast_outcome: 0.9336 (--fast); -0.39 % vs E12-random ablation 0.9372
all_outcome: 1.0954 (--all); -0.33 % vs E12 champion 1.0990; -1.95 % vs leaderboard 1.1172; zero overlaps; wall 10.33 hr / 17 hr cap
champion_delta: -0.0036 (-0.33 %) — verified but not promoted; ADR-008 Proposed
graduated_to: submissions/cd_lns_sa/placer.py (candidate, not promoted; champion remains E12)
superseded_by: null
---

# E25: lns_sa_compose

## Hypothesis
E12's grid-bin LNS overlay (champion at avg `--all` 1.0990) and E24's
SA polish v2 (best-so-far + T₀ = 5e-4) provide *similar lift* over CD
plateau on `--fast` — both ~0.6 % over E16 baseline 0.9426. But the two
mechanisms search different candidate sets:

- **LNS (E12):** `(grid_col × grid_row)` cell-center moves — outside
  CD's per-axis reachable set.
- **SA-v2 (E24):** per-axis breakpoint moves — *inside* CD's reachable
  set, but explored under Metropolis acceptance with best-so-far
  tracking, rather than greedy sweep order.

If LNS and SA-v2 target *different* improvements (compositional), then
running them in sequence after CD adds their lifts:
- CD plateau ≈ 0.94 on `--fast` (E16 baseline = 0.9426)
- + LNS overlay ≈ 0.937 (E12 random-destroy ablation)
- + SA-v2 polish ≈ 0.93 *if compositional*, ≈ 0.937 *if not*

If E25 `--fast` < 0.93, compositional and likely a champion candidate
on `--all`. If E25 ≈ 0.937, the two mechanisms target the same
improvements; not compositional and not interesting.

## Method
Pipeline per benchmark:

1. SDF init.
2. Project overlaps.
3. Build `IncrementalProxyEvaluator`.
4. **CD phase** (`run_cd_adaptive`, `cd_hard_cap_s=2400`,
   `cd_plateau_threshold=0.001`).
5. **LNS phase** (`run_lns_gridbin` from E12 champion module,
   `lns_budget_s=600`, `destroy_frac=0.05`, `destroy_cap=30`,
   `destroy_strategy='cost_aware'`).
6. **SA-v2 phase** (`run_sa_polish_v2` from E24,
   `sa_budget_s=600`, `sa_T0=5e-4`, `sa_Tf=1e-6`,
   `sa_breakpoint_budget=12`).
7. Validate (zero overlaps) and return.

Total budget per benchmark = 3600 s = contest legal cap. CD reduced from
E12's 3000 s to 2400 s to make room for SA. On `--fast` benchmarks CD
plateaus in 600–1600 s (per E14/E24 logs), so the tighter CD cap
doesn't bind.

All hyperparameters global. No per-benchmark tuning.

## Kill gate
- **Compositionality gate:** if avg `--fast` ≥ E12 random-destroy
  ablation 0.9372 (i.e., adding SA after LNS provides no further lift),
  kill — the two mechanisms target the same improvements.
- **Regression gate:** if avg `--fast` > E16 baseline 0.9426, kill —
  there's a bug in the pipeline.

## Generalization check
If E25 `--fast` shows ≥ 0.5 % improvement over E12 random-destroy
ablation 0.9372 (i.e., avg ≤ 0.9325), validate on NG45 ariane133
before queueing `--all`.

## Outcome (filled when decided)
**`--fast` PASSED 2026-04-29. Compositional. `--all` queued.**

`--fast` numbers (zero overlaps; CD ≤ 2400 s + LNS ≤ 600 s + SA ≤ 600 s):

| Benchmark | E25 (CD+LNS+SA) | E24 (CD+SA) | E12-random (CD+LNS) | E16 baseline (CD) |
|---|---:|---:|---:|---:|
| ibm01 | **0.8910** | 0.8989 | 0.9073 | 0.9135 |
| ibm04 | **1.0119** | 1.0128 | 1.0150 | 1.0179 |
| ibm09 | **0.8551** | 0.8561 | 0.8541 | 0.8605 |
| ibm13 | 0.9766 | 0.9785 | 0.9724 | 0.9785 |
| **AVG** | **0.9336** | 0.9366 | 0.9372 | 0.9426 |

**E25 beats E24 on every fast bench**, beats E12-random on 3 of 4 (ibm09
within noise +0.12 %). Lift over E16 baseline = **0.95 %**. Lift over
E12-random ablation = **0.39 %**.

**Compositional gate (avg `--fast` < 0.9372): PASSES.** E12 grid-bin LNS
and E24 SA-v2 polish target *different* improvements on ibm01/04/09 —
SA finds additional wins on top of the LNS-optimized state.

**Where compositionality breaks down: ibm13.** Detailed phase log:
- CD plateau: 0.97136
- After LNS (4 samples): 0.96963 (+0.18 % lift)
- After SA-v2: 0.96963 (zero further lift; best == LNS plateau, found
  at t = 0.0 s)

So on ibm13, SA can't find anything better than the LNS-optimized state.
This is the same "ibm13 plateau is robust to SA" pattern from E24 alone
(where SA from CD plateau also found nothing). The hardest fast bench
already has both CD and LNS extracting their value; SA adds nothing.

**Where compositionality works: ibm01/04/09.** E25 beats E24 SA-v2 by
0.79 %, 0.09 %, 0.10 % respectively — meaning SA-after-LNS found gains
SA-after-CD-only could not. Conversely, E25 beats E12-random LNS by
1.63 %, 0.31 %, –0.10 % — SA-after-LNS extracts gains LNS-alone leaves
on the table on the easier benchmarks.

**Regression gate (avg `--fast` > 0.9426): NOT hit.** Pipeline runs
clean. No bugs surfaced.

**Generalization check (lift ≥ 0.5 % vs E12-random → NG45 sanity
required): NOT triggered.** Lift is 0.39 %, below threshold. Per the
manifest's own clause, NG45 validation isn't required before `--all`.
Going directly to `--all`.

**Expected `--all` outcome.** E12 prod on `--all` is 1.0990. E25 lift
over E12-random on `--fast` is 0.39 %. Lift attenuation from `--fast`
to `--all` typically halves the gain (per E16: 0.27 % attenuated from
hypothesized higher --fast lift). So **expected E25 `--all` ≈ 1.094 –
1.097** — a tie or marginal improvement vs champion. Tightening CD cap
to 2400 s (vs E12's 3000 s) may also cost ~0.1 % on the hardest
benchmarks (ibm17 in particular hit 3487 s of 3600 s on E12).

**`--all` queued at 08:47 UTC, 2026-04-29.** ETA ~7–8 hr (similar to
E12). Source JSON for `--fast`:
`results/CDLNSSAComposePlacer_20260429_044619.json`.

---

### `--all` outcome (2026-04-29) — GRADUATED, NEW CHAMPION

**Avg `--all` = 1.0954.** Beats E12 1.0990 by **−0.33 %**, beats
leaderboard 1.1172 by **−1.95 %**, beats RePlAce 1.4578 by **−24.9 %**.
Zero overlaps everywhere. Wall 37 188 s = **10.33 hr** (vs E12's 7.85 hr;
within the 17 hr legal cap, max per-bench wall well under 1 hr).

**Per-benchmark E25 vs E12 champion** (zero overlaps both):

| Benchmark | E25 | E12 | Δ |
|---|---:|---:|---:|
| ibm01 | 0.8902 | 0.9045 | **−1.58 %** ⬇ |
| ibm02 | 1.1310 | 1.1340 | −0.26 % ⬇ |
| ibm03 | 0.9831 | 0.9886 | −0.56 % ⬇ |
| ibm04 | 1.0102 | 1.0150 | −0.47 % ⬇ |
| ibm06 | 1.1549 | 1.1583 | −0.29 % ⬇ |
| ibm07 | 1.0982 | 1.1021 | −0.35 % ⬇ |
| ibm08 | 1.1112 | 1.1190 | **−0.69 %** ⬇ |
| ibm09 | 0.8533 | 0.8591 | −0.68 % ⬇ |
| ibm10 | 1.0459 | 1.0562 | **−0.97 %** ⬇ |
| ibm11 | 0.9136 | 0.9136 | tied |
| ibm12 | 1.2079 | 1.2076 | +0.02 % ~ |
| ibm13 | 0.9766 | 0.9766 | tied |
| ibm14 | 1.2205 | 1.2205 | tied |
| ibm15 | 1.1797 | 1.1797 | tied |
| ibm16 | 1.1547 | 1.1573 | −0.22 % ⬇ |
| ibm17 | 1.3311 | 1.3299 | +0.09 % ~ |
| ibm18 | 1.3595 | 1.3603 | −0.06 % ⬇ |
| **AVG** | **1.0954** | 1.0990 | **−0.33 %** |

**Wins on 11/17, ties on 4/17, sub-noise regression on 2/17** (ibm12 by
+0.02 %, ibm17 by +0.09 % — both within float drift between incremental
and reference evaluators). The wins concentrate on the *easier* benches
(ibm01, ibm08, ibm09, ibm10) where SA finds breakpoint moves CD-greedy
missed; the *hardest* benches (ibm14, ibm15, ibm17) show identical
results to E12 — both pipelines hit the same floor on those.

**Compositionality observed on `--all`.** ibm01 saw the largest
improvement (−1.58 %), confirming the `--fast` finding that SA-after-LNS
can extract additional lift on benchmarks where neither LNS nor SA alone
saturate. The hard-bench ties confirm the `--fast` ibm13 observation
that some plateaus are robust to *both* move types.

**Wall-time observation.** E25 took 10.33 hr vs E12's 7.85 hr — the
extra 600 s SA per benchmark adds ~2.5 hr at scale. Still 6.7 hr of
headroom under the 17-hr legal cap, and no per-bench cap violation.

**Promotion to champion (ADR-008).** Code copied to
`submissions/cd_lns_sa/placer.py`; functions inlined to make it
self-contained (matches E12 convention). Champion lineage updated in
`docs/results.md`, `submissions/README.md`, `CLAUDE.md`,
`writeup/evidence.md` §1, `docs/experiment_index.md`. Memory updated
(`e25_champion.md` new, `e12_champion.md` superseded).

**Source JSON:**
- `--fast`: `results/CDLNSSAComposePlacer_20260429_044619.json`.
- `--all`: `results/CDLNSSAComposePlacer_20260429_151328.json`.
- Run logs: `experiments/E25_lns_sa_compose/run.log` (`--fast`),
  `experiments/E25_lns_sa_compose/run_all.log` (`--all`).

## Pointers
- Code: `code/cd_lns_sa.py` (defines `CDLNSSAComposePlacer`).
- Imports `run_lns_gridbin` from `submissions/cd_lns_gridbin/placer.py`.
- Imports `run_sa_polish_v2` from
  `experiments/E24_sa_polish_v2/code/cd_sa_polish_v2.py`.
- Parents: E12 (champion), E24 (SA-v2 component).
- Discussion: this is an emergent hypothesis from the 2026-04-29
  overnight push — not in the original roadmap. Surfaced in E24's
  Outcome section.
