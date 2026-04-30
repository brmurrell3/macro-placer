---
id: E17
name: random_init
status: falsified
parent: E25
created: 2026-04-29
decided: 2026-04-29
champion_at_time: 1.0990
outcome: 1.08653 (--fast); +16.4 % vs E25 fast 0.9336; kill gate fired (gate = +5 %); zero per-bench wins.
champion_delta: +0.1530 (+16.4 %)
graduated_to: null
superseded_by: null
---

# E17: random_init

## Hypothesis
SDF init may be a contractive basin: every champion since CDOnly starts
from `sdf_init(benchmark)`, then drives that single starting point
through CD → LNS → SA. If SDF is putting us in a basin from which the
downstream pipeline (CD plateau + grid-bin LNS + SA-v2) cannot escape on
some benchmarks, then a *different* legal initialization should expose
multi-basin structure: at least one benchmark should improve under a
random init that the SDF-anchored pipeline cannot reach. We replace
SDF with a uniform-random legal placement, run the full E25 pipeline
unchanged, and compare per-benchmark. If random init beats E25 on ANY
benchmark → multi-basin signal, feeds into E27 diagnostic. If random
init never beats E25 → SDF is at worst basin-equivalent on what we
test, and the multi-basin hypothesis is at least weakly falsified at
the per-benchmark level.

## Method
Identical to E25 (champion candidate `submissions/cd_lns_sa/placer.py`)
except step 1 "SDF init" is replaced by `random_legal_init(benchmark,
seed)`. Random init places each hard movable macro at a uniformly random
legal center via shuffled-greedy rejection sampling: for each hard
movable macro (in shuffled order, seeded), sample
(cx, cy) ∼ U([half_w, canvas_w − half_w] × [half_h, canvas_h − half_h])
and accept if it does not overlap any already-placed hard macro
(`_is_legal_2d` from cd_core/E25). Retry up to 1000 attempts; if budget
exhausted, fall back to that macro's SDF position. Fixed macros stay at
`benchmark.macro_positions`. Soft macros take their SDF positions
(soft macros never block legality). Pipeline: random_legal_init →
project_overlaps → IncrementalProxyEvaluator → CD (≤2400 s, plateau
threshold 0.001) → LNS gridbin (≤600 s, cost-aware destroy, K=5 % cap
30) → SA-v2 (≤600 s, T₀=5e-4, T_f=1e-6, breakpoint budget 12,
best-so-far tracking) → validate zero overlaps → return. seed=42 by
default, overridable via constructor kwarg `random_init_seed`.

## Kill gate
If avg `--fast` > E25 fast 0.9336 by ≥ 5 % (i.e., avg `--fast` ≥
0.9803), kill — random init lands too far off-basin for the downstream
pipeline to recover. (5 % regression on a 4-bench `--fast` average
swamps any per-benchmark multi-basin signal we'd care about.)

## Generalization check
If any single `--fast` benchmark BEATS its E25 score (E25 `--fast`:
ibm01 0.8910, ibm04 1.0119, ibm09 0.8551, ibm13 0.9766), log the
per-benchmark delta and queue `--all` regardless of the `--fast`
average. Per-benchmark wins under a random init are the multi-basin
signal we're testing for; the `--fast` average is only the kill gate.

## Outcome (filled when decided)
**FALSIFIED 2026-04-29.** Random init avg `--fast` = **1.08653**, +16.4 %
worse than E25 fast 0.9336. Kill gate (≥5 %) fires.

**Per-benchmark `--fast` (zero overlaps):**

| Benchmark | E17 random | E25 SDF | Δ vs E25 |
|---|---:|---:|---:|
| ibm01 | 1.0841 | 0.8910 | **+21.7 %** |
| ibm04 | 1.1518 | 1.0119 | **+13.8 %** |
| ibm09 | 0.9470 | 0.8551 | **+10.7 %** |
| ibm13 | 1.1632 | 0.9766 | **+19.1 %** |
| AVG | 1.08653 | 0.9336 | **+16.4 %** |

**Zero per-benchmark wins.** Random init lands in a structurally worse
basin on every benchmark — CD plateau ~1.21 (vs SDF's ~0.93 on the same
benches), LNS recovers ~10 %, SA recovers ~3 %, but the post-pipeline
final never approaches E25's SDF-anchored result.

**Implication for E27:** consistent with the "useful basin = SDF basin
(or DPO basin, see E18)" reading — random init does NOT find a basin
below E25 floor, so it doesn't motivate E28-E30 multi-day builds.
Random-init trajectories ARE in a different placement-space basin (per
E27 cluster diagnostic), but that basin is structurally above the E25
floor by ~10-20 %. **No multi-basin signal that justifies post-E25
research.**

Source JSON: `results/CDLNSSARandomInitPlacer_20260429_223308.json`.

## Pointers
- Code: `code/cd_lns_sa_random_init.py` (defines `CDLNSSARandomInitPlacer`).
- Reference: `submissions/cd_lns_sa/placer.py` (E25 candidate, full pipeline).
- Parent: E25 (CD + LNS + SA-v2 candidate at avg `--all` 1.0954).
- Wall: ~6 hr `--all` expected (similar to E25's 10.33 hr but no
  guarantee — random init may extend CD time on some benches).
- Discussion: motivates E27 multi-basin diagnostic if any per-bench win.
