# Closing the Gap to 1.11

## 🏆 LEADERBOARD BEATEN 2026-04-27 17:22 — E9 Adaptive avg 1.1055

**CDAdaptivePlacer --all final: avg_proxy_cost = 1.1055 across all 17 IBM benchmarks. Zero overlaps. Runtime 17480s (4.85 hr).**

| Method | Avg | Delta |
| --- | --- | --- |
| **CDAdaptivePlacer (E9, NEW)** | **1.1055** | (this run) |
| Leaderboard vmallela target | 1.1172 | **-1.05%** |
| CDOnly --all (prior champion) | 1.1193 | -1.23% |
| DPO best_of_v2 | 1.3834 | -20.1% |
| RePlAce baseline | 1.4578 | -24.2% |

**The plateau-detection bet worked.** CDOnly's 600s/bench was one-size-fits-all: easy benchmarks stopped descending around 5-6 min, hard ones still had Δ≈0.001-0.003 at the 600s mark. E9's plateau detection lets each benchmark exit when its 3-sweep delta-window drops below 0.005 OR after 1hr (competition rule). Net: hard benchmarks 12/14/15/16/17/18 each gained 1.7-3.6%; easy benchmarks tied within ±0.4%. All 17 exited via plateau, none hit the 1hr cap.

**Per-bench wins (E9 vs CDOnly):**

| Bench | E9 | CDOnly | Δ | E9 Wall (s) |
| --- | --- | --- | --- | --- |
| ibm10 | 1.0749 | 1.1000 | -2.3% | 1332 |
| ibm12 | 1.2153 | 1.2418 | -2.1% | 1424 |
| ibm13 | 0.9772 | 0.9939 | -1.7% | 1143 |
| ibm14 | 1.2234 | 1.2478 | -2.0% | 1572 |
| ibm15 | 1.1809 | 1.2109 | -2.5% | 1609 |
| ibm16 | 1.1610 | 1.1919 | -2.6% | 1518 |
| ibm17 | 1.3326 | 1.3830 | **-3.6%** | 2238 |
| ibm18 | 1.3633 | 1.3865 | -1.7% | 1589 |

The 1-hour cap was never hit. Plateau is the right exit condition, not a wall.

**E3 LNS verdict (this branch):** Falsified at single-macro local-window architecture. E3 v2 ibm17 ran 15 LNS iterations in 600s (vs v1's 1-2 in 428s) but final VALID 1.3824 = essentially flat vs CDOnly 1.3830. Cost-based destroy selector saturates after 1-2 accepts; 5×5 window can't move clusters. Real LNS would need cluster-level joint reinsertion or randomized destroy strategies. NOT pursued — E9 already beats the leaderboard.

**Submission deadline May 21 — 24 days away. Champion entry secured.**

---

## 🎯 CD --all COMPLETE 2026-04-27 04:01 — avg 1.1193 (matches leaderboard +0.2%) [SUPERSEDED BY E9]

CDOnlyPlacer --all final: **avg_proxy_cost = 1.1193 across all 17 IBM benchmarks. Zero overlaps. Runtime 10316s (172 min).**

| Method | Avg | Delta |
| --- | --- | --- |
| CDOnlyPlacer --all (10 min/bench) | 1.1193 | (this run) |
| Leaderboard vmallela target | 1.1172 | +0.18% |
| DPO best_of_v2 prior champion | 1.3834 | -19.1% |
| RePlAce baseline | 1.4578 | -23.2% |
| SA baseline | 2.1251 | -47.3% |

**Per-bench (all VALID, zero overlaps):**

| Bench | CD final | DPO BoV2 | Delta |
| --- | --- | --- | --- |
| ibm01 | 0.9133 | 1.1285 | -19% |
| ibm02 | 1.1534 | 1.6888 | -32% |
| ibm03 | 0.9942 | 1.2508 | -21% |
| ibm04 | 1.0193 | 1.3239 | -23% |
| ibm06 | 1.1656 | 1.6434 | -29% |
| ibm07 | 1.1105 | 1.3846 | -20% |
| ibm08 | 1.1307 | 1.3816 | -18% |
| ibm09 | 0.8606 | 1.0130 | -15% |
| ibm10 | 1.1000 | 1.2540 | -12% |
| ibm11 | 0.9248 | 1.0657 | -13% |
| ibm12 | 1.2418 | 1.6497 | -25% |
| ibm13 | 0.9939 | 1.2089 | -18% |
| ibm14 | 1.2478 | 1.4725 | -15% |
| ibm15 | 1.2109 | 1.3802 | -12% |
| ibm16 | 1.1919 | 1.3594 | -12% |
| ibm17 | 1.3830 | 1.5888 | -13% |
| ibm18 | 1.3865 | 1.6352 | -15% |
| **AVG** | **1.1193** | **1.3134** | **-15%** |

Every single benchmark improves over DPO. No regressions. The hard benchmarks (ibm17/18 at 1.38, ibm14 at 1.25) limit how low the average can drop — they need more budget.

## Path below 1.117

The leaderboard winner is at 1.1172, we are at 1.1193. Closing the 0.002 gap:

1. **Per-benchmark budget allocation (E9):** Give ibm17/18/14/12 longer (20-30 min); easy ones (ibm09 at 0.86) need only 3-5 min. Total wall stays around 3 hr but avg drops.
2. **LNS on hard benchmarks (E3):** ibm17 finished sweep 3 still improving slowly — CD plateaued. LNS rip-up-and-reinsert can escape the plateau by relocating clusters of macros at once. May crack ibm17/18 below 1.20.
3. **Combine both:** Long CD run + LNS at the end on the worst benchmarks.

Submission deadline May 21 — 24 days away. Cushion to iterate.

---

## Remaining experiments (2026-04-27 status)

After CD --all hit 1.1193, the experiment hierarchy reshuffles. **DPO-direction experiments (E10, E11, more seeds, Steiner-tree congestion, SA polish) are obsolete** — they target a baseline we've replaced. CD-direction experiments are what's live.

### Live (CD-direction)

| ID | Name | Effort | Expected delta | Status |
| --- | --- | --- | --- | --- |
| **E9** | Per-benchmark budget allocation | 0.5 day | ~0.02 drop → ~1.10 avg | Highest priority — cheap win |
| **E3** | LNS rip-up-and-reinsert | 2-3 days | Could crack ibm17/18 below 1.20 | Likely the leaderboard's actual recipe (vmallela's name = "Incremental CD+LNS") |
| **E4** | GPU-batched CD via conflict-graph coloring | 3-4 days | 5-20× speedup per sweep | Only useful if we exceed budget; current 600s is fine |
| **E7** | Differentiable LNS (DPO-style soft destroy) | 1 week | Speculative | Run only if E3 works as a refinement |

### Frozen / obsolete

| ID | Name | Why frozen |
| --- | --- | --- |
| E5 | Batched DPO seeds | Falsified — same-basin collapse |
| E6 | DPO seed → CD polish | DPO seed offers no advantage over SDF for CD; superseded by CDOnly direct |
| E10 | Congestion-only DPO | Marginal -0.33%, deepens basin lock on ibm02 |
| E11 | DPO from diverse priors | Flat on --all; ibm02/12 got worse |
| Steiner-tree congestion | (other agent's DPO improvement) | Targets DPO's RUDY model; CD doesn't use it |
| SA polish, adaptive phases | (other agent's DPO improvement) | Targets DPO; CD doesn't need it |

### Sequencing — May 21 deadline (24 days)

| When | Action | Expected result |
| --- | --- | --- |
| Today | Run E9 (per-bench budget) — ~3 hr wall clock | avg ~1.09-1.10 |
| Days 2-4 | Build E3 (LNS); validate on ibm17/18 single-bench | ibm17/18 to 1.20-1.25 |
| Days 5-7 | E3 --all + ablation against E9-only | avg ~1.05-1.08 |
| Days 8-14 | Polish, multi-seed validation, NG45 hidden-test prep | confirm robustness |
| Days 15-21 | Writeup + final submission | innovation-prize report + champion submission |

The other Claude instance should pivot from DPO experiments to **drafting the writeup**. The methodology section now leads with: incremental evaluator → LP-HPWL diagnostic showing congestion = 74% → full-proxy CD as the basin-changing optimizer that DPO couldn't achieve.

### Decision rules (updated)

- **If E9 hits ≤ 1.10:** Submit it as a verified champion baseline; LNS becomes optional polish.
- **If E9 plateaus at ~1.115:** LNS (E3) is mandatory; the per-benchmark plateau on ibm17/18 is structural, not budget-bound.
- **If E3 doesn't push us below 1.05 by Day 14:** Accept the result, prioritize writeup quality and NG45 robustness.
- **If we hit avg ≤ 1.00:** That's well below the leaderboard; we likely have the prize-winning entry.

---

## 🏆 CHAMPION TRAJECTORY 2026-04-27 — CD --all running 10/17 benches at avg 1.0372

CDOnlyPlacer `--all` (10 min/bench, full-proxy CD on incremental evaluator). 10 of 17 benchmarks complete; running average **1.0372 vs DPO best_of_v2 1.3134 (-21%)**, **vs leaderboard target 1.117 (-7.1%)**.

| Bench | DPO BoV2 | CD --all | Delta |
| --- | --- | --- | --- |
| ibm01 | 1.1285 | 0.9133 | -19% |
| ibm02 | 1.6888 | 1.1534 | -32% |
| ibm03 | 1.2508 | 0.9942 | -21% |
| ibm04 | 1.3239 | 1.0193 | -23% |
| ibm06 | 1.6434 | 1.1656 | -29% |
| ibm07 | 1.3846 | 1.1105 | -20% |
| ibm08 | 1.3816 | 1.1307 | -18% |
| ibm09 | 1.0130 | 0.8606 | -15% |
| ibm10 | 1.2540 | 1.1000 | -12% |
| ibm11 | 1.0657 | 0.9248 | -13% |
| **avg10** | **1.3134** | **1.0372** | **-21%** |

7 benchmarks remaining (ibm12-18, currently on ibm12 init 1.650). If trajectory holds, full-17 projection is **~1.04-1.09 avg**, beating the leaderboard 1.117 by 5-15%.

The recipe that closed the gap:
- **Incremental evaluator (E1)** — 4657× speedup on per-move cost queries; the unlock
- **Full-proxy coordinate descent** — discrete moves on each macro's two axes via breakpoint enumeration (no golden-section fallbacks needed)
- **SDF init** — same prior we've used all along
- **10-minute budget per benchmark** — captures ~85% of 40-minute value
- **No DPO, no LNS, no diverse priors** — pure CD on a fast evaluator

Total wall clock: ~170 minutes for 17 benchmarks, well under the 1-hour-per-benchmark hidden-test budget.

## ⚡ BREAKTHROUGH 2026-04-27 — E2 hits 1.0632 on ibm10 from CD-only

E2 (40-min full-proxy CD on incremental evaluator, SDF init) on ibm10 reached **proxy 1.0632** (full-eval cross-check 1.0724). For comparison:

| Method | ibm10 proxy |
| --- | --- |
| SDF init | 1.4112 |
| RePlAce baseline (avg) | 1.4578 |
| DPO best_of_v2 (our champion) | 1.254 |
| **E2 CD-only (40 min from SDF)** | **1.0632 (1.07 full)** |
| Leaderboard avg target (all 17) | 1.117 |

CD beat our DPO by **15.2%**, beat the leaderboard's average target on a single benchmark. WL went UP (0.0687 to 0.0790) — CD trades cheap WL for expensive density (-25%) and congestion (-27%), exactly as E8 predicted.

13 sweeps, 14932 accepted moves, 0 overlaps, golden-section fallbacks 0 (all closed-form breakpoint enumeration). Fast convergence: sweep 1 (3.6 min) hit 1.18; diminishing returns thereafter. **8-10 min/benchmark may capture 90% of the value.**

**Next:** Build `CDOnlyPlacer` from the diagnostic, validate on hard benchmarks (ibm02 high-density, ibm18 high-cong), then graduate to `--all`.

### Validation 2026-04-27 01:03 — CD generalizes

Three single-benchmark `-b` runs at 600s CD budget each:

| Benchmark | SDF init | DPO best_of_v2 | CD-only 10min | Delta vs DPO |
| --- | --- | --- | --- | --- |
| ibm10 | 1.411 | 1.254 | 1.1039 | -12% |
| ibm02 | 1.689 | 1.689 basin-locked | 1.1536 | -32% |
| ibm18 | 1.793 | 1.635 | 1.3929 | -15% |
| avg of 3 | 1.631 | 1.526 | 1.2168 | -20% |

CD broke the **ibm02 basin lock** that DPO could not escape regardless of seed (multi-seed showed all 4 seeds at 1.6888 byte-identical). Same trade-off pattern across all 3: WL up 13-15%, density down 21-32%, congestion down 23-40%. Zero overlaps everywhere.

10-min budget on ibm10 hit 1.1039 vs 40-min budget's 1.0724 — confirms **10 min captures ~85% of the value**, plenty for a tractable `--all` run.

**Status:** `--all` run launched at 01:04 with 600s CD budget per benchmark (17 × 10 min ≈ 170 min wall clock). Expected avg ~1.10-1.15 if 3-benchmark validation scales.

**Implication:** This invalidates the "IBM benchmarks must run ~50-60s" memory entry. vmallela's confirmed 40-min/benchmark on the leaderboard means the rules allow our 10-min-or-more runs. Update `time_budget_constraint` memory once CD --all confirms champion.

---



Created 2026-04-26 in response to leaderboard entry **vmallela "Incremental
CD+LNS" — 1.1172 (unverified)**, ~40 min/benchmark, pure Python+numpy,
single-threaded, zero overlaps.

> **Coordination note:** A parallel Claude Code agent owns `writeup/todo.md`
> (DPO ablations, multi-seed runs, congestion-gradient literature check,
> per-benchmark figures). This doc is **orthogonal**: score-pushing
> experiments inspired by the leaderboard delta and our unused GPU. Do not
> duplicate writeup ablations here.

---

## Reading the leaderboard

| Submission | Score | Compute | Hardware | Gap to ours (1.3834) |
|---|---|---|---|---|
| vmallela "Incremental CD+LNS" | **1.1172** | ~40 min/bench | CPU, single-thread | **−19.2%** |
| Our DPO best-of-v2 | 1.3834 | ~60 s/bench | M3 MPS GPU | — |
| RePlAce baseline | 1.4578 | — | — | +5.4% |

**The implied recipe** (from the submission name + runtime profile):

1. **Coordinate descent on the actual HPWL.** For one macro held against the
   rest, HPWL is piecewise-linear in `x` with breakpoints at incident-net
   endpoints. The optimum is the **weighted median of net-endpoints** —
   closed form, O(degree·log degree). One sweep ≈ a wirelength projection.
2. **Incremental evaluator.** Per-net min/max trackers + bin-density grid
   with delta updates. A single macro move costs O(degree) instead of
   O(macros·nets). At ~1000× speedup per move, 40 min single-thread ≈ tens
   of millions of accepted moves.
3. **LNS to escape CD local minima.** Destroy *k* macros (5–50), re-insert
   via constrained CD or a small ILP. Standard VLSI detail-placement recipe.

**The two levers we are not pulling:**
- **Compute** — we run ~60 s/IBM. They run ~40 min. ~40× gap.
- **Move quality** — we use gradient steps (DPO) and simulated annealing.
  They use exact 1D optima per coordinate.

We have a third lever they don't: **GPU**. M3 MPS is genuinely fast for
batched tensor ops. CD and LNS are usually framed as serial, but most of
the work parallelizes.

---

## ⚠ Strategic update 2 — 2026-04-27 (post-multi-seed)

The other agent's 4-seed `--all` run on `best_of_v2` finished. Per-seed averages 1.3790-1.3927 (range 0.014). **Best-per-bench across all 4 seeds = 1.3710** — only 0.012 (0.9%) better than baseline 1.3834.

Even more telling: **ibm02 and ibm12 have ZERO seed variance** — DPO converges to byte-identical placement regardless of seed. These are the high-density (0.84/0.81), congestion-dominant benchmarks where the SDF basin is so deep DPO can't escape via Adam init noise. E5's perturbed-seed collapse isn't unique to seed perturbation; it's structural.

**Implication:** Within-DPO improvements (seeds, iters, perturbation, even diverse priors) cap at ~1-2%. To reach 1.117 we need a mechanism that **changes the basin**, not refines within it. That's what E2/E3/E10/E6 test.

## ⚠ Strategic update — 2026-04-26 (post-E8)

E8 (LP HPWL diagnostic, see `docs/lp_hpwl_diagnostic.md`) decomposes our
1.3834 proxy as:

| Component | Contribution | Pct of proxy |
| --- | --- | --- |
| Wirelength (coef 1.0) | 0.078 | 5.6% |
| Density (coef 0.5) | 0.279 | 20.2% |
| Congestion (coef 0.5) | 1.026 | 74.2% |

Closing 100% of the WL gap to the LP floor saves at most 0.069 — about a
quarter of the gap to 1.117. **Congestion is the dominant lever.** The
leaderboard winner must be doing efficient congestion-aware descent, not
just HPWL median sweeps.

This invalidates the "weighted median" framing for E2 — that only optimizes
WL. The redesign:

- **E2 reframed:** CD on the full proxy via numerical 1D line search per
  coordinate (golden section or PWL evaluation at breakpoints), NOT closed-
  form HPWL median. The incremental evaluator (E1) must expose
  congestion+density deltas with the same speedup as HPWL deltas, otherwise
  the whole program is bottlenecked.
- **E3 promoted:** LNS becomes the highest-leverage downstream experiment.
- **E6 unchanged:** DPO→CD polish still valid because DPO already optimizes
  the full proxy via gradients.
- **New E10 below:** congestion-only attack on existing DPO output.

---

## Experiments

Ranked by expected value-per-day. Each one is a single hypothesis, runnable
with the existing `evaluate` harness. Update `results/experiment_log.jsonl`
after each `--all` run.

### Tier 1 — replicate vmallela's recipe (highest expected value)

#### E1. Incremental HPWL/density evaluator — **VALIDATED 2026-04-26**
**Result:** Built and verified. **4657× speedup on ibm10** (6.45 ms/move incremental vs 30.06 s/call full). Bit-for-bit parity at machine precision (worst diff 1.1e-15 abs) on ibm01 and ibm10 across 130 random moves; revert test passes within 1e-9 relative.

**Critical finding:** RUDY congestion IS decomposable per single-macro move — no fallbacks needed. Per-net congestion contributions, per-cell macro routing, per-net WL bbox, per-cell density all stored and updated in place. Smoothing recomputed per `current_cost()` call via vectorized cumsum.

**Files (930 + 146 + 112 lines):**
- `macro_place/incremental_evaluator.py` — `IncrementalProxyEvaluator` class with `move(idx, xy)`, `current_cost()`, `revert()`
- `test/test_incremental_evaluator.py` — parity + revert tests
- `scripts/bench_incremental.py` — speedup micro-benchmark

**Quirks worth knowing for downstream agents:**
- Float precision: cast input placement to float64 for parity; macro sizes must be read from `plc.modules_w_pins[i].get_width()/get_height()`, NOT `benchmark.macro_sizes` (float32 vs float64 mismatch)
- Net weights come from the driver pin's `get_weight()`, not `benchmark.net_weights` (zeros after loader)
- `compute_proxy_cost` monkey-patches `__get_grid_cell_location` to clamp; the evaluator mirrors clamped semantics
- Smoothing pass on `current_cost()` is the dominant per-call cost; `move()` is much cheaper than `current_cost()`. Plan CD line-search to minimize redundant cost queries.

**Unblocks:** E2, E3, E6.

---

#### E2. Coordinate-median sweeps (CPU, 40-min budget, single benchmark)
**Hypothesis:** CD-only on incremental evaluator at vmallela's budget gets
us most of the way to 1.11, before any LNS.

**Run:** Pick `ibm10` (large, our weakness). Initialize from DPO output. Run
weighted-median sweeps until convergence or 40 min wall clock. Report
proxy cost vs DPO-only baseline.

**Effort:** 1 day after E1. **Kill gate:** if ibm10 proxy doesn't drop ≥10%
vs DPO seed in 40 min, the median-sweep recipe alone isn't the answer and
LNS (E3) becomes the load-bearing piece.

---

#### E3. LNS rip-up-and-reinsert — **NEXT (priority 2, biggest lever)**

**Hypothesis:** CD plateaus on hard benchmarks (ibm17/18 at 1.38) because it respects topology — it can move each macro independently but can't rearrange clusters. LNS rips up *k* macros at once and reinserts them in a different region, escaping the CD-fixed-point. vmallela's submission is named "**Incremental CD+LNS**" — the LNS is exactly what we don't have, and the most likely reason their hard benchmarks land lower than ours.

**Where to attack:** Concentrate LNS on the 4 worst CDOnly benchmarks: ibm17 (1.3830), ibm18 (1.3865), ibm14 (1.2478), ibm12 (1.2418). Each ~0.20 above the easy-benchmark floor.

**Algorithm sketch:**
1. **CD to convergence** (or near it — 600s).
2. **LNS phase** for remaining time:
   - **Region selector:** rank macros by per-macro cost contribution (delta to proxy if removed, computed via the incremental evaluator). Pick top-*k* (k=5-30, sweep).
   - **Or:** spatial cluster — pick all macros in the most congested grid cell + neighbors.
   - **Destroy:** mark these k macros as "free"; record their original positions for revert.
   - **Reinsert:** for each free macro in cost-descending order, do a full canvas search via the incremental evaluator (every grid bin center is a candidate). Pick the proxy-minimizing position. Or: serial CD on the destroyed subset only.
   - **Accept:** keep iff total proxy improves; else revert.
3. **Repeat** with different *k* values (annealing).

**Why the incremental evaluator is critical:** a full canvas search for one macro = ~1500 grid bin candidates × O(degree) per candidate = ~1.5ms per candidate × 1500 = 2 sec per macro reinsertion. Without E1's 4657× speedup, this would be hours per LNS iteration.

**Build:**
- New module `submissions/cd/lns.py` with `LNSDestroyAndReinsert` class
- New placer `submissions/cd/cd_lns_placer.py` = CDOnly + LNS phase
- Default config: 600s CD + 600s LNS per benchmark (1200s total budget for hard benchmarks)

**Validation plan:** Run on ibm17 and ibm18 single-bench with 1200s total. Each should drop ≥10% from CDOnly's 1.38 → 1.20-1.25.

**Effort:** 2-3 days. **Acceptance:** 4-bench (ibm12/14/17/18) avg drops from current 1.32 to ≤ 1.20. Total --all drops to 1.05-1.08.

**Risk:** LNS without overlap-aware reinsertion can produce overlaps that the incremental evaluator doesn't track explicitly. Mitigation: post-LNS legalization pass (existing projection.py) before final cost validation.

---

### Tier 2 — GPU exploitation (our differentiator)

#### E4. GPU-batched coordinate descent
**Hypothesis:** CD is "serial" only because each macro's optimum depends on
its neighbors. Build a **conflict graph** (macros sharing nets), color it,
and update all macros of one color in parallel on GPU. Typical chromatic
number on netlists is 50–200 — we get 5–20× parallelism within sweeps.

**Effort:** 3–4 days. Builds on E1's evaluator. **Acceptance:** wallclock
per sweep ≤ 1/5 of E2's CPU baseline at equal acceptance count.

---

#### E5. Massively parallel DPO seeds (best-of-N at scale) — **FALSIFIED 2026-04-26**
**Original hypothesis:** Vectorize across seeds into `[B, M, 2]`; B=64 at ≤2× wall clock with best-of-N quality win.

**Result:** B=64 wall clock = **8.9× B=1** (RUDY congestion kernel scales 27× on MPS — ironic given E8 says congestion is 74% of cost). Quality essentially flat: B=64 fast-set 1.1698 vs B=1 1.1682. SDF-perturbed seeds at sigma=0.04·canvas all collapse to the same basin.

**Files:** `submissions/dpo/batched_seeds_placer.py` (679 lines), `submissions/dpo/batched_seeds_b1.py` (24 lines). Logged as `e5_batched_seeds` and `e5_batched_seeds_b1`.

**Lesson:** Best-of-N within a single basin doesn't help. The fix is prior diversity — see **E11** below.

---

#### E6. DPO seed → CD polish (the obvious hybrid)
**Hypothesis:** DPO finds the right *topology* (good convergence basin); CD
tightens within that basin to the local HPWL/density optimum. Sequential
composition.

**Run:** DPO best-of-v2 → 30 s of weighted-median CD per benchmark on the
incremental evaluator. Total budget ≈ 90 s/IBM, still within hidden-test
1-hour budget for NG45.

**Effort:** 1 day after E1. **Acceptance:** improves over DPO-only on
≥12/17 IBM benchmarks. **High prior** this works — both methods are valid
in isolation; combining them only fails if DPO output is already at a CD
fixed point (unlikely given its gradient-not-median objective).

---

### Tier 3 — speculative / structural

#### E11. DPO from diverse priors — **VALIDATED on --fast 2026-04-26, --all running**

**Result on --fast (4 benchmarks, ibm01/04/09/13):**

| Strategy | Avg proxy | Delta vs best_of_v2 1.1749 |
| --- | --- | --- |
| All 4 priors sdf+will+greedy+random | 1.1704 | -0.4% |
| 2 priors sdf+random | 1.1659 | -0.8% |
| sdf only | 1.1846 | +0.8% infra noise |

**Per-prior fast-set avg:** sdf 1.1647, will 1.1862, greedy 1.5378, random 1.4146.

**Per-bench winner:** sdf wins ibm01/04; will wins ibm09/13. greedy and random NEVER win on fast set but don't hurt because best-of selection.

**Basin diversity confirmed:** Greedy lands DPO in a cramped basin (Y-spread 3.1 vs SDF 5.6) that's consistently 0.5 worse. Random reaches 1.41-1.65 but stays feasible after projection. SDF and Will land in genuinely different basins (4-7% per-bench difference, alternating winners). This contrasts with E5's perturbed-seed collapse (all 64 seeds within 0.001).

**Files:** `submissions/dpo/init_strategies.py` (196), `submissions/dpo/diverse_priors_placer.py` (177), 2 ablation wrappers. Total 423 lines.

**Wall clock:** all-4 fast = 450 s (vs 4× sdf-only 562 s — passes ≤4× constraint).

**Status:** Graduated to `--all` (running, ~25 min). Will report back. Expected `--all` ≈ 1.36-1.37 if fast-set delta scales linearly. Suggest follow-up `e11_sdf_will` 2-prior ablation since greedy/random never win on fast.

**--all RESULT 2026-04-27 01:04: avg = 1.3839 vs baseline 1.3834 (+0.04% — FLAT).** Fast-set improvement did not scale. Per-bench: 9 small wins (best -2.34% on ibm13) balanced by 8 small losses. Critically, **ibm02 (+1.46%) and ibm12 (+1.71%) — the basin-locked benchmarks — got WORSE**, suggesting alternative priors (will/greedy/random) land in even deeper local minima for high-density. The fast set's positive result was driven by basins ibm09/ibm13 where Will-prior happens to win, but those gains don't generalize to hard cases. **E11 dead end. CD is the path.**

---

#### E10. Congestion-only attack — **MARGINAL --all = 1.3788 (-0.33%) 2026-04-27**

**Fast set (3 variants):** CongestionRefine 1.1750 (flat), Aggressive 1.1693 (-0.5%), **Mild 1.1636 (-1.0%)**.

**--all (Mild):** 1.3788 vs baseline 1.3834 = **-0.33%**. Per-bench wins on cong-heavy: ibm06 -3.0%, ibm10 -0.9%, ibm17 -0.6%. But **ibm02 got WORSE +2.3%** — congestion refinement deepened the basin lock instead of breaking it.

**Files:** `submissions/dpo/congestion_refine_placer.py`, `submissions/dpo/_e10_sweep_mild.py`, `_e10_sweep_aggressive.py`. **Verdict:** marginal lever, not a champion path. Within-DPO refinement still hits the basin ceiling.

#### E10. Congestion-only attack on existing DPO output (NEW, post-E8)
**Hypothesis:** Since congestion is 74% of cost and DPO already converged on
WL+density, a focused congestion-only refinement on top of DPO output may
unlock disproportionate improvement. Use the existing DPO congestion
gradient (already in `submissions/dpo/best_of_v2_placer.py`) but freeze WL
and density terms and run for 30 s of additional iterations after
legalization.

**Effort:** 1 day. No E1 dependency — uses existing DPO machinery.
**Acceptance:** ≥3% drop on the average over the 5 most congested
benchmarks (ibm06, ibm17, ibm18, ibm02, ibm12).
**Why high prior:** the DPO gradient on congestion is already implemented;
this just changes the loss-term weighting late in training. Cheapest
possible test of "is congestion-aware refinement the missing piece."

---

#### E7. Differentiable LNS
**Hypothesis:** Replace LNS's discrete destroy-and-reinsert with a soft
relaxation: temporarily decrease the overlap penalty for a region, run
DPO gradient steps, reproject. Inspired by diffusion-style annealing.

**Effort:** 1 week. **Run only if E3 works** — confirms LNS-style escapes
help before investing in the soft variant.

---

#### E8. HPWL lower-bound diagnostic
**Hypothesis:** Solve the LP relaxation of HPWL per benchmark (drop overlap
constraints, keep fixed macros). The gap between LP-HPWL and our HPWL tells
us what fraction of the remaining 1.38 → 1.11 gap is wirelength vs
density/congestion.

**Effort:** 1 day (LP solver already in repo). **Output:** per-benchmark
table of `(LP-HPWL, our-HPWL, gap%)` and the same for density bound. Tells
us where the score is actually hiding.

---

#### E9. Adaptive per-benchmark budget with plateau detection — **NEXT (priority 1)**

**Hypothesis:** Static per-bench tiers require guessing benchmark difficulty in advance — fragile, doesn't transfer to hidden NG45. **Auto-adaptive: each benchmark runs until plateau OR 1hr cap, whichever first.** Easy benchmarks exit early; hard ones can use the full hour if still improving.

**Why this is strictly better than static tiers:**
1. ibm09 plateaus at sweep 4-5 (~3 min) — early exit reclaims ~7 min for harder benchmarks
2. ibm17/18 may want up to 1hr if still improving (current 600s ran out mid-descent)
3. Transfers to NG45 hidden test without per-bench tuning
4. Graceful 1hr stop instead of hitting a hard rule

**Trajectory evidence from current --all run:** Last 6 ibm10 sweeps (24 min) gained 0.005 vs first sweep gaining 0.23. ~95% efficiency loss in the second half. Plateau detection captures the elbow.

**Algorithm:**
```python
hard_cap_s = 3600          # 1 hr per benchmark (matches competition rule)
min_time_s = 300           # don't check plateau before 5 min
patience = 3               # consecutive sweeps allowed below threshold
plateau_threshold = 0.005  # absolute proxy delta per sweep

# Exit when:
#   wall_clock >= hard_cap_s, OR
#   wall_clock >= min_time_s AND
#     last `patience` sweeps each had delta_proxy < plateau_threshold
```

**Build:**
- Modify `run_cd(...)` in `submissions/cd/cd_only_placer.py` (or factored module) to accept `(min_time_s, hard_cap_s, patience, plateau_threshold)` and track per-sweep deltas
- Track a deque of last `patience` sweep-deltas; exit when all are below threshold AND we're past `min_time_s`
- Default: `(300, 3600, 3, 0.005)`. Make all four configurable for tuning.
- New file `submissions/cd/cd_adaptive_placer.py` with these defaults

**Acceptance:** avg --all proxy ≤ 1.10 with total wall ≤ 4 hr. **Kill gate:** if total wall exceeds 6 hr (some benchmark hits the 1hr cap multiple times), tighten patience or threshold.

**Tuning sweep (do on --fast first):** vary `(patience, threshold)` over 3×3 = 9 settings. Pick the Pareto-optimal point on the (avg proxy, wall clock) curve.

**Why this beats static tiers:** the plateau detection is **per-benchmark, per-run**, so it adapts to actual trajectory rather than priors. On a benchmark we've never seen (NG45), this is the only sane policy.

---

## Sequencing (original 2026-04-26 plan — superseded, see "Remaining experiments" section above)

The original plan above this point was written before the CD breakthrough. Live sequencing is in the "Remaining experiments" section near the top of this file.
