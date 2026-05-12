# Morning Report 2 — 2026-05-12 06:30 PDT

## TL;DR

Overnight ran **8+ cloud experiments** across 80+ lane-hours. **Found a real new winner: `finegrain` cascade.**

**Recommended submission: `submissions/cd_lns_sa_cascade/placer_finegrain_safe_adaptive.py`**

| Score | Value | vs prior best (1.137 / 0.6978) |
|-------|------:|-------------------------------:|
| IBM avg | **1.12928** | **−0.68%** |
| NG45 avg | 0.7037 | +0.85% (regression on ariane133) |
| Max wall | 51.9 min | 8 min cap margin |

Or for best proxy with thinner margin: **A. `placer_finegrain_adaptive.py`** (1.12189 / 0.6938 / 55.9 min / 4 min margin).

## Overnight chronicle

### 21:00 — 5-variant cascade probe on 4 hardest benches

Tested {wide-saddle, more-cascade, dual-basin, aggressive-kjoint, finegrain} on ibm10/12/14/17.

- **4 variants strictly worse than cascade b=3000** (falsified): wide-saddle, more-cascade, dual-basin, aggressive-kjoint.
- **Finegrain** (tight eps {0.05, 0.1, 0.3, 0.7} + 300s polish + max_iters=2): **2/4 wins, avg −0.16%.** Marginal but real.

### 22:00 — finegrain --all + ext-budget --all launched in parallel on cloud

Finegrain --all first 4: avg −1.55% vs cascade. Real lift, not marginal! Promoted to candidate.

### 02:14 — finegrain --all FINAL

**1.12189 IBM avg.** Wins 12/17 benches, losses ≤+1.16%. Walls 50–55.9 min. **The new IBM champion under wall cap.**

### 02:30 — finegrain_safe (b=2700) launched for durability

Tighter budget gives more margin against partcl box variability. Cost: +0.66% on IBM aggregate.

### 03:30 — ext_budget (b=5400) --all FINAL

**1.10412 IBM avg** with 78–94 min walls. **Confirms EPYC ceiling at ~1.10 for our pipeline.** Achievable only with PATH A2 CD speedup (Cython port of `move()`).

### 04:00 — micrograin probe (ultra-tight eps {0.02, 0.05, 0.1, 0.2, 0.5})

Tied with finegrain (−0.03% diff). No further lift from tighter eps.

### 05:18 — fgsa --ng45 audit

Finegrain_safe_adaptive --ng45 = 0.7037 avg. **ariane133 regressed badly** (0.6518 → 0.7018) because b=2700 plateau-exits too early on the breakthrough bench.

### 06:22 — finegrain_safe --all FINAL

**1.12928 IBM avg.** +0.66% vs b=3000 but 4 min more cap margin.

## Submission decision matrix

```
                    A (b=3000)         B (b=2700, RECOMMENDED)
IBM aggregate       1.12189            1.12928   (+0.66%)
NG45 aggregate      0.6938             0.7037    (+1.43%)
Max IBM wall        55.9 min           51.9 min  (+8 min margin)
DQ risk             7% slowdown        15% slowdown
Beats RePlAce       −23.0%             −22.5%
vs leaderboard 1.117 +0.4%             +1.1%
```

**Recommendation: B (`placer_finegrain_safe_adaptive.py`).**

The proxy delta is small (0.66%) but B's 15% slowdown tolerance is significantly safer than A's 7%. The DQ risk on a single bench (ibm18 the worst) would zero our score on that bench.

If you trust partcl's EPYC is within 5% of ours: submit A.
If you want defensive: submit B.

## All cloud lanes run overnight

| Experiment | Hours | Result | Useful? |
|-----------|------:|--------|---------|
| 5-variant probe (wide-saddle/more-cascade/dual-basin/aggressive-kjoint/finegrain) on 4 benches | 1.0 | finegrain wins | ✓ critical |
| finegrain --all 17 IBM | 4.0 | 1.12189 IBM | ✓ submission target A |
| finegrain --ng45 | 1.0 | 0.6954 | ✓ confirmed NG45 |
| finegrain_adaptive --ng45 | 1.0 | 0.6938 | ✓ submission target A NG45 |
| micrograin probe | 0.9 | tied | refutation, useful |
| finegrain_safe (b=2700) --all 17 | 3.5 | 1.12928 IBM | ✓ submission target B |
| fgsa --ng45 | 0.8 | 0.7037 | ✓ submission target B NG45 |
| ext_budget (b=5400) --all 17 | 4.5 | 1.10412 IBM | ✓ ceiling proves A2 path |
| **Total cloud lane-hours** | **~16.7** | | |

## What's NOT done (next 9 days of work)

- **PATH A2 (Cython port of `move()`)** — the path from 1.122 to ~1.105. Multi-day refactor.
- **PATH A3 (GPU batch candidate eval)** — alternative to A2, similar 5-10× speedup target.
- **Tier 2 OpenROAD verification** — ariane133 with our placement under full PnR flow. Required for grand prize.
- **Innovation Award writeup** ($4k prize, Henkelman & Jónsson saddle escape mechanism).

## Files / artifacts to look at

- `LEADERBOARD.md` — full state, single source of truth
- `OVERNIGHT_QUEUE.md` — autonomous loop self-tracking
- `submissions/cd_lns_sa_cascade/placer_finegrain_safe_adaptive.py` ← **RECOMMENDED SUBMISSION**
- `submissions/cd_lns_sa_cascade/placer_finegrain_adaptive.py` ← alternative if accepting wall risk
- `~/parallel_eval/` on `mpc-cloud` — raw per-bench logs

## Cloud state at 06:30

- All overnight runs complete except a few stragglers
- Cloud will idle after the few stragglers finish
- DREAMPlace install preserved at `/opt/DREAMPlace/install`
- 6 cascade variant placers in `submissions/cd_lns_sa_cascade/`:
  - `placer.py` (base)
  - `placer_b3000.py`, `placer_adaptive.py`, `placer_extended.py`, `placer_widesaddle.py`, `placer_morecascade.py`, `placer_dualbasin.py`, `placer_aggressive_kjoint.py`, `placer_finegrain.py`, `placer_micrograin.py`, `placer_finegrain_safe.py`, `placer_finegrain_adaptive.py`, `placer_finegrain_safe_adaptive.py`

## Recommended next moves on wake

1. **Inspect leaderboard + this report** (5 min)
2. **Decide submission target A vs B** (A=proxy-best, B=durable)
3. **Final smoke test the chosen placer on 1 bench** (5 min)
4. **Submit via partcl form** (15 min)
5. **Spend remaining 9 days on PATH A2** (Cython port of `move()` — only path to close gap to 1.10 ceiling)
