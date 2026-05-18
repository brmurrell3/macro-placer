# 2026-05-18 next submission candidate

The CURRENT submission (entry `placer.py` → `cd_lns_sa_cascade_dp_lane`)
lands IBM 1.06650 / NG45 0.68086 / combined 0.993. Solid but not a
champion.

The **NEXT submission candidate** is
[`submissions/cd_lns_sa_cascade_stacked_periphery/placer.py`](../../submissions/cd_lns_sa_cascade_stacked_periphery/placer.py)
(`CDLNSSACascadeStackedPeripheryPlacer`).

## Verified result

IBM `--all`: **1.05750**
NG45 `--ng45`: **0.68930**
Combined (21 benches): **0.987**
Overlaps: 0 across all benches.

Verified 2026-05-17 on M3 Max via `evaluate_parallel.py --jobs 4`. Total
wall: ~3.8 hr for IBM, ~50 min for NG45.

| | Next candidate (C) | Current (B) | Δ vs current |
|---|---:|---:|---:|
| IBM `--all` | **1.05750** | 1.06650 | **−0.85 %** |
| NG45 `--ng45` | 0.68930 | 0.68086 | +1.24 % |
| Combined (21) | **0.987** | 0.993 | **−0.60 %** |
| External deps | none | DREAMPlace optional | — |

## Architecture

Sequential cascade-then-portfolio stacking:

1. **E25 lane** — SDF init → CD → LNS → SA-v2 polish (with the new
   `project_overlaps` fallback before final validity check).
2. **E41 lane** — DPO init → CD → LNS → SA-v2 → K-joint K=3.
3. **Plateau pick** = best of {E25, E41} (with `try/except` on each
   lane — if one crashes on residual overlaps, the other carries the
   placer; fixes the NG45 nvdla case that crashed the first stacked run).
4. **Phase 3a: cascade_saddle** — canonical (1, 0.5, 0.5) Hessian
   eigvec, `max_iters=5`, polish=180s.
5. **Phase 3b: portfolio_saddle** — 3 non-canonical Hessian weights
   `[(1,0,1), (1,1,0), (0,1,1)]`, `max_iters=2`, `K_eps=2`, polish=60s.
6. **Periphery wrapper** — α=0.01 edge push + CD polish,
   strict-conservative accept. Every push was REJECTed in the --all run,
   so the wrapper acts purely as safety; same numbers without it.

## Why it beats the current submission

The current submission (`cd_lns_sa_cascade_dp_lane`) uses 3 init lanes
{E25, E41, DP-polished} feeding a single cascading_saddle escape.
The next candidate adds the **portfolio_saddle** layer on top —
3 additional non-canonical Hessian eigvec directions per iteration.

From the E100 spike: cascade alone hits ~λ_canonical-min plateau;
adding the cong-only Hessian's softest eigvec (different direction)
gives an additional −1 % lift on hard benches. Composing across 3
weights compounds.

Portfolio chose as the final winner on **17 of 17** IBM benches in the
verified run; the 4-layer chain (E25 → E41 → cascade → portfolio) gave
an aggregate **−3.3 %** descent from E25 plateau to final.

## Switch instructions

The submission entry placer (`placer.py` at repo root) currently points
to `cd_lns_sa_cascade_dp_lane`. To switch:

1. Edit `placer.py`:
   - Change `_PLACER_PATH = _REPO / "submissions" / "cd_lns_sa_cascade_dp_lane" / "placer.py"`
   - To: `_PLACER_PATH = _REPO / "submissions" / "cd_lns_sa_cascade_stacked_periphery" / "placer.py"`
   - And `class Placer(_mod.CDLNSSACascadeDPLanePlacer)` → `class Placer(_mod.CDLNSSACascadeStackedPeripheryPlacer)`
2. Update `README.md` and `SUBMISSION.md` to drop the DREAMPlace mount
   instruction (next candidate has no external deps).
3. Verify locally: `uv run evaluate placer.py -b ibm03 --json`.
4. Push to remote.

## Open variants still running (may produce sub-1.05)

- **tabu_stacked --all** on M3 (started 2026-05-17 ~9 PM, ETA ~1 AM
  2026-05-18): cascading_saddle replaced with E99 `tabu_levy_saddle_escape`
  which forces orthogonal eigvecs across cascade iters (more direction
  diversity, complementary to portfolio's weight diversity). First WINNER
  was ibm01 at 0.85111 (vs original 0.85963 = −1.0 %), so the variant is
  showing real lift. Aggregate unclear until all 17 are in.
- **no_e41_deep --all** queued (will run on M3 or aws-gpu CPU after tabu):
  skips E41 lane, reallocates its 0.32×B to cascade (0.20→0.36) and
  portfolio (0.13→0.29). Hypothesis: more saddle iters compensate for
  worse starting plateau.

If either lands a new best, document and re-promote.

## How to verify

```bash
# Run the next candidate in our partcl eval_docker
./eval_docker/run_eval.sh thinkorplace \
    submissions/cd_lns_sa_cascade_stacked_periphery/placer.py .

# Or outside docker (dev)
uv run evaluate submissions/cd_lns_sa_cascade_stacked_periphery/placer.py --all --json
uv run evaluate submissions/cd_lns_sa_cascade_stacked_periphery/placer.py --ng45 --json
```

Expected: IBM 1.057-1.058 / NG45 0.689-0.69 / zero overlaps on all 21
benches.

## Files

- [`submissions/cd_lns_sa_cascade_stacked_periphery/placer.py`](../../submissions/cd_lns_sa_cascade_stacked_periphery/placer.py) — entry
- [`submissions/cd_lns_sa_cascade_stacked/placer.py`](../../submissions/cd_lns_sa_cascade_stacked/placer.py) — inner cascade→portfolio
- [`experiments/E109_stacked_cascade_portfolio/manifest.md`](../../experiments/E109_stacked_cascade_portfolio/manifest.md) — research record
- [`experiments/E84_cascading_saddle/code/cascading_saddle.py`](../../experiments/E84_cascading_saddle/code/cascading_saddle.py) — cascade saddle
- [`experiments/E100_weight_portfolio_saddle/code/portfolio_saddle.py`](../../experiments/E100_weight_portfolio_saddle/code/portfolio_saddle.py) — portfolio saddle
- Memory: `champion_2026_05_17_stacked_periphery.md`
