# E68 engineering notes — what changed and where

Audit trail for the §4.5 work-bounded refactor.
**No champion files were modified.** All changes live in
`code/cd_lns_sa_workbounded.py` as a flat copy of E25 + E41 + E48 with
the four termination loci edited.

## File map

| New entity in `code/cd_lns_sa_workbounded.py` | Source (verbatim copy unless noted) | Phase |
|---|---|---|
| `_cost_aware_destroy`, `_is_legal_2d`, `_is_legal_2d_excluded`, `_gridbin_reinsert` | `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py` | helpers |
| `_adjacency_scores`, `_ktuples_pairwise_legal`, `_enumerate_topN_for_macro` | same | helpers |
| `run_lns_gridbin_wb`            | derived from `run_lns_gridbin` (E39 / E25) — termination edited | LNS |
| `run_sa_polish_v2_wb`           | derived from `run_sa_polish_v2` (E39 / E25) — termination edited | SA |
| `run_kjoint_lns_wb`             | derived from `run_kjoint_lns` (E39) — termination edited | K-joint |
| `CDLNSSAPlacerWB`               | derived from `CDLNSSAPlacer` (E25) — soft caps loosened | lane 1 |
| `CDLNSSADPOKJointPlacerWB`      | derived from `CDLNSSADPOKJointPlacer` (E41) — soft caps loosened | lane 2 |
| `CDLNSSAHybridWBPlacer`         | derived from `CDLNSSAHybridPlacer` (E48) — uses WB lanes | hybrid |

CD's `run_cd_adaptive` is NOT copied — it lives in `macro_place/cd_core.py`
and the existing implementation already matches the §4.5 prescription
(plateau on `delta < threshold` for `patience` consecutive sweeps with
defaults `patience=3, plateau_threshold=0.001`). The E68 placers simply
call it with the loosened soft cap.

## Per-phase before / after

### CD — `macro_place/cd_core.py:481` `run_cd_adaptive`

| Aspect | E48 (parent) | E68 |
|---|---|---|
| Plateau check (cd_core.py:563-569, unchanged) | `len(deltas) == patience AND all(d < plateau_threshold) AND elapsed >= min_time_s` | same |
| `patience` arg passed in | 3 (E25 / E41 default) | 3 (matches §4.5: "3 consecutive sweeps") |
| `plateau_threshold` arg passed in | 0.001 (E25 / E41 default) | 0.001 (matches §4.5: "delta < 0.001") |
| `min_time_s` arg passed in | 300.0 | 300.0 (unchanged) |
| Soft wall cap (`hard_cap_s`) | **2400.0** | **3000.0** ← §4.5 |

Code reference: `experiments/E68_workbounded_refactor/code/cd_lns_sa_workbounded.py`
constructor defaults — `cd_hard_cap_s: float = 3000.0` in both
`CDLNSSAPlacerWB` (line ~700) and `CDLNSSADPOKJointPlacerWB` (line ~860).

The plateau primary is unchanged in semantics. The §4.5 directive is
realized by loosening the soft cap so that on slower hardware the cap
fires later than the plateau, never first.

### LNS — `submissions/cd_lns_sa/placer.py` `run_lns_gridbin` and the verbatim clone in `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py`

Old saturation locus (cd_lns_sa/placer.py:247-250 and kjoint:263-266):

```python
if abs(sample_delta) < 1e-7:
    if log_fn is not None:
        log_fn(f"  LNS converged at sample {sample} (no improvement)")
    break                                  # <-- breaks on first non-improving sample
```

New saturation in `run_lns_gridbin_wb` (this experiment, ~line 245):

```python
if abs(sample_delta) < 1e-7:
    consecutive_no_improve += 1
else:
    consecutive_no_improve = 0
...
if consecutive_no_improve >= lns_patience:
    ...
    break                                  # §4.5: 5 consecutive non-improving
```

| Aspect | E48 (parent) | E68 |
|---|---|---|
| Saturation criterion | 1 non-improving sample | **5 consecutive non-improving samples** ← §4.5 |
| Soft wall cap | 600 s | **900 s** ← §4.5 |

Note: §4.5 calls this "looser saturation" relative to the current code
(which actually breaks on the *first* non-improvement). On M3 Max, the
LNS phase already plateau-exits in ≤397 s on every E41 bench (per E66
manifest), so loosening to 5-streak should rarely fire; the 900 s cap
is purely the slow-hardware floor.

### SA-v2 — `submissions/cd_lns_sa/placer.py` `run_sa_polish_v2` and the verbatim clone in kjoint

Old SA loop (cd_lns_sa/placer.py:322-324 and kjoint:328-330):

```python
while True:
    elapsed = time.perf_counter() - t_start
    if elapsed >= time_budget_s:
        break                              # <-- only termination is wall cap
```

The existing best-tracking code at lines 381-384 / 387-390 already
records `best_proxy` + `best_placement` + `best_found_at_t` whenever
`cur_proxy < best_proxy - 1e-12`. We add a counter alongside it.

New termination in `run_sa_polish_v2_wb` (this experiment, ~line 385):

```python
if accepted and cur_proxy < best_proxy - 1e-12:
    best_proxy = cur_proxy
    best_placement = evaluator.placement.detach().clone()
    best_found_at_t = time.perf_counter() - t_start
    moves_since_best = 0                   # §4.5: reset on improvement
else:
    moves_since_best += 1                  # §4.5: tick on every PROPOSED move

if moves_since_best >= sa_no_improve_moves:
    ...
    break                                  # §4.5: 1000-move saturation
```

| Aspect | E48 (parent) | E68 |
|---|---|---|
| Primary termination | wall cap only | **no best-so-far improvement in last 1000 proposed moves** ← §4.5 |
| Soft wall cap | 600 s | **900 s** ← §4.5 |
| Counter unit | n/a | `proposed` (a Metropolis step where `lo, hi`, candidate filter, and breakpoint generator all produced a usable candidate). Skipped iterations (`hi-lo < 1e-5` or empty `cands`) do **not** tick the counter. |
| Counter reset on | n/a | improvement to best-so-far (matches existing `best_proxy < best_proxy - 1e-12` test) |

Ambiguity / choice made: §4.5 says "no improvement in last 1000 moves"
without specifying *what* counts as a move. SA's loop has three counters
(`proposed`, `accepted_better+accepted_worse`, `skipped`). I chose to
tick on every `proposed` increment (i.e., every Metropolis step where a
candidate was successfully generated), since that's the natural unit of
"work" — skipped iterations represent search-space dead-ends, not real
work. This matches the spirit of "work-bounded" termination. On the
margin between this and "tick on every iter regardless of skip", the
choice favors a slightly looser termination (more skipped iters means
counter advances more slowly), which is conservative.

### K-joint — `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py` `run_kjoint_lns`

Old saturation locus (kjoint.py:828-832):

```python
# Stop if a full pass produced no commit.
if pass_committed == 0:
    if log_fn is not None:
        log_fn(f"  K-joint converged at pass {pass_idx} (no improvement)")
    break                                  # <-- pass-bound, not tuple-bound
```

New saturation in `run_kjoint_lns_wb` (this experiment, ~line 660):

```python
# Per-tuple sliding counter. Does NOT reset on pass boundary.
if committed_this_tuple:
    consecutive_no_commit = 0
else:
    consecutive_no_commit += 1

if consecutive_no_commit >= kjoint_no_commit_tuples:
    ...
    saturated = True
    break                                  # §4.5: 30-tuple sliding window
```

| Aspect | E48 (parent) | E68 |
|---|---|---|
| Saturation criterion | 1 full pass with 0 commits | **30 consecutive K-tuples without commit (sliding window across passes)** ← §4.5 |
| Soft wall cap | 600 s | **900 s** ← §4.5 |
| Counter scope | per-pass | global (does not reset on pass boundary) |

Ambiguity / choice made: §4.5 says "no commits in last 30 K-tuples"
without specifying whether the counter spans pass boundaries. The
random-pass mode draws fresh K-tuples from the top-3K pool each pass,
so resetting on pass boundary would let an unlucky early-pass run-of-30
non-commits be erased by a single late commit. Spanning pass boundaries
matches the more honest "30 tuples in a row produced no improvement"
reading and matches §4.5's analogy with LNS's "5 consecutive
non-improving samples" (which also doesn't have a per-pass concept).
The counter only resets on an actual commit.

The E66 manifest shows 14 / 17 E41 K-joint phases hit the 600 s cap on
M3 Max, indicating K-joint is fundamentally budget-bound on hard
benches. On those benches the §4.5 change is mostly the soft-cap
loosening (600 → 900), giving 50 % more wall to commit more K-tuples
before cap. The 30-tuple saturation will rarely fire on those benches;
on easy benches (where ibm10 K-joint committed 72 tuples in 600 s
under thread-limit per E66) the saturation may fire if a late
saturation kicks in.

## Soft-cap totals (per lane)

| Phase | E48 lane budget | E68 lane budget |
|---|---|---|
| Init (SDF or DPO) | ~30 s — 600 s | unchanged |
| CD               | 2400 s          | 3000 s |
| LNS              | 600 s           | 900 s |
| SA-v2            | 600 s           | 900 s |
| K-joint (lane 2 only) | 600 s      | 900 s |
| **Total upper bound (lane 2)** | **4800 s** | **6300 s** |

The +1500 s growth is purely cap-headroom; all M3 Max benches plateau /
saturate well below the soft caps per E66 phase profile. Realistic per-
lane wall on M3 Max should remain in the ~30-60 min range.

## What this experiment does NOT do

Per the task spec:

- **No adaptive caps.** The task forbade extending §4.5 with later-
  phase budget reduction (E66 phase-3 ideas like "if CD ate 2800 s,
  shrink LNS to 200 s"). All caps are global constants.
- **No hardware probe at startup** (§4.2 item 5). The §4.5 prescription
  doesn't include this and the task explicitly bans adding it.
- **No telemetry beyond existing print statements.** The new counters
  are surfaced in the existing log lines (e.g., LNS's
  `no-improve-streak={n}/{patience}`) but no new file artifacts or
  metrics are written.

## What's untouched (champion preserved)

- `submissions/cd_lns_sa_hybrid/placer.py` (E48 hybrid)
- `submissions/cd_lns_sa/placer.py` (E25 lane source)
- `experiments/E41_dpo_kjoint/code/cd_lns_sa_dpo_kjoint.py` (E41 lane source)
- `experiments/E39_kmacro_joint_lns/code/cd_lns_sa_kjoint.py` (E39 inner primitives)
- `experiments/E18_dpo_init/code/cd_lns_sa_dpo_init.py` (DPO init helper)
- `macro_place/cd_core.py` (`run_cd_adaptive` plateau check)

## Validation pending

The placer was static-checked (`ast.parse` succeeds) but **not run**.
The directive from the parent agent: another probe is using compute on
ibm12; running another placer would contaminate timings. Next-session
plan:

1. `uv run evaluate experiments/E68_workbounded_refactor/code/cd_lns_sa_workbounded.py --fast --json --hypothesis E68`
2. Compare to E48 baseline --fast 0.92024.
3. If --fast clears, run `--ng45` (sentinel: ariane133 ≤ 0.7214).
4. If both clear, run `--all`.

Promotion threshold: --all ≤ 1.08151 AND --ng45 ≤ 0.6922 AND no
overlaps. Kill thresholds in manifest.
