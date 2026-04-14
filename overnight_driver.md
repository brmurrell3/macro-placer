# Overnight Driver Prompt

Paste this verbatim after `/loop ` (no interval — let the model self-pace):

---

You are the overnight experiment driver for the Macro Placement Challenge project. Your job is to mechanically execute `queue.md` until a stop condition triggers. Read `CLAUDE.md` for project context if you need it, but do not deviate from this protocol.

**Loop length note:** The queue holds 22 items across 4 stages. Worst case (all-run-all): 22 × 22min ≈ 8hr. With kills mixed in, realistic is 4–6hr. The 10hr budget should just fit with combine phases.

**Per-iteration protocol:**

1. **Check stop conditions first** (abort before picking new work):
   - `queue.md` has no `[pending]` items → STOP with `QUEUE_EMPTY`.
   - Driver has been running >10 hours (check first-log-line timestamp vs now) → STOP with `TIME_UP`.
   - Kill-streaks are NOT a stop condition. A kill is ~2min of work (fast_gate only). Keep going through the full queue regardless of how many consecutive fails appear — queue order shouldn't determine coverage.
   - Threshold crossings (<1.50, <1.46) are NOT a stop condition. Log as `NEW_BEST` / `CHAMPION_CROSSED` and keep searching — we want the global best.

2. **Pop** the top `[pending]` item in `queue.md`. Mark it `[running]` in-place with a timestamp.

3. **Bootstrap the worktree** (required — `git worktree add` does NOT include submodules). Before spawning the subagent, create the worktree yourself and symlink the submodule in:

   ```bash
   SLUG=<hypothesis_name>
   git worktree add -b exp/$SLUG .worktrees/$SLUG
   rm -rf .worktrees/$SLUG/external  # remove stub dir
   ln -s /Users/brendan/Developer/macro-place-challenge-2026/external .worktrees/$SLUG/external
   ```

   Then spawn a subagent via the Agent tool with `subagent_type: "general-purpose"` (NOT `isolation: "worktree"` — that would create a second worktree without the bootstrap). Tell the subagent the absolute worktree path and that it should `cd` into it before running anything.

   Prompt the subagent with:
   - The absolute worktree path and a hard requirement to `cd` into it for all commands.
   - The full text of the queue item (goal, touch, change, fast_gate, kill).
   - "Implement the change. Run `uv run evaluate submissions/polyhedra/placer.py --fast --json --hypothesis <name>`. Parse the JSON (location: `results/PolyhedraNavigationPlacer_*.json` newest, or `results/experiment_log.jsonl` tail). If the fast_gate passes, run `--all --json --hypothesis <name>`. Report back: avg_proxy_cost for each mode, total_overlaps, fast_gate pass/fail, any kill trigger, and the worktree path. Under 300 words. Do NOT commit; I will decide."
   - "The `external/` directory in your worktree is a symlink to the main repo's submodule — this is intentional, do not touch it."
   - "If your change causes a Python error or overlaps, report the error and stop — do not retry more than once."

4. **Interpret the subagent report:**
   - If overlaps > 0 anywhere → mark queue item `[killed: overlaps]`.
   - If fast_gate fail → mark `[killed: fast_gate]`.
   - If the kill criterion spells a specific reason (e.g. `no_partitioner`, `no_replace_positions`, `no_pin_data`, `no_hook`, `depends-on-killed`) → mark `[killed: <reason>]`.
   - If `--all` ran → mark `[done: avg=<x.xxxx>]`, and add `[new_best]` / `[new_best_stage]` tags where applicable.
   - Else → mark `[done: fast_only avg=<x.xxxx>]`.
   - **Never stop on avg crossing 1.50 or 1.46.** Keep going.

5. **Track best-so-far (global AND per-stage).** Before writing the log line:
   - Parse the queue item's `stage:<tag>` field.
   - Scan `results/overnight_run.log` for prior `--all` avg values (globally, and filtered by same `stage=`).
   - Compute `best_so_far` (global min) and `best_in_stage` (per-stage min).
   - If this run's `--all` avg beats the prior global best → append a `NEW_BEST` line.
   - If this run's `--all` avg beats the prior stage best → append a `NEW_BEST_STAGE` line.

6. **Append one log line** to `results/overnight_run.log`:
   `<ISO-timestamp> <hypothesis> stage=<tag> <fast|all|kill> avg=<x.xxxx> overlaps=<n> gate=<pass|fail|kill> best_so_far=<x.xxxx> best_in_stage=<x.xxxx> worktree=<path>`

   Plus `NEW_BEST` / `NEW_BEST_STAGE` lines as applicable.

7. **Do not merge worktrees.** Leave the branch for the human to review in the morning. Do `git worktree list` at the end of each iteration. The current global best worktree is the primary merge target; per-stage winners are the inputs to combine-phase items.

8. **Combine-phase items are special.** When the queue item has `stage:combine`:
   - Before spawning the subagent, grep `results/overnight_run.log` for `NEW_BEST_STAGE stage=sp3`, `stage=sp1`, `stage=sp4` — the latest line per stage is the winner; extract its `worktree=` path and infer its branch (`git -C <path> rev-parse --abbrev-ref HEAD`).
   - If any required stage has no `NEW_BEST_STAGE` entry, look for its lowest-avg `[new_best]`-tagged `--all` run in the log; if still none, mark the combine item `[skipped: no_winner_for_stage_<X>]` and continue.
   - Pass the winning branch names into the subagent prompt. The subagent creates a fresh worktree off `main`, applies each winner's diff (`git diff main <branch> -- submissions/ | git apply`), runs `--fast` then `--all`, reports back as usual.
   - If `git apply` fails for any patch, the subagent should report the conflict summary; mark the item `[killed: patch_conflict <branch>]`.

9. **Loop back** to step 1.

**Hard rules:**
- Never modify `macro_place/` (the harness). Only `submissions/polyhedra/` and related submission files.
- Never `git push`, `git reset --hard`, `git branch -D`, or remove worktrees with `--force`.
- Never skip the kill criteria.
- Never ask the human a question — act on the protocol. If the protocol is ambiguous, STOP with `AMBIGUOUS: <reason>`.
- Each subagent iteration should take 10–25 min (fast ~2min, all ~20min). Use `ScheduleWakeup` between iterations only if needed; normally just loop.

**Telemetry recap each wake-up:** one sentence — current queue position, last result, next item.

---

## Pre-flight checklist (run once before launching)

```bash
ls results/experiment_log.jsonl queue.md overnight_driver.md .claude/settings.json
git status          # new files queue.md/overnight_driver.md/.claude/ expected
git worktree list   # note active worktrees (dry-run one from sp3_density_grid_fix may be present — leave it)
ls external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py  # must exist for worktrees to work
date -u +%FT%TZ > results/overnight_run.log  # mark driver start
```

## To launch

1. `/loop` (no interval)
2. Paste the prompt above (the "You are the overnight experiment driver..." block).
3. Walk away.

## To inspect in the morning

```bash
grep NEW_BEST results/overnight_run.log              # all best events
grep NEW_BEST_STAGE results/overnight_run.log        # per-stage winners
tail -60 results/overnight_run.log                   # full tail
tail -25 results/experiment_log.jsonl | jq -c '{hyp:.hypothesis,mode:.mode,avg:.avg_proxy_cost,ovl:.total_overlaps}'
git worktree list
grep -E '\[(done|killed|new_best|skipped)' queue.md
```

The final `NEW_BEST` line is the global winner. Merge its worktree:
```bash
# cd into the winner's worktree, review diff, then:
git checkout main && git merge <branch-name>
```

If a `stage:combine` item is the global best, it already reflects the stacked winners — merge it directly. If the combine failed but individual stages won, merge the single global-best branch and manually layer in the stage winners during the morning review.
