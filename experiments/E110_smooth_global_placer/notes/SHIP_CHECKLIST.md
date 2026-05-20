# Submission ship checklist — execute when winner cfg is decided

## Pre-flight (do this in order)

### 1. Confirm winner placer + class name

```bash
ls submissions/cd_lns_sa_cascade_stacked_periphery_*  # see options
# For ovl10:
#   path: submissions/cd_lns_sa_cascade_stacked_periphery_e110_ovl10/placer.py
#   class: CDLNSSACascadeStackedPeripheryE110Ovl10Placer
```

### 2. Update placer.py launcher

```bash
uv run python experiments/E110_smooth_global_placer/code/update_launcher.py \
  submissions/cd_lns_sa_cascade_stacked_periphery_e110_ovl10 \
  CDLNSSACascadeStackedPeripheryE110Ovl10Placer

# Verify:
uv run python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('p', 'placer.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('OK:', m.Placer.__bases__[0].__name__)
"
```

### 3. Smoke test on ibm01

```bash
uv run evaluate placer.py -b ibm01 --json
# Verify: produces valid placement, zero overlaps, proxy in expected range
```

### 4. EPYC cross-validation (tomorrow May 20)

```bash
./experiments/E110_smooth_global_placer/code/epyc_validate.sh placer.py
# Wait ~4 hr, verify EPYC avg projected ~+2.4% over M3
```

### 5. Update CLAUDE.md champion entry

Edit `CLAUDE.md` `## Current submission floor` section with new winner.

### 6. Update docs/results.md

Add per-bench numbers for the winner.

### 7. Commit + push

```bash
git add submissions/ experiments/E110_smooth_global_placer/ placer.py CLAUDE.md docs/
git commit -m "$(cat <<'EOF'
E110 gradient lane: ship <winner cfg> at <IBM avg> IBM / <NG45 avg> NG45

Lane-4 architecture with E110 SmoothGlobalPlacer as parallel candidate
alongside cascade. <cfg-specific notes>.

EPYC cross-validated at <epyc avg>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

git push origin main
```

### 8. Submit via form

URL: <https://forms.gle/YDRtYV5Vq68SZgKW9>
- Team name: thinkorplace
- Repo: <github URL>
- Open-source license: Apache 2.0

### 9. Test eval_docker locally (if Docker available)

```bash
./eval_docker/run_eval.sh thinkorplace placer.py
# Check eval_docker/results/thinkorplace.log
```

## Fallback if winner cfg has any issue

```bash
# Restore Option C launcher
cp placer.py.bak placer.py
# OR
uv run python experiments/E110_smooth_global_placer/code/update_launcher.py \
  submissions/cd_lns_sa_cascade_stacked_periphery \
  CDLNSSACascadeStackedPeripheryPlacer
```

## Key reminders

- Zero overlaps requirement is HARD. Any overlap = disqualified.
- 60-min/bench cap. Our budget_seconds=3300s leaves headroom.
- Pre-existing `submit_deps/` has DREAMPlace CPU-only install (irrelevant
  for Option C/ovl10 which don't use DREAMPlace).
- The launcher placer.py adds repo root to sys.path so cross-dir
  imports work in the eval_docker mount.
