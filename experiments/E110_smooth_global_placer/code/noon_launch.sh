#!/bin/bash
# Noon launch script — fires next phase when morning jobs are done.
# Runs analyze_results.py first to summarize, then launches Triple + NG45 jobs.

set -e
cd /Users/brendan/Developer/macro-place-challenge-2026

echo "=========================================="
echo "NOON LAUNCH — $(date)"
echo "=========================================="

# Check morning jobs done.
RUNNING=$(ps aux | grep -E "evaluate_parallel.py" | grep -v grep | grep -E "E110_lane4|E110_ovl10|E110_nocong|E110_minimal_600s|E110_lane4_default_ng45" | wc -l)
if [ "$RUNNING" -gt 0 ]; then
    echo "WARNING: $RUNNING morning jobs still running:"
    ps aux | grep -E "evaluate_parallel.py" | grep -v grep | grep -E "E110_lane4|E110_ovl10|E110_nocong|E110_minimal_600s|E110_lane4_default_ng45" | awk '{print "  ", $2, $13, $14}'
    echo ""
    echo "ABORT — wait for morning jobs to finish."
    exit 1
fi

# Step 1: summarize.
echo ""
echo "STEP 1: Analyze morning results"
echo "------------------------------"
uv run python experiments/E110_smooth_global_placer/code/analyze_results.py

# Step 2: Launch afternoon experiments
echo ""
echo "STEP 2: Launch afternoon"
echo "------------------------"

# Triple variant --all (multi-cfg plateau, most promising untested)
nohup uv run python scripts_parallel/evaluate_parallel.py \
  submissions/cd_lns_sa_cascade_stacked_periphery_e110_triple/placer.py \
  --all --json --hypothesis E110_triple --jobs 4 \
  > /tmp/e110_triple_all.log 2>&1 &
TRIPLE_PID=$!
echo "Triple --all PID: $TRIPLE_PID  ETA ~4 hr"

sleep 5

# NG45 on ovl10 (test generalization) — but we already kicked this off earlier
# Check if NG45 ovl10 already running
NG45_OVL10_RUNNING=$(ps aux | grep -E "E110_ovl10_ng45" | grep -v grep | wc -l)
if [ "$NG45_OVL10_RUNNING" -gt 0 ]; then
    echo "NG45 ovl10 already running (started earlier)"
else
    nohup uv run python scripts_parallel/evaluate_parallel.py \
      submissions/cd_lns_sa_cascade_stacked_periphery_e110_ovl10/placer.py \
      --ng45 --json --hypothesis E110_ovl10_ng45 --jobs 2 \
      > /tmp/e110_ovl10_ng45.log 2>&1 &
    echo "NG45 ovl10 PID: $!  ETA ~1.5 hr"
fi

echo ""
echo "=========================================="
echo "Launched. Check progress:"
echo "  tail /tmp/e110_triple_all.log"
echo "  tail /tmp/e110_ovl10_ng45.log"
echo ""
echo "ETAs (assuming low CPU contention):"
echo "  ~13:30: NG45 ovl10 done"
echo "  ~16:30: Triple --all done"
echo ""
echo "Re-run analyze_results.py at 16:30 for final picture."
echo "=========================================="
