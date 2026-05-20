#!/bin/bash
# Launch the next phase of experiments based on current results.
# Run at noon (or whenever ovl10/nocong --all + minimal-600s + NG45 all complete).

set -e
cd /Users/brendan/Developer/macro-place-challenge-2026

# Verify all morning jobs done.
RUNNING=$(ps aux | grep -E "evaluate_parallel|sweep_harness" | grep -v grep | wc -l)
if [ "$RUNNING" -gt 0 ]; then
    echo "WARNING: $RUNNING jobs still running. Wait before queuing next phase."
    ps aux | grep -E "evaluate_parallel|sweep_harness" | grep -v grep | awk '{print "  ", $2, $9, $13, $14}'
    exit 1
fi

echo "All morning jobs done. Analyzing results..."
uv run python experiments/E110_smooth_global_placer/code/analyze_results.py

echo ""
echo "Queuing afternoon experiments..."

# Always test: NG45 on ovl10 (whatever the IBM result, we need NG45 data)
echo "Starting NG45 ovl10 (generalization test)..."
nohup uv run python scripts_parallel/evaluate_parallel.py \
  submissions/cd_lns_sa_cascade_stacked_periphery_e110_ovl10/placer.py \
  --ng45 --json --hypothesis E110_ovl10_ng45 --jobs 2 \
  > /tmp/e110_ovl10_ng45.log 2>&1 &
echo "  NG45 ovl10 PID: $!"

# NG45 on nocong (alternate generalization test)
sleep 5
echo "Starting NG45 nocong..."
nohup uv run python scripts_parallel/evaluate_parallel.py \
  submissions/cd_lns_sa_cascade_stacked_periphery_e110_nocong/placer.py \
  --ng45 --json --hypothesis E110_nocong_ng45 --jobs 2 \
  > /tmp/e110_nocong_ng45.log 2>&1 &
echo "  NG45 nocong PID: $!"

# Triple variant --all (addresses per-bench tuning by multi-cfg plateau)
sleep 5
echo "Starting Triple variant --all (multi-cfg plateau-pick)..."
nohup uv run python scripts_parallel/evaluate_parallel.py \
  submissions/cd_lns_sa_cascade_stacked_periphery_e110_triple/placer.py \
  --all --json --hypothesis E110_triple --jobs 4 \
  > /tmp/e110_triple_all.log 2>&1 &
echo "  Triple --all PID: $!"

echo ""
echo "Afternoon queue launched. ETAs:"
echo "  NG45 ovl10:  ~1.5-2 hr"
echo "  NG45 nocong: ~1.5-2 hr"
echo "  Triple --all: ~4 hr"
echo ""
echo "Check progress: tail /tmp/e110_*_ng45.log /tmp/e110_triple_all.log"
