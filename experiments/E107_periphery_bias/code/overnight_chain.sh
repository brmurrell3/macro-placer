#!/bin/bash
# Run E107 multi-seed tests sequentially overnight.
# Each test: 8 cases × ~120s = ~16 min.
# Total budget: ~3-4 hours of CPU time.

cd /Users/brendan/Developer/macro-place-challenge-2026

LOG_DIR="experiments/E107_periphery_bias/results"
mkdir -p "$LOG_DIR"

run_test() {
    local bench=$1
    local alpha=$2
    local logfile="$LOG_DIR/multiseed_${bench}_a${alpha//./}.log"
    echo "[$(date +%H:%M:%S)] START $bench α=$alpha → $logfile"
    PYTHONUNBUFFERED=1 uv run python experiments/E107_periphery_bias/code/multiseed_periphery.py "$bench" "$alpha" > "$logfile" 2>&1
    echo "[$(date +%H:%M:%S)] DONE  $bench α=$alpha"
    tail -8 "$logfile"
    echo "---"
}

# Phase 1: alpha sweep on ariane133 (proxy 0.66993 baseline)
run_test ariane133 0.005
run_test ariane133 0.01
run_test ariane133 0.02

# Phase 2: alpha=0.01 on cached IBM benches
run_test ibm09 0.01
run_test ibm10 0.01
run_test ibm14 0.01
run_test ibm17 0.01

# Phase 3: alpha=0.005 (gentler) on IBM benches
run_test ibm10 0.005
run_test ibm14 0.005

# Phase 4: alpha=0.01 on remaining NG45 (cached via symlink)
run_test ariane136 0.01
run_test mempool_tile 0.01
run_test nvdla 0.01

echo "[$(date +%H:%M:%S)] OVERNIGHT CHAIN COMPLETE"
