#!/bin/bash
# E74 full-wave: produce missing E25/E41 placements + run hessian saddle on each bench.
# Sequential per bench to avoid thrashing; run multiple benches in parallel via separate
# invocations.

set -e
BENCH=$1
ROOT=/Users/brendan/Developer/macro-place-challenge-2026
PLACEMENTS_DIR=$ROOT/experiments/E69_sequence_pair_search/results/placements
LOGS_DIR=$ROOT/experiments/E74_hessian_saddle/results/logs

# Step 1: produce E25 if missing.
if [ ! -f "$PLACEMENTS_DIR/e25_$BENCH.pt" ]; then
  echo "[wave/$BENCH] producing E25..."
  uv run python $ROOT/experiments/E69_sequence_pair_search/code/produce_basin_placement.py e25 $BENCH $PLACEMENTS_DIR/e25_$BENCH.pt > $LOGS_DIR/e25_$BENCH.log 2>&1
fi

# Step 2: produce E41 if missing.
if [ ! -f "$PLACEMENTS_DIR/e41_$BENCH.pt" ]; then
  echo "[wave/$BENCH] producing E41..."
  uv run python $ROOT/experiments/E69_sequence_pair_search/code/produce_basin_placement.py e41 $BENCH $PLACEMENTS_DIR/e41_$BENCH.pt > $LOGS_DIR/e41_$BENCH.log 2>&1
fi

# Step 3: run E74.
echo "[wave/$BENCH] running E74 hessian saddle..."
uv run python $ROOT/experiments/E74_hessian_saddle/code/run_hessian_saddle.py $BENCH --n-eigvecs 2 --epsilons 0.3,1.0,3.0 --polish-budget 240 > $LOGS_DIR/hessian_$BENCH.log 2>&1

echo "[wave/$BENCH] DONE"
