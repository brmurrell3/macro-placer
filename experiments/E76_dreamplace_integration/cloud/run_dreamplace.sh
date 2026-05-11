#!/bin/bash
# Cloud-side runner: assumes DREAMPlace is installed (e.g., via Docker image
# limbo018/dreamplace-cuda or built from github.com/limbo018/DREAMPlace), and
# this directory contains <bench>/<bench>.{aux,nodes,nets,pl,scl,wts} for each
# IBM benchmark.
#
# Usage on cloud box:
#   ./run_dreamplace.sh <bench_dir> <dreamplace_root>
# e.g.
#   ./run_dreamplace.sh ibm01 /opt/DREAMPlace
#
# Output: <bench_dir>/<bench>.gp.pl and possibly <bench>.lg.pl
#
# After all 17 benches done, rsync the .gp.pl files back to local machine:
#   rsync -av --include='*.gp.pl' cloud:dreamplace_runs/ ./

set -e

BENCH_DIR=$1
DP_ROOT=${2:-/opt/DREAMPlace}

if [ -z "$BENCH_DIR" ]; then
  echo "usage: $0 <bench_dir> [dreamplace_root]"
  exit 1
fi

BENCH=$(basename "$BENCH_DIR")
AUX_PATH="$BENCH_DIR/$BENCH.aux"

if [ ! -f "$AUX_PATH" ]; then
  echo "ERROR: missing $AUX_PATH"
  exit 2
fi

# Build a minimal DREAMPlace config. Defaults from Place.json with the
# inputs pointed at our Bookshelf files.
CONFIG_JSON=$(cat <<EOF
{
  "aux_input": "$AUX_PATH",
  "target_density": 0.85,
  "density_weight": 8e-5,
  "gpu": 1,
  "global_place_stages": [
    {"num_bins_x": 1024, "num_bins_y": 1024,
     "iteration": 1000, "learning_rate": 0.01,
     "wirelength": "weighted_average", "optimizer": "nesterov"}
  ],
  "legalize_flag": 1,
  "detailed_place_flag": 0,
  "stop_overflow": 0.07,
  "result_dir": "$BENCH_DIR"
}
EOF
)

CONFIG_PATH="$BENCH_DIR/dreamplace_config.json"
echo "$CONFIG_JSON" > "$CONFIG_PATH"

echo "[run_dreamplace] running on $BENCH (config=$CONFIG_PATH)..."
cd "$DP_ROOT"
python dreamplace/Placer.py "$CONFIG_PATH"

echo "[run_dreamplace] DONE $BENCH; .gp.pl should be at $BENCH_DIR/$BENCH.gp.pl"
ls -la "$BENCH_DIR/" | grep -E "\.pl$" || true
