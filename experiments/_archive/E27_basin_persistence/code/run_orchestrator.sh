#!/usr/bin/env bash
# E27 — orchestrator. Sequence run_trajectory.py over (bench, init, seed).
#
# Sequential by design — the parent session may parallelize across
# benchmarks via background processes (`bash run_orchestrator.sh ibm11 &`,
# `bash run_orchestrator.sh ibm13 &`, ...). The K=11 trajectories per
# benchmark stay serial inside one bench so M3 Max isn't oversubscribed.
#
# Trajectories: 4 benches × 11 trajectories = 44 runs.
#   sdf:        seeds {42, 1, 2}            (3)
#   sdf_jitter: seeds {42, 1, 2}            (3)
#   uniform:    seeds {42, 1, 2}            (3)
#   greedy:     deterministic               (1)
#   dpo:        seed 42                     (1)
#
# Total per bench = 11 trajectories × 600s budget = 6600s = ~1.83 hr.
# Total across 4 benches sequential = ~7.3 hr. Parallel across benches
# = ~1.83 hr wall.
#
# Usage:
#   bash run_orchestrator.sh                # all 4 benchmarks
#   bash run_orchestrator.sh ibm11          # just one bench
#   BUDGET_S=300 bash run_orchestrator.sh   # short-budget smoke
#
# Output: trajectories/<bench>_<init>_<seed>.json

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAJ_DIR="$THIS_DIR/trajectories"
mkdir -p "$TRAJ_DIR"

BUDGET_S="${BUDGET_S:-600}"

# Default benchmarks: ibm11/13/14/15 (where E12 and E25 tie).
if [ "$#" -ge 1 ]; then
    BENCHES=("$@")
else
    BENCHES=("ibm11" "ibm13" "ibm14" "ibm15")
fi

run_one() {
    local bench="$1"
    local init="$2"
    local seed="$3"
    local out="$TRAJ_DIR/${bench}_${init}_${seed}.json"

    if [ -f "$out" ]; then
        echo "[E27] SKIP exists: $out"
        return 0
    fi

    local log="$TRAJ_DIR/${bench}_${init}_${seed}.log"
    echo "[E27] RUN bench=$bench init=$init seed=$seed budget=${BUDGET_S}s"
    cd "$REPO_ROOT"
    uv run python "$THIS_DIR/run_trajectory.py" \
        --benchmark "$bench" --init "$init" --seed "$seed" \
        --budget-s "$BUDGET_S" --out "$out" \
        2>&1 | tee "$log"
}

for bench in "${BENCHES[@]}"; do
    echo "===== $bench ====="
    for seed in 42 1 2; do
        run_one "$bench" "sdf" "$seed"
        run_one "$bench" "sdf_jitter" "$seed"
        run_one "$bench" "uniform" "$seed"
    done
    run_one "$bench" "greedy" 0
    run_one "$bench" "dpo" 42
done

echo "[E27] orchestrator done. Trajectories in $TRAJ_DIR"
