#!/bin/bash
# Wait for E68 --ng45 chain to finish, then launch E70 --fast.
#
# Inputs:
#   NG45_CHAIN_PID  PID of the chain_overnight.sh process running --ng45.
# Output:
#   experiments/E70_kjoint_hungarian_integrated/results/chain.log
#   experiments/E70_kjoint_hungarian_integrated/results/fast_run.log

set -u
LOG=experiments/E70_kjoint_hungarian_integrated/results/chain.log
PLACER=experiments/E70_kjoint_hungarian_integrated/code/cd_lns_sa_hybrid_hungarian.py
FAST_LOG=experiments/E70_kjoint_hungarian_integrated/results/fast_run.log
KILL_GATE=0.9248

NG45_CHAIN_PID="${NG45_CHAIN_PID:-${1:-}}"
if [ -z "$NG45_CHAIN_PID" ]; then
    echo "ERROR: NG45_CHAIN_PID not set" | tee -a "$LOG"
    exit 1
fi

mkdir -p experiments/E70_kjoint_hungarian_integrated/results
echo "[$(date '+%Y-%m-%d %H:%M:%S')] E70 chain start; waiting for E68 chain PID=$NG45_CHAIN_PID" | tee -a "$LOG"

# 1. Wait for E68 chain (which runs --ng45 inside) to exit.
while kill -0 "$NG45_CHAIN_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] E68 chain exited" | tee -a "$LOG"
sleep 10  # let JSON flush

# 2. Launch E70 --fast.
> "$FAST_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting E70 --fast → $FAST_LOG" | tee -a "$LOG"
uv run evaluate "$PLACER" --fast --json --hypothesis e70_hungarian_fast \
    > "$FAST_LOG" 2>&1
RC=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] E70 --fast exited rc=$RC" | tee -a "$LOG"

# 3. Read result + gate check.
LAST_JSON=$(grep -l 'e70_hungarian_fast' results/CDLNSSAHybridHungarianPlacer_*.json 2>/dev/null | tail -1)
if [ -z "$LAST_JSON" ]; then
    LAST_JSON=$(ls -t results/CDLNSSAHybridHungarianPlacer_*.json 2>/dev/null | head -1)
fi
if [ -n "$LAST_JSON" ] && [ -f "$LAST_JSON" ]; then
    AVG=$(uv run python -c "
import json
d = json.load(open('$LAST_JSON'))
print(d.get('avg_proxy_cost', float('nan')))
")
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] E70 --fast avg_proxy_cost = $AVG (gate < $KILL_GATE)" | tee -a "$LOG"
    PASS=$(uv run python -c "
avg = $AVG
gate = $KILL_GATE
print('pass' if avg < gate else 'fail')
")
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] gate: $PASS" | tee -a "$LOG"
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] E70 chain complete" | tee -a "$LOG"
