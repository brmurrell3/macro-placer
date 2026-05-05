#!/bin/bash
# Overnight chain for E68: wait for --fast → check kill gate → launch --ng45.
#
# Inputs (env or positional):
#   FAST_PID  PID of the running E68 --fast process.
# Output:
#   experiments/E68_workbounded_refactor/results/chain_overnight.log

set -u
LOG=experiments/E68_workbounded_refactor/results/chain_overnight.log
KILL_GATE=0.9248   # E48 0.92024 + 0.5%
PLACER=experiments/E68_workbounded_refactor/code/cd_lns_sa_workbounded.py

FAST_PID="${FAST_PID:-${1:-}}"
if [ -z "$FAST_PID" ]; then
    echo "ERROR: FAST_PID not set (env var or arg 1)" | tee -a "$LOG"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] chain start; waiting for FAST_PID=$FAST_PID" | tee -a "$LOG"

# 1. Wait for --fast to exit.
while kill -0 "$FAST_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAST_PID=$FAST_PID exited" | tee -a "$LOG"

# 2. Find latest JSON for this hypothesis. The harness writes
#    results/CDLNSSAHybridWBPlacer_<timestamp>.json AND appends to
#    experiment_log.jsonl. Use the log entry tagged e68_workbounded_fast.
sleep 5  # let the harness flush JSON

LAST_FAST_JSON=$(grep -l 'e68_workbounded_fast' results/CDLNSSAHybridWBPlacer_*.json 2>/dev/null | head -1)
if [ -z "$LAST_FAST_JSON" ]; then
    # Fallback: latest JSON file by mtime.
    LAST_FAST_JSON=$(ls -t results/CDLNSSAHybridWBPlacer_*.json 2>/dev/null | head -1)
fi
if [ -z "$LAST_FAST_JSON" ] || [ ! -f "$LAST_FAST_JSON" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: no E68 --fast JSON result found" | tee -a "$LOG"
    exit 2
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] reading $LAST_FAST_JSON" | tee -a "$LOG"

# 3. Extract avg_proxy_cost (Python is more reliable than jq for floats).
AVG=$(uv run python -c "
import json, sys
d = json.load(open('$LAST_FAST_JSON'))
print(d.get('avg_proxy_cost', float('nan')))
")
echo "[$(date '+%Y-%m-%d %H:%M:%S')] --fast avg_proxy_cost = $AVG" | tee -a "$LOG"

# 4. Kill-gate check.
PASS=$(uv run python -c "
avg = $AVG
gate = $KILL_GATE
print('pass' if avg < gate else 'fail')
")
if [ "$PASS" = "pass" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --fast PASSED gate ($AVG < $KILL_GATE); launching --ng45" | tee -a "$LOG"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --fast FAILED gate ($AVG >= $KILL_GATE); halting chain" | tee -a "$LOG"
    exit 0
fi

# 5. Launch --ng45.
NG45_LOG=experiments/E68_workbounded_refactor/results/ng45_run.log
> "$NG45_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting --ng45 → $NG45_LOG" | tee -a "$LOG"
uv run evaluate "$PLACER" --ng45 --json --hypothesis e68_workbounded_ng45 \
    > "$NG45_LOG" 2>&1
NG45_RC=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] --ng45 exited rc=$NG45_RC" | tee -a "$LOG"

# 6. Summarize.
LAST_NG45_JSON=$(grep -l 'e68_workbounded_ng45' results/CDLNSSAHybridWBPlacer_*.json 2>/dev/null | tail -1)
if [ -z "$LAST_NG45_JSON" ]; then
    LAST_NG45_JSON=$(ls -t results/CDLNSSAHybridWBPlacer_*.json 2>/dev/null | head -1)
fi
if [ -n "$LAST_NG45_JSON" ] && [ -f "$LAST_NG45_JSON" ]; then
    NG45_AVG=$(uv run python -c "
import json
d = json.load(open('$LAST_NG45_JSON'))
print(d.get('avg_proxy_cost', float('nan')))
")
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --ng45 avg_proxy_cost = $NG45_AVG (E48 baseline 0.6922)" | tee -a "$LOG"
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] chain complete" | tee -a "$LOG"
