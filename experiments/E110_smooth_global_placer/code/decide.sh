#!/bin/bash
# Decision script for after ovl10 jobs=8 completes.

set -e
cd /Users/brendan/Developer/macro-place-challenge-2026

# Wait for ovl10 jobs=8 to complete.
RUNNING=$(ps aux | grep -E "E110_ovl10_jobs8" | grep -v grep | wc -l)
if [ "$RUNNING" -gt 0 ]; then
    echo "ovl10 jobs=8 still running. Last activity:"
    tail -3 /tmp/e110_ovl10_jobs8_all.log
    echo ""
    echo "Wait until done before running this script."
    exit 1
fi

# Find the result JSON
RESULT=$(ls -t results/CDLNSSACascadeStackedPeripheryE110Ovl10Placer_*.json 2>/dev/null | head -1)
if [ -z "$RESULT" ]; then
    echo "ERROR: No ovl10 result JSON found."
    exit 1
fi

# Parse avg
AVG=$(python3 -c "import json; d=json.load(open('$RESULT')); print(d['avg_proxy_cost'])")
OVL=$(python3 -c "import json; d=json.load(open('$RESULT')); print(d['total_overlaps'])")
QUAL=$(python3 -c "import json; d=json.load(open('$RESULT')); print(d['qualified'])")

echo "=========================================="
echo "OVL10 RESULT"
echo "=========================================="
echo "  avg: $AVG"
echo "  overlaps: $OVL"
echo "  qualified: $QUAL"
echo ""
echo "vs Option C 1.0575:"
python3 -c "
ibm = $AVG
oc = 1.0575
delta = ibm - oc
pct = delta / oc * 100
print(f'  Δ = {delta:+.5f}  ({pct:+.2f}%)')
"

# Decision
AVG_F=$(python3 -c "print(float($AVG))")
DECISION=$(python3 -c "
ibm = $AVG_F
if ibm < 1.050:
    print('STRONG_WIN')
elif ibm < 1.055:
    print('MODEST_WIN')
elif ibm < 1.060:
    print('TIE')
else:
    print('LOSS')
")

echo ""
echo "Decision: $DECISION"
echo ""

case "$DECISION" in
    STRONG_WIN|MODEST_WIN)
        echo "Action: launch NG45 ovl10 to verify generalization"
        echo "Command:"
        echo "  nohup uv run python scripts_parallel/evaluate_parallel.py \\"
        echo "    submissions/cd_lns_sa_cascade_stacked_periphery_e110_ovl10/placer.py \\"
        echo "    --ng45 --json --hypothesis E110_ovl10_ng45_v2 --jobs 4 > /tmp/e110_ovl10_ng45_v2.log 2>&1 &"
        echo ""
        echo "After NG45 OK: update launcher"
        echo "  uv run python experiments/E110_smooth_global_placer/code/update_launcher.py \\"
        echo "    submissions/cd_lns_sa_cascade_stacked_periphery_e110_ovl10 \\"
        echo "    CDLNSSACascadeStackedPeripheryE110Ovl10Placer"
        ;;
    TIE|LOSS)
        echo "Action: ship Option C as primary (already in launcher)"
        echo "Optional: Launch Triple --all to try multi-cfg approach"
        echo "  nohup uv run python scripts_parallel/evaluate_parallel.py \\"
        echo "    submissions/cd_lns_sa_cascade_stacked_periphery_e110_triple/placer.py \\"
        echo "    --all --json --hypothesis E110_triple --jobs 8 > /tmp/e110_triple_all.log 2>&1 &"
        ;;
esac
echo "=========================================="
