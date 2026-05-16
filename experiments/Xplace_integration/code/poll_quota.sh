#!/usr/bin/env bash
# Poll AWS GPU quota; auto-launch g5.xlarge spot when approved.
# Runs as a background daemon — fires once and exits.
#
# Usage: nohup bash poll_quota.sh > poll_quota.log 2>&1 &

set +e
INTERVAL_S=${1:-1800}  # 30 min
LAUNCHER="$(dirname "$0")/launch_gpu_box.sh"
QUOTA_LOG=/tmp/aws_quota_poll.log

echo "[$(date -u +%H:%M:%S)] starting quota poll (interval=${INTERVAL_S}s)" | tee -a $QUOTA_LOG

while true; do
  Q=$(aws service-quotas get-service-quota \
    --service-code ec2 --quota-code L-DB2E81BA \
    --query 'Quota.Value' --output text 2>/dev/null)
  echo "[$(date -u +%H:%M:%S)] G/VT on-demand quota = $Q" | tee -a $QUOTA_LOG

  # Spot fallback
  QS=$(aws service-quotas get-service-quota \
    --service-code ec2 --quota-code L-3819A6DF \
    --query 'Quota.Value' --output text 2>/dev/null)
  echo "[$(date -u +%H:%M:%S)] G/VT spot quota = $QS" | tee -a $QUOTA_LOG

  if [ "$(echo "$Q >= 4" | bc 2>/dev/null)" = "1" ] || \
     [ "$(echo "$QS >= 4" | bc 2>/dev/null)" = "1" ]; then
    echo "[$(date -u +%H:%M:%S)] QUOTA APPROVED — launching g5.xlarge spot" | tee -a $QUOTA_LOG
    bash "$LAUNCHER" g5.xlarge 2>&1 | tee -a $QUOTA_LOG
    echo "[$(date -u +%H:%M:%S)] LAUNCH COMPLETE — bootstrapping..." | tee -a $QUOTA_LOG

    # Auto-bootstrap GPU box: copy setup + rsync code + run setup
    REPO_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
    SETUP="$REPO_DIR/experiments/Xplace_integration/code/setup_gpu_box.sh"

    # Wait for SSH alias to resolve
    sleep 20
    echo "[$(date -u +%H:%M:%S)] copying setup script + rsync repo to aws-gpu" | tee -a $QUOTA_LOG
    scp -i ~/.ssh/tier2-key.pem -o StrictHostKeyChecking=no "$SETUP" ubuntu@aws-gpu:~/setup_gpu_box.sh 2>&1 | tee -a $QUOTA_LOG
    rsync -az --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
      --exclude 'results/cloud_snapshot_*' --exclude '*.pt' \
      -e "ssh -i ~/.ssh/tier2-key.pem -o StrictHostKeyChecking=no" \
      "$REPO_DIR/" ubuntu@aws-gpu:~/macro-place-challenge-2026/ 2>&1 | tail -5 | tee -a $QUOTA_LOG

    echo "[$(date -u +%H:%M:%S)] running setup_gpu_box.sh on aws-gpu" | tee -a $QUOTA_LOG
    ssh -i ~/.ssh/tier2-key.pem -o StrictHostKeyChecking=no ubuntu@aws-gpu \
      "setsid nohup bash ~/setup_gpu_box.sh > ~/setup.log 2>&1 < /dev/null & disown" \
      2>&1 | tee -a $QUOTA_LOG

    echo "[$(date -u +%H:%M:%S)] FULL BOOTSTRAP DISPATCHED — poll exiting" | tee -a $QUOTA_LOG
    echo "  Setup runs in background on aws-gpu. Check: ssh aws-gpu 'tail -20 ~/setup.log'" | tee -a $QUOTA_LOG
    exit 0
  fi

  sleep $INTERVAL_S
done
