#!/bin/bash
# EPYC validation script for the chosen submission placer.
# Run tomorrow (May 20) to verify the M3 → EPYC variance.
#
# Pre-reqs:
#   1. AWS credentials configured
#   2. SSH key authorized for the EPYC instance
#   3. Spot instance request approved
#
# Usage:
#   ./experiments/E110_smooth_global_placer/code/epyc_validate.sh \
#     submissions/cd_lns_sa_cascade_stacked_periphery_e110_ovl10/placer.py

PLACER_PATH="${1:?Usage: $0 <placer_path>}"

# AWS spot instance (c6a.4xlarge, ~$0.29/hr)
INSTANCE_TYPE="c6a.4xlarge"
KEY_NAME="${AWS_KEY_NAME:-mpc-validate}"
REGION="${AWS_REGION:-us-west-2}"

echo "=========================================="
echo "EPYC validation: $PLACER_PATH"
echo "=========================================="
echo ""

# Step 1: launch spot instance (using existing AWS setup)
# This assumes there's already a tagged spot or persistent instance.
# Check first.

INSTANCE_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=mpc-epyc" "Name=instance-state-name,Values=running" \
  --region $REGION \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text 2>/dev/null)

if [ -z "$INSTANCE_IP" ] || [ "$INSTANCE_IP" = "None" ]; then
    echo "No running mpc-epyc instance found."
    echo "Either:"
    echo "  1. Launch one: aws ec2 run-instances --instance-type $INSTANCE_TYPE ..."
    echo "  2. Use existing instance with: ssh ubuntu@<ip>"
    exit 1
fi

echo "Found EPYC instance at $INSTANCE_IP"
echo ""

# Step 2: sync code (rsync)
echo "Syncing code to EPYC..."
rsync -avz --exclude='.venv' --exclude='results/cloud_*' --exclude='external/MacroPlacement/.git' \
  -e "ssh -o StrictHostKeyChecking=no" \
  /Users/brendan/Developer/macro-place-challenge-2026/ \
  ubuntu@$INSTANCE_IP:/home/ubuntu/macro-place-challenge-2026/

# Step 3: run --all
echo ""
echo "Launching --all on EPYC (~4 hr ETA)..."
ssh ubuntu@$INSTANCE_IP "cd /home/ubuntu/macro-place-challenge-2026 && \
  OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  nohup uv run python scripts_parallel/evaluate_parallel.py $PLACER_PATH \
    --all --json --hypothesis epyc_validation --jobs 4 \
    > /tmp/epyc_validation.log 2>&1 < /dev/null &"

# Step 4: wait + retrieve
echo ""
echo "Running. To monitor:"
echo "  ssh ubuntu@$INSTANCE_IP 'tail -f /tmp/epyc_validation.log'"
echo ""
echo "When done (4 hr), retrieve results:"
echo "  rsync -avz ubuntu@$INSTANCE_IP:/home/ubuntu/macro-place-challenge-2026/results/*epyc_validation* ./results/"
