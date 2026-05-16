#!/usr/bin/env bash
# Launch a GPU spot instance once AWS quota is approved.
# Fires automatically from the polling loop; safe to invoke manually too.
#
# Usage: bash launch_gpu_box.sh [instance_type=g5.xlarge]
#
# Picks cheapest spot across AZs; tags as brendan-gpu-xplace.
set -e

ITYPE="${1:-g5.xlarge}"
AMI="${2:-}"
KEY=tier2-key
SG=sg-0ee1d2f825ff9cfb1
# Use most recent Deep Learning AMI (Ubuntu 22, PyTorch + CUDA preinstalled)
if [ -z "$AMI" ]; then
  AMI=$(aws ec2 describe-images \
    --owners 898082745236 \
    --filters \
      "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch * Ubuntu 22.04*" \
      "Name=state,Values=available" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' \
    --output text 2>/dev/null)
fi
if [ -z "$AMI" ] || [ "$AMI" = "None" ]; then
  echo "Falling back to plain Ubuntu 22.04 (will need to install CUDA)"
  AMI=$(aws ec2 describe-images --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
              "Name=state,Values=available" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
fi
echo "AMI: $AMI"

# Find cheapest spot AZ
read -r SUBNET MAXPRICE < <(
  aws ec2 describe-spot-price-history \
    --instance-types $ITYPE \
    --product-descriptions "Linux/UNIX" \
    --start-time $(date -u -v-1H +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -u -d '-1 hour' +%Y-%m-%dT%H:%M:%S) \
    --query 'SpotPriceHistory[0:5].[AvailabilityZone,SpotPrice]' --output text \
  | sort -k2 -n | head -1 | awk '{print $1, $2}')

# Map AZ → subnet
SUBNET_ID=$(aws ec2 describe-subnets --query "Subnets[?AvailabilityZone=='$SUBNET'].SubnetId" --output text | head -1)
if [ -z "$SUBNET_ID" ]; then
  echo "ERROR: no subnet in $SUBNET"; exit 1
fi
echo "Spot in $SUBNET (subnet $SUBNET_ID, current spot=$MAXPRICE)"

# Pad max price 1.5x current
MAXBID=$(awk -v p="$MAXPRICE" 'BEGIN{printf "%.4f", p * 1.5}')

INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI \
  --instance-type $ITYPE \
  --key-name $KEY \
  --security-group-ids $SG \
  --subnet-id $SUBNET_ID \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3,DeleteOnTermination=true}" \
  --instance-market-options "MarketType=spot,SpotOptions={MaxPrice=$MAXBID,SpotInstanceType=one-time}" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=brendan-gpu-xplace},{Key=Owner,Value=brendan-claude}]" \
  --query 'Instances[].InstanceId' --output text)

echo "Launched $INSTANCE_ID; waiting for running state..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID
IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[].Instances[].PublicIpAddress' --output text)
echo "Running at $IP"

# Add ssh alias
grep -q "Host aws-gpu" ~/.ssh/config || cat >> ~/.ssh/config <<EOF

Host aws-gpu
    HostName $IP
    User ubuntu
    IdentityFile ~/.ssh/tier2-key.pem
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
    ServerAliveInterval 60
EOF
echo "SSH alias: aws-gpu"

# Wait for SSH
echo "Waiting for SSH..."
until ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/tier2-key.pem ubuntu@$IP "echo up" 2>/dev/null | grep -q up; do sleep 5; done
echo "SSH ready"
echo ""
echo "Next steps:"
echo "  1. ssh aws-gpu"
echo "  2. bash ~/setup_gpu_box.sh  (after copying setup_gpu_box.sh)"
echo "  3. Rsync code: rsync -az ./ aws-gpu:~/macro-place-challenge-2026/"
