#!/bin/bash
# One-shot setup for Tier-2 ORFS on a fresh Intel cloud box.
#
# Prereq: user has spun up an Ubuntu 22.04 Intel box and provided IP.
# Local prereq: /tmp/tier2_handoff/*.pt placements present.
#
# Usage:
#   ./scripts/setup_intel_tier2.sh <new-ip> [user]
#
# What it does:
#   1. scp placements + supporting files to new box
#   2. install docker + pull openroad/orfs:latest
#   3. run gcd smoke test (verify CTS works on new CPU)
#   4. if smoke passes, sync MacroPlacement submodule + scripts/evaluate_with_orfs.py
#   5. launch ORFS chain (touching + cleaned in sequence) in background
#
# Outputs:
#   - ~/orfs_results/ (touching)
#   - ~/orfs_results_cleaned/ (cleaned)
#   - ~/run_tier2_chain.log
set -e

NEW_IP=${1:?usage: $0 <new-ip> [user]}
USR=${2:-ubuntu}

REPO_DIR=/Users/brendan/Developer/macro-place-challenge-2026

echo "=== Step 1: probe ssh + record CPU info ==="
ssh -o ConnectTimeout=10 ${USR}@${NEW_IP} 'cat /proc/cpuinfo | grep "model name" | head -1; cat /proc/cpuinfo | grep flags | head -1 | tr " " "\n" | grep -i avx | sort -u'

echo "=== Step 2: install docker ==="
ssh ${USR}@${NEW_IP} 'command -v docker >/dev/null && echo "docker present" || (
    sudo apt-get update -q
    sudo apt-get install -y docker.io git rsync python3 python3-pip
    sudo usermod -aG docker $USER
    sudo systemctl enable --now docker
)'
# Need a new shell for docker group to take effect.
ssh ${USR}@${NEW_IP} 'sg docker -c "docker version --format \"{{.Server.Version}}\""'

echo "=== Step 3: pull ORFS image (~3 GB, may take 5-10 min) ==="
ssh ${USR}@${NEW_IP} 'sg docker -c "docker pull openroad/orfs:latest"'

echo "=== Step 4: gcd smoke test (verify CTS works) ==="
ssh ${USR}@${NEW_IP} 'mkdir -p /tmp/gcd_test
sg docker -c "docker run --rm -i -u 1000:1000 \
    -e FLOW_HOME=/OpenROAD-flow-scripts/flow/ \
    -e WORK_HOME=/work \
    -v /tmp/gcd_test:/work:Z \
    openroad/orfs:latest \
    bash -c \"set -e; cd /OpenROAD-flow-scripts/flow; timeout 600 make DESIGN_CONFIG=./designs/nangate45/gcd/config.mk finish\"" 2>&1 | tail -20'

ssh ${USR}@${NEW_IP} 'ls /tmp/gcd_test/results/nangate45/gcd/base/6_*.def 2>/dev/null'
if [ $? -ne 0 ]; then
    echo
    echo "ERROR: gcd smoke FAILED on ${NEW_IP} — Intel box also has ORFS incompat."
    echo "Check the docker log above for error type. Abort."
    exit 1
fi

echo
echo "=== Step 4a: GCD SMOKE PASSED — CTS works on ${NEW_IP} ==="

echo "=== Step 5: scp placements + repo essentials ==="
ssh ${USR}@${NEW_IP} 'mkdir -p ~/ng45_placements ~/mpc-work/repo/scripts'
scp /tmp/tier2_handoff/*.pt ${USR}@${NEW_IP}:~/ng45_placements/
scp /tmp/tier2_handoff/*.json ${USR}@${NEW_IP}:~/ng45_placements/

# Sync the scripts/ dir + key macro_place + analysis files needed for evaluation.
rsync -avz --exclude='__pycache__' --exclude='.git' \
    "${REPO_DIR}/scripts" \
    "${REPO_DIR}/macro_place" \
    "${REPO_DIR}/analysis/macro_clearance_diagnostic" \
    "${REPO_DIR}/external" \
    "${REPO_DIR}/pyproject.toml" \
    "${REPO_DIR}/uv.lock" \
    ${USR}@${NEW_IP}:~/mpc-work/repo/

echo "=== Step 6: install uv + python deps ==="
ssh ${USR}@${NEW_IP} 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
cd ~/mpc-work/repo && uv sync --frozen 2>&1 | tail -5'

echo "=== Step 7: clone ORFS-flow-scripts (for evaluate_with_orfs.py) ==="
ssh ${USR}@${NEW_IP} 'test -d ~/mpc-work/OpenROAD-flow-scripts || git clone --depth=1 https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts ~/mpc-work/OpenROAD-flow-scripts'

echo
echo "=== Setup complete on ${NEW_IP} ==="
echo "  Repo:      ~/mpc-work/repo/"
echo "  ORFS:      ~/mpc-work/OpenROAD-flow-scripts/"
echo "  Placements: ~/ng45_placements/"
echo
echo "Next step: launch ORFS chain (touching + cleaned in series)."
echo "Run: ssh ${USR}@${NEW_IP} 'nohup bash ~/mpc-work/repo/scripts/run_tier2_orfs_chain.sh > ~/run_tier2_chain.log 2>&1 &'"
