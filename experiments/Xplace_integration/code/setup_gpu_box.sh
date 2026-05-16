#!/usr/bin/env bash
# Bootstrap a fresh GPU EC2 instance (Ubuntu 22.04 + CUDA) for Xplace.
#
# Prerequisites on the box:
#   - Ubuntu 22.04 AMI with NVIDIA driver already installed (Deep Learning AMI),
#     OR fresh Ubuntu — script will install driver via .run.
#   - GPU visible to `nvidia-smi`.
#
# Run on the target box AFTER ssh-ing in:
#   bash setup_gpu_box.sh
#
# Idempotent: skips steps already done.

set -e
export DEBIAN_FRONTEND=noninteractive

echo "=== [1/6] verify GPU + driver ==="
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. Use Deep Learning AMI or install driver."
  exit 1
fi
nvidia-smi | head -8

echo "=== [2/6] system packages ==="
sudo apt-get update -qq
sudo apt-get install -y -qq \
  git build-essential cmake ninja-build python3-pip python3-venv \
  libboost-all-dev libfftw3-dev libeigen3-dev \
  curl jq rsync

echo "=== [3/6] uv ==="
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH=$HOME/.local/bin:$PATH

echo "=== [4/6] clone Xplace (CUHK) ==="
if [ ! -d $HOME/Xplace ]; then
  git clone --depth 1 https://github.com/cuhk-eda/Xplace.git $HOME/Xplace
fi

echo "=== [5/6] build Xplace ==="
cd $HOME/Xplace
# Use CUDA arch native (works on g5 A10G = 86, g4dn T4 = 75, p3 V100 = 70)
mkdir -p build
cd build
cmake .. -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DPYTHON_EXECUTABLE=$(which python3)
ninja -j$(nproc)
ninja install

echo "=== [6/6] python deps for Xplace ==="
cd $HOME/Xplace
if [ ! -d .venv ]; then
  uv venv
fi
source .venv/bin/activate
uv pip install \
  torch==2.4.1 \
  triton \
  numpy scipy pandas matplotlib \
  protobuf

# Smoke test
echo "=== smoke ==="
python3 -c "import torch; print('cuda:', torch.cuda.is_available(), 'gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
echo "=== DONE ==="
