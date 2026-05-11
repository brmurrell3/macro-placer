#!/usr/bin/env bash
# Run wall-safe E74-DP placer across all 17 IBM in parallel on cloud.
# Targets: prove wall-safe variant fits 1-hr cap and beats E48 on canonical.
#
# Run from cloud: ssh mpc-cloud "bash ~/macro-place-challenge-2026/run_cloud_walltight_all.sh"
set -euo pipefail

export PATH=$HOME/.local/bin:$PATH
export DREAMPLACE_ROOT=/opt/DREAMPlace/install
# OpenBLAS defaults to 1 thread on this box → numpy was 9× slower than M3
# Without these, ibm01 wall-safe takes ~175 min on cloud vs ~55 min on M3.
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

cd ~/macro-place-challenge-2026

VARIANT="${1:-dp}"  # "dp" | "cascade" | "hessian"
JOBS="${2:-4}"
LABEL="${3:-walltight_$(date -u +%H%M)}"

case "$VARIANT" in
  dp)      PLACER=submissions/cd_lns_sa_hessian_dp/placer.py ;;
  cascade) PLACER=submissions/cd_lns_sa_cascade/placer.py ;;
  hessian) PLACER=submissions/cd_lns_sa_hessian/placer.py ;;
  *)       echo "unknown variant: $VARIANT"; exit 1 ;;
esac

echo "=== running --all on $PLACER, --jobs $JOBS, label $LABEL ==="
echo "  budget_seconds default = 3300s (55 min/bench)"
echo "  DREAMPLACE_ROOT=$DREAMPLACE_ROOT"
echo ""

uv run evaluate "$PLACER" --all --jobs "$JOBS" --json --hypothesis "$LABEL" 2>&1 | tee ~/walltight_${LABEL}.log
