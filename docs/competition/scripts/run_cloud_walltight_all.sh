#!/usr/bin/env bash
# Run the partcl submission entry across all 17 IBM in parallel on cloud.
#
# Run from cloud:
#   ssh mpc-cloud "bash ~/macro-place-challenge-2026/run_cloud_walltight_all.sh"
set -euo pipefail

export PATH=$HOME/.local/bin:$PATH

# OpenBLAS defaults to 1 thread on this box; without these, ibm01 takes
# ~175 min on cloud vs ~55 min on M3. See memory/cloud_openblas_gotcha.md.
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

cd ~/macro-place-challenge-2026

PLACER="${1:-submissions/cd_lns_sa_cascade/placer_adaptive.py}"
JOBS="${2:-4}"
LABEL="${3:-walltight_$(date -u +%H%M)}"

echo "=== running --all on $PLACER, --jobs $JOBS, label $LABEL ==="
echo "  budget_seconds default = 3000s (50 min/bench, fits 60-min cap)"
echo ""

uv run evaluate "$PLACER" --all --jobs "$JOBS" --json --hypothesis "$LABEL" 2>&1 \
  | tee ~/walltight_${LABEL}.log
