#!/bin/bash
# Run ORFS evaluation chain on 4 NG45 placements (touching + cleaned).
# Assumes: ~/ng45_placements/*.pt present, ~/mpc-work/repo cloned,
# ~/mpc-work/OpenROAD-flow-scripts/ cloned, openroad/orfs:latest pulled.
#
# Output: ~/orfs_results/ (touching) + ~/orfs_results_cleaned/ (cleaned)
set +e

log() { echo "[$(date +%Y-%m-%dT%H:%M:%SZ)] $*"; }

export PATH=$HOME/.local/bin:$PATH
ORFS_ROOT=$HOME/mpc-work/OpenROAD-flow-scripts
cd $HOME/mpc-work/repo

mkdir -p ~/orfs_results ~/orfs_results_cleaned

run_one() {
    local d=$1           # design name (ariane133_ng45 etc)
    local pt=$2          # placement .pt path
    local out=$3         # output dir
    local label=$4       # log label

    local short=$(echo $d | sed "s/_ng45//")
    log "  [${label}] launching $d with $pt"
    timeout 21600 uv run python scripts/evaluate_with_orfs.py \
        --benchmark "$d" \
        --placement "$pt" \
        --orfs-root "$ORFS_ROOT" \
        --skip-synthesis \
        --output "$out/${d}" \
        > "$out/${d}.log" 2>&1
    log "  [${label}] $d done (rc=$?)"
}

# === Phase A: touching, ariane133 + 136 parallel-2 ===
log "=== Phase A: touching ariane133 + ariane136 parallel-2 ==="
run_one ariane133_ng45 ~/ng45_placements/ariane133_dp_lane.pt ~/orfs_results A-touching &
P1=$!
sleep 15
run_one ariane136_ng45 ~/ng45_placements/ariane136_dp_lane.pt ~/orfs_results A-touching &
P2=$!
wait $P1 $P2
log "Phase A done."

# Clear flow work dir between batches (avoid stale state).
sudo rm -rf $ORFS_ROOT/flow/results/nangate45/* 2>/dev/null || true
sudo rm -rf $ORFS_ROOT/flow/objects/nangate45/* 2>/dev/null || true
sudo rm -rf $ORFS_ROOT/flow/logs/nangate45/* 2>/dev/null || true
sudo chown -R $USER:$USER $ORFS_ROOT/flow/ 2>/dev/null || true

# === Phase B: touching, mempool_tile + nvdla parallel-2 ===
# NOTE: ORFS upstream does not have mempool_tile/nvdla designs by default.
# evaluate_with_orfs.py is supposed to patch these in. May need extra setup.
log "=== Phase B: touching mempool_tile + nvdla parallel-2 ==="
run_one mempool_tile_ng45 ~/ng45_placements/mempool_tile_dp_lane.pt ~/orfs_results B-touching &
P3=$!
sleep 15
run_one nvdla_ng45 ~/ng45_placements/nvdla_dp_lane.pt ~/orfs_results B-touching &
P4=$!
wait $P3 $P4
log "Phase B done."

sudo rm -rf $ORFS_ROOT/flow/results/nangate45/* 2>/dev/null || true
sudo rm -rf $ORFS_ROOT/flow/objects/nangate45/* 2>/dev/null || true
sudo rm -rf $ORFS_ROOT/flow/logs/nangate45/* 2>/dev/null || true
sudo chown -R $USER:$USER $ORFS_ROOT/flow/ 2>/dev/null || true

# === Phase C: cleaned, ariane133 + 136 parallel-2 ===
log "=== Phase C: cleaned ariane133 + ariane136 parallel-2 ==="
run_one ariane133_ng45 ~/ng45_placements/ariane133_dp_lane_12um.pt ~/orfs_results_cleaned C-cleaned &
P5=$!
sleep 15
run_one ariane136_ng45 ~/ng45_placements/ariane136_dp_lane_12um.pt ~/orfs_results_cleaned C-cleaned &
P6=$!
wait $P5 $P6
log "Phase C done."

sudo rm -rf $ORFS_ROOT/flow/results/nangate45/* 2>/dev/null || true
sudo rm -rf $ORFS_ROOT/flow/objects/nangate45/* 2>/dev/null || true
sudo rm -rf $ORFS_ROOT/flow/logs/nangate45/* 2>/dev/null || true
sudo chown -R $USER:$USER $ORFS_ROOT/flow/ 2>/dev/null || true

# === Phase D: cleaned, mempool_tile + nvdla parallel-2 ===
log "=== Phase D: cleaned mempool_tile + nvdla parallel-2 ==="
run_one mempool_tile_ng45 ~/ng45_placements/mempool_tile_dp_lane_12um.pt ~/orfs_results_cleaned D-cleaned &
P7=$!
sleep 15
run_one nvdla_ng45 ~/ng45_placements/nvdla_dp_lane_12um.pt ~/orfs_results_cleaned D-cleaned &
P8=$!
wait $P7 $P8
log "Phase D done."

log "=== ALL DONE ==="
log "Touching results: ~/orfs_results/"
log "Cleaned results:  ~/orfs_results_cleaned/"
log "Compare via:"
log "  for d in ariane133 ariane136 mempool_tile nvdla; do"
log "    echo \"=== \$d ===\""
log "    cat ~/orfs_results/\${d}_ng45/evaluation_summary.json"
log "    cat ~/orfs_results_cleaned/\${d}_ng45/evaluation_summary.json"
log "  done"
