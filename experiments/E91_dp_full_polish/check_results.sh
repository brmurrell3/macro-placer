#!/usr/bin/env bash
# Pull and summarize all E91 B-R0' results.
set -eo pipefail

CLOUD="ubuntu@129.213.18.245"

echo "Syncing results from cloud..."
rsync -az "${CLOUD}:~/macro-place-challenge-2026/experiments/E91_dp_full_polish/results/" \
  experiments/E91_dp_full_polish/results/ 2>/dev/null

echo
echo "=== B-R0' results so far ==="
python3 - <<'PYEOF'
import json
import glob

# Cascade ibm-bench references (capped wall-safe ~3300s)
REF_CAPPED = {
    "ibm01": 0.85,  # uncapped
    "ibm10": 1.0775,
    "ibm12": 1.3031,
    "ibm14": 1.2919,
    "ibm17": 1.4546,
}
print(f"{'bench':8s} | {'dp_basin':>9s} | {'legal':>9s} | {'cd':>9s} | {'lns':>9s} | {'sa':>9s} | {'cascade':>9s} | {'ref':>9s} | {'delta':>8s}")
print("-" * 100)
for f in sorted(glob.glob('experiments/E91_dp_full_polish/results/*_dp_full_polish.json')):
    bench = f.split('/')[-1].replace('_dp_full_polish.json', '')
    d = json.load(open(f))
    ref = REF_CAPPED.get(bench, 1.0)
    if 'final_proxy' not in d:
        print(f"{bench:8s} | INCOMPLETE | status={d.get('status','?')}")
        continue
    delta = (d['final_proxy'] - ref) / ref * 100
    print(f"{bench:8s} | "
          f"{d['dp_basin_proxy']:>9.5f} | "
          f"{d.get('legal_proxy', 0):>9.5f} | "
          f"{d.get('cd_proxy', 0):>9.5f} | "
          f"{d.get('lns_proxy', 0):>9.5f} | "
          f"{d.get('sa_proxy', 0):>9.5f} | "
          f"{d['final_proxy']:>9.5f} | "
          f"{ref:>9.5f} | "
          f"{delta:>+7.2f}%")
PYEOF

echo
echo "=== In-flight runs on cloud ==="
ssh ${CLOUD} 'for b in ibm10 ibm12 ibm14 ibm17 ibm01; do
  if [ -f /tmp/E91_${b}.log ]; then
    echo "--- $b ---";
    tail -3 /tmp/E91_${b}.log;
  fi;
done' 2>&1
