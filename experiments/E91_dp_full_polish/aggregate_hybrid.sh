#!/usr/bin/env bash
# Pull cloud results and aggregate hybrid --all proxy vs cascade reference.
set -eo pipefail

CLOUD="ubuntu@129.213.18.245"

echo "=== Pulling latest experiment_log.jsonl from cloud ==="
rsync -az "${CLOUD}:~/macro-place-challenge-2026/results/experiment_log.jsonl" \
  /tmp/experiment_log_cloud.jsonl

echo
echo "=== Hybrid (E91hybrid) per-bench + aggregate ==="
python3 - <<'PYEOF'
import json
hybrid = {}
cascade = {}
with open("/tmp/experiment_log_cloud.jsonl") as f:
    for line in f:
        try:
            d = json.loads(line)
        except Exception:
            continue
        pb = d.get("per_benchmark", {})
        h = d.get("hypothesis", "")
        if h == "E91hybrid":
            for b, v in pb.items():
                # keep most recent per-bench
                if b not in hybrid or d.get("timestamp", "") > hybrid[b][1]:
                    hybrid[b] = (v, d.get("timestamp", ""))
        if "a4v2_postLNS" in h and len(pb) >= 17:
            for b, v in pb.items():
                cascade[b] = v

ibm17 = [f"ibm{n:02d}" for n in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
ng45 = ["ariane133", "ariane136", "mempool_tile", "nvdla"]

print(f"{'bench':14s} | {'hybrid':>9s} | {'cascade':>9s} | {'delta':>8s}")
print("-" * 60)
hsum, csum, ncount = 0.0, 0.0, 0
for b in ibm17:
    h_v = hybrid.get(b, [None])[0]
    c_v = cascade.get(b)
    if h_v is None or c_v is None:
        print(f"{b:14s} | {'?' if h_v is None else f'{h_v:.5f}':>9s} | {'?' if c_v is None else f'{c_v:.5f}':>9s} | -")
        continue
    delta = (h_v - c_v) / c_v * 100
    print(f"{b:14s} | {h_v:>9.5f} | {c_v:>9.5f} | {delta:>+7.2f}%")
    hsum += h_v
    csum += c_v
    ncount += 1
print("-" * 60)
if ncount:
    print(f"{'IBM avg ('+str(ncount)+')':14s} | {hsum/ncount:>9.5f} | {csum/ncount:>9.5f} | {(hsum-csum)/csum*100:>+7.2f}%")
print()
print("NG45 hybrid:")
for b in ng45:
    h_v = hybrid.get(b, [None])[0]
    if h_v is None:
        print(f"  {b:14s} | not run")
    else:
        print(f"  {b:14s} | {h_v:>9.5f}")
PYEOF

echo
echo "=== In-flight processes ==="
ssh ${CLOUD} 'ps -ef | grep -E "hybrid|cascade_dp_lane" | grep -v grep | head -10'
