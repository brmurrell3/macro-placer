"""E113 — Just run the placer via evaluate on key benches.

Quick smoke check that the placer handles ibm09 (where V3 baseline
fails) and ibm17 (the hard bench).
"""
import subprocess, sys, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PLACER = "submissions/cd_lns_sa_xplace_patched/placer.py"

# Run --fast first, then ibm09 and ibm17 separately.
for cmd in [
    ["uv", "run", "evaluate", PLACER, "--fast", "--json", "--hypothesis", "E113_fast"],
]:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          env={**os.environ, "PYTHONUNBUFFERED": "1"})
    wall = time.time() - t0
    print(proc.stdout)
    if proc.returncode != 0:
        print("STDERR:")
        print(proc.stderr)
    print(f"  wall={wall:.0f}s exit={proc.returncode}")
