"""Calibration: how does E122 perform on the SDF init placement (the
actual starting point for Adam descent), vs E111 base?

The cached cascade placement is the OUTPUT after polish — but the actual
Adam descent starts at SDF init. The calibration should hold there too.

This is a sanity check that the bias closure isn't just a property of
the cascade-polished placement.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, sdf_init
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

from per_net_trace_proxy import PerNetTraceCongestion
from three_pin_steiner import PerNetTraceCongestionSteiner3Pin


def calibrate_sdf(bench_name: str = "ibm17"):
    print(f"\n=== SDF-init calibration: {bench_name} ===", flush=True)
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    pos = sdf_init(benchmark)
    pos, _ = project_overlaps(pos, benchmark)
    pos = pos.to(torch.float32)
    print(f"  SDF placement (after project_overlaps)", flush=True)

    base = PerNetTraceCongestion(benchmark, plc, device="cpu")
    e122 = PerNetTraceCongestionSteiner3Pin(benchmark, plc, device="cpu")

    can = compute_proxy_cost(pos, benchmark, plc)
    can_cong = float(can["congestion_cost"])
    can_proxy = float(can["proxy_cost"])

    with torch.no_grad():
        sm_base = float(base.compute_congestion(pos))
        sm_e122 = float(e122.compute_congestion(pos))

    rel_base = (sm_base - can_cong) / max(abs(can_cong), 1e-6) * 100.0
    rel_e122 = (sm_e122 - can_cong) / max(abs(can_cong), 1e-6) * 100.0
    closure = abs(rel_base) - abs(rel_e122)
    print(f"  canonical proxy: {can_proxy:.5f}  cong: {can_cong:.5f}", flush=True)
    print(f"  E111 cong:  {sm_base:.5f}  ({rel_base:+.1f}%)", flush=True)
    print(f"  E122 cong:  {sm_e122:.5f}  ({rel_e122:+.1f}%)", flush=True)
    print(f"  Bias closure: {closure:+.1f} pp", flush=True)


if __name__ == "__main__":
    benches = sys.argv[1:] if len(sys.argv) > 1 else ["ibm17"]
    for b in benches:
        calibrate_sdf(b)
