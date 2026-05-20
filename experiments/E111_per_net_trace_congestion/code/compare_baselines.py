"""Compare E88 bbox-uniform vs E111 per-net-trace vs canonical on ibm17.

Run from repo root: uv run python experiments/E111_per_net_trace_congestion/code/compare_baselines.py [bench]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments" / "E88_diff_proxy" / "code"))
sys.path.insert(0, str(_ROOT / "experiments" / "E95_diff_proxy_v2" / "code"))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

from per_net_trace_proxy import PerNetTraceCongestion
from diff_proxy_v2 import DiffProxyV2


def compare(bench_name: str = "ibm17", n_perturb: int = 16, perturb_frac: float = 0.02):
    print(f"\n=== compare baselines: {bench_name} ===", flush=True)
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    cached = _ROOT / "experiments" / "E84_cascading_saddle" / "results" / f"cascade_{bench_name}.pt"
    if cached.exists():
        data = torch.load(cached, weights_only=False, map_location="cpu")
        start = (data.get("placement") if "placement" in data else data["polished_placement"]).to(torch.float32)
        print(f"  using cached cascade placement", flush=True)
    else:
        start = benchmark.macro_positions.clone().float()
        print(f"  using macro_positions", flush=True)

    trace = PerNetTraceCongestion(benchmark, plc, device="cpu")
    bbox = DiffProxyV2(benchmark, plc, device="cpu", gamma_frac=5e-4)

    can = compute_proxy_cost(start, benchmark, plc)
    can_cong = float(can["congestion_cost"])
    with torch.no_grad():
        sm_trace = float(trace.compute_congestion(start))
        _, parts = bbox.cost(start, include_congestion=True)
        sm_bbox = float(parts["cong"])
    print(f"  canonical cong:   {can_cong:.5f}", flush=True)
    print(f"  trace cong:       {sm_trace:.5f}   ({(sm_trace-can_cong)/can_cong*100:+.1f}%)", flush=True)
    print(f"  bbox  cong (E88): {sm_bbox:.5f}   ({(sm_bbox-can_cong)/can_cong*100:+.1f}%)", flush=True)

    rng = np.random.default_rng(42)
    fixed = benchmark.macro_fixed.cpu().numpy()
    cw = float(benchmark.canvas_width)
    movable_hard = [i for i in range(benchmark.num_hard_macros) if not bool(fixed[i])]
    scale = perturb_frac * cw

    can_deltas, tr_deltas, bb_deltas = [], [], []
    for k in range(n_perturb):
        p = start.clone()
        targets = rng.choice(movable_hard, size=min(5, len(movable_hard)), replace=False)
        for t in targets:
            dx = float(rng.normal(0.0, scale))
            dy = float(rng.normal(0.0, scale))
            x0 = float(benchmark.macro_sizes[t, 0]) / 2.0
            y0 = float(benchmark.macro_sizes[t, 1]) / 2.0
            p[t, 0] = torch.clamp(p[t, 0] + dx, x0, cw - x0)
            p[t, 1] = torch.clamp(p[t, 1] + dy, y0, float(benchmark.canvas_height) - y0)
        c_can = float(compute_proxy_cost(p, benchmark, plc)["congestion_cost"])
        with torch.no_grad():
            c_tr = float(trace.compute_congestion(p))
            _, parts2 = bbox.cost(p, include_congestion=True)
            c_bb = float(parts2["cong"])
        can_deltas.append(c_can - can_cong)
        tr_deltas.append(c_tr - sm_trace)
        bb_deltas.append(c_bb - sm_bbox)

    can_arr = np.asarray(can_deltas)
    tr_arr = np.asarray(tr_deltas)
    bb_arr = np.asarray(bb_deltas)

    def report(label, sm_arr):
        pearson = float(np.corrcoef(can_arr, sm_arr)[0, 1])
        rk_c = np.argsort(np.argsort(can_arr))
        rk_s = np.argsort(np.argsort(sm_arr))
        spearman = float(np.corrcoef(rk_c, rk_s)[0, 1])
        sign_agree = float(np.mean(np.sign(can_arr) == np.sign(sm_arr)))
        print(f"  {label:8s} pearson={pearson:+.3f}  spearman={spearman:+.3f}  sign={sign_agree:.0%}", flush=True)

    print(f"  Δcong from {n_perturb} perturbations:", flush=True)
    report("trace", tr_arr)
    report("bbox", bb_arr)


if __name__ == "__main__":
    benches = sys.argv[1:] if len(sys.argv) > 1 else ["ibm17"]
    for b in benches:
        compare(b)
