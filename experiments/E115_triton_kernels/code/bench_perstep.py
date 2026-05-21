"""E115 per-step bench: compare {V3, V4, V4+compile} across benchmarks.

Output: pure per-step wall (excluding setup), so the speedup figures
isolate the gradient-step work.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E110_smooth_global_placer" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.cd_core import project_overlaps, sdf_init

from smooth_global_placer_v3 import DiffProxyV3, loss_with_penalty_v3
from fast_proxy import FastDiffProxy, fast_loss_with_penalty


def time_eager(proxy, loss_fn, positions, fixed_mask, n=20, device="cuda"):
    opt = torch.optim.Adam([positions], lr=10.0)
    for _ in range(3):
        opt.zero_grad()
        l, _ = loss_fn(proxy, positions, 50.0)
        l.backward()
        with torch.no_grad():
            positions.grad[fixed_mask] = 0
        opt.step()
    if device.startswith("cuda"): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        opt.zero_grad()
        l, _ = loss_fn(proxy, positions, 50.0)
        l.backward()
        with torch.no_grad():
            positions.grad[fixed_mask] = 0
        opt.step()
    if device.startswith("cuda"): torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n


def time_compiled(proxy, positions, fixed_mask, n=20, device="cuda"):
    @torch.compile(mode="reduce-overhead")
    def closure(positions):
        loss, _ = fast_loss_with_penalty(proxy, positions, 50.0, include_congestion=True, boundary_lambda=50.0)
        return loss

    opt = torch.optim.Adam([positions], lr=10.0)
    for _ in range(10):
        opt.zero_grad()
        l = closure(positions)
        l.backward()
        with torch.no_grad():
            positions.grad[fixed_mask] = 0
        opt.step()
    if device.startswith("cuda"): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        opt.zero_grad()
        l = closure(positions)
        l.backward()
        with torch.no_grad():
            positions.grad[fixed_mask] = 0
        opt.step()
    if device.startswith("cuda"): torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n


def main(benches, device, n=20):
    rows = []
    for bench_name in benches:
        bench_dir = find_benchmark_dir(bench_name)
        benchmark, plc = load_benchmark_from_dir(str(bench_dir))
        init_pos = sdf_init(benchmark)
        init_pos, _ = project_overlaps(init_pos, benchmark)

        # V3 baseline
        proxy_b = DiffProxyV3(benchmark, plc, device=device, gamma_frac=5e-4)
        pos = init_pos.clone().detach().to(device).requires_grad_(True)
        mask = benchmark.macro_fixed.bool().to(device)
        ms_v3 = time_eager(proxy_b, loss_with_penalty_v3, pos, mask, n=n, device=device)

        # V4 fast
        proxy_f = FastDiffProxy(benchmark, plc, device=device, gamma_frac=5e-4)
        pos = init_pos.clone().detach().to(device).requires_grad_(True)
        ms_v4 = time_eager(proxy_f, fast_loss_with_penalty, pos, mask, n=n, device=device)

        # V4 compiled (only on CUDA)
        if device.startswith("cuda"):
            pos = init_pos.clone().detach().to(device).requires_grad_(True)
            ms_v5 = time_compiled(proxy_f, pos, mask, n=n, device=device)
        else:
            ms_v5 = None

        row = {
            "bench": bench_name,
            "num_macros": int(benchmark.num_macros),
            "num_hard": int(benchmark.num_hard_macros),
            "grid": f"{benchmark.grid_rows}x{benchmark.grid_cols}",
            "ms_v3": ms_v3,
            "ms_v4": ms_v4,
            "ms_v5": ms_v5,
            "speedup_v4_over_v3": ms_v3 / ms_v4,
            "speedup_v5_over_v3": ms_v3 / ms_v5 if ms_v5 else None,
        }
        rows.append(row)

    print(f"\n=== Per-step wall (ms) on {device}, {n} steps each ===", flush=True)
    print(f"  {'bench':<8s} {'macros':>7s} {'grid':>9s} {'V3':>8s} {'V4':>8s} {'V5':>8s} {'V4/V3':>7s} {'V5/V3':>7s}", flush=True)
    for r in rows:
        v5s = f"{r['ms_v5']:8.2f}" if r['ms_v5'] is not None else f"{'n/a':>8s}"
        v5x = f"{r['speedup_v5_over_v3']:6.1f}x" if r['speedup_v5_over_v3'] is not None else "  n/a "
        print(f"  {r['bench']:<8s} {r['num_macros']:>7d} {r['grid']:>9s} {r['ms_v3']:8.2f} {r['ms_v4']:8.2f} {v5s} {r['speedup_v4_over_v3']:6.1f}x {v5x}", flush=True)

    out_path = _HERE.parent / "results" / f"perstep_{device}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "device": device, "n": n}, f, indent=2)
    print(f"\n  wrote {out_path}", flush=True)
    return rows


if __name__ == "__main__":
    benches = sys.argv[1].split(",") if len(sys.argv) > 1 else ["ibm01", "ibm07", "ibm10", "ibm17"]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    main(benches, device, n)
