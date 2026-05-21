"""E115 baseline: time PyTorch CPU autograd vs CUDA, vs Triton.

Measure per-step wall time for the full smooth-proxy cost (LSE-HPWL +
grid-density + per-net-trace-congestion) + backward + Adam step on ibm17.

Usage:
  .venv/bin/python experiments/E115_triton_kernels/code/bench_baseline.py [bench] [device] [n_steps]

Defaults: ibm17, cpu, 30 steps.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
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


def main(bench_name: str = "ibm17", device: str = "cpu", n_steps: int = 30):
    print(f"\n=== E115 baseline: bench={bench_name} device={device} steps={n_steps} ===", flush=True)
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"  num_macros={benchmark.num_macros} num_hard={benchmark.num_hard_macros} "
          f"grid={benchmark.grid_rows}x{benchmark.grid_cols}", flush=True)

    init_pos = sdf_init(benchmark)
    init_pos, _ = project_overlaps(init_pos, benchmark)

    proxy = DiffProxyV3(benchmark, plc, device=device, gamma_frac=5e-4)
    positions = init_pos.clone().detach().to(device).requires_grad_(True)
    fixed_mask = benchmark.macro_fixed.bool().to(device)
    optimizer = torch.optim.Adam([positions], lr=0.005 * float(benchmark.canvas_width))

    # Warmup
    print("  warmup 3 steps...", flush=True)
    for _ in range(3):
        optimizer.zero_grad()
        loss, parts = loss_with_penalty_v3(proxy, positions, 50.0, include_congestion=True, boundary_lambda=50.0)
        loss.backward()
        with torch.no_grad():
            if positions.grad is not None:
                positions.grad[fixed_mask] = 0.0
        optimizer.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Measure forward + backward + step separately
    fwd_times, bwd_times, opt_times = [], [], []
    cong_times, wl_times, dens_times = [], [], []
    print(f"  timing {n_steps} steps...", flush=True)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_total0 = time.perf_counter()
    for step in range(n_steps):
        # Component timings
        if device.startswith("cuda"): torch.cuda.synchronize()
        t0 = time.perf_counter()
        # Time congestion alone
        from diff_proxy import _lse_hpwl, _grid_density
        clamped = proxy._clamp_to_canvas(positions)

        # WL
        if device.startswith("cuda"): torch.cuda.synchronize()
        t_wl0 = time.perf_counter()
        wl = _lse_hpwl(clamped, proxy.net_data, proxy.port_base, proxy.gamma) / proxy.wl_norm
        if device.startswith("cuda"): torch.cuda.synchronize()
        wl_times.append(time.perf_counter() - t_wl0)

        # Density
        t_d0 = time.perf_counter()
        density = _grid_density(
            clamped, proxy.sizes,
            proxy.cell_x_min, proxy.cell_x_max,
            proxy.cell_y_min, proxy.cell_y_max,
            proxy.cell_area, proxy.grid_rows, proxy.grid_cols,
        )
        if device.startswith("cuda"): torch.cuda.synchronize()
        dens_times.append(time.perf_counter() - t_d0)

        # Congestion
        t_c0 = time.perf_counter()
        cong = proxy.trace.compute_congestion(clamped)
        if device.startswith("cuda"): torch.cuda.synchronize()
        cong_times.append(time.perf_counter() - t_c0)

        # Total forward (full loss with penalties)
        optimizer.zero_grad()
        if device.startswith("cuda"): torch.cuda.synchronize()
        t_f0 = time.perf_counter()
        loss, parts = loss_with_penalty_v3(proxy, positions, 50.0, include_congestion=True, boundary_lambda=50.0)
        if device.startswith("cuda"): torch.cuda.synchronize()
        fwd_times.append(time.perf_counter() - t_f0)

        # Backward
        if device.startswith("cuda"): torch.cuda.synchronize()
        t_b0 = time.perf_counter()
        loss.backward()
        if device.startswith("cuda"): torch.cuda.synchronize()
        bwd_times.append(time.perf_counter() - t_b0)

        # Optimizer step
        if device.startswith("cuda"): torch.cuda.synchronize()
        t_o0 = time.perf_counter()
        with torch.no_grad():
            if positions.grad is not None:
                positions.grad[fixed_mask] = 0.0
        optimizer.step()
        if device.startswith("cuda"): torch.cuda.synchronize()
        opt_times.append(time.perf_counter() - t_o0)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_total = time.perf_counter() - t_total0

    import statistics as st
    def summarize(times, label):
        ms = [t * 1000 for t in times]
        return {"label": label, "mean_ms": st.mean(ms), "median_ms": st.median(ms), "min_ms": min(ms), "max_ms": max(ms)}

    results = {
        "bench": bench_name,
        "device": device,
        "n_steps": n_steps,
        "num_macros": int(benchmark.num_macros),
        "num_hard": int(benchmark.num_hard_macros),
        "grid_rows": int(benchmark.grid_rows),
        "grid_cols": int(benchmark.grid_cols),
        "total_wall_s": t_total,
        "per_step_mean_ms": t_total / n_steps * 1000,
        "components": [
            summarize(wl_times, "wl"),
            summarize(dens_times, "density"),
            summarize(cong_times, "congestion"),
            summarize(fwd_times, "fwd_total"),
            summarize(bwd_times, "bwd_total"),
            summarize(opt_times, "opt_step"),
        ],
    }

    print(f"\n  results:", flush=True)
    print(f"    total: {t_total:.2f}s ({results['per_step_mean_ms']:.1f} ms/step)", flush=True)
    for c in results["components"]:
        print(f"    {c['label']:12s} mean={c['mean_ms']:7.2f}ms  median={c['median_ms']:7.2f}ms  min={c['min_ms']:7.2f}ms", flush=True)
    return results


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm17"
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    r = main(bench, device, n)
    out_path = _HERE.parent / "results" / f"baseline_{bench}_{device}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(r, f, indent=2)
    print(f"  wrote {out_path}", flush=True)
