"""Quick calibration: scalar canonical mismatch for E111 vs E122 on a bench.

Skips the perturbation Pearson/Spearman loop (which costs 16x canonical
evals at ~30 s each on ibm17 = ~8 min). Just reports scalar relative
mismatch on the cached cascade placement.

Run: uv run python experiments/E122_3pin_steiner/code/calibrate_quick.py ibm17
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _ROOT,
    _HERE,
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

from per_net_trace_proxy import PerNetTraceCongestion
from three_pin_steiner import PerNetTraceCongestionSteiner3Pin


def quick_calibrate(bench_name: str = "ibm17"):
    print(f"\n=== E122 quick scalar calibration: {bench_name} ===", flush=True)
    t0 = time.time()
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"  load: {time.time()-t0:.1f}s", flush=True)

    cached = _ROOT / "experiments" / "E84_cascading_saddle" / "results" / f"cascade_{bench_name}.pt"
    if cached.exists():
        data = torch.load(cached, weights_only=False, map_location="cpu")
        start = (data.get("placement") if "placement" in data else data["polished_placement"]).to(torch.float32)
        print(f"  using cached cascade placement", flush=True)
    else:
        start = benchmark.macro_positions.clone().float()
        print(f"  using macro_positions", flush=True)

    t1 = time.time()
    base = PerNetTraceCongestion(benchmark, plc, device="cpu")
    print(f"  E111 build: {time.time()-t1:.1f}s  n_pairs={base.n_pairs}", flush=True)
    t1 = time.time()
    e122 = PerNetTraceCongestionSteiner3Pin(benchmark, plc, device="cpu")
    print(
        f"  E122 build: {time.time()-t1:.1f}s  n_pairs={e122.n_pairs}  n_3pin={e122.n_3pin}",
        flush=True,
    )

    # Pin-degree distribution
    from per_net_trace_proxy import _extract_net_data
    nd = _extract_net_data(benchmark, plc)
    n_pins_each = nd.mask.sum(dim=1)
    n_total = int((nd.weights > 0).sum().item())
    n_2 = int(((n_pins_each == 2) & (nd.weights > 0)).sum().item())
    n_3 = int(((n_pins_each == 3) & (nd.weights > 0)).sum().item())
    n_4 = int(((n_pins_each >= 4) & (nd.weights > 0)).sum().item())
    print(
        f"  pin-degree: total={n_total}  "
        f"2-pin={n_2} ({100*n_2/max(1,n_total):.1f}%)  "
        f"3-pin={n_3} ({100*n_3/max(1,n_total):.1f}%)  "
        f"4+pin={n_4} ({100*n_4/max(1,n_total):.1f}%)",
        flush=True,
    )

    t1 = time.time()
    can = compute_proxy_cost(start, benchmark, plc)
    can_cong = float(can["congestion_cost"])
    can_proxy = float(can["proxy_cost"])
    can_wl = float(can["wirelength_cost"])
    can_density = float(can["density_cost"])
    print(f"  canonical eval: {time.time()-t1:.1f}s", flush=True)
    print(
        f"  canonical: proxy={can_proxy:.5f} wl={can_wl:.5f} d={can_density:.5f} c={can_cong:.5f}",
        flush=True,
    )

    t1 = time.time()
    with torch.no_grad():
        sm_base = float(base.compute_congestion(start))
    print(f"  E111 eval: {time.time()-t1:.1f}s", flush=True)
    t1 = time.time()
    with torch.no_grad():
        sm_e122 = float(e122.compute_congestion(start))
    print(f"  E122 eval: {time.time()-t1:.1f}s", flush=True)

    rel_base = (sm_base - can_cong) / max(abs(can_cong), 1e-6) * 100.0
    rel_e122 = (sm_e122 - can_cong) / max(abs(can_cong), 1e-6) * 100.0
    delta_e122 = (sm_e122 - sm_base) / max(abs(sm_base), 1e-6) * 100.0
    abs_close = (abs(rel_base) - abs(rel_e122))

    print(f"  E111 cong:  {sm_base:.5f}  ({rel_base:+.1f}% vs canonical)", flush=True)
    print(f"  E122 cong:  {sm_e122:.5f}  ({rel_e122:+.1f}% vs canonical)", flush=True)
    print(f"  E122 - E111: {sm_e122 - sm_base:+.5f}  ({delta_e122:+.1f}%)", flush=True)
    print(f"  Bias closure: |rel_base| - |rel_e122| = {abs_close:+.1f}pp", flush=True)

    return {
        "bench": bench_name,
        "canonical": can_cong,
        "base": sm_base,
        "base_rel_pct": rel_base,
        "e122": sm_e122,
        "e122_rel_pct": rel_e122,
        "delta_pct": delta_e122,
        "bias_closure_pp": abs_close,
        "n_total": n_total,
        "n_3pin": n_3,
    }


if __name__ == "__main__":
    benches = sys.argv[1:] if len(sys.argv) > 1 else ["ibm17"]
    results = []
    for b in benches:
        try:
            r = quick_calibrate(b)
            results.append(r)
        except Exception as e:
            print(f"  ERROR on {b}: {e}", flush=True)
            import traceback
            traceback.print_exc()
    print("\n=== Summary ===", flush=True)
    for r in results:
        print(
            f"  {r['bench']:10s} base={r['base_rel_pct']:+.1f}%  "
            f"e122={r['e122_rel_pct']:+.1f}%  closure={r['bias_closure_pp']:+.1f}pp  "
            f"(3pin nets {r['n_3pin']}/{r['n_total']})",
            flush=True,
        )
