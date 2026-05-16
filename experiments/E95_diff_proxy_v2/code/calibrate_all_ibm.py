"""E95 — calibrate smooth vs canonical decomposition across all 17 IBM benches.

For each cached cascade output, report:
  - canonical proxy (cascade quality)
  - smooth proxy (with WL norm fix from DiffProxyV2)
  - per-component decomposition: WL, density, congestion
  - smooth/canonical ratio per component

Identifies which benches have RUDY mismatch (smooth_cong >> canonical_cong)
and which match. Useful for scoping where C1 could theoretically work
without the RUDY rewrite.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
_E88_DIR = _ROOT / "experiments" / "E88_diff_proxy" / "code"
for p in (_ROOT, _E88_DIR, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from diff_proxy_v2 import DiffProxyV2


IBM_BENCHES = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08", "ibm09",
    "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15", "ibm16", "ibm17",
    "ibm18",
]


def main():
    rows = []
    t_global = time.time()
    for bench_name in IBM_BENCHES:
        bench_dir = Path(f"external/MacroPlacement/Testcases/ICCAD04/{bench_name}")
        if not bench_dir.exists():
            rows.append({"bench": bench_name, "error": "benchmark dir missing"})
            continue
        cache_path = Path(f"experiments/E84_cascading_saddle/results/cascade_{bench_name}.pt")
        if not cache_path.exists():
            rows.append({"bench": bench_name, "error": "no cached cascade output"})
            continue
        try:
            benchmark, plc = load_benchmark_from_dir(str(bench_dir))
            proxy = DiffProxyV2(benchmark, plc, gamma_frac=0.005)
            cached = torch.load(str(cache_path), weights_only=False, map_location="cpu")
            pos = cached["placement"].clone().float()
            with torch.no_grad():
                smooth, parts = proxy.cost(pos)
            canon = compute_proxy_cost(pos, benchmark, plc)
            r = {
                "bench": bench_name,
                "num_macros": pos.shape[0],
                "num_hard": benchmark.num_hard_macros,
                "len_nets": len(plc.nets),
                "plc_net_cnt": float(plc.net_cnt),
                "canvas_w": float(benchmark.canvas_width),
                "canvas_h": float(benchmark.canvas_height),
                "grid_cols": benchmark.grid_cols,
                "grid_rows": benchmark.grid_rows,
                "canon_proxy": float(canon["proxy_cost"]),
                "canon_wl": float(canon["wirelength_cost"]),
                "canon_density": float(canon["density_cost"]),
                "canon_cong": float(canon["congestion_cost"]),
                "canon_ovl": int(canon["overlap_count"]),
                "smooth_proxy": float(smooth),
                "smooth_wl": float(parts["wl"]),
                "smooth_density": float(parts["density"]),
                "smooth_cong": float(parts["cong"]),
            }
            r["cong_ratio"] = r["smooth_cong"] / max(1e-6, r["canon_cong"])
            r["wl_ratio"] = r["smooth_wl"] / max(1e-6, r["canon_wl"])
            r["density_ratio"] = r["smooth_density"] / max(1e-6, r["canon_density"])
            r["proxy_ratio"] = r["smooth_proxy"] / max(1e-6, r["canon_proxy"])
            rows.append(r)
            print(f"{bench_name}: macros={r['num_macros']:5d}  nets={r['len_nets']:6d}  "
                  f"canon_proxy={r['canon_proxy']:.4f}  smooth_proxy={r['smooth_proxy']:.4f}  "
                  f"cong_ratio={r['cong_ratio']:.2f}× (canon={r['canon_cong']:.4f} smooth={r['smooth_cong']:.4f})")
        except Exception as exc:
            rows.append({"bench": bench_name, "error": repr(exc)})
            print(f"{bench_name}: ERROR {exc}")

    out_path = Path("experiments/E95_diff_proxy_v2/results/calibrate_all_ibm.json")
    out_path.write_text(json.dumps(rows, indent=2, default=str))

    # Markdown table
    print()
    print(f"{'bench':<8} {'macros':>6} {'nets':>6} {'canon':>8} {'smooth':>8} {'wl_r':>6} {'dens_r':>6} {'cong_r':>6} {'prox_r':>6}")
    for r in rows:
        if "error" in r:
            print(f"{r['bench']:<8} ERROR: {r['error']}")
            continue
        print(f"{r['bench']:<8} {r['num_macros']:>6} {r['len_nets']:>6} "
              f"{r['canon_proxy']:>8.4f} {r['smooth_proxy']:>8.4f} "
              f"{r['wl_ratio']:>6.2f} {r['density_ratio']:>6.2f} "
              f"{r['cong_ratio']:>6.2f} {r['proxy_ratio']:>6.2f}")
    print(f"\nTotal wall: {time.time() - t_global:.0f}s")


if __name__ == "__main__":
    main()
