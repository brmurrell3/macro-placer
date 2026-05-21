"""E123 — Fast ε sweep on a cached placement.

Sweeps Sinkhorn ε values across {0.5, 0.3, 0.1, 0.05, 0.02, 0.01} and
reports both scalar Sinkhorn-vs-canonical mismatch and gradient
distribution properties (how many cells have non-trivial gradient
relative to the K-th gradient magnitude).

Output: results/eps_sweep_<bench>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for p in (
    _HERE,
    _ROOT,
    _ROOT / "experiments" / "E111_per_net_trace_congestion" / "code",
    _ROOT / "experiments" / "E88_diff_proxy" / "code",
    _ROOT / "experiments" / "E95_diff_proxy_v2" / "code",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

from sinkhorn_per_net_trace import SinkhornPerNetTraceCongestion


def gradient_stats(proxy, positions: torch.Tensor) -> dict:
    """Run forward+backward on positions and report gradient distribution.

    Returns dict with:
      - top_k_mean_grad: average gradient magnitude across the cells
        that are in the top-K according to the smooth value.
      - n_cells_with_grad: count of cells where |grad| > 0.05 * top_k_grad.
    """
    # Make positions trainable
    pos = positions.detach().clone().requires_grad_(True)
    cong = proxy.compute_congestion(pos)
    cong.backward()
    grad_mag = pos.grad.abs().sum(dim=1)  # per-macro grad magnitude
    if grad_mag.numel() == 0:
        return {"top10_grad": 0.0, "n_movers": 0}
    top10 = torch.topk(grad_mag, min(10, grad_mag.numel())).values
    top10_mean = float(top10.mean())
    threshold = top10_mean * 0.05
    n_movers = int((grad_mag > threshold).sum())
    return {
        "top10_grad": top10_mean,
        "n_movers": n_movers,
        "total_macros": int(grad_mag.numel()),
    }


def main():
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm17"
    print(f"=== E123 ε sweep on {bench} ===", flush=True)
    bench_dir = find_benchmark_dir(bench)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    cached = _ROOT / "experiments" / "E84_cascading_saddle" / "results" / f"cascade_{bench}.pt"
    if cached.exists():
        data = torch.load(cached, weights_only=False, map_location="cpu")
        start = (data.get("placement") if "placement" in data else data["polished_placement"]).to(torch.float32)
        source = "cached_cascade"
    else:
        start = benchmark.macro_positions.clone().float()
        source = "macro_positions"
    print(f"  source: {source}, n_movable_hard={int((~benchmark.macro_fixed.bool()).sum())}", flush=True)

    # Canonical reference (use precomputed values for hard benches to avoid
    # the slow plc.get_congestion_cost call on ibm17 ≈ 90s).
    PRECOMPUTED_CANON = {
        "ibm17": 1.89497,
        "ibm12": 1.654,
        "ibm10": 1.301,
    }
    if bench in PRECOMPUTED_CANON:
        can_cong = PRECOMPUTED_CANON[bench]
        print(f"  canonical cong (precomputed): {can_cong:.5f}", flush=True)
    else:
        t = time.time()
        can = compute_proxy_cost(start, benchmark, plc)
        can_cong = float(can["congestion_cost"])
        print(f"  canonical cong: {can_cong:.5f}  (wall={time.time()-t:.1f}s)", flush=True)

    rows = []

    # Reference: torch.topk
    proxy_topk = SinkhornPerNetTraceCongestion(
        benchmark, plc, device="cpu", use_sinkhorn=False,
    )
    with torch.no_grad():
        topk_cong = float(proxy_topk.compute_congestion(start))
    topk_grad = gradient_stats(proxy_topk, start)
    rel_topk = (topk_cong - can_cong) / abs(can_cong) * 100
    print(f"\n  torch.topk:  cong={topk_cong:.5f}  rel={rel_topk:+.2f}%  "
          f"n_movers={topk_grad['n_movers']}/{topk_grad['total_macros']}  "
          f"top10_grad={topk_grad['top10_grad']:.4e}",
          flush=True)
    rows.append({"variant": "torch.topk", **topk_grad,
                 "cong": topk_cong, "rel_pct": rel_topk})

    # Sweep ε values
    eps_values = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
    print(f"\n  {'eps':>6s} {'iters':>6s} {'cong':>9s} {'rel%':>7s} {'movers':>8s} {'top10g':>10s}",
          flush=True)
    for eps in eps_values:
        # auto-bump iters scales with 1/eps; cap at 200
        iters = max(50, min(200, int(50 / max(eps, 0.01))))
        proxy_sk = SinkhornPerNetTraceCongestion(
            benchmark, plc, device="cpu",
            sinkhorn_eps=eps, sinkhorn_iters=iters, use_sinkhorn=True,
        )
        with torch.no_grad():
            sk_cong = float(proxy_sk.compute_congestion(start))
        sk_grad = gradient_stats(proxy_sk, start)
        rel_sk = (sk_cong - can_cong) / abs(can_cong) * 100
        print(f"  {eps:>6.3f} {iters:>6d} {sk_cong:>9.5f} {rel_sk:>+7.2f}% "
              f"{sk_grad['n_movers']:>5d}/{sk_grad['total_macros']:<3d} "
              f"{sk_grad['top10_grad']:>10.4e}",
              flush=True)
        rows.append({"variant": f"sinkhorn_eps={eps}", "iters": iters,
                     **sk_grad, "cong": sk_cong, "rel_pct": rel_sk})

    out = {
        "bench": bench,
        "canonical": can_cong,
        "source": source,
        "rows": rows,
    }
    out_path = _HERE.parent / "results" / f"eps_sweep_{bench}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
