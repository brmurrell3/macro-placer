"""E88 calibration probe — does diff proxy track canonical across placements?

Eval on:
  - cascade-cached ibm01 (canonical 0.84528)
  - cached + small Gaussian perturbations (5 seeds, sigma = 1% canvas)
  - cached + larger perturbations (5 seeds, sigma = 5% canvas)
  - random uniform init (5 seeds)

For each placement: compute smooth proxy (diff_proxy.cost) and canonical
proxy (macro_place.objective.compute_proxy_cost), report (smooth, canonical,
gap_pct, overlap_count). Spike continues only if (a) at the cascade optimum
gap < 5 %, (b) Spearman rank correlation across placements is positive
and strong (>0.7) — meaning gradient descent on smooth basically descends
canonical too.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from diff_proxy import DiffProxy


def _spearman(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    rx = [r for r, _ in sorted(enumerate(xs), key=lambda kv: kv[1])]
    ry = [r for r, _ in sorted(enumerate(ys), key=lambda kv: kv[1])]
    rank_x = [0.0] * n
    rank_y = [0.0] * n
    for new_rank, idx in enumerate(rx):
        rank_x[idx] = float(new_rank)
    for new_rank, idx in enumerate(ry):
        rank_y[idx] = float(new_rank)
    mx = sum(rank_x) / n
    my = sum(rank_y) / n
    num = sum((rank_x[i] - mx) * (rank_y[i] - my) for i in range(n))
    dx = (sum((r - mx) ** 2 for r in rank_x)) ** 0.5
    dy = (sum((r - my) ** 2 for r in rank_y)) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def main():
    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    proxy = DiffProxy(benchmark, plc, device="cpu")

    cached = torch.load(
        "experiments/E84_cascading_saddle/results/cascade_ibm01.pt",
        weights_only=False,
        map_location="cpu",
    )
    base = cached["placement"].clone().float()
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    half = benchmark.macro_sizes / 2.0

    def clamp(p: torch.Tensor) -> torch.Tensor:
        p = p.clone()
        p[:, 0] = p[:, 0].clamp(half[:, 0], cw - half[:, 0])
        p[:, 1] = p[:, 1].clamp(half[:, 1], ch - half[:, 1])
        return p

    placements = []
    placements.append(("cascade_cached", base))
    g = torch.Generator().manual_seed(0)
    for s in range(5):
        torch.manual_seed(s)
        noise = torch.randn_like(base) * torch.tensor([cw * 0.01, ch * 0.01])
        placements.append((f"perturb_1pct_seed{s}", clamp(base + noise)))
    for s in range(5):
        torch.manual_seed(100 + s)
        noise = torch.randn_like(base) * torch.tensor([cw * 0.05, ch * 0.05])
        placements.append((f"perturb_5pct_seed{s}", clamp(base + noise)))
    for s in range(5):
        torch.manual_seed(200 + s)
        rand = torch.rand_like(base)
        rand[:, 0] = rand[:, 0] * (cw - 2 * half[:, 0]) + half[:, 0]
        rand[:, 1] = rand[:, 1] * (ch - 2 * half[:, 1]) + half[:, 1]
        placements.append((f"uniform_random_seed{s}", rand))

    rows = []
    smooth_vals = []
    canonical_vals = []
    t0 = time.time()
    for name, pos in placements:
        smooth, _ = proxy.cost(pos)
        smooth_v = float(smooth.detach())
        canonical = compute_proxy_cost(pos, benchmark, plc)
        canonical_v = float(canonical["proxy_cost"])
        gap_pct = (smooth_v - canonical_v) / canonical_v * 100.0
        ov = int(canonical["overlap_count"])
        rows.append({
            "name": name,
            "smooth": smooth_v,
            "canonical": canonical_v,
            "gap_pct": gap_pct,
            "wl": float(canonical["wirelength_cost"]),
            "density": float(canonical["density_cost"]),
            "cong": float(canonical["congestion_cost"]),
            "overlap_count": ov,
        })
        smooth_vals.append(smooth_v)
        canonical_vals.append(canonical_v)
        print(f"{name:30s}  smooth={smooth_v:.5f}  canonical={canonical_v:.5f}  gap={gap_pct:+.2f}%  ovl={ov}")

    rho = _spearman(smooth_vals, canonical_vals)
    wall = time.time() - t0
    print(f"\nSpearman(smooth, canonical) over {len(rows)} placements: rho={rho:.4f}")
    print(f"Wall: {wall:.1f}s")

    cascade_gap = abs(rows[0]["gap_pct"])
    decision = {
        "cascade_gap_pct": cascade_gap,
        "spearman_rho": rho,
        "gate_cascade_under_5pct": cascade_gap < 5.0,
        "gate_rho_over_0p7": rho > 0.7,
        "verdict": "pass" if (cascade_gap < 5.0 and rho > 0.7) else "fail",
    }
    print(f"\nDecision: {decision}")

    out = {
        "rows": rows,
        "decision": decision,
        "wall_seconds": wall,
    }
    out_path = Path("experiments/E88_diff_proxy/results/calibrate_ibm01.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
