"""E98 — congestion-gradient polish: L-BFGS on smooth congestion only.

Track 3 implementation. Optimizes ONLY the congestion term of the smooth proxy
(plus overlap + boundary Lagrangians), starting from a cascade-quality plateau.

Key design choices that avoid E88/E95 failure modes:
  1. Optimize congestion ONLY (not full smooth proxy with WL). WL gradient was
     the dominant E88 destabilizer per memory; congestion direction alone is
     less prone to wandering.
  2. Tiny initial lr. L-BFGS with strong Wolfe line search adapts step size
     and rejects basin-blowing updates.
  3. Frequent overlap projection (every N steps) keeps placement feasible.
  4. Cascade-saddle-equivalent recovery polish (CD) after gradient descent
     to restore WL/density and re-tune to canonical valley.

Pipeline:
  cascade plateau → congestion-gradient descent (~50-200 L-BFGS steps) →
  project_overlaps → CD-adaptive recovery polish (~60-180s) → final.

Test on cached ibm10/12/14/17 cascade plateaus locally first.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_E88 = _ROOT / "experiments" / "E88_diff_proxy" / "code"
_E84 = _ROOT / "experiments" / "E84_cascading_saddle" / "results"
for p in (_E88, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from diff_proxy import DiffProxy
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def cong_only_loss(
    proxy: DiffProxy,
    positions: torch.Tensor,
    overlap_lambda: float,
    boundary_lambda: float,
) -> Tuple[torch.Tensor, dict]:
    """Differentiable: congestion + overlap penalty + boundary penalty.
    Returns (loss, parts dict)."""
    _, parts = proxy.cost(positions, include_congestion=True)
    cong = parts["cong"] if isinstance(parts["cong"], torch.Tensor) else \
        torch.tensor(parts["cong"], device=positions.device, dtype=positions.dtype)
    # parts["cong"] is detached above. Re-run to get differentiable:
    _, parts_diff = proxy.cost(positions, include_congestion=True)
    # Hmm, DiffProxy.cost returns ".detach()'d" parts but the total IS differentiable.
    # The trick: total = wl + 0.5*density + 0.5*cong. We want grad of cong only.
    # We'll do: loss = total - wl_only - density_only (computed separately).
    total_full, _ = proxy.cost(positions, include_congestion=True)
    total_no_cong, _ = proxy.cost(positions, include_congestion=False)
    cong_diff = (total_full - total_no_cong) * 2.0  # 0.5*cong → cong
    penalty = proxy.overlap_penalty(positions)
    boundary = proxy.out_of_canvas_penalty(positions)
    canvas_area = proxy.cw * proxy.ch
    penalty_norm = penalty / canvas_area
    boundary_norm = boundary / canvas_area
    loss = cong_diff + overlap_lambda * penalty_norm + boundary_lambda * boundary_norm
    return loss, {
        "cong": cong_diff.detach(),
        "overlap_norm": penalty_norm.detach(),
        "boundary_norm": boundary_norm.detach(),
    }


def cong_gradient_polish(
    init_state: torch.Tensor,
    benchmark,
    plc,
    *,
    n_lbfgs_steps: int = 100,
    lbfgs_lr: float = 0.5,
    lbfgs_history: int = 10,
    overlap_lambda_start: float = 1000.0,
    overlap_lambda_end: float = 10000.0,
    boundary_lambda: float = 500.0,
    project_every_n: int = 25,
    cd_recovery_budget_s: float = 90.0,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, dict]:
    """Congestion-gradient polish on a feasible placement.

    Returns (final_placement, stats).
    """
    if log is None:
        log = lambda s: print(s, flush=True)

    init_proxy = float(compute_proxy_cost(init_state, benchmark, plc)["proxy_cost"])
    init_cong = float(compute_proxy_cost(init_state, benchmark, plc)["congestion_cost"])
    log(f"[cong-pol] init: proxy={init_proxy:.5f} congestion={init_cong:.5f}")

    proxy = DiffProxy(benchmark, plc, device="cpu")
    pos = init_state.detach().clone().float().requires_grad_(True)
    t_start = time.time()

    # L-BFGS descent on congestion + overlap + boundary
    opt = torch.optim.LBFGS(
        [pos], lr=lbfgs_lr, max_iter=n_lbfgs_steps,
        history_size=lbfgs_history, line_search_fn="strong_wolfe",
        tolerance_grad=1e-8, tolerance_change=1e-10,
    )

    best_canon = init_proxy
    best_state = init_state.detach().clone()
    step = [0]  # mutable closure counter
    ov_lambda = [overlap_lambda_start]
    lambda_growth = (overlap_lambda_end / overlap_lambda_start) ** (1.0 / max(1, n_lbfgs_steps))

    def closure():
        opt.zero_grad()
        loss, parts = cong_only_loss(proxy, pos, ov_lambda[0], boundary_lambda)
        loss.backward()
        if step[0] % 5 == 0:
            with torch.no_grad():
                proxy_curr = float(compute_proxy_cost(pos.detach(), benchmark, plc)["proxy_cost"])
            log(f"  [cong-pol] step={step[0]} loss={float(loss):.5f} "
                f"cong={float(parts['cong']):.4f} ovl_n={float(parts['overlap_norm']):.4f} "
                f"bnd_n={float(parts['boundary_norm']):.4f} canon={proxy_curr:.5f} "
                f"λ_ovl={ov_lambda[0]:.0f}")
        step[0] += 1
        ov_lambda[0] *= lambda_growth
        return loss

    try:
        opt.step(closure)
    except Exception as exc:
        log(f"  [cong-pol] L-BFGS exception: {exc}")

    descent_wall = time.time() - t_start
    log(f"  [cong-pol] descent done in {descent_wall:.1f}s after {step[0]} closures")

    # Project to feasible (zero overlap)
    with torch.no_grad():
        post_descent = pos.detach().clone().float()
    proxy_pre_project = float(compute_proxy_cost(post_descent, benchmark, plc)["proxy_cost"])
    cong_pre_project = float(compute_proxy_cost(post_descent, benchmark, plc)["congestion_cost"])
    ovl_pre = int(compute_overlap_metrics(post_descent, benchmark)["overlap_count"])
    log(f"  [cong-pol] post-descent: canon={proxy_pre_project:.5f} cong={cong_pre_project:.5f} ovl={ovl_pre}")

    legalized, _ = project_overlaps(post_descent, benchmark)
    proxy_legal = float(compute_proxy_cost(legalized, benchmark, plc)["proxy_cost"])
    ovl_legal = int(compute_overlap_metrics(legalized, benchmark)["overlap_count"])
    log(f"  [cong-pol] post-project: canon={proxy_legal:.5f} ovl={ovl_legal}")

    if ovl_legal > 0:
        log(f"  [cong-pol] WARNING: {ovl_legal} residual overlaps after project_overlaps; "
            f"falling back to init")
        legalized = init_state.detach().clone()
        proxy_legal = init_proxy

    # CD recovery polish (restore WL/density basin without losing congestion gain)
    log(f"  [cong-pol] CD recovery (budget={cd_recovery_budget_s:.0f}s)")
    placement_f64 = legalized.detach().clone().to(torch.float64)
    ev = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable = [i for i in range(benchmark.num_macros) if not bool(fixed[i])]
    run_cd_adaptive(
        evaluator=ev, benchmark=benchmark, plc=plc, movable=movable,
        min_time_s=min(20.0, cd_recovery_budget_s * 0.2),
        hard_cap_s=cd_recovery_budget_s,
        patience=3, plateau_threshold=0.001, log_fn=None,
    )
    final_state = ev.placement.detach().clone().to(torch.float32)
    final_state, _ = project_overlaps(final_state, benchmark)
    final_proxy = float(compute_proxy_cost(final_state, benchmark, plc)["proxy_cost"])
    final_cong = float(compute_proxy_cost(final_state, benchmark, plc)["congestion_cost"])
    final_ovl = int(compute_overlap_metrics(final_state, benchmark)["overlap_count"])
    total_wall = time.time() - t_start
    log(f"[cong-pol] FINAL: canon={final_proxy:.5f} cong={final_cong:.5f} ovl={final_ovl} wall={total_wall:.1f}s")

    return final_state, {
        "init_proxy": init_proxy, "init_cong": init_cong,
        "post_descent_proxy": proxy_pre_project, "post_descent_cong": cong_pre_project,
        "post_descent_ovl": ovl_pre,
        "post_project_proxy": proxy_legal, "post_project_ovl": ovl_legal,
        "final_proxy": final_proxy, "final_cong": final_cong, "final_ovl": final_ovl,
        "wall_s": total_wall, "n_closures": step[0],
        "improvement": init_proxy - final_proxy,
        "improvement_frac": (init_proxy - final_proxy) / init_proxy,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--start-pt", help="path to .pt with 'placement' key (cascade plateau)")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--lambda-start", type=float, default=1000.0)
    ap.add_argument("--lambda-end", type=float, default=10000.0)
    ap.add_argument("--recovery-cd", type=float, default=90.0)
    args = ap.parse_args()

    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(args.bench)))
    if args.start_pt:
        data = torch.load(args.start_pt, weights_only=False)
        start = (data["placement"] if "placement" in data else data["polished_placement"]).to(torch.float32)
    else:
        cached = _E84 / f"cascade_{args.bench}.pt"
        if not cached.exists():
            raise SystemExit(f"no cached cascade for {args.bench}; pass --start-pt")
        data = torch.load(cached, weights_only=False)
        start = (data["placement"] if "placement" in data else data["polished_placement"]).to(torch.float32)

    result, stats = cong_gradient_polish(
        start, bench, plc,
        n_lbfgs_steps=args.steps, lbfgs_lr=args.lr,
        overlap_lambda_start=args.lambda_start,
        overlap_lambda_end=args.lambda_end,
        cd_recovery_budget_s=args.recovery_cd,
    )

    out_dir = _HERE.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.bench}_cong_polish_steps{args.steps}_lr{args.lr}.json"
    out_path.write_text(json.dumps({**stats, "bench": args.bench, "steps": args.steps, "lr": args.lr}, indent=2))
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
