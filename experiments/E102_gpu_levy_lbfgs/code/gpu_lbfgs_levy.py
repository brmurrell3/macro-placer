"""E102 — GPU L-BFGS gradient descent on Lévy --all output.

Tests whether L-BFGS with strong-Wolfe line search + annealed γ + Lagrangian
overlap penalty can lift cascade plateau placements when run on GPU (A100).

This is the Day-1 spike for PATH C1 (differentiable canonical proxy). It uses
the EXISTING E88 diff_proxy (which has the known smooth-RUDY ≠ canonical-RUDY
mismatch). If THIS lifts canonical proxy: the optimizer can recover even with
imperfect RUDY → great signal. If it fails (likely): confirms we MUST build
the differentiable trace-route RUDY before C1 progresses.

Pipeline per bench:
  1. Load placement from cascade-cached or Lévy --all cached result.
  2. Build DiffProxy on CUDA.
  3. L-BFGS on (cost + λ_ovl·overlap + λ_bnd·boundary), with:
     - γ annealed: 0.0005·cw → 0.00005·cw over the run
     - λ_ovl annealed: 1000 → 10000
     - Periodic canonical eval; rollback to best canonical-checkpoint
     - Strong Wolfe line search (the E88 AdamW-step-blowup fix)
  4. Project to feasible (greedy_macro_legalize).
  5. Final cascade polish via run_cd_adaptive.

Acceptance metric: final canonical < Lévy --all baseline by ≥0.5% on the same bench.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_E88 = _ROOT / "experiments" / "E88_diff_proxy" / "code"
for p in (_E88, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import diff_proxy as diff_proxy_mod
from diff_proxy import DiffProxy
from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


# === CUDA device patches for DPO primitives ===
_orig_rudy = diff_proxy_mod._rudy_congestion


def _cuda_safe_rudy(positions, net_data, port_base, gamma,
                    cell_x_min, cell_x_max, cell_y_min, cell_y_max,
                    grid_h_routes, grid_v_routes, grid_rows, grid_cols):
    device = positions.device
    if positions.numel() == 0:
        return torch.tensor(0.0, device=device)
    all_pos = torch.cat([positions, port_base], dim=0)
    pin_macro_idx = net_data.pin_macro_idx
    pin_offsets = net_data.pin_offsets
    mask = net_data.mask
    pin_pos = all_pos[pin_macro_idx] + pin_offsets
    mask_float = mask.float()
    mask_inv = (1.0 - mask_float)
    x = pin_pos[..., 0] * mask_float + (-1e9) * mask_inv
    y = pin_pos[..., 1] * mask_float + (-1e9) * mask_inv
    x_neg = pin_pos[..., 0] * mask_float + 1e9 * mask_inv
    y_neg = pin_pos[..., 1] * mask_float + 1e9 * mask_inv
    bbox_x_max = gamma * torch.logsumexp(x / gamma, dim=1)
    bbox_y_max = gamma * torch.logsumexp(y / gamma, dim=1)
    bbox_x_min = -gamma * torch.logsumexp(-x_neg / gamma, dim=1)
    bbox_y_min = -gamma * torch.logsumexp(-y_neg / gamma, dim=1)
    bbox_w = torch.clamp(bbox_x_max - bbox_x_min, min=1e-3)
    bbox_h = torch.clamp(bbox_y_max - bbox_y_min, min=1e-3)
    num_nets = pin_macro_idx.shape[0]
    w = net_data.weights
    h_coeff = w / (bbox_w * grid_h_routes)
    v_coeff = w / (bbox_h * grid_v_routes)
    h_cong = torch.zeros(grid_rows, grid_cols, device=device)
    v_cong = torch.zeros(grid_rows, grid_cols, device=device)
    batch_size = 2000
    for b_start in range(0, num_nets, batch_size):
        b_end = min(b_start + batch_size, num_nets)
        b = slice(b_start, b_end)
        x_ol = torch.clamp(
            torch.min(bbox_x_max[b].unsqueeze(1), cell_x_max.unsqueeze(0))
            - torch.max(bbox_x_min[b].unsqueeze(1), cell_x_min.unsqueeze(0)),
            min=0,
        )
        y_ol = torch.clamp(
            torch.min(bbox_y_max[b].unsqueeze(1), cell_y_max.unsqueeze(0))
            - torch.max(bbox_y_min[b].unsqueeze(1), cell_y_min.unsqueeze(0)),
            min=0,
        )
        ol_area = y_ol.unsqueeze(2) * x_ol.unsqueeze(1)  # [batch, R, C]
        h_cong = h_cong + (h_coeff[b].unsqueeze(1).unsqueeze(2) * ol_area).sum(0)
        v_cong = v_cong + (v_coeff[b].unsqueeze(1).unsqueeze(2) * ol_area).sum(0)
    abu_5 = max(1, int(h_cong.numel() * 0.05))
    total = (h_cong + v_cong).flatten()
    top, _ = torch.topk(total, abu_5)
    return top.mean()


# Replace in module so DiffProxy.cost() uses CUDA-safe version
diff_proxy_mod._rudy_congestion = _cuda_safe_rudy
print("[gpu_lbfgs_levy] patched diff_proxy._rudy_congestion for CUDA", flush=True)


def gpu_lbfgs_descent(
    init_state: torch.Tensor,
    benchmark,
    plc,
    *,
    device: str = "cuda",
    n_lbfgs_steps: int = 80,
    history_size: int = 20,
    gamma_frac_start: float = 0.0005,
    gamma_frac_end: float = 0.00005,
    overlap_lambda_start: float = 1000.0,
    overlap_lambda_end: float = 10000.0,
    boundary_lambda: float = 500.0,
    canonical_check_every: int = 5,
    cd_recovery_budget_s: float = 120.0,
    log=None,
) -> Tuple[torch.Tensor, dict]:
    """Run L-BFGS on smooth proxy, periodically check canonical, rollback if regress."""
    if log is None:
        log = lambda s: print(s, flush=True)

    log(f"[gpu-lbfgs] device={device} steps={n_lbfgs_steps}")
    init_canonical = float(compute_proxy_cost(init_state, benchmark, plc)["proxy_cost"])
    log(f"[gpu-lbfgs] init canonical: {init_canonical:.5f}")

    # Position tensor on GPU; gradient enabled.
    pos = init_state.detach().clone().float().to(device).requires_grad_(True)
    t_start = time.time()

    proxy = DiffProxy(benchmark, plc, device=device, gamma_frac=gamma_frac_start)

    # Track best canonical-checkpoint
    best_canon = init_canonical
    best_state = init_state.detach().clone()
    step = [0]
    gamma_growth = (gamma_frac_end / gamma_frac_start) ** (1.0 / max(1, n_lbfgs_steps))
    lambda_growth = (overlap_lambda_end / overlap_lambda_start) ** (1.0 / max(1, n_lbfgs_steps))
    ov_lambda = [overlap_lambda_start]
    gamma_frac = [gamma_frac_start]

    opt = torch.optim.LBFGS(
        [pos], lr=0.5, max_iter=n_lbfgs_steps, history_size=history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-8, tolerance_change=1e-10,
    )

    def closure():
        opt.zero_grad()
        # Update γ on proxy in-place: rebuild expensive bits cheap, just multiplier.
        proxy.gamma = gamma_frac[0] * proxy.cw
        smooth, _parts = proxy.cost(pos, include_congestion=True)
        penalty = proxy.overlap_penalty(pos)
        boundary = proxy.out_of_canvas_penalty(pos)
        canvas_area = proxy.cw * proxy.ch
        loss = smooth + (ov_lambda[0] / canvas_area) * penalty + (boundary_lambda / canvas_area) * boundary
        loss.backward()
        s = step[0]
        if s % canonical_check_every == 0:
            with torch.no_grad():
                pos_cpu = pos.detach().cpu()
                canon = float(compute_proxy_cost(pos_cpu, benchmark, plc)["proxy_cost"])
                ovl = int(compute_overlap_metrics(pos_cpu, benchmark)["overlap_count"])
            log(f"  step={s} loss={float(loss):.5f} smooth={float(smooth):.5f} "
                f"canon={canon:.5f} ovl={ovl} λ_ovl={ov_lambda[0]:.0f} γ_frac={gamma_frac[0]:.5f}")
            nonlocal_best_canon = best_canon if False else None  # python scoping shim
            if canon < best_canon and ovl == 0:
                # Update outer scope via list
                _best_canon[0] = canon
                _best_state[0] = pos.detach().cpu().clone()
                log(f"    >> NEW best canonical {canon:.5f} ovl=0")
        step[0] += 1
        ov_lambda[0] *= lambda_growth
        gamma_frac[0] *= gamma_growth
        return loss

    # Use list trick to mutate from closure
    _best_canon = [init_canonical]
    _best_state = [init_state.detach().clone()]

    try:
        opt.step(closure)
    except Exception as exc:
        log(f"  L-BFGS exception: {exc}")

    descent_wall = time.time() - t_start
    log(f"  L-BFGS descent done in {descent_wall:.1f}s ({step[0]} closures)")

    # Final canonical eval
    post_pos = pos.detach().cpu()
    post_canon = float(compute_proxy_cost(post_pos, benchmark, plc)["proxy_cost"])
    post_ovl = int(compute_overlap_metrics(post_pos, benchmark)["overlap_count"])
    log(f"  final after descent: canon={post_canon:.5f} ovl={post_ovl}")

    # Pick best of {init, best_checkpoint, post_descent}
    candidates = [
        (init_canonical, init_state.detach().clone(), "init", 0),
        (_best_canon[0], _best_state[0], "best_ckpt", 0),
    ]
    if post_ovl == 0:
        candidates.append((post_canon, post_pos.clone(), "post_descent", 0))
    candidates.sort(key=lambda c: c[0])
    chosen_proxy, chosen_state, chosen_label, _ = candidates[0]
    log(f"  pick before legalize: {chosen_label} {chosen_proxy:.5f}")

    legalized, _ = project_overlaps(chosen_state, benchmark)
    leg_proxy = float(compute_proxy_cost(legalized, benchmark, plc)["proxy_cost"])
    leg_ovl = int(compute_overlap_metrics(legalized, benchmark)["overlap_count"])
    if leg_ovl > 0:
        log(f"  project_overlaps left {leg_ovl} residuals; reverting to init")
        legalized = init_state.detach().clone()
        leg_proxy = init_canonical

    # Final CD recovery polish to remove descent artifacts
    log(f"  CD recovery (budget={cd_recovery_budget_s:.0f}s)")
    placement_f64 = legalized.detach().clone().to(torch.float64)
    ev = IncrementalProxyEvaluator(benchmark, plc, placement_f64)
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable = [i for i in range(benchmark.num_macros) if not bool(fixed[i])]
    run_cd_adaptive(
        evaluator=ev, benchmark=benchmark, plc=plc, movable=movable,
        min_time_s=min(30.0, cd_recovery_budget_s * 0.2),
        hard_cap_s=cd_recovery_budget_s,
        patience=3, plateau_threshold=0.001, log_fn=None,
    )
    final_state = ev.placement.detach().clone().to(torch.float32)
    final_state, _ = project_overlaps(final_state, benchmark)
    final_proxy = float(compute_proxy_cost(final_state, benchmark, plc)["proxy_cost"])
    final_ovl = int(compute_overlap_metrics(final_state, benchmark)["overlap_count"])
    total_wall = time.time() - t_start
    log(f"[gpu-lbfgs] FINAL: canon={final_proxy:.5f} ovl={final_ovl} wall={total_wall:.0f}s "
        f"Δ_vs_init={(final_proxy - init_canonical)/init_canonical*100:+.3f}%")

    return final_state, {
        "init_canonical": init_canonical, "final_canonical": final_proxy,
        "best_ckpt_canonical": _best_canon[0],
        "post_descent_canonical": post_canon, "post_descent_ovl": post_ovl,
        "n_closures": step[0], "wall_seconds": total_wall,
        "lift_frac": (init_canonical - final_proxy) / init_canonical,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--init-pt", required=False, help="path to .pt with starting placement")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--cd-recovery", type=float, default=120.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(args.bench)))

    if args.init_pt:
        data = torch.load(args.init_pt, weights_only=False)
        if "placement" in data:
            init = data["placement"]
        elif "polished_placement" in data:
            init = data["polished_placement"]
        else:
            raise SystemExit(f"no placement in {args.init_pt}: {list(data.keys())}")
        init = init.to(torch.float32)
    else:
        # Fall back to cached cascade plateau (E84)
        cached = _ROOT / "experiments" / "E84_cascading_saddle" / "results" / f"cascade_{args.bench}.pt"
        if not cached.exists():
            raise SystemExit(f"no cached cascade for {args.bench}; pass --init-pt")
        data = torch.load(cached, weights_only=False)
        init = (data.get("placement") if "placement" in data else data["polished_placement"]).to(torch.float32)

    result, stats = gpu_lbfgs_descent(
        init, bench, plc, device=args.device,
        n_lbfgs_steps=args.steps, cd_recovery_budget_s=args.cd_recovery,
    )

    out_dir = _HERE.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.bench}_gpu_lbfgs.json"
    out.write_text(json.dumps({**stats, "bench": args.bench, "steps": args.steps}, indent=2))
    print(f"results: {out}")


if __name__ == "__main__":
    main()
