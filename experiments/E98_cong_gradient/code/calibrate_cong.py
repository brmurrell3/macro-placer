"""E98 — calibrate differentiable RUDY congestion vs canonical.

Track 3 spike. Before committing to congestion-gradient polish, we need to know:
  1. Does smooth `_rudy_congestion` track canonical `plc.get_congestion_cost()` in
     ABSOLUTE value? (memory says 3-4× divergence on hard benches → likely no.)
  2. Does it track in DIRECTION? i.e., if we perturb the placement and recompute
     both, do they correlate? (this is the actual signal we need for gradient descent.)
  3. Does its GRADIENT (autograd) point in the same direction as the canonical
     finite-difference gradient on a small sample of macros?

Test protocol per bench (ibm10/12/14/17):
  A. Load cached cascade plateau (.pt).
  B. Compute canonical congestion + smooth congestion at the plateau.
  C. Perturb 20 macros by small random offsets, recompute both.
  D. Report:
     - absolute mismatch (smooth - canonical) / canonical
     - rank correlation between perturbation effect on smooth vs canonical
     - sign-agreement of finite-difference congestion grad on a 10-macro subset

If rank correlation > 0.7: green light to build the gradient polish.
If rank correlation < 0.5: smooth RUDY is too detached; need per-net trace-route.

Test wall: ~2-3 min per bench locally.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


def calibrate_bench(bench_name: str, n_perturbs: int = 16, perturb_scale_frac: float = 0.02,
                    seed: int = 42, log=print) -> dict:
    log(f"\n=== calibrate {bench_name} ===")
    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))
    cw, ch = float(bench.canvas_width), float(bench.canvas_height)

    cascade_pt = _E84 / f"cascade_{bench_name}.pt"
    if not cascade_pt.exists():
        log(f"  no cached cascade for {bench_name}; skip")
        return {"bench": bench_name, "skipped": True}
    data = torch.load(cascade_pt, weights_only=False)
    start = data["placement"].to(torch.float32) if "placement" in data else data["polished_placement"].to(torch.float32)
    n_hard = bench.num_hard_macros

    diff = DiffProxy(bench, plc, device="cpu")

    # === A. Baseline values at plateau ===
    canonical = compute_proxy_cost(start, bench, plc)
    can_cong = float(canonical["congestion_cost"])
    can_total = float(canonical["proxy_cost"])

    with torch.no_grad():
        smooth_total, parts = diff.cost(start, include_congestion=True)
    sm_cong = float(parts["cong"])

    log(f"  canonical: total={can_total:.4f}  congestion={can_cong:.4f}")
    log(f"  smooth:    total={float(smooth_total):.4f}  congestion={sm_cong:.4f}")
    log(f"  cong abs mismatch: {(sm_cong - can_cong) / max(can_cong, 1e-6) * 100:+.1f}%")

    # === B. Perturbation correlation ===
    rng = np.random.default_rng(seed)
    fixed = bench.macro_fixed.cpu().numpy()
    movable_hard = [i for i in range(n_hard) if not bool(fixed[i])]
    can_deltas, sm_deltas = [], []
    scale = perturb_scale_frac * cw  # e.g., 2% of canvas

    log(f"  {n_perturbs} perturbations of scale {scale:.2f} (2% canvas)...")
    base_can = can_cong
    base_sm = sm_cong
    for k in range(n_perturbs):
        perturb = start.clone()
        # Move 5 random movable hard macros by Gaussian offsets
        targets = rng.choice(movable_hard, size=min(5, len(movable_hard)), replace=False)
        for t in targets:
            dx = float(rng.normal(0.0, scale))
            dy = float(rng.normal(0.0, scale))
            perturb[t, 0] = torch.clamp(perturb[t, 0] + dx,
                                         bench.macro_sizes[t, 0] / 2,
                                         cw - bench.macro_sizes[t, 0] / 2)
            perturb[t, 1] = torch.clamp(perturb[t, 1] + dy,
                                         bench.macro_sizes[t, 1] / 2,
                                         ch - bench.macro_sizes[t, 1] / 2)
        c_can = float(compute_proxy_cost(perturb, bench, plc)["congestion_cost"])
        with torch.no_grad():
            _, p = diff.cost(perturb, include_congestion=True)
            c_sm = float(p["cong"])
        can_deltas.append(c_can - base_can)
        sm_deltas.append(c_sm - base_sm)

    can_arr = np.array(can_deltas)
    sm_arr = np.array(sm_deltas)
    if len(can_arr) > 2:
        pearson = float(np.corrcoef(can_arr, sm_arr)[0, 1])
        # Rank correlation (Spearman)
        rk_c = np.argsort(np.argsort(can_arr))
        rk_s = np.argsort(np.argsort(sm_arr))
        spearman = float(np.corrcoef(rk_c, rk_s)[0, 1])
        sign_agree = float(np.mean(np.sign(can_arr) == np.sign(sm_arr)))
    else:
        pearson = spearman = sign_agree = float('nan')

    log(f"  Δcong perturbation correlation:")
    log(f"    Pearson  ρ = {pearson:+.3f}")
    log(f"    Spearman ρ = {spearman:+.3f}")
    log(f"    sign-agree = {sign_agree:.2%}")

    # === C. Finite-difference gradient on a sample ===
    # Take 5 random movable hard macros. For each, compute ∂cong/∂x_i, ∂cong/∂y_i
    # via finite difference on canonical, and via autograd on smooth. Compare directions.
    h = 0.005 * cw  # 0.5% canvas step
    sample_idx = list(rng.choice(movable_hard, size=min(5, len(movable_hard)), replace=False))
    log(f"  finite-difference vs autograd on {len(sample_idx)} macros (step h={h:.3f})...")

    # Autograd of smooth congestion
    pos_grad = start.clone().detach().requires_grad_(True)
    _, parts_g = diff.cost(pos_grad, include_congestion=True)
    cong_g = parts_g["cong"].requires_grad_(True) if not parts_g["cong"].requires_grad else parts_g["cong"]
    # Re-run forward with congestion-only loss to get autograd gradient
    _, parts2 = diff.cost(pos_grad, include_congestion=True)
    # cost returns smooth_total = wl + 0.5*density + 0.5*cong. We want grad of cong alone.
    # Re-derive: redo cost with include_congestion=False to get wl+density part, subtract.
    # Simpler: re-call _rudy_congestion via a thin wrapper. But too involved here —
    # use 0.5*cong from total. We can extract: total grad = ∇wl + 0.5*∇density + 0.5*∇cong.
    # Just use 0.5*∇cong = ∇(0.5*cong) = ∇(total - wl - 0.5*density).
    pos_grad2 = start.clone().detach().requires_grad_(True)
    smooth_no_cong, _ = diff.cost(pos_grad2, include_congestion=False)
    smooth_no_cong.backward()
    grad_no_cong = pos_grad2.grad.detach().clone()

    pos_grad3 = start.clone().detach().requires_grad_(True)
    smooth_full, _ = diff.cost(pos_grad3, include_congestion=True)
    smooth_full.backward()
    grad_full = pos_grad3.grad.detach().clone()
    # Gradient of 0.5 * cong = grad_full - grad_no_cong
    grad_cong = (grad_full - grad_no_cong).numpy() * 2.0  # times 2 to get ∇cong not ∇(0.5*cong)

    # Finite difference of canonical congestion
    base_can_cong = can_cong
    fd_dirs = []
    sm_dirs = []
    for idx in sample_idx:
        for axis in (0, 1):
            perturb = start.clone()
            perturb[idx, axis] += h
            c_can_plus = float(compute_proxy_cost(perturb, bench, plc)["congestion_cost"])
            fd_g = (c_can_plus - base_can_cong) / h
            sm_g = float(grad_cong[idx, axis])
            fd_dirs.append(fd_g)
            sm_dirs.append(sm_g)
    fd_arr = np.array(fd_dirs)
    sm_arr_g = np.array(sm_dirs)
    if len(fd_arr) > 2:
        grad_pearson = float(np.corrcoef(fd_arr, sm_arr_g)[0, 1])
        grad_sign_agree = float(np.mean(np.sign(fd_arr) == np.sign(sm_arr_g)))
    else:
        grad_pearson = grad_sign_agree = float('nan')

    log(f"  ∂cong/∂x correlation (finite-diff canonical vs autograd smooth):")
    log(f"    Pearson    ρ = {grad_pearson:+.3f}")
    log(f"    sign-agree   = {grad_sign_agree:.2%}")

    # === Verdict ===
    # Green if either Spearman > 0.7 OR grad_sign_agree > 0.75
    # (we just need the descent direction to be roughly right)
    green = (spearman > 0.7) or (grad_sign_agree > 0.75)
    verdict = "GREEN — congestion-gradient polish viable" if green else "RED — smooth RUDY too detached"
    log(f"  VERDICT: {verdict}")

    return {
        "bench": bench_name,
        "canonical_cong": can_cong,
        "smooth_cong": sm_cong,
        "abs_mismatch_pct": (sm_cong - can_cong) / max(can_cong, 1e-6) * 100,
        "perturb_pearson": pearson,
        "perturb_spearman": spearman,
        "perturb_sign_agree": sign_agree,
        "grad_pearson": grad_pearson,
        "grad_sign_agree": grad_sign_agree,
        "green_light": bool(green),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", nargs="+", default=["ibm10", "ibm12", "ibm14", "ibm17"])
    ap.add_argument("--n-perturbs", type=int, default=16)
    ap.add_argument("--perturb-scale-frac", type=float, default=0.02)
    args = ap.parse_args()

    results = []
    for b in args.benches:
        try:
            r = calibrate_bench(b, n_perturbs=args.n_perturbs,
                                perturb_scale_frac=args.perturb_scale_frac)
            results.append(r)
        except Exception as exc:
            print(f"  {b}: ERROR {exc}")
            results.append({"bench": b, "error": repr(exc)})

    out_dir = _HERE.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibration.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")

    print("\n=== SUMMARY ===")
    for r in results:
        if r.get("error") or r.get("skipped"):
            print(f"  {r['bench']}: skipped/errored")
            continue
        light = "🟢" if r["green_light"] else "🔴"
        print(f"  {r['bench']:6s} {light}  abs_mismatch={r['abs_mismatch_pct']:+6.1f}%  "
              f"perturb_ρ={r['perturb_spearman']:+.2f}  grad_sign={r['grad_sign_agree']:.0%}")


if __name__ == "__main__":
    main()
