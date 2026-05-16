"""E95 — C1 spike v2: L-BFGS / projected SGD with γ-annealing on diff proxy.

Diagnosis from E88:
  - LSE-HPWL bias (fixed γ=0.0005·canvas) produces non-zero gradient at
    cascade's canonical optimum. AdamW's sign-of-gradient first step
    amplifies that bias into a basin-blowing displacement.

Spike redesign:
  - Anneal γ from γ_start=0.005·canvas → γ_end=1e-4·canvas in 4 stages.
  - Use torch.optim.LBFGS with strong_wolfe line search (variant A) or
    projected SGD with tiny-lr warmup (variant B). Both reject the
    sign-of-gradient first step that killed E88.
  - Stronger overlap and boundary Lagrangian schedules.
  - Honest best-canon tracker requiring 0 overlaps.

4 variants on ibm01: {cascade init, SDF init} × {L-BFGS, projected SGD}.
Gate: any post-legalize canonical proxy ≤ 0.898 (cascade × 1.05).
"""
from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
_E88_DIR = _ROOT / "experiments" / "E88_diff_proxy" / "code"
_E76_DIR = _ROOT / "experiments" / "E76_dreamplace_integration" / "code"
for p in (_ROOT, _E88_DIR, _E76_DIR, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.sdf_init import SDFPlacer
from diff_proxy_v2 import DiffProxyV2, loss_with_penalty
from macro_legalizer import greedy_macro_legalize


# ---------- helpers ----------

def _clamp_to_canvas(positions: torch.Tensor, benchmark) -> torch.Tensor:
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    half = benchmark.macro_sizes / 2.0
    p = positions.clone()
    with torch.no_grad():
        p[:, 0] = p[:, 0].clamp(half[:, 0], cw - half[:, 0])
        p[:, 1] = p[:, 1].clamp(half[:, 1], ch - half[:, 1])
    return p


def _eval_canonical(positions, benchmark, plc):
    pos_clamped = _clamp_to_canvas(positions.detach(), benchmark)
    c = compute_proxy_cost(pos_clamped, benchmark, plc)
    return (
        float(c["proxy_cost"]),
        int(c["overlap_count"]),
        pos_clamped,
    )


def _gamma_schedule(stage_idx: int, num_stages: int,
                    gamma_start: float, gamma_end: float) -> float:
    """Geometric anneal γ from start to end across stages."""
    if num_stages <= 1:
        return gamma_end
    frac = stage_idx / (num_stages - 1)
    log_gamma = math.log(gamma_start) + frac * (math.log(gamma_end) - math.log(gamma_start))
    return math.exp(log_gamma)


def _lambda_schedule(stage_idx: int, num_stages: int,
                     lam_start: float, lam_end: float) -> float:
    frac = stage_idx / max(1, num_stages - 1)
    return lam_start + frac * (lam_end - lam_start)


# ---------- descent variants ----------

def descend_lbfgs(
    proxy: DiffProxyV2,
    benchmark,
    plc,
    init: torch.Tensor,
    *,
    num_stages: int = 4,
    steps_per_stage: int = 25,
    gamma_start_frac: float = 0.005,
    gamma_end_frac: float = 1e-4,
    lam_start: float = 500.0,
    lam_end: float = 10000.0,
    boundary_lam: float = 500.0,
    lbfgs_history: int = 20,
    lbfgs_lr: float = 1.0,
    label: str = "lbfgs",
):
    """L-BFGS with strong Wolfe; γ and λ annealed across stages."""
    pos = init.clone().detach().requires_grad_(True)
    best_canon = float("inf")
    best_pos = init.clone()
    log = []
    t0 = time.time()

    for stage in range(num_stages):
        gamma_frac = _gamma_schedule(stage, num_stages, gamma_start_frac, gamma_end_frac)
        lam = _lambda_schedule(stage, num_stages, lam_start, lam_end)
        proxy.set_gamma_frac(gamma_frac)

        # Fresh L-BFGS per stage (history meaningless across γ jumps).
        opt = torch.optim.LBFGS(
            [pos],
            lr=lbfgs_lr,
            max_iter=steps_per_stage,
            history_size=lbfgs_history,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-8,
            tolerance_change=1e-12,
        )

        def closure():
            opt.zero_grad()
            total, _ = loss_with_penalty(
                proxy, pos, overlap_lambda=lam,
                boundary_lambda=boundary_lam,
            )
            total.backward()
            return total

        try:
            opt.step(closure)
        except Exception as exc:  # L-BFGS can throw on degenerate Wolfe
            log.append({
                "stage": stage, "exception": repr(exc),
                "gamma_frac": gamma_frac, "lam": lam,
            })
            continue

        # Stage-end canonical eval
        canon_v, ovl, pos_clamped = _eval_canonical(pos, benchmark, plc)
        with torch.no_grad():
            total, parts = loss_with_penalty(
                proxy, pos, overlap_lambda=lam,
                boundary_lambda=boundary_lam,
            )
        log.append({
            "stage": stage,
            "gamma_frac": gamma_frac,
            "lam": lam,
            "loss": float(total),
            "smooth": float(parts["smooth_cost"]),
            "ov_norm": float(parts["overlap_area_norm"]),
            "bnd_norm": float(parts["boundary_norm"]),
            "canon": canon_v,
            "ovl": ovl,
        })
        print(f"[{label}] stage={stage} γfrac={gamma_frac:.5f} λ={lam:.0f} "
              f"loss={float(total):.4f} smooth={float(parts['smooth_cost']):.4f} "
              f"canon={canon_v:.4f} ovl={ovl}")
        if ovl == 0 and canon_v < best_canon:
            best_canon = canon_v
            best_pos = pos_clamped.clone()

    final_pos = _clamp_to_canvas(pos, benchmark).detach()
    return final_pos, log, time.time() - t0, best_pos, best_canon


def descend_sgd(
    proxy: DiffProxyV2,
    benchmark,
    plc,
    init: torch.Tensor,
    *,
    num_stages: int = 4,
    steps_per_stage: int = 100,
    gamma_start_frac: float = 0.005,
    gamma_end_frac: float = 1e-4,
    lam_start: float = 500.0,
    lam_end: float = 10000.0,
    boundary_lam: float = 500.0,
    lr_warmup: float = 1e-4,
    lr_peak: float = 5e-3,
    label: str = "sgd",
):
    """Projected SGD with tiny-lr warmup; γ annealed across stages.

    First 50 steps: tiny lr to avoid blowing the basin. Then ramp.
    """
    pos = init.clone().detach().requires_grad_(True)
    best_canon = float("inf")
    best_pos = init.clone()
    log = []
    t0 = time.time()
    total_steps = num_stages * steps_per_stage
    warmup_steps = max(1, total_steps // 8)

    step_global = 0
    for stage in range(num_stages):
        gamma_frac = _gamma_schedule(stage, num_stages, gamma_start_frac, gamma_end_frac)
        lam = _lambda_schedule(stage, num_stages, lam_start, lam_end)
        proxy.set_gamma_frac(gamma_frac)

        for inner_step in range(steps_per_stage):
            # Cosine warmup → peak → cosine decay.
            if step_global < warmup_steps:
                lr_t = lr_warmup + (lr_peak - lr_warmup) * (step_global / warmup_steps)
            else:
                frac = (step_global - warmup_steps) / max(1, total_steps - warmup_steps)
                lr_t = lr_peak * (0.5 * (1 + math.cos(math.pi * frac)))

            total, parts = loss_with_penalty(
                proxy, pos, overlap_lambda=lam,
                boundary_lambda=boundary_lam,
            )
            grad = torch.autograd.grad(total, pos)[0]
            with torch.no_grad():
                # Plain SGD step (no momentum, to avoid AdamW-style sign reduction).
                pos.sub_(lr_t * grad)
                # Project to canvas.
                cw = float(benchmark.canvas_width)
                ch = float(benchmark.canvas_height)
                half = benchmark.macro_sizes / 2.0
                pos[:, 0].clamp_(half[:, 0], cw - half[:, 0])
                pos[:, 1].clamp_(half[:, 1], ch - half[:, 1])
            step_global += 1

        canon_v, ovl, pos_clamped = _eval_canonical(pos, benchmark, plc)
        log.append({
            "stage": stage,
            "gamma_frac": gamma_frac,
            "lam": lam,
            "lr_final": lr_t,
            "loss": float(total),
            "smooth": float(parts["smooth_cost"]),
            "ov_norm": float(parts["overlap_area_norm"]),
            "bnd_norm": float(parts["boundary_norm"]),
            "canon": canon_v,
            "ovl": ovl,
        })
        print(f"[{label}] stage={stage} γfrac={gamma_frac:.5f} λ={lam:.0f} lr={lr_t:.4g} "
              f"loss={float(total):.4f} smooth={float(parts['smooth_cost']):.4f} "
              f"canon={canon_v:.4f} ovl={ovl}")
        if ovl == 0 and canon_v < best_canon:
            best_canon = canon_v
            best_pos = pos_clamped.clone()

    final_pos = _clamp_to_canvas(pos, benchmark).detach()
    return final_pos, log, time.time() - t0, best_pos, best_canon


# ---------- evaluation ----------

def evaluate_with_legalize(pos, benchmark, plc, *, label):
    canon_pre = compute_proxy_cost(pos, benchmark, plc)
    t0 = time.time()
    legal, leg_stats = greedy_macro_legalize(pos, benchmark, verbose=False)
    wall = time.time() - t0
    canon_post = compute_proxy_cost(legal, benchmark, plc)
    print(
        f"[{label}] pre-leg canon={canon_pre['proxy_cost']:.4f} "
        f"ovl={int(canon_pre['overlap_count'])} | "
        f"post-leg canon={canon_post['proxy_cost']:.4f} "
        f"ovl={int(canon_post['overlap_count'])} | leg_wall={wall:.1f}s"
    )
    return {
        "pre_legalize": {
            "proxy_cost": float(canon_pre["proxy_cost"]),
            "overlap_count": int(canon_pre["overlap_count"]),
        },
        "post_legalize": {
            "proxy_cost": float(canon_post["proxy_cost"]),
            "overlap_count": int(canon_post["overlap_count"]),
        },
        "legalize_stats": leg_stats,
        "legalize_wall": wall,
        "placement": legal,
    }


def run_variant(name, descent_fn, proxy, benchmark, plc, init, **kwargs):
    try:
        print(f"\n=== RUN {name} ===")
        pos_final, log, wall, best_pos, best_canon = descent_fn(
            proxy, benchmark, plc, init, label=name, **kwargs
        )
    except Exception as exc:
        print(f"[{name}] DESCENT FAILED: {exc}")
        traceback.print_exc()
        return {
            "name": name,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }

    eval_last = evaluate_with_legalize(pos_final, benchmark, plc, label=f"{name}_last")
    eval_best = evaluate_with_legalize(best_pos, benchmark, plc, label=f"{name}_best")
    pick = min(
        eval_last["post_legalize"]["proxy_cost"],
        eval_best["post_legalize"]["proxy_cost"],
    )
    return {
        "name": name,
        "descent_log": log,
        "descent_wall": wall,
        "best_canon_during_descent": best_canon,
        "result_last_step": {
            "pre_legalize": eval_last["pre_legalize"],
            "post_legalize": eval_last["post_legalize"],
        },
        "result_best_step": {
            "pre_legalize": eval_best["pre_legalize"],
            "post_legalize": eval_best["post_legalize"],
        },
        "best_post_legalize": pick,
        "best_placement": eval_best["placement"] if eval_best["post_legalize"]["proxy_cost"] <= eval_last["post_legalize"]["proxy_cost"] else eval_last["placement"],
    }


def main():
    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    proxy = DiffProxyV2(benchmark, plc, device="cpu", gamma_frac=0.005)

    cached = torch.load(
        "experiments/E84_cascading_saddle/results/cascade_ibm01.pt",
        weights_only=False,
        map_location="cpu",
    )
    init_cascade = cached["placement"].clone().float()

    sdf = SDFPlacer(seed=42, n_iters=500)
    init_sdf = sdf.place(benchmark).float()
    assert init_sdf.shape == init_cascade.shape

    cascade_baseline = compute_proxy_cost(init_cascade, benchmark, plc)
    sdf_baseline = compute_proxy_cost(init_sdf, benchmark, plc)
    print(f"[baseline] cascade init: canon={cascade_baseline['proxy_cost']:.5f} "
          f"ovl={int(cascade_baseline['overlap_count'])}")
    print(f"[baseline] sdf init    : canon={sdf_baseline['proxy_cost']:.5f} "
          f"ovl={int(sdf_baseline['overlap_count'])}")

    variants = []
    for init_name, init in [("cascade", init_cascade), ("sdf", init_sdf)]:
        for opt_name, fn in [("lbfgs", descend_lbfgs), ("sgd", descend_sgd)]:
            variants.append((f"{init_name}_{opt_name}", fn, init))

    results = {}
    for name, fn, init in variants:
        r = run_variant(name, fn, proxy, benchmark, plc, init.clone())
        # Save best placement for the variant.
        if "best_placement" in r:
            torch.save(
                {"placement": r["best_placement"], "bench_name": "ibm01", "label": name},
                f"experiments/E95_diff_proxy_v2/results/spike_v2_{name}.pt",
            )
            del r["best_placement"]
        results[name] = r

    GATE = 0.898
    valid_picks = [
        (name, r["best_post_legalize"])
        for name, r in results.items()
        if "best_post_legalize" in r
    ]
    if valid_picks:
        winner_name, winner_proxy = min(valid_picks, key=lambda x: x[1])
        verdict = "pass" if winner_proxy <= GATE else "fail"
    else:
        winner_name = None
        winner_proxy = float("inf")
        verdict = "all_errored"

    decision = {
        "gate": GATE,
        "best_proxy": winner_proxy,
        "winner_variant": winner_name,
        "verdict": verdict,
        "cascade_baseline": float(cascade_baseline["proxy_cost"]),
        "sdf_baseline": float(sdf_baseline["proxy_cost"]),
    }
    print(f"\n=== SPIKE V2 GATE: {GATE:.4f} ===")
    print(f"Winner: {winner_name} → {winner_proxy:.4f} → {verdict}")

    out_path = Path("experiments/E95_diff_proxy_v2/results/spike_v2_ibm01.json")
    out_path.write_text(json.dumps({"runs": results, "decision": decision}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
