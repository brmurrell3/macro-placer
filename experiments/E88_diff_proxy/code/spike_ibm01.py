"""E88 — C1 spike: AdamW on differentiable proxy, then legalize, then canonical eval.

Tests two inits on ibm01:
  (A) cascade-cached placement (canonical 0.84528) — does descent IMPROVE the basin?
  (B) SDFPlacer init                                — does descent reach a competitive
                                                       basin from a clean start?

For each: run AdamW for N steps with ramped overlap-penalty Lagrangian, snap
to feasible with greedy_macro_legalize, evaluate canonical proxy.

Spike gate: best canonical ibm01 proxy ≤ 0.898 (cascade init 0.85527 × 1.05).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_ROOT / "experiments" / "E76_dreamplace_integration" / "code"))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.sdf_init import SDFPlacer
from diff_proxy import DiffProxy, loss_with_penalty
from macro_legalizer import greedy_macro_legalize


def _clamp_to_canvas(positions: torch.Tensor, benchmark) -> torch.Tensor:
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    half = benchmark.macro_sizes / 2.0
    p = positions.clone()
    with torch.no_grad():
        p[:, 0] = p[:, 0].clamp(half[:, 0], cw - half[:, 0])
        p[:, 1] = p[:, 1].clamp(half[:, 1], ch - half[:, 1])
    return p


def descent(
    proxy: DiffProxy,
    benchmark,
    plc,
    init: torch.Tensor,
    *,
    steps: int = 400,
    lr: float = 0.5,
    lambda_start: float = 100.0,
    lambda_end: float = 1000.0,
    boundary_lambda: float = 100.0,
    eval_every: int = 50,
    label: str = "",
):
    """AdamW on positions with ramped overlap-penalty + always-on boundary penalty.

    Tracks best canonical-proxy placement across the trajectory (legalized inline
    every eval_every steps) and returns it. AdamW can leave good basins; pre-leg
    canonical eval lets us pick the best step rather than the last step.
    """
    pos = init.clone().detach().requires_grad_(True)
    opt = torch.optim.AdamW([pos], lr=lr, weight_decay=0.0)
    log = []
    best_canon = float("inf")
    best_pos = init.clone()
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad()
        frac = step / max(1, steps - 1)
        lam = lambda_start + (lambda_end - lambda_start) * frac
        total, parts = loss_with_penalty(
            proxy, pos, overlap_lambda=lam, boundary_lambda=boundary_lambda,
        )
        total.backward()
        opt.step()
        if step % eval_every == 0 or step == steps - 1:
            with torch.no_grad():
                pos_clamped = _clamp_to_canvas(pos, benchmark)
                canon = compute_proxy_cost(pos_clamped.detach(), benchmark, plc)
                canon_v = float(canon["proxy_cost"])
                if canon_v < best_canon and int(canon["overlap_count"]) == 0:
                    best_canon = canon_v
                    best_pos = pos_clamped.detach().clone()
                msg = (
                    f"[{label}] step={step:4d} loss={float(total):.5f} "
                    f"smooth={float(parts['smooth_cost']):.5f} "
                    f"ov_norm={float(parts['overlap_area_norm']):.5f} "
                    f"bnd_norm={float(parts['boundary_norm']):.5f} "
                    f"lam={lam:.1f} "
                    f"canon={canon_v:.5f} ovl={int(canon['overlap_count'])} "
                    f"best_canon={best_canon:.5f}"
                )
                print(msg)
                log.append({
                    "step": step,
                    "loss": float(total),
                    "smooth": float(parts["smooth_cost"]),
                    "overlap_area_norm": float(parts["overlap_area_norm"]),
                    "boundary_norm": float(parts["boundary_norm"]),
                    "lam": lam,
                    "canon_proxy": canon_v,
                    "canon_overlap_count": int(canon["overlap_count"]),
                    "canon_overlap_area": float(canon["total_overlap_area"]),
                    "best_canon_so_far": best_canon,
                })
    wall = time.time() - t0
    return _clamp_to_canvas(pos, benchmark).detach(), log, wall, best_pos, best_canon


def evaluate_with_legalize(pos: torch.Tensor, benchmark, plc, *, label: str):
    canon_pre = compute_proxy_cost(pos, benchmark, plc)
    t0 = time.time()
    legal, leg_stats = greedy_macro_legalize(pos, benchmark, verbose=False)
    wall = time.time() - t0
    canon_post = compute_proxy_cost(legal, benchmark, plc)
    print(
        f"[{label}] pre-leg: canon={canon_pre['proxy_cost']:.5f} "
        f"ovl={int(canon_pre['overlap_count'])} | "
        f"post-leg: canon={canon_post['proxy_cost']:.5f} "
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


def main():
    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    proxy = DiffProxy(benchmark, plc, device="cpu")

    cached = torch.load(
        "experiments/E84_cascading_saddle/results/cascade_ibm01.pt",
        weights_only=False,
        map_location="cpu",
    )
    init_cascade = cached["placement"].clone().float()

    sdf = SDFPlacer(seed=42, n_iters=500)
    init_sdf = sdf.place(benchmark).float()
    assert init_sdf.shape == init_cascade.shape, (
        f"SDF init shape {tuple(init_sdf.shape)} != cascade init {tuple(init_cascade.shape)}"
    )
    print("[init_sdf] full SDFPlacer.place() output (hard + soft).")

    cascade_baseline = compute_proxy_cost(init_cascade, benchmark, plc)
    sdf_baseline = compute_proxy_cost(init_sdf, benchmark, plc)
    print(
        f"[baseline] cascade init: canon={cascade_baseline['proxy_cost']:.5f} "
        f"ovl={int(cascade_baseline['overlap_count'])}"
    )
    print(
        f"[baseline] sdf init    : canon={sdf_baseline['proxy_cost']:.5f} "
        f"ovl={int(sdf_baseline['overlap_count'])}"
    )

    runs = {}

    print("\n=== RUN A: descent from cascade-cached init ===")
    pos_a, log_a, wall_a, best_pos_a, best_canon_a = descent(
        proxy, benchmark, plc, init_cascade,
        steps=400, lr=0.5,
        lambda_start=100.0, lambda_end=2000.0,
        boundary_lambda=500.0,
        eval_every=25,
        label="A_cascade",
    )
    print(f"[A_cascade] best canon during descent: {best_canon_a:.5f}")
    eval_a_last = evaluate_with_legalize(pos_a, benchmark, plc, label="A_cascade_last")
    eval_a_best = evaluate_with_legalize(best_pos_a, benchmark, plc, label="A_cascade_best")
    pick_a = min(
        eval_a_last["post_legalize"]["proxy_cost"],
        eval_a_best["post_legalize"]["proxy_cost"],
    )
    runs["A_cascade"] = {
        "descent_log": log_a,
        "descent_wall": wall_a,
        "best_canon_during_descent": best_canon_a,
        "result_last_step": {
            "pre_legalize": eval_a_last["pre_legalize"],
            "post_legalize": eval_a_last["post_legalize"],
        },
        "result_best_step": {
            "pre_legalize": eval_a_best["pre_legalize"],
            "post_legalize": eval_a_best["post_legalize"],
        },
        "baseline_canon": float(cascade_baseline["proxy_cost"]),
    }
    torch.save(
        {"placement": eval_a_best["placement"], "bench_name": "ibm01", "label": "A_cascade_best"},
        "experiments/E88_diff_proxy/results/spike_A_cascade.pt",
    )

    print("\n=== RUN B: descent from SDF init ===")
    pos_b, log_b, wall_b, best_pos_b, best_canon_b = descent(
        proxy, benchmark, plc, init_sdf,
        steps=600, lr=1.0,
        lambda_start=100.0, lambda_end=5000.0,
        boundary_lambda=500.0,
        eval_every=25,
        label="B_sdf",
    )
    print(f"[B_sdf] best canon during descent: {best_canon_b:.5f}")
    eval_b_last = evaluate_with_legalize(pos_b, benchmark, plc, label="B_sdf_last")
    eval_b_best = evaluate_with_legalize(best_pos_b, benchmark, plc, label="B_sdf_best")
    pick_b = min(
        eval_b_last["post_legalize"]["proxy_cost"],
        eval_b_best["post_legalize"]["proxy_cost"],
    )
    runs["B_sdf"] = {
        "descent_log": log_b,
        "descent_wall": wall_b,
        "best_canon_during_descent": best_canon_b,
        "result_last_step": {
            "pre_legalize": eval_b_last["pre_legalize"],
            "post_legalize": eval_b_last["post_legalize"],
        },
        "result_best_step": {
            "pre_legalize": eval_b_best["pre_legalize"],
            "post_legalize": eval_b_best["post_legalize"],
        },
        "baseline_canon": float(sdf_baseline["proxy_cost"]),
    }
    torch.save(
        {"placement": eval_b_best["placement"], "bench_name": "ibm01", "label": "B_sdf_best"},
        "experiments/E88_diff_proxy/results/spike_B_sdf.pt",
    )

    GATE = 0.898  # cascade init 0.85527 × 1.05
    best_proxy = min(pick_a, pick_b)
    verdict = "pass" if best_proxy <= GATE else "fail"
    print(f"\n=== SPIKE GATE: {GATE:.4f} (cascade init × 1.05) ===")
    print(f"Best post-legalize canonical proxy: {best_proxy:.5f} → {verdict}")

    decision = {
        "gate": GATE,
        "best_proxy": best_proxy,
        "verdict": verdict,
        "winning_run": "A_cascade" if pick_a <= pick_b else "B_sdf",
        "cascade_baseline": float(cascade_baseline["proxy_cost"]),
    }

    out_path = Path("experiments/E88_diff_proxy/results/spike_ibm01.json")
    out_path.write_text(json.dumps({"runs": runs, "decision": decision}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
