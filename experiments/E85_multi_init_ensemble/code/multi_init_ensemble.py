"""Multi-init Hessian ensemble.

Run N distinct init → polish → Hessian saddle escape trajectories on a
benchmark; take the per-bench best.

For init diversity we use a mix of:
  1. Cached E25 plateau (from E69_sequence_pair_search/results/placements/).
  2. Cached E41 plateau (same).
  3. Jittered SDF init: SDFPlacer output + σ·N(0,1) per-macro perturbation,
     then project_overlaps + CD polish (300 s budget).
  4. (optional) Random uniform feasible placement + CD polish.

Each candidate is then put through the standard Hessian saddle escape
(k=2 eigvecs, ε={0.3, 1.0, 3.0}, polish 180 s/trial).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_E74 = _ROOT / "experiments" / "E74_hessian_saddle" / "code"
if str(_E74) not in sys.path:
    sys.path.insert(0, str(_E74))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hessian_saddle import saddle_escape

from macro_place.bench_paths import find_benchmark_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive, sdf_init
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost


def _cd_polish(placement, benchmark, plc, *, budget_s):
    """Short CD polish to land in a local basin from any init."""
    placement = placement.detach().clone()
    ev = IncrementalProxyEvaluator(benchmark, plc, placement)
    n_hard = benchmark.num_hard_macros
    fixed = benchmark.macro_fixed.cpu().numpy()
    movable = [i for i in range(n_hard) if not bool(fixed[i])]
    run_cd_adaptive(
        ev, benchmark, plc, movable,
        min_time_s=30.0, hard_cap_s=budget_s,
        patience=3, plateau_threshold=0.001, log_fn=None,
    )
    return ev.placement.detach().clone().to(torch.float32)


def _jittered_sdf_init(benchmark, plc, sigma_frac: float, seed: int):
    """SDF + Gaussian jitter scaled to sigma_frac × canvas, then project."""
    sdf = sdf_init(benchmark)
    rng = np.random.default_rng(seed)
    n_hard = benchmark.num_hard_macros
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    sigma_x = sigma_frac * cw
    sigma_y = sigma_frac * ch
    jit = sdf.detach().clone()
    pos_np = jit.cpu().numpy().astype(np.float64)
    pos_np[:n_hard, 0] += rng.normal(0, sigma_x, n_hard)
    pos_np[:n_hard, 1] += rng.normal(0, sigma_y, n_hard)
    hw = benchmark.macro_sizes[:n_hard, 0].cpu().numpy() / 2.0
    hh = benchmark.macro_sizes[:n_hard, 1].cpu().numpy() / 2.0
    pos_np[:n_hard, 0] = np.clip(pos_np[:n_hard, 0], hw, cw - hw)
    pos_np[:n_hard, 1] = np.clip(pos_np[:n_hard, 1], hh, ch - hh)
    jit = torch.tensor(pos_np, dtype=jit.dtype)
    jit, _ = project_overlaps(jit, benchmark)
    return jit


def run_ensemble(
    benchmark, plc,
    *,
    bench_name: str,
    inits_to_try: List[str],
    sdf_sigma_fracs: Tuple[float, ...] = (0.03, 0.08),
    init_polish_budget: float = 300.0,
    saddle_polish_budget: float = 180.0,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, dict]:
    if log is None:
        log = lambda s: print(s, flush=True)

    e69 = _ROOT / "experiments" / "E69_sequence_pair_search" / "results" / "placements"
    candidates: List[dict] = []

    for label in inits_to_try:
        log(f"[ensemble] === init: {label} ===")
        t_init = time.time()
        if label == "E25_cached":
            pt = e69 / f"e25_{bench_name}.pt"
            if not pt.exists():
                log(f"  skipped: no cached E25 for {bench_name}")
                continue
            plateau = torch.load(pt, weights_only=False)["placement"].to(torch.float32)
        elif label == "E41_cached":
            pt = e69 / f"e41_{bench_name}.pt"
            if not pt.exists():
                log(f"  skipped: no cached E41 for {bench_name}")
                continue
            plateau = torch.load(pt, weights_only=False)["placement"].to(torch.float32)
        elif label.startswith("SDF_jitter_"):
            seed = int(label.split("_")[-1])
            sigma_frac = sdf_sigma_fracs[seed % len(sdf_sigma_fracs)]
            log(f"  SDF jitter σ={sigma_frac:.3f}·canvas, seed={seed}")
            init = _jittered_sdf_init(benchmark, plc, sigma_frac, seed)
            log(f"  CD polish from jittered init ({init_polish_budget:.0f} s)")
            plateau = _cd_polish(init, benchmark, plc, budget_s=init_polish_budget)
        elif label == "DREAMPlace_cached":
            # Load DREAMPlace output (must be produced via E76 cloud workflow).
            dp_pt = _ROOT / "experiments" / "E76_dreamplace_integration" / "results" / f"dreamplace_{bench_name}.pt"
            if not dp_pt.exists():
                log(f"  skipped: no cached DREAMPlace .pt for {bench_name} "
                    f"(run E76 cloud workflow first)")
                continue
            dp = torch.load(dp_pt, weights_only=False)
            log(f"  DREAMPlace cached proxy={dp['proxy']:.5f}; CD polish to land basin "
                f"({init_polish_budget:.0f} s)")
            # The DREAMPlace output is a placement; we polish briefly to land
            # in a stable basin for the saddle escape.
            plateau = _cd_polish(dp["placement"].to(torch.float32),
                                 benchmark, plc, budget_s=init_polish_budget)
        else:
            log(f"  unknown init label: {label}; skipping")
            continue

        plateau_proxy = float(compute_proxy_cost(plateau, benchmark, plc)["proxy_cost"])
        plateau_ovl = compute_overlap_metrics(plateau, benchmark)["overlap_count"]
        init_wall = time.time() - t_init
        log(f"  plateau proxy={plateau_proxy:.5f} ovl={plateau_ovl} init_wall={init_wall:.0f}s")
        if plateau_ovl > 0:
            log(f"  plateau has overlaps; skipping this init")
            continue

        # Hessian saddle escape from this plateau.
        log(f"  running Hessian saddle escape from {label}")
        t_saddle = time.time()
        saddle_state, saddle_stats = saddle_escape(
            plateau, benchmark, plc,
            n_eigvecs=2,
            epsilon_values=(0.3, 1.0, 3.0),
            cd_polish_budget=saddle_polish_budget,
        )
        saddle_wall = time.time() - t_saddle
        saddle_proxy = float(saddle_stats.get("best_proxy", plateau_proxy))
        log(f"  saddle best={saddle_proxy:.5f} (Δ vs plateau {saddle_proxy - plateau_proxy:+.5f}) "
            f"saddle_wall={saddle_wall:.0f}s")
        candidates.append({
            "label": label,
            "plateau_proxy": plateau_proxy,
            "saddle_proxy": saddle_proxy,
            "saddle_lift": plateau_proxy - saddle_proxy,
            "init_wall": init_wall,
            "saddle_wall": saddle_wall,
            "saddle_state": saddle_state,
        })

    if not candidates:
        raise RuntimeError("All inits failed; no candidates to ensemble")

    candidates.sort(key=lambda c: c["saddle_proxy"])
    winner = candidates[0]
    log(f"[ensemble] WINNER: {winner['label']} at {winner['saddle_proxy']:.5f}")
    for c in candidates:
        log(f"  {c['label']}: plateau={c['plateau_proxy']:.5f} → saddle={c['saddle_proxy']:.5f}")

    return winner["saddle_state"], {
        "candidates": [
            {k: v for k, v in c.items() if k != "saddle_state"}
            for c in candidates
        ],
        "winner": {k: v for k, v in winner.items() if k != "saddle_state"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument(
        "--inits",
        default="E25_cached,E41_cached,SDF_jitter_1,SDF_jitter_7",
        help="comma-separated init labels",
    )
    ap.add_argument("--init-polish", type=float, default=300.0)
    ap.add_argument("--saddle-polish", type=float, default=180.0)
    args = ap.parse_args()

    bench_name = args.bench
    inits = args.inits.split(",")
    print(f"[E85] {bench_name}: inits={inits}")

    benchmark, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))
    t0 = time.time()
    result, stats = run_ensemble(
        benchmark, plc, bench_name=bench_name,
        inits_to_try=inits,
        init_polish_budget=args.init_polish,
        saddle_polish_budget=args.saddle_polish,
    )
    total_wall = time.time() - t0
    final_proxy = float(compute_proxy_cost(result, benchmark, plc)["proxy_cost"])
    final_ovl = compute_overlap_metrics(result, benchmark)["overlap_count"]
    print(f"[E85] DONE: final={final_proxy:.5f} ovl={final_ovl} wall={total_wall:.0f}s")

    out_summary = _HERE.parent / "results" / f"ensemble_{bench_name}.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps({
        "bench_name": bench_name,
        "final_proxy": final_proxy,
        "final_overlap": final_ovl,
        "wall_seconds": total_wall,
        "candidates": stats["candidates"],
        "winner": stats["winner"],
    }, indent=2))
    out_pt = _HERE.parent / "results" / f"ensemble_{bench_name}.pt"
    torch.save({"placement": result.cpu(), "stats": stats, "bench_name": bench_name}, out_pt)
    print(f"[E85] saved -> {out_summary}, {out_pt}")


if __name__ == "__main__":
    main()
