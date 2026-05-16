"""E91 B-R0': Stock DP basin + full cascade-equivalent polish.

The proper falsification test that was never run. PATH B autopsy was
made on DP-basin-only quality (sweep) and 15-min CD polish probe; the
"DP basin + same polish budget as cascade gives E25/E41" was never run.

Pipeline per bench:
  1. Run stock DREAMPlace (Docker) — config matches cd_lns_sa_hessian_dp.
  2. greedy_macro_legalize → project_overlaps cleanup.
  3. Full E25-equivalent polish on DP init: CD-adaptive + LNS-gridbin + SA-v2.
  4. Cascading saddle escape.
  5. Compare final proxy to cascade reference.

Wall budget per bench: 3300s (55 min) — matches partcl cap.

Usage on cloud (lambda.ai with DREAMPlace built at ~/DREAMPlace/install):

  cd ~/macro-place-challenge-2026
  OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \\
    DREAMPLACE_ROOT=/home/ubuntu/DREAMPlace/install \\
    DP_DOCKER_IMAGE=limbo018/dreamplace:cuda \\
    uv run python experiments/E91_dp_full_polish/code/dp_full_polish.py ibm10

Output: experiments/E91_dp_full_polish/results/<bench>_dp_full_polish.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))
_E76 = _ROOT / "experiments" / "E76_dreamplace_integration" / "code"
sys.path.insert(0, str(_E76))
_E84 = _ROOT / "experiments" / "E84_cascading_saddle" / "code"
sys.path.insert(0, str(_E84))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir
from macro_place.cd_core import project_overlaps, run_cd_adaptive
from macro_place.incremental_evaluator import IncrementalProxyEvaluator
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

# E76 I/O helpers
import tilos_to_bookshelf as bk_writer
import bookshelf_to_pt as bk_reader
from macro_legalizer import greedy_macro_legalize

# Stricter legalizer (E71 jitter+project loop) for NG45-class stuck cases.
sys.path.insert(0, str(_HERE))
from extended_legalize import extended_legalize

# E25 polish components (LNS gridbin + SA-v2)
import importlib.util
_E25_PATH = _ROOT / "submissions" / "cd_lns_sa" / "placer.py"
_E25_SPEC = importlib.util.spec_from_file_location("e25_placer", str(_E25_PATH))
_E25_MOD = importlib.util.module_from_spec(_E25_SPEC)
_E25_SPEC.loader.exec_module(_E25_MOD)
run_lns_gridbin = _E25_MOD.run_lns_gridbin
run_sa_polish_v2 = _E25_MOD.run_sa_polish_v2

# Cascading saddle (E84)
from cascading_saddle import cascading_saddle_escape


def _run_dp_with_config(bench, log, target_density, stop_overflow, dp_iter, dp_lr,
                       config_label):
    """Run DREAMPlace once with the given config. Internal — used by both
    stages of run_dp_docker.
    """
    dp_image = os.environ.get("DP_DOCKER_IMAGE", "limbo018/dreamplace:cuda")
    dp_root_host = os.environ.get("DREAMPLACE_ROOT", "/home/ubuntu/DREAMPlace/install")
    SCALE = float(bk_writer.SCALE)

    with tempfile.TemporaryDirectory(prefix=f"dp_{bench.name}_", dir="/tmp") as tmp:
        tmp = Path(tmp)
        bk_writer._write_nodes(bench, tmp)
        bk_writer._write_pl(bench, tmp)
        bk_writer._write_nets(bench, tmp)
        bk_writer._write_scl(bench, tmp)
        bk_writer._write_wts(bench, tmp)
        bk_writer._write_aux(bench, tmp)
        cfg = tmp / "dp.json"

        log(f"  [DP {config_label}] target_density={target_density:.3f} "
            f"stop_overflow={stop_overflow} iter={dp_iter} lr={dp_lr}")
        cfg.write_text(json.dumps({
            "aux_input": f"/work/{bench.name}.aux",
            "target_density": target_density,
            "density_weight": 8e-5,
            "gpu": int(os.environ.get("DP_USE_GPU", "1")),
            "num_threads": int(os.environ.get("DP_NUM_THREADS", "8")),
            "global_place_stages": [{
                "num_bins_x": 1024, "num_bins_y": 1024,
                "iteration": dp_iter, "learning_rate": dp_lr,
                "wirelength": "weighted_average", "optimizer": "nesterov",
            }],
            "legalize_flag": 0,
            "detailed_place_flag": 0,
            "stop_overflow": stop_overflow,
            "result_dir": "/work",
        }))

        uid = os.getuid()
        gid = os.getgid()
        docker_args = ["sudo", "docker", "run", "--rm",
                       "--user", f"{uid}:{gid}",
                       "-v", f"{tmp}:/work",
                       "-v", f"{dp_root_host}:/dp_install:ro",
                       "-w", "/work"]
        if int(os.environ.get("DP_USE_GPU", "1")):
            docker_args.extend(["--gpus", "all"])
        cmd = docker_args + [
            dp_image,
            "python", "/dp_install/dreamplace/Placer.py", "/work/dp.json",
        ]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            return None, time.time() - t0, "timeout"
        wall = time.time() - t0
        if proc.returncode != 0:
            err = proc.stderr[-500:] if proc.stderr else "<empty stderr>"
            log(f"  [DP {config_label}] FAILED exit={proc.returncode}")
            log(f"  [DP {config_label}] stderr tail: {err}")
            return None, wall, f"dp_exit_{proc.returncode}"

        gp_pl = tmp / f"{bench.name}.gp.pl"
        if not gp_pl.exists():
            for c in tmp.rglob("*.gp.pl"):
                gp_pl = c
                break
            if not gp_pl.exists():
                return None, wall, "no_output"

        pl_map = bk_reader.parse_pl(gp_pl)
        n = bench.num_macros
        sizes = bench.macro_sizes.cpu().numpy()
        placement = bench.macro_positions.clone().detach()
        for i in range(n):
            if i in pl_map:
                llx, lly = pl_map[i]
                placement[i, 0] = llx / SCALE + float(sizes[i, 0]) / 2.0
                placement[i, 1] = lly / SCALE + float(sizes[i, 1]) / 2.0
        return placement.to(torch.float32), wall, "ok"


def run_dp_docker(bench, plc, log) -> Tuple[Optional[torch.Tensor], float, str]:
    """Two-stage DREAMPlace: try the fast/lenient config first (works for
    IBM out of the box); if greedy_legalize can't fully clean overlaps,
    retry with a tighter, adaptive config that's been shown to work for
    NG45-class commercial designs.

    Single algorithm applied to every input — rule-compliant. The branch
    is on observed overlap count after legalize, not on benchmark name.

    Stage A: original config (target_density=0.85, stop_overflow=0.07,
             iter=1000, lr=0.01). Fast, ~12-90 s. Works for IBM benches.
    Stage B (only if Stage A's basin can't legalize):
             target_density derived from macro_density (clipped [0.40, 0.85]),
             stop_overflow=0.02, iter=2000, lr=0.005. Slower (~50-90 s)
             but produces cleaner basins for tight-tolerance benches.
    """
    # Stage A
    pl_a, wall_a, status_a = _run_dp_with_config(
        bench, log,
        target_density=0.85, stop_overflow=0.07, dp_iter=1000, dp_lr=0.01,
        config_label="stageA",
    )
    if pl_a is None:
        return None, wall_a, status_a

    ovl_a = compute_overlap_metrics(pl_a, bench)["overlap_count"]

    # Quick greedy legalize attempt to test feasibility of Stage A basin.
    # If greedy lands fully legal, Stage A wins (IBM-friendly fast path).
    legal_a, leg_a_stats = greedy_macro_legalize(pl_a, bench, step_size_frac=0.01)
    legal_a, _ = project_overlaps(legal_a, bench)
    ovl_legal_a = compute_overlap_metrics(legal_a, bench)["overlap_count"]
    log(f"  [DP stageA] basin_ovl={ovl_a} → greedy_legalize: ovl={ovl_legal_a}")

    if ovl_legal_a == 0:
        # Stage A is sufficient. Return the raw Stage A placement so the
        # caller's pipeline can repeat its standard legalize+polish flow.
        return pl_a, wall_a, "ok"

    # Stage B: retry with tighter config. Derived from observable bench geometry.
    sizes = bench.macro_sizes.cpu().numpy()
    hard_idx = bench.num_hard_macros
    macro_area = float((sizes[:hard_idx, 0] * sizes[:hard_idx, 1]).sum())
    canvas_area = float(bench.canvas_width * bench.canvas_height)
    macro_density = macro_area / max(canvas_area, 1e-9)
    target_b = float(min(0.85, max(0.40, macro_density * 1.5)))
    log(f"  [DP stageA basin can't legalize ({ovl_legal_a} residuals); "
        f"retrying stageB with macro_density={macro_density:.3f}→target={target_b:.3f}]")
    pl_b, wall_b, status_b = _run_dp_with_config(
        bench, log,
        target_density=target_b, stop_overflow=0.02, dp_iter=2000, dp_lr=0.005,
        config_label="stageB",
    )
    if pl_b is None:
        # Stage B failed too; return Stage A's placement (best we have).
        log(f"  [DP stageB] failed ({status_b}); falling back to stageA placement")
        return pl_a, wall_a + wall_b, "ok_stageA_fallback"
    return pl_b, wall_a + wall_b, "ok_stageB"


def run_full_polish_on_init(
    init_placement,
    bench,
    plc,
    *,
    cd_hard_cap_s: float,
    lns_budget_s: float,
    sa_budget_s: float,
    cascade_budget_s: float,
    cascade_max_iters: int = 5,
    cascade_eps_values: Tuple[float, ...] = (0.3, 1.0, 3.0),
    cascade_polish_budget: float = 180.0,
    log=print,
):
    """Run the E25 polish pipeline + cascading saddle on a given init placement.

    Equivalent to what cd_lns_sa_cascade does on the E48 plateau pick, but
    with the init coming from an external source (DP / legalize) rather
    than from sdf_init.
    """
    n_hard = bench.num_hard_macros
    fixed = bench.macro_fixed.cpu().numpy()
    movable = [i for i in range(bench.num_macros) if not bool(fixed[i])]
    hard_movable = [i for i in range(n_hard) if not bool(fixed[i])]

    placement = init_placement.detach().clone().to(torch.float32)
    placement, _ = project_overlaps(placement, bench)
    init_proxy = float(compute_proxy_cost(placement, bench, plc)["proxy_cost"])
    log(f"  init proxy (post-project): {init_proxy:.5f}")

    placement_f64 = placement.detach().clone().to(torch.float64)
    ev = IncrementalProxyEvaluator(bench, plc, placement_f64)

    log(f"  [polish] CD-adaptive (cap={cd_hard_cap_s:.0f}s)")
    t0 = time.time()
    run_cd_adaptive(
        evaluator=ev, benchmark=bench, plc=plc, movable=movable,
        min_time_s=min(60.0, cd_hard_cap_s * 0.1),
        hard_cap_s=cd_hard_cap_s,
        patience=3, plateau_threshold=0.001, log_fn=None,
    )
    cd_proxy = ev.current_cost()["proxy"]
    log(f"  [polish] CD done in {time.time() - t0:.0f}s, proxy={cd_proxy:.5f}")

    log(f"  [polish] LNS-gridbin (budget={lns_budget_s:.0f}s)")
    t0 = time.time()
    run_lns_gridbin(
        evaluator=ev, benchmark=bench, plc=plc,
        hard_movable=hard_movable, time_budget_s=lns_budget_s,
        destroy_frac=0.05, destroy_cap=30, seed=42, log_fn=None,
    )
    lns_proxy = ev.current_cost()["proxy"]
    log(f"  [polish] LNS done in {time.time() - t0:.0f}s, proxy={lns_proxy:.5f}")

    log(f"  [polish] SA-v2 (budget={sa_budget_s:.0f}s)")
    t0 = time.time()
    run_sa_polish_v2(
        evaluator=ev, benchmark=bench, plc=plc,
        hard_movable=hard_movable, time_budget_s=sa_budget_s,
        T0=5e-4, Tf=1e-6, seed=42, breakpoint_budget=12, log_fn=None,
    )
    sa_proxy = ev.current_cost()["proxy"]
    log(f"  [polish] SA-v2 done in {time.time() - t0:.0f}s, proxy={sa_proxy:.5f}")

    plateau = ev.placement.detach().clone().to(torch.float32)
    plateau, _ = project_overlaps(plateau, bench)
    plateau_proxy = float(compute_proxy_cost(plateau, bench, plc)["proxy_cost"])
    log(f"  [polish] plateau (post-project): {plateau_proxy:.5f}")

    log(f"  [polish] cascading saddle (budget={cascade_budget_s:.0f}s, "
        f"max_iters={cascade_max_iters})")
    t0 = time.time()
    cascade_state, stats = cascading_saddle_escape(
        plateau, bench, plc,
        max_iters=cascade_max_iters,
        eps_values=cascade_eps_values,
        polish_budget=cascade_polish_budget,
        total_budget_s=cascade_budget_s,
        log=lambda s: None,
    )
    cascade_state, _ = project_overlaps(
        cascade_state.to(torch.float32), bench,
    )
    cascade_proxy = float(compute_proxy_cost(cascade_state, bench, plc)["proxy_cost"])
    log(f"  [polish] cascade done in {time.time() - t0:.0f}s, "
        f"proxy={cascade_proxy:.5f} iters={stats.get('iters_run','?')}")

    return {
        "init_proxy": init_proxy,
        "cd_proxy": float(cd_proxy),
        "lns_proxy": float(lns_proxy),
        "sa_proxy": float(sa_proxy),
        "plateau_proxy": plateau_proxy,
        "cascade_proxy": cascade_proxy,
        "final_state": cascade_state,
        "final_proxy": cascade_proxy,
        "overlap_count": int(compute_overlap_metrics(
            cascade_state, bench)["overlap_count"]),
    }


def main(bench_name: str, budget_s: float = 3300.0):
    log = lambda s: print(s, flush=True)
    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))
    log(f"\n=== E91 B-R0' DP+full-polish on {bench_name} "
        f"(n_macros={bench.num_macros}, n_hard={bench.num_hard_macros}) ===")
    log(f"  budget={budget_s:.0f}s")

    t_total0 = time.time()
    out_path = _HERE.parent / "results" / f"{bench_name}_dp_full_polish.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1: DP
    log("  Phase 1: DREAMPlace (stock config)")
    placement, dp_wall, dp_status = run_dp_docker(bench, plc, log)
    if placement is None:
        result = {
            "bench": bench_name, "status": f"dp_{dp_status}",
            "dp_wall": dp_wall, "total_wall": time.time() - t_total0,
        }
        out_path.write_text(json.dumps(result, indent=2))
        log(f"  ABORT: DP failed ({dp_status})")
        return
    ovl_dp = compute_overlap_metrics(placement, bench)["overlap_count"]
    proxy_dp = float(compute_proxy_cost(placement, bench, plc)["proxy_cost"])
    log(f"  Phase 1 done: DP wall={dp_wall:.0f}s ovl={ovl_dp} basin_proxy={proxy_dp:.5f}")

    # Phase 2: greedy legalize, then extended-legalize fallback if residuals
    log("  Phase 2: greedy_macro_legalize")
    t0 = time.time()
    legal, leg_stats = greedy_macro_legalize(placement, bench, step_size_frac=0.01)
    legal, _ = project_overlaps(legal, bench)
    ovl_leg = compute_overlap_metrics(legal, bench)["overlap_count"]
    proxy_leg = float(compute_proxy_cost(legal, bench, plc)["proxy_cost"])
    log(f"  Phase 2 done: wall={time.time() - t0:.1f}s ovl={ovl_leg} "
        f"legal_proxy={proxy_leg:.5f} moved={leg_stats.get('n_moved','?')} "
        f"failed={leg_stats.get('n_failed','?')}")
    if ovl_leg > 0:
        log(f"  Phase 2b: {ovl_leg} residual overlaps — running extended_legalize "
            f"(jitter+project loop, max 20 passes)")
        t0 = time.time()
        legal, ext_stats = extended_legalize(
            legal, bench, max_passes=20, jitter_scale=0.5, seed=42,
        )
        ovl_leg = ext_stats["final_overlap"]
        proxy_leg = float(compute_proxy_cost(legal, bench, plc)["proxy_cost"])
        log(f"  Phase 2b done: wall={time.time() - t0:.1f}s ovl={ovl_leg} "
            f"passes={ext_stats['passes_used']} history={ext_stats['history']}")
    if ovl_leg > 0:
        log(f"  WARNING: {ovl_leg} residual overlaps even after extended_legalize; "
            f"polish may not be able to clean (skip cascade saddle / fall back)")

    # Phase 3-5: full polish + cascade
    remaining = budget_s - (time.time() - t_total0)
    # Same proportions as cd_lns_sa_cascade: E25 portion = 32% of budget,
    # cascade portion = remaining after subtracting polish.
    cd_cap = remaining * 0.20
    lns_cap = remaining * 0.06
    sa_cap = remaining * 0.06
    cascade_cap = remaining * 0.60
    log(f"  Polish budgets: CD={cd_cap:.0f}s LNS={lns_cap:.0f}s "
        f"SA={sa_cap:.0f}s cascade={cascade_cap:.0f}s "
        f"(remaining={remaining:.0f}s)")

    polish_result = run_full_polish_on_init(
        legal, bench, plc,
        cd_hard_cap_s=cd_cap,
        lns_budget_s=lns_cap,
        sa_budget_s=sa_cap,
        cascade_budget_s=cascade_cap,
        log=log,
    )

    total_wall = time.time() - t_total0
    final_state = polish_result.pop("final_state")
    polish_result["dp_basin_proxy"] = proxy_dp
    polish_result["dp_basin_ovl"] = int(ovl_dp)
    polish_result["legal_proxy"] = proxy_leg
    polish_result["legal_ovl"] = int(ovl_leg)
    polish_result["dp_wall"] = dp_wall
    polish_result["total_wall"] = total_wall
    polish_result["bench"] = bench_name
    polish_result["status"] = "ok"

    out_path.write_text(json.dumps(polish_result, indent=2))
    pt_path = out_path.with_suffix(".pt")
    torch.save({"placement": final_state, "proxy": polish_result["final_proxy"]}, pt_path)

    log(f"\n=== E91 {bench_name} RESULT ===")
    log(f"  DP basin       : {proxy_dp:.5f}  (ovl={ovl_dp})")
    log(f"  + greedy legal : {proxy_leg:.5f}  (ovl={ovl_leg})")
    log(f"  + CD adaptive  : {polish_result['cd_proxy']:.5f}")
    log(f"  + LNS gridbin  : {polish_result['lns_proxy']:.5f}")
    log(f"  + SA-v2        : {polish_result['sa_proxy']:.5f}")
    log(f"  + cascade      : {polish_result['final_proxy']:.5f}  "
        f"(ovl={polish_result['overlap_count']})")
    log(f"  total wall: {total_wall:.0f}s")
    log(f"  → comparison baselines (autopsy table):")
    refs = {"ibm10": 1.0775, "ibm12": 1.3031, "ibm14": 1.2919, "ibm17": 1.4546}
    if bench_name in refs:
        ref = refs[bench_name]
        delta = (polish_result["final_proxy"] - ref) / ref * 100
        log(f"     cascade b=3000 ref: {ref:.5f}  → delta {delta:+.2f}%")
    log(f"  results written to: {out_path}")


if __name__ == "__main__":
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm10"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 3300.0
    main(bench, budget)
