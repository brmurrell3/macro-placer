"""E116 Path B — run patched DREAMPlace on a benchmark.

This script is intended to run on the AWS instance where DREAMPlace
is properly built (Python 3.11 + ABI=1). It:

  1. Writes the benchmark in Bookshelf format (via E76 converter).
  2. Loads DREAMPlace's Params + PlaceDB + PlaceObj.
  3. Monkey-patches PlaceObj.obj_fn with our challenge-proxy loss.
  4. Runs DREAMPlace's standard global placement (Nesterov + Lgamma).
  5. Reads back the .gp.pl, converts to our (num_macros, 2) format.
  6. Runs greedy_macro_legalize + CD polish.
  7. Reports final proxy.

Designed for remote execution on AWS via ssh:
    ssh ubuntu@aws_ip 'cd ~/dreamplace_src_install && python dp_runner.py ibm01'

Two passes are required because DREAMPlace's runtime needs the
Bookshelf/aux files locally on the AWS box, AND our challenge proxy
needs our macro_place/ code path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def setup_paths(repo_root: Path):
    """Configure Python path so we can import both our code and DREAMPlace."""
    paths = [
        str(repo_root),
        str(repo_root / "experiments" / "E88_diff_proxy" / "code"),
        str(repo_root / "experiments" / "E95_diff_proxy_v2" / "code"),
        str(repo_root / "experiments" / "E76_dreamplace_integration" / "code"),
        str(repo_root / "experiments" / "E110_smooth_global_placer" / "code"),
        str(repo_root / "experiments" / "E111_per_net_trace_congestion" / "code"),
        str(repo_root / "experiments" / "E116_dreamplace_patched" / "code"),
    ]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)


def write_bookshelf(benchmark, plc, out_dir: Path) -> Path:
    """Write Bookshelf files for the given benchmark. Returns aux path."""
    import tilos_to_bookshelf as bk_writer
    out_dir.mkdir(parents=True, exist_ok=True)
    bk_writer._write_nodes(benchmark, out_dir)
    bk_writer._write_pl(benchmark, out_dir)
    bk_writer._write_nets(benchmark, out_dir)
    bk_writer._write_scl(benchmark, out_dir)
    bk_writer._write_wts(benchmark, out_dir)
    bk_writer._write_aux(benchmark, out_dir)
    return out_dir / f"{benchmark.name}.aux"


def run_patched_dp(
    benchmark, plc, dp_install_root: Path, out_dir: Path,
    *, num_iter: int = 1000, target_density: float = 0.85,
    challenge_weight: float = 1.0, edensity_weight: float = 0.05,
    verbose: bool = True,
):
    """Run patched DREAMPlace on the benchmark.

    Returns (positions_tensor, stats).
    """
    import torch

    sys.path.insert(0, str(dp_install_root))
    import dreamplace.Params as DpParams
    import dreamplace.PlaceDB as DpPlaceDB
    import dreamplace.PlaceObj as DpPlaceObj
    import dreamplace.NonLinearPlace as DpNonLinearPlace

    from diff_proxy_v2 import DiffProxyV2  # noqa: E402
    from diff_proxy import _grid_density, _lse_hpwl  # noqa: E402
    from per_net_trace_proxy import PerNetTraceCongestion  # noqa: E402
    from dp_loss_patch import install_patch  # noqa: E402

    # 1. Bookshelf write
    aux_path = write_bookshelf(benchmark, plc, out_dir)
    if verbose:
        print(f"[runner] Bookshelf written: {aux_path}")

    # 2. DP params
    params_dict = {
        "aux_input": str(aux_path),
        "target_density": target_density,
        "density_weight": 8e-5,
        "gpu": 1 if torch.cuda.is_available() else 0,
        "num_threads": 4,
        "deterministic_flag": 1,
        "global_place_stages": [{
            "num_bins_x": 1024, "num_bins_y": 1024,
            "iteration": num_iter, "learning_rate": 0.01,
            "wirelength": "weighted_average", "optimizer": "nesterov",
        }],
        "legalize_flag": 0, "detailed_place_flag": 0,
        "stop_overflow": 0.07,
        "result_dir": str(out_dir),
        "global_place_flag": 1,
        "macro_place_flag": 0,
        "scale_factor": 1.0,
        "random_center_init_flag": 1,
        "sort_nets_by_degree": 0,
        "num_bins_x": 1024,
        "num_bins_y": 1024,
        "global_swap_flag": 0,
        "k_reorder_flag": 0,
        "independent_set_matching_flag": 0,
        "macro_halo_x": 0,
        "macro_halo_y": 0,
        "regioned_global_place_flag": 0,
        "routability_opt_flag": 0,
        "timing_opt_flag": 0,
        "dtype": "float32",
        "force_cpu_flag": 0,
        "filler_size_x_ratio": 0.5,
        "filler_size_y_ratio": 0.5,
    }
    params = DpParams.Params()
    params.fromJson(params_dict)
    params.printParams()

    # 3. PlaceDB
    placedb = DpPlaceDB.PlaceDB()
    placedb(params)

    # 4. PlaceObj + monkey-patch
    # DP's NonLinearPlace constructs PlaceObj inside its __call__. We
    # need to monkey-patch before the optimization runs. The cleanest
    # way is to install the patch on the obj_fn AFTER construction but
    # BEFORE the first Nesterov step.
    #
    # We do this by subclassing NonLinearPlace and overriding the place
    # objective construction. Simpler approach: just monkey-patch
    # `PlaceObj.PlaceObj.obj_fn` at the class level, then it's used by
    # all instances.
    #
    # Even simpler: wrap the placer to add a hook. The Nesterov
    # optimizer reads `model.obj_and_grad_fn` which calls `model.obj_fn`.
    # So if we patch `model.obj_fn` after PlaceObj construction, it
    # works.

    # Run with default obj_fn first to set up internal state
    t0 = time.time()
    placer = DpNonLinearPlace.NonLinearPlace(params, placedb, timer=None)

    # Now install patch on placer.model — but NonLinearPlace creates
    # the PlaceObj inside __call__, so we need to hook in there.
    # Workaround: call placer with a flag that pauses after PlaceObj
    # construction. There's no such flag, so we monkey-patch the
    # PlaceObj CLASS itself:

    original_obj_fn = DpPlaceObj.PlaceObj.obj_fn
    patch_state = {"installed": False, "patched_state": None}

    def hooked_obj_fn(self, pos):
        # On first call, install the patched obj_fn (we now have a
        # valid model with all ops set up).
        if not patch_state["installed"]:
            patched_state = install_patch(
                self,
                placedb=placedb, benchmark=benchmark, plc=plc,
                per_net_trace_cls=PerNetTraceCongestion,
                diff_proxy_v2_cls=DiffProxyV2,
                grid_density_fn=_grid_density,
                lse_hpwl_fn=_lse_hpwl,
                challenge_weight=challenge_weight,
                edensity_weight=edensity_weight,
                trace_kwargs=None,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            patch_state["installed"] = True
            patch_state["patched_state"] = patched_state
            # `self.obj_fn` is now our patched version
            return self.obj_fn(pos)
        # All subsequent calls: just delegate to the patched obj_fn
        return self.obj_fn(pos)

    # Actually, since install_patch replaces self.obj_fn (instance
    # attribute), subsequent calls to model.obj_fn will hit our
    # version directly. We just need to make sure the patch is
    # installed on first call.
    DpPlaceObj.PlaceObj.obj_fn = hooked_obj_fn

    try:
        # Run global placement
        placer(params, placedb, learning_rate_value=0.01)
        wall = time.time() - t0
        if verbose:
            print(f"[runner] DREAMPlace done in {wall:.1f}s")
            if patch_state["patched_state"]:
                ps = patch_state["patched_state"]
                print(f"  last challenge={ps.get('last_challenge')}")
                print(f"  last edensity={ps.get('last_edensity')}")
    finally:
        # Restore class-level obj_fn
        DpPlaceObj.PlaceObj.obj_fn = original_obj_fn

    # 5. Read final positions from placer.pos[0]
    # pos = [x_0..x_N-1, y_0..y_N-1] for N = num_nodes
    if patch_state["patched_state"]:
        state = patch_state["patched_state"]
        pos = placer.pos[0].detach()
        final_centers = state["dp_pos_to_macro_centers"](pos).cpu()
    else:
        # Fallback: read .gp.pl
        import bookshelf_to_pt as bk_reader
        gp_pl = out_dir / f"{benchmark.name}.gp.pl"
        if not gp_pl.exists():
            for c in out_dir.rglob("*.gp.pl"):
                gp_pl = c; break
        pl_map = bk_reader.parse_pl(gp_pl)
        sizes = benchmark.macro_sizes.cpu().numpy()
        final_centers = benchmark.macro_positions.clone().detach()
        SCALE = 1000.0
        for i in range(benchmark.num_macros):
            if i in pl_map:
                llx, lly = pl_map[i]
                final_centers[i, 0] = llx / SCALE + float(sizes[i, 0]) / 2.0
                final_centers[i, 1] = lly / SCALE + float(sizes[i, 1]) / 2.0
        final_centers = final_centers.to(torch.float32)

    stats = {"wall_s": wall, "patched": patch_state["installed"]}
    return final_centers, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", default="ibm01", nargs="?")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--dp-root", default=str(Path.home() / "dreamplace_src_install"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--num-iter", type=int, default=1000)
    ap.add_argument("--cd-budget", type=float, default=300.0)
    ap.add_argument("--challenge-w", type=float, default=1.0)
    ap.add_argument("--edensity-w", type=float, default=0.05)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    setup_paths(repo_root)

    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost, compute_overlap_metrics
    from macro_place.cd_core import project_overlaps, run_cd_adaptive
    from macro_place.incremental_evaluator import IncrementalProxyEvaluator
    from macro_legalizer import greedy_macro_legalize

    benchmark, plc = load_benchmark_from_dir(str(find_benchmark_dir(args.bench)))
    print(f"\n=== {args.bench} (n_macros={benchmark.num_macros} "
          f"n_hard={benchmark.num_hard_macros}) ===")

    with tempfile.TemporaryDirectory(prefix=f"dp_{args.bench}_") as tmp:
        tmp_path = Path(tmp)
        positions, dp_stats = run_patched_dp(
            benchmark, plc, Path(args.dp_root), tmp_path,
            num_iter=args.num_iter,
            challenge_weight=args.challenge_w,
            edensity_weight=args.edensity_w,
        )

        ovl_dp = compute_overlap_metrics(positions, benchmark)["overlap_count"]
        proxy_dp = float(compute_proxy_cost(positions, benchmark, plc)["proxy_cost"])
        print(f"[dp] raw_proxy={proxy_dp:.5f} ovl={ovl_dp} wall={dp_stats['wall_s']:.1f}s")

        # 6. Legalize
        legal_pos, leg_stats = greedy_macro_legalize(
            positions, benchmark, search_radius_steps=80, step_size_frac=0.02,
            verbose=False,
        )
        ovl_leg = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        if ovl_leg > 0:
            legal_pos, _ = project_overlaps(legal_pos, benchmark)
            ovl_leg = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        proxy_leg = float(compute_proxy_cost(legal_pos, benchmark, plc)["proxy_cost"])
        print(f"[legalize] proxy={proxy_leg:.5f} ovl={ovl_leg} moved={leg_stats.get('n_moved')}")

        # 7. CD polish
        if args.cd_budget > 0:
            ev = IncrementalProxyEvaluator(benchmark, plc, legal_pos.clone())
            fixed = benchmark.macro_fixed.cpu().numpy()
            movable = [i for i in range(benchmark.num_hard_macros) if not bool(fixed[i])]
            t0 = time.time()
            run_cd_adaptive(
                ev, benchmark, plc, movable,
                min_time_s=args.cd_budget * 0.5,
                hard_cap_s=args.cd_budget,
                patience=3, plateau_threshold=0.001, log_fn=None,
            )
            polished = ev.placement.detach().clone().to(positions.dtype)
            polished, _ = project_overlaps(polished, benchmark)
            ovl_final = compute_overlap_metrics(polished, benchmark)["overlap_count"]
            proxy_final = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
            print(f"[cd polish] proxy={proxy_final:.5f} ovl={ovl_final} wall={time.time()-t0:.1f}s")
        else:
            polished = legal_pos
            proxy_final = proxy_leg
            ovl_final = ovl_leg

        result = {
            "bench": args.bench,
            "proxy_dp_raw": proxy_dp,
            "proxy_legalized": proxy_leg,
            "proxy_final": proxy_final,
            "ovl_final": int(ovl_final),
            "dp_wall_s": dp_stats["wall_s"],
        }
        print(f"\nFINAL: {json.dumps(result, indent=2)}")
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
