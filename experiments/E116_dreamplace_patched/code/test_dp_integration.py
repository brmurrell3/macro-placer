"""E116 — minimal DREAMPlace integration test.

Runs DP with patched obj_fn on ibm01. The patch replaces DP's
`wirelength + density_weight * eDensity` with our challenge proxy
(LSE-HPWL + 0.5*top-K-density + 0.5*PerNetTraceCong).

This script runs on the AWS instance where DP is built (Python 3.11
+ ABI=1 + CUDA). Local execution will fail due to ABI mismatch.

Usage on AWS:
    source ~/dp_venv/bin/activate
    cd ~/macro-place-challenge-2026/experiments/E116_dreamplace_patched/code
    export LD_LIBRARY_PATH=~/dp_venv/lib/python3.11/site-packages/torch/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    python test_dp_integration.py ibm01 --challenge-w 1.0 --edensity-w 0.0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

# Path setup
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
DP_SRC = Path.home() / "dreamplace_src_install"

# Add our code paths
for p in [
    _REPO,
    _REPO / "experiments" / "E88_diff_proxy" / "code",
    _REPO / "experiments" / "E95_diff_proxy_v2" / "code",
    _REPO / "experiments" / "E76_dreamplace_integration" / "code",
    _REPO / "experiments" / "E110_smooth_global_placer" / "code",
    _REPO / "experiments" / "E111_per_net_trace_congestion" / "code",
    _REPO / "experiments" / "E116_dreamplace_patched" / "code",
]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Add DP install path
sys.path.insert(0, str(DP_SRC))


def write_bookshelf(benchmark, out_dir: Path) -> Path:
    import tilos_to_bookshelf as bk_writer
    out_dir.mkdir(parents=True, exist_ok=True)
    bk_writer._write_nodes(benchmark, out_dir)
    bk_writer._write_pl(benchmark, out_dir)
    bk_writer._write_nets(benchmark, out_dir)
    bk_writer._write_scl(benchmark, out_dir)
    bk_writer._write_wts(benchmark, out_dir)
    bk_writer._write_aux(benchmark, out_dir)
    return out_dir / f"{benchmark.name}.aux"


def patch_dp_obj_fn(
    placedb, benchmark, plc,
    *, challenge_weight: float = 1.0, edensity_weight: float = 0.0,
    device: str = "cuda",
):
    """Monkey-patch dreamplace.PlaceObj.PlaceObj.obj_fn at class level.

    This is the only way to intercept DP's obj_fn since NonLinearPlace
    creates PlaceObj instances internally. The patch reads the relevant
    fixed data (sizes, fixed positions, etc.) and computes the challenge
    proxy.
    """
    import torch
    import dreamplace.PlaceObj as DpPlaceObj
    from diff_proxy_v2 import DiffProxyV2
    from diff_proxy import _grid_density, _lse_hpwl
    from per_net_trace_proxy import PerNetTraceCongestion

    # Build our challenge-proxy objects ONCE (not per-step)
    device_t = torch.device(device)
    diff = DiffProxyV2(benchmark, plc, device=str(device_t), gamma_frac=5e-3)
    trace = PerNetTraceCongestion(benchmark, plc, device=str(device_t))

    # Build mapping: our macro index → DP node id
    num_macros = int(benchmark.num_macros)
    num_movable = int(placedb.num_movable_nodes)
    num_filler = int(placedb.num_filler_nodes)
    num_physical = int(placedb.num_physical_nodes)
    num_nodes = int(placedb.num_nodes)

    print(f"  PlaceDB: num_macros={num_macros} num_movable={num_movable} "
          f"num_filler={num_filler} num_physical={num_physical} num_nodes={num_nodes}")

    macro_to_dp = torch.zeros(num_macros, dtype=torch.long, device=device_t)
    for i in range(num_macros):
        name_b = f"n{i}".encode("utf-8")
        name_s = f"n{i}"
        if name_b in placedb.node_name2id_map:
            macro_to_dp[i] = placedb.node_name2id_map[name_b]
        elif name_s in placedb.node_name2id_map:
            macro_to_dp[i] = placedb.node_name2id_map[name_s]
        else:
            raise KeyError(f"Macro n{i} not in DP node_name2id_map")

    # Half-sizes for center conversion (from LLX/LLY to center)
    macro_half_w = (benchmark.macro_sizes[:, 0] / 2.0).to(device_t)
    macro_half_h = (benchmark.macro_sizes[:, 1] / 2.0).to(device_t)
    macro_sizes = benchmark.macro_sizes.to(device_t)

    # Canvas bounds in DP's internal coords (after scale_pl)
    xl, yl, xh, yh = float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh)
    cw_ours = float(benchmark.canvas_width)
    ch_ours = float(benchmark.canvas_height)
    # Determine scale factor: DP scaled positions by some factor relative to ours
    # E.g. ibm01 our canvas=22.95 microns, DP canvas xh=2296 → 100x scale
    dp_to_our_scale = cw_ours / (xh - xl) if xh > xl else 1.0
    print(f"  DP canvas: xl={xl} yl={yl} xh={xh} yh={yh}")
    print(f"  Our canvas: cw={cw_ours} ch={ch_ours}")
    print(f"  DP → ours scale factor: {dp_to_our_scale:.6f} (DP coords × this = our microns)")

    # Stash state to share between patched obj_fn and inspection
    state = {"call_count": 0, "last_challenge": None, "last_edensity": None}

    original_obj_fn = DpPlaceObj.PlaceObj.obj_fn
    print(f"  Original obj_fn: {original_obj_fn}")
    print(f"  Original obj_and_grad_fn: {DpPlaceObj.PlaceObj.obj_and_grad_fn}")

    def patched_obj_fn(self, pos):
        if state["call_count"] == 0:
            print(f"  *** FIRST CALL to patched_obj_fn (pos shape={tuple(pos.shape)} dtype={pos.dtype}) ***")
        # 1. Compute DP's original loss (for the eDensity term + sanity)
        if edensity_weight > 0.0:
            dp_loss = original_obj_fn(self, pos)
        else:
            dp_loss = torch.zeros(1, device=pos.device)

        # 2. Extract macro center positions in OUR coordinate system.
        # DP's pos[i] for i < num_nodes is the LLX in DP's internal
        # scaled coords (after PlaceDB.scale_pl). Convert to our scale
        # by multiplying by dp_to_our_scale.
        dp_x = pos[:num_nodes][macro_to_dp.to(pos.device)]   # [num_macros]
        dp_y = pos[num_nodes:2 * num_nodes][macro_to_dp.to(pos.device)]
        # Apply scale to map DP coords back to our microns.
        scaled_x = dp_x * dp_to_our_scale
        scaled_y = dp_y * dp_to_our_scale
        # Convert LL to center (sizes already in our microns)
        centers_x = scaled_x + macro_half_w.to(pos.device)
        centers_y = scaled_y + macro_half_h.to(pos.device)
        macro_centers = torch.stack([centers_x, centers_y], dim=1)

        # 3. Compute challenge proxy
        # Clamp to canvas (our canvas, not DP's)
        clamped_x = macro_centers[:, 0].clamp(macro_half_w, cw_ours - macro_half_w)
        clamped_y = macro_centers[:, 1].clamp(macro_half_h, ch_ours - macro_half_h)
        clamped = torch.stack([clamped_x, clamped_y], dim=1)

        wl = _lse_hpwl(clamped, diff.net_data, diff.port_base, diff.gamma) / diff.wl_norm
        density = _grid_density(
            clamped, diff.sizes,
            diff.cell_x_min, diff.cell_x_max,
            diff.cell_y_min, diff.cell_y_max,
            diff.cell_area, diff.grid_rows, diff.grid_cols,
        )
        cong = trace.compute_congestion(clamped)
        challenge = wl + 0.5 * density + 0.5 * cong

        state["call_count"] += 1
        state["last_challenge"] = float(challenge.item())
        state["last_edensity"] = float(dp_loss.item()) if isinstance(dp_loss, torch.Tensor) else float(dp_loss)
        state["last_wl"] = float(wl.item())
        state["last_density"] = float(density.item())
        state["last_cong"] = float(cong.item())

        if state["call_count"] % 50 == 1:
            print(f"  [obj #{state['call_count']}] challenge={state['last_challenge']:.5f} "
                  f"(wl={state['last_wl']:.4f} d={state['last_density']:.4f} c={state['last_cong']:.4f}) "
                  f"edensity={state['last_edensity']:.5f}")

        return challenge_weight * challenge + edensity_weight * dp_loss

    # Also need to patch obj_and_grad_fn to ensure our obj_fn is called
    original_obj_and_grad_fn = DpPlaceObj.PlaceObj.obj_and_grad_fn

    def patched_obj_and_grad_fn(self, pos):
        if pos.grad is not None:
            pos.grad.zero_()
        obj = patched_obj_fn(self, pos)
        if obj.requires_grad:
            obj.backward()
        # Apply DP's preconditioner if it exists
        try:
            self.op_collections.precondition_op(
                pos.grad, self.density_weight,
                self.update_mask, self.fix_nodes_mask,
            )
        except Exception as e:
            print(f"  precond skipped: {e}")
        return obj, pos.grad

    # Also patch the top-level PlaceObj module (the one NonLinearPlace
    # imports via `import PlaceObj` not `from dreamplace import PlaceObj`).
    import PlaceObj as TopPlaceObj  # type: ignore
    DpPlaceObj.PlaceObj.obj_fn = patched_obj_fn
    DpPlaceObj.PlaceObj.obj_and_grad_fn = patched_obj_and_grad_fn
    TopPlaceObj.PlaceObj.obj_fn = patched_obj_fn
    TopPlaceObj.PlaceObj.obj_and_grad_fn = patched_obj_and_grad_fn
    # Verify patch installed
    print(f"  PATCH INSTALLED: dreamplace.PlaceObj.PlaceObj.obj_fn = {DpPlaceObj.PlaceObj.obj_fn}")
    print(f"  PATCH INSTALLED: top-level PlaceObj.PlaceObj.obj_fn = {TopPlaceObj.PlaceObj.obj_fn}")
    print(f"  DpPlaceObj module path: {DpPlaceObj.__file__}")
    print(f"  TopPlaceObj module path: {TopPlaceObj.__file__}")
    print(f"  Same class? {DpPlaceObj.PlaceObj is TopPlaceObj.PlaceObj}")
    state["original_obj_and_grad_fn"] = original_obj_and_grad_fn

    def restore():
        setattr(DpPlaceObj.PlaceObj, "obj_fn", original_obj_fn)
        setattr(DpPlaceObj.PlaceObj, "obj_and_grad_fn", original_obj_and_grad_fn)
        try:
            import PlaceObj as TopPlaceObj
            setattr(TopPlaceObj.PlaceObj, "obj_fn", original_obj_fn)
            setattr(TopPlaceObj.PlaceObj, "obj_and_grad_fn", original_obj_and_grad_fn)
        except Exception:
            pass
    state["restore"] = restore
    state["macro_to_dp"] = macro_to_dp
    state["macro_half_w"] = macro_half_w
    state["macro_half_h"] = macro_half_h
    state["num_nodes"] = num_nodes
    state["dp_to_our_scale"] = dp_to_our_scale
    return state


def run_dp_on_bench(
    bench_name: str, *, num_iter: int = 1000,
    challenge_w: float = 1.0, edensity_w: float = 0.0,
    target_density: float = 0.85,
):
    """Run patched DP on a single benchmark."""
    import torch
    from macro_place.bench_paths import find_benchmark_dir
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost, compute_overlap_metrics
    from macro_place.cd_core import project_overlaps, run_cd_adaptive
    from macro_place.incremental_evaluator import IncrementalProxyEvaluator
    from macro_legalizer import greedy_macro_legalize
    import bookshelf_to_pt as bk_reader

    print(f"\n=== {bench_name} ===")
    bench_dir = find_benchmark_dir(bench_name)
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))

    import dreamplace.Params as DpParams
    import dreamplace.PlaceDB as DpPlaceDB
    import dreamplace.NonLinearPlace as DpNonLinearPlace

    with tempfile.TemporaryDirectory(prefix=f"dp_{bench_name}_") as tmp:
        tmp_path = Path(tmp)
        aux_path = write_bookshelf(benchmark, tmp_path)
        print(f"  bookshelf: {aux_path}")

        params_dict = {
            "aux_input": str(aux_path),
            "target_density": target_density,
            "density_weight": 8e-5,
            "gpu": 1 if torch.cuda.is_available() else 0,
            "num_threads": 8,
            "deterministic_flag": 1,
            "global_place_stages": [{
                "num_bins_x": 1024, "num_bins_y": 1024,
                "iteration": num_iter, "learning_rate": 0.01,
                "wirelength": "weighted_average", "optimizer": "nesterov",
            }],
            "legalize_flag": 0, "detailed_place_flag": 0,
            "stop_overflow": 0.07,
            "result_dir": str(tmp_path),
            "global_place_flag": 1, "macro_place_flag": 0,
            "scale_factor": 1.0, "random_center_init_flag": 0,
            "sort_nets_by_degree": 0,
            "num_bins_x": 1024, "num_bins_y": 1024,
            "global_swap_flag": 0, "k_reorder_flag": 0,
            "independent_set_matching_flag": 0,
            "macro_halo_x": 0, "macro_halo_y": 0,
            "regioned_global_place_flag": 0,
            "routability_opt_flag": 0,
            "timing_opt_flag": 0,
            "dtype": "float32",
            "force_cpu_flag": 0,
            "filler_size_x_ratio": 0.5, "filler_size_y_ratio": 0.5,
        }
        params = DpParams.Params()
        params.fromJson(params_dict)
        # params.printParams()

        placedb = DpPlaceDB.PlaceDB()
        placedb(params)

        # Install the challenge-proxy patch BEFORE NonLinearPlace runs
        patch_state = patch_dp_obj_fn(
            placedb, benchmark, plc,
            challenge_weight=challenge_w,
            edensity_weight=edensity_w,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        try:
            t0 = time.time()
            placer = DpNonLinearPlace.NonLinearPlace(params, placedb, timer=None)
            placer(params, placedb, learning_rate_value=0.01)
            wall = time.time() - t0
            print(f"  DP wall: {wall:.1f}s, obj_fn calls: {patch_state['call_count']}")
        finally:
            patch_state["restore"]()

        # Read final placement positions: use placer.pos[0] directly
        with torch.no_grad():
            pos = placer.pos[0].detach()
            num_nodes = patch_state["num_nodes"]
            macro_to_dp = patch_state["macro_to_dp"].to(pos.device)
            dp_x = pos[:num_nodes][macro_to_dp]
            dp_y = pos[num_nodes:2 * num_nodes][macro_to_dp]
            # Apply scale to convert DP coords → our microns
            scale = patch_state.get("dp_to_our_scale", 1.0)
            scaled_x = dp_x * scale
            scaled_y = dp_y * scale
            cx = scaled_x + patch_state["macro_half_w"].to(pos.device)
            cy = scaled_y + patch_state["macro_half_h"].to(pos.device)
            final_centers = torch.stack([cx, cy], dim=1).cpu().to(torch.float32)

        # Patch fixed macros (terminals) to their original positions
        # since DP may have shifted them numerically
        fixed_mask = benchmark.macro_fixed.bool()
        final_centers[fixed_mask] = benchmark.macro_positions[fixed_mask].to(torch.float32)

        ovl_dp = compute_overlap_metrics(final_centers, benchmark)["overlap_count"]
        proxy_dp = float(compute_proxy_cost(final_centers, benchmark, plc)["proxy_cost"])
        print(f"  DP raw: proxy={proxy_dp:.5f} ovl={ovl_dp}")

        # Legalize
        legal_pos, leg_stats = greedy_macro_legalize(
            final_centers, benchmark,
            search_radius_steps=80, step_size_frac=0.02,
            verbose=False,
        )
        ovl_leg = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        if ovl_leg > 0:
            legal_pos, _ = project_overlaps(legal_pos, benchmark)
            ovl_leg = compute_overlap_metrics(legal_pos, benchmark)["overlap_count"]
        proxy_leg = float(compute_proxy_cost(legal_pos, benchmark, plc)["proxy_cost"])
        print(f"  Legalize: proxy={proxy_leg:.5f} ovl={ovl_leg} "
              f"moved={leg_stats.get('n_moved')} max_disp={leg_stats.get('max_displacement', -1):.1f}")

        # CD polish
        cd_budget = 300.0
        ev = IncrementalProxyEvaluator(benchmark, plc, legal_pos.clone())
        fixed = benchmark.macro_fixed.cpu().numpy()
        movable = [i for i in range(benchmark.num_hard_macros) if not bool(fixed[i])]
        run_cd_adaptive(
            ev, benchmark, plc, movable,
            min_time_s=cd_budget * 0.5, hard_cap_s=cd_budget,
            patience=3, plateau_threshold=0.001, log_fn=None,
        )
        polished = ev.placement.detach().clone().to(torch.float32)
        polished, _ = project_overlaps(polished, benchmark)
        ovl_final = compute_overlap_metrics(polished, benchmark)["overlap_count"]
        proxy_final = float(compute_proxy_cost(polished, benchmark, plc)["proxy_cost"])
        print(f"  CD polish: proxy={proxy_final:.5f} ovl={ovl_final}")

        return {
            "bench": bench_name,
            "proxy_dp_raw": proxy_dp,
            "proxy_legalized": proxy_leg,
            "proxy_final": proxy_final,
            "ovl_final": int(ovl_final),
            "dp_wall_s": wall,
            "challenge_w": challenge_w,
            "edensity_w": edensity_w,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", default="ibm01", nargs="?")
    ap.add_argument("--num-iter", type=int, default=1000)
    ap.add_argument("--challenge-w", type=float, default=1.0)
    ap.add_argument("--edensity-w", type=float, default=0.0)
    ap.add_argument("--target-density", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = run_dp_on_bench(
        args.bench,
        num_iter=args.num_iter,
        challenge_w=args.challenge_w,
        edensity_w=args.edensity_w,
        target_density=args.target_density,
    )
    print(f"\n=== FINAL ===")
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
