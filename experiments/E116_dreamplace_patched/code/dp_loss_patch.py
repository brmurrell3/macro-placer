"""E116 Path B — Monkey-patch DREAMPlace's obj_fn to use the challenge proxy.

DREAMPlace's stock loss is `wirelength + density_weight * eDensity`. We
need `LSE-HPWL + 0.5*top-K-density + 0.5*PerNetTraceCongestion`. The
patch replaces `PlaceObj.obj_fn` with our version that:

  1. Extracts movable macro positions from DREAMPlace's `pos` tensor
     using the name-to-id map (since Bookshelf reorders nodes).
  2. Combines with fixed macro positions to build a (num_macros, 2)
     tensor in OUR macro ordering.
  3. Computes challenge proxy via E88 LSE-HPWL + E88 top-K density +
     E111 PerNetTraceCongestion.
  4. Adds DREAMPlace's eDensity term as a SOFT SPREADING term, with
     small coefficient (gives gradient signal to push movables apart
     even when the challenge proxy gradient is locally flat).

The eDensity term is critical: without it, the descent doesn't have
the strong spreading signal that DREAMPlace's Nesterov + density-weight
ramping is designed for. With it, we get DP's optimizer benefits +
our challenge-aligned loss.

Run as: this module exposes `patch_place_obj(model, benchmark, plc)`
which monkey-patches `model.obj_fn` in place. Called by `dp_runner.py`
between PlaceObj construction and Nesterov optimization.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch

# Path setup expected to be done by the caller (dp_runner.py)
# since this module needs to import from macro_place/ on the user's side.


def make_patched_obj_fn(
    model,  # dreamplace.PlaceObj instance
    placedb,
    benchmark,  # our macro_place.Benchmark
    plc,
    per_net_trace_cls,
    diff_proxy_v2_cls,
    grid_density_fn,
    lse_hpwl_fn,
    *,
    challenge_weight: float = 1.0,
    edensity_weight: float = 0.05,
    trace_kwargs: Optional[Dict] = None,
    device: str = "cuda",
):
    """Build a replacement `obj_fn` that returns
    challenge_weight * challenge_proxy + edensity_weight * edensity.

    Returns a tuple `(new_obj_fn, state_dict)` where state_dict carries
    intermediate tensors for inspection.

    Notes:
      - DREAMPlace stores positions as `pos = [x_0..x_N-1, y_0..y_N-1]`
        with `N = num_nodes = num_physical + num_filler`.
      - The first `num_movable_nodes` entries are movable macros.
        Order matches Bookshelf .nodes file ordering.
      - In our Bookshelf write, we use node names `n{i}` for macro i.
        DREAMPlace's `placedb.node_name2id_map` maps these back.
    """
    state = {}
    device_t = torch.device(device)
    num_macros = int(benchmark.num_macros)
    num_movable = int(placedb.num_movable_nodes)
    num_nodes = int(placedb.num_nodes)
    num_physical = int(placedb.num_physical_nodes)

    # Build mapping: our macro index `i` → DP node index `dp_i`
    # via `node_name2id_map["n{i}"]`.
    macro_to_dp = torch.zeros(num_macros, dtype=torch.long, device=device_t)
    for i in range(num_macros):
        name = f"n{i}".encode("utf-8")
        if name in placedb.node_name2id_map:
            macro_to_dp[i] = placedb.node_name2id_map[name]
        else:
            # Some bookshelf parsers store names as str. Fallback.
            macro_to_dp[i] = placedb.node_name2id_map[f"n{i}"]
    state["macro_to_dp"] = macro_to_dp

    # Identify fixed macros (terminals in DP terms, fixed in ours).
    # Movable macros: macro_to_dp[i] < num_movable
    # Fixed macros: num_movable <= macro_to_dp[i] < num_physical
    fixed_mask = benchmark.macro_fixed.bool().to(device_t)
    state["fixed_mask"] = fixed_mask

    # Bookshelf scale factor — we wrote positions multiplied by SCALE.
    # Read from environment / config.
    SCALE = float(state.get("SCALE", 1000))
    state["SCALE"] = SCALE

    # Half-sizes for converting llx,lly → centers
    macro_half_w = (benchmark.macro_sizes[:, 0] / 2.0).to(device_t)
    macro_half_h = (benchmark.macro_sizes[:, 1] / 2.0).to(device_t)
    state["macro_half_w"] = macro_half_w
    state["macro_half_h"] = macro_half_h

    # Build PerNetTraceCongestion + DiffProxyV2 in our challenge basis.
    diff_proxy = diff_proxy_v2_cls(
        benchmark, plc, device=str(device_t), gamma_frac=5e-3,
    )
    state["diff_proxy"] = diff_proxy
    trace = per_net_trace_cls(
        benchmark, plc, device=str(device_t), **(trace_kwargs or {})
    )
    state["trace"] = trace

    # Save originals so we can call DP's eDensity / WL ops too.
    state["original_obj_fn"] = model.obj_fn
    state["original_wl_op"] = model.op_collections.wirelength_op
    state["original_density_op"] = model.op_collections.density_op

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    def dp_pos_to_macro_centers(pos: torch.Tensor) -> torch.Tensor:
        """Extract (num_macros, 2) centers in MICRONS from DP's flat pos.

        DP's pos is in Bookshelf-scaled units (LLx/LLy of each node).
        Convert: center = (llx_in_microns + half_w) where
                 llx_in_microns = (llx_in_bookshelf - placedb.xl) / SCALE
        ... but actually DP normalizes positions during PlaceDB.scale_pl.

        Simpler: read placedb.xl, placedb.yl, infer normalization, and
        use the formula. For now: assume pos is in micron units already
        (we'll calibrate).
        """
        # DP stores LLX/LLY of each node. To get centers in OUR
        # coordinate space (microns from origin), we need:
        #   center_x = (pos_llx - placedb.xl) + node_size_x/2   [for movable]
        # ... but during scale_pl, DP shifts and scales to internal
        # coordinates. The safest approach is to use placedb.unscale_pl()
        # — but that's expensive per step.
        #
        # Alternative: we know our bookshelf put macro `i` at LLX
        # corresponding to (cx - w/2) * SCALE. DP's scale_pl divides
        # by SCALE typically (= unit conversion). So DP's internal pos
        # for our macro `i` is approximately (cx - w/2) in microns.
        # Thus:
        #   our_cx = pos_x[dp_i] + macro_half_w[i]
        #   our_cy = pos_y[dp_i] + macro_half_h[i]
        dp_x = pos[:num_nodes][macro_to_dp]  # [num_macros]
        dp_y = pos[num_nodes:2*num_nodes][macro_to_dp]
        cx = dp_x + macro_half_w
        cy = dp_y + macro_half_h
        return torch.stack([cx, cy], dim=1)  # [num_macros, 2]

    state["dp_pos_to_macro_centers"] = dp_pos_to_macro_centers

    def patched_obj_fn(pos: torch.Tensor) -> torch.Tensor:
        """Returns challenge_weight * challenge_proxy + edensity_weight * eDensity.

        pos: DP's flat position tensor [2*num_nodes], with autograd.
        Returns: scalar tensor (with grad).
        """
        # 1. eDensity from DP (preserves gradient on pos)
        edensity = state["original_density_op"](pos)
        if isinstance(edensity, torch.Tensor) and edensity.dim() > 0:
            # In DP, density_op may return a multi-element tensor (fence regions)
            edensity_scalar = edensity.sum()
        else:
            edensity_scalar = edensity

        # 2. Challenge proxy on our macro positions
        macro_centers = dp_pos_to_macro_centers(pos)
        # Clamp to canvas (so the proxy doesn't blow up early in descent)
        clamped = torch.stack([
            macro_centers[:, 0].clamp(macro_half_w, cw - macro_half_w),
            macro_centers[:, 1].clamp(macro_half_h, ch - macro_half_h),
        ], dim=1)

        wl = lse_hpwl_fn(clamped, diff_proxy.net_data, diff_proxy.port_base, diff_proxy.gamma) / diff_proxy.wl_norm
        density = grid_density_fn(
            clamped, diff_proxy.sizes,
            diff_proxy.cell_x_min, diff_proxy.cell_x_max,
            diff_proxy.cell_y_min, diff_proxy.cell_y_max,
            diff_proxy.cell_area, diff_proxy.grid_rows, diff_proxy.grid_cols,
        )
        cong = trace.compute_congestion(clamped)
        challenge = wl + 0.5 * density + 0.5 * cong

        state["last_challenge"] = challenge.detach()
        state["last_edensity"] = edensity_scalar.detach()
        state["last_wl"] = wl.detach()
        state["last_density"] = density.detach()
        state["last_cong"] = cong.detach()

        return challenge_weight * challenge + edensity_weight * edensity_scalar

    return patched_obj_fn, state


def install_patch(model, **kwargs):
    """Convenience: install the patched obj_fn on model.

    Returns the `state` dict for later inspection / debugging.
    """
    new_fn, state = make_patched_obj_fn(model, **kwargs)
    model.obj_fn = new_fn
    return state
