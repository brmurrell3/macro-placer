"""B-R2: monkey-patch DREAMPlace's PlaceObj.obj_fn to add canonical losses.

Usage from a DREAMPlace runner:

    from placeobj_patch import patch_obj_fn
    place_obj = PlaceObj(...)
    patch_obj_fn(place_obj, lambda_topk=1.0, lambda_rudy=0.5,
                 num_bins_x=32, num_bins_y=32, K_frac=0.10, tau=0.05)
    # Now place_obj.obj_fn returns wl + dw * eDensity + λ_topk * topk + λ_rudy * rudy

The patch wraps the original `obj_fn` with a closure that adds our
canonical losses. Original obj_fn is preserved as `_obj_fn_original`.

Important caveats:
  - DREAMPlace's `pos` is a flat tensor: pos = [x_0, ..., x_N, y_0, ..., y_N].
    We slice to extract movable-node positions.
  - DREAMPlace optimizes movable nodes + filler nodes. We compute canonical
    losses only on movable nodes (num_movable_nodes excludes fillers).
  - Sizes come from `data_collections.node_size_x` / `node_size_y`.
"""
from __future__ import annotations

import torch
import sys
import os

# Make canonical_losses importable
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from canonical_losses import topk_density_loss, rudy_congestion_loss


def patch_obj_fn(
    place_obj,
    lambda_topk: float = 0.0,
    lambda_rudy: float = 0.0,
    num_bins_x: int = 32,
    num_bins_y: int = 32,
    K_frac: float = 0.10,
    tau: float = 0.05,
    skip_fillers: bool = True,
) -> None:
    """Monkey-patch place_obj.obj_fn to add canonical losses.

    Args:
      place_obj: an instance of DREAMPlace's PlaceObj
      lambda_topk: weight for top-K density loss
      lambda_rudy: weight for RUDY congestion loss
      num_bins_x, num_bins_y: grid for canonical density/congestion
      K_frac: top-K fraction (canonical uses 0.10 = top 10%)
      tau: softmax temperature for soft top-K
      skip_fillers: only compute losses on non-filler movable nodes
    """
    if hasattr(place_obj, '_obj_fn_original'):
        # already patched; re-patch with new lambdas
        original = place_obj._obj_fn_original
    else:
        original = place_obj.obj_fn
        place_obj._obj_fn_original = original

    placedb = place_obj.placedb
    data_collections = place_obj.data_collections
    n_mov = placedb.num_movable_nodes
    n_filler = placedb.num_filler_nodes
    n_total = placedb.num_nodes

    # Macros are typically num_movable_nodes - num_filler_nodes for our
    # bookshelf-from-tilos conversion. For ICCAD04 IBM (200-760 macros)
    # we have no filler nodes (added by DP based on placement density).
    if skip_fillers:
        n_canon = n_mov - n_filler
    else:
        n_canon = n_mov

    xl, yl, xh, yh = placedb.xl, placedb.yl, placedb.xh, placedb.yh

    # Sizes (fixed, no grad)
    sz_x = data_collections.node_size_x[:n_canon].detach()
    sz_y = data_collections.node_size_y[:n_canon].detach()
    sizes = torch.stack([sz_x, sz_y], dim=1)

    # Build netpin tensors. DP keeps netpin_start (CSR pointers) and
    # flat_netpin (pin indices). Pin positions are macro centers in our setup.
    # We need to filter to nets/pins where the pin's owner is a movable macro.
    if lambda_rudy > 0:
        netpin_start_full = data_collections.netpin_start
        flat_netpin_full = data_collections.flat_netpin
        # Pin-to-node mapping in DP: pin2node_map maps pin idx -> node idx.
        pin2node = data_collections.pin2node_map
        # Net weights
        net_weights = data_collections.net_weights.detach() if hasattr(data_collections, 'net_weights') else torch.ones(netpin_start_full.shape[0] - 1, dtype=torch.float32, device=sz_x.device)
        # We use the existing structure as-is; for non-macro pins we treat their
        # owning node's center as the pin position. This requires building a
        # full position vector below.

    def new_obj_fn(pos):
        # Original DREAMPlace objective
        result = original(pos)
        if lambda_topk <= 0.0 and lambda_rudy <= 0.0:
            return result

        # Extract per-node positions. pos format: pos[0:n_total] = x, pos[n_total:] = y.
        # We slice the first n_canon for movable, non-filler nodes.
        # Canon positions: macros (non-fixed movable).
        mov_x = pos[:n_canon]
        mov_y = pos[n_total:n_total + n_canon]
        mov_pos = torch.stack([mov_x, mov_y], dim=1)

        if lambda_topk > 0.0:
            topk = topk_density_loss(
                mov_pos, sizes, xl, yl, xh, yh,
                num_bins_x=num_bins_x, num_bins_y=num_bins_y,
                K_frac=K_frac, tau=tau,
            )
            result = result + lambda_topk * topk

        if lambda_rudy > 0.0:
            # For RUDY we need per-pin positions. Pin position = owner node center.
            # Build all-node center positions (movable + fixed), then index by pin2node.
            all_x = pos[:n_total]
            all_y = pos[n_total:2 * n_total]
            # All nodes get their own (x, y). Sizes determine pin offset, but
            # our setup uses pin-at-center (offset 0,0). So pin_pos = node_pos.
            all_pos = torch.stack([all_x, all_y], dim=1)
            pin_pos = all_pos[pin2node]
            rudy = rudy_congestion_loss(
                pin_pos, netpin_start_full, flat_netpin_full, net_weights,
                xl, yl, xh, yh,
                num_bins_x=num_bins_x, num_bins_y=num_bins_y,
                K_frac=K_frac, tau=tau,
            )
            result = result + lambda_rudy * rudy

        return result

    place_obj.obj_fn = new_obj_fn
    place_obj.lambda_topk = lambda_topk
    place_obj.lambda_rudy = lambda_rudy


def calibrate_lambdas_at(
    place_obj,
    pos_sample: torch.Tensor,
    target_ratio_topk: float = 1.0,
    target_ratio_rudy: float = 0.5,
    num_bins_x: int = 32, num_bins_y: int = 32,
    K_frac: float = 0.10, tau: float = 0.05,
) -> tuple[float, float]:
    """Compute lambda values so canonical terms contribute target_ratio
    times the wirelength at the given placement.

    Args:
      place_obj: PlaceObj
      pos_sample: a representative pos tensor (e.g., post-init)
      target_ratio_topk: lambda_topk * topk(pos) / wl(pos) ≈ this
      target_ratio_rudy: lambda_rudy * rudy(pos) / wl(pos) ≈ this

    Returns: (lambda_topk, lambda_rudy)
    """
    placedb = place_obj.placedb
    data_collections = place_obj.data_collections
    n_mov = placedb.num_movable_nodes
    n_filler = placedb.num_filler_nodes
    n_total = placedb.num_nodes
    n_canon = n_mov - n_filler
    xl, yl, xh, yh = placedb.xl, placedb.yl, placedb.xh, placedb.yh

    with torch.no_grad():
        # Original wirelength
        place_obj._obj_fn_original = getattr(place_obj, '_obj_fn_original', place_obj.obj_fn)
        wl = float(place_obj.op_collections.wirelength_op(pos_sample))

        sz_x = data_collections.node_size_x[:n_canon].detach()
        sz_y = data_collections.node_size_y[:n_canon].detach()
        sizes = torch.stack([sz_x, sz_y], dim=1)

        mov_x = pos_sample[:n_canon]
        mov_y = pos_sample[n_total:n_total + n_canon]
        mov_pos = torch.stack([mov_x, mov_y], dim=1)

        topk = float(topk_density_loss(mov_pos, sizes, xl, yl, xh, yh,
                                        num_bins_x, num_bins_y, K_frac, tau))

        net_weights = data_collections.net_weights.detach() if hasattr(data_collections, 'net_weights') else torch.ones(data_collections.netpin_start.shape[0] - 1, dtype=torch.float32, device=sz_x.device)
        pin2node = data_collections.pin2node_map
        all_pos = torch.stack([pos_sample[:n_total], pos_sample[n_total:2 * n_total]], dim=1)
        pin_pos = all_pos[pin2node]
        rudy = float(rudy_congestion_loss(
            pin_pos, data_collections.netpin_start, data_collections.flat_netpin,
            net_weights, xl, yl, xh, yh, num_bins_x, num_bins_y, K_frac, tau))

    lambda_topk = target_ratio_topk * wl / max(topk, 1e-9)
    lambda_rudy = target_ratio_rudy * wl / max(rudy, 1e-9)
    return lambda_topk, lambda_rudy
