"""Build Xplace design_info .pt directly from TILOS benchmark, bypassing
Xplace's broken C++ bookshelf parser (which infinite-recurses on our format).

Output: a .pt file Xplace can load with `--load_from_raw False`.

Usage:
  uv run python tilos_to_xplace_pt.py <bench_name> [<out_dir>]

Then run Xplace with:
  python main.py --custom_path 'benchmark:<bench>,design_name:<bench>,aux:dummy' \\
    --load_from_raw False --dataset_root <out_root>
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.bench_paths import find_benchmark_dir
from macro_place.loader import load_benchmark_from_dir


def build_design_info(bench, plc, microns: int = 1000) -> dict:
    """Build the Xplace design_info dict from a TILOS Benchmark.

    Includes both macros (movable) and port pads (fixed IOPin).
    All Xplace coordinates are integer-strict (multiplied by `microns`).
    """
    n_macros = bench.num_macros
    sizes_m = bench.macro_sizes.cpu().float()  # (M, 2) μm
    fixed_m = bench.macro_fixed.cpu().bool()  # (M,)
    pos_m = bench.macro_positions.cpu().float()  # (M, 2) μm centers

    # Port pads: fixed IO pin pads, indexed M..M+P-1 in net_nodes
    if hasattr(bench, 'port_positions') and bench.port_positions is not None \
       and bench.port_positions.numel() > 0:
        port_pos = bench.port_positions.cpu().float()
        n_ports = int(port_pos.shape[0])
    else:
        port_pos = torch.zeros((0, 2), dtype=torch.float32)
        n_ports = 0

    n = n_macros + n_ports

    # Concatenate macros + ports
    sizes = torch.cat([sizes_m, torch.full((n_ports, 2), 0.001)], dim=0)  # tiny port size
    positions = torch.cat([pos_m, port_pos], dim=0)
    fixed = torch.cat([fixed_m, torch.ones(n_ports, dtype=torch.bool)], dim=0)

    canvas_w = float(bench.canvas_width)
    canvas_h = float(bench.canvas_height)

    # die_info = [dieLX, dieHX, dieLY, dieHY] in microns-scaled units
    die_info = torch.tensor(
        [0.0, canvas_w * microns, 0.0, canvas_h * microns]
    ).float()

    # node_pos = (N, 2) centers in scaled units
    node_pos = (positions * microns).contiguous()
    node_size = (sizes * microns).contiguous()
    node_lpos = (node_pos - node_size * 0.5).contiguous()

    # Build pin info from net_nodes (list of LongTensor of macro indices)
    pin_id2node_id_list = []  # one entry per (net, macro) — pin id
    hyperedge_index_pin = []  # pin ids
    hyperedge_index_net = []  # net ids
    hyperedge_list = []  # flat list of macros, grouped by net
    hyperedge_list_end = []  # cumulative
    pin_rel_cpos_list = []  # offset from macro center (we use 0)
    pin_rel_lpos_list = []  # offset from macro lower-left (= +size/2 since cpos=0)
    pin_size_list = []  # pin physical size (use small)
    pin_names = []
    node2pin_map = {i: [] for i in range(n)}  # node_id → list of pin ids

    cum = 0
    pin_id = 0
    for net_idx, net in enumerate(bench.net_nodes):
        macros = net.cpu().long().tolist() if hasattr(net, "tolist") else list(net)
        for m in macros:
            m = int(m)
            pin_id2node_id_list.append(m)
            hyperedge_index_pin.append(pin_id)
            hyperedge_index_net.append(net_idx)
            hyperedge_list.append(m)
            pin_rel_cpos_list.append([0.0, 0.0])
            # rel_lpos = pin position relative to macro lower-left = +size/2
            sz = sizes[m]
            pin_rel_lpos_list.append([float(sz[0]) * 0.5 * microns,
                                      float(sz[1]) * 0.5 * microns])
            pin_size_list.append([1.0, 1.0])  # nominal 1 unit
            pin_names.append(f"p{pin_id}")
            node2pin_map[m].append(pin_id)
            pin_id += 1
        cum += len(macros)
        hyperedge_list_end.append(cum)

    pin_id2node_id = torch.tensor(pin_id2node_id_list, dtype=torch.long)
    hyperedge_index = torch.tensor([hyperedge_index_pin, hyperedge_index_net],
                                    dtype=torch.long)
    hyperedge_list_t = torch.tensor(hyperedge_list, dtype=torch.long)
    hyperedge_list_end_t = torch.tensor(hyperedge_list_end, dtype=torch.long)
    pin_rel_cpos = torch.tensor(pin_rel_cpos_list, dtype=torch.float32)
    pin_rel_lpos = torch.tensor(pin_rel_lpos_list, dtype=torch.float32)
    pin_size = torch.tensor(pin_size_list, dtype=torch.float32)

    # node2pin
    node2pin_list_flat = []
    node2pin_list_end = []
    cum = 0
    for i in range(n):
        pins = node2pin_map[i]
        node2pin_list_flat.extend(pins)
        cum += len(pins)
        node2pin_list_end.append(cum)
    node2pin_list_t = torch.tensor(node2pin_list_flat, dtype=torch.long)
    node2pin_list_end_t = torch.tensor(node2pin_list_end, dtype=torch.long)
    # node2pin_index is unused per source — set to a placeholder shape
    node2pin_index = torch.zeros((2, len(node2pin_list_flat)), dtype=torch.long)
    # Xplace also uses (hyperedge_index, hyperedge_list, hyperedge_list_end)
    # Let's reverse-engineer node2pin_index: it pairs (pin_id, node_id) similar
    # to hyperedge_index. For now, fill with the same pin->node mapping.
    node2pin_index[0] = torch.arange(len(node2pin_list_flat), dtype=torch.long)
    node2pin_index[1] = pin_id2node_id[node2pin_list_flat]

    # Regions / fence: no fences in our problem, so single region covering canvas
    node_id2region_id = torch.zeros(n, dtype=torch.long)
    region_boxes = torch.tensor(
        [[0.0, canvas_w * microns, 0.0, canvas_h * microns]],
        dtype=torch.float32,
    )
    region_boxes_end = torch.tensor([1], dtype=torch.long)

    # Node type ordering: Mov (connected, movable), FloatMov (unconnected, movable),
    # Fix (connected, fixed), IOPin, Blkg, FloatIOPin, FloatFix
    # In our problem: all macros are "Mov" (connected, movable, unless macro_fixed)
    is_connected = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        if len(node2pin_map[i]) > 0:
            is_connected[i] = True

    # Macros are i < n_macros; ports are i >= n_macros (always fixed, treat as IOPin)
    mov_ids = [i for i in range(n_macros)
               if is_connected[i] and not fixed[i]]
    float_mov_ids = [i for i in range(n_macros)
                     if not is_connected[i] and not fixed[i]]
    fix_ids = [i for i in range(n_macros)
               if is_connected[i] and fixed[i]]
    iopin_ids = [i for i in range(n_macros, n)
                 if is_connected[i]]
    blkg_ids = []
    float_iopin_ids = [i for i in range(n_macros, n)
                       if not is_connected[i]]
    float_fix_ids = [i for i in range(n_macros)
                     if not is_connected[i] and fixed[i]]

    # We must REORDER nodes so that they are grouped: Mov, FloatMov, Fix, IOPin, Blkg, FloatIOPin, FloatFix
    new_order = mov_ids + float_mov_ids + fix_ids + iopin_ids + blkg_ids + \
                float_iopin_ids + float_fix_ids
    perm = torch.tensor(new_order, dtype=torch.long)
    inv_perm = torch.zeros(n, dtype=torch.long)
    inv_perm[perm] = torch.arange(n, dtype=torch.long)

    # Permute node tensors
    node_pos = node_pos[perm].contiguous()
    node_lpos = node_lpos[perm].contiguous()
    node_size = node_size[perm].contiguous()

    # Remap pin_id2node_id under the permutation
    pin_id2node_id = inv_perm[pin_id2node_id]

    # hyperedge_list is node ids — remap
    hyperedge_list_t = inv_perm[hyperedge_list_t]

    # node2pin: rebuild after perm
    new_node2pin = {}
    for new_idx, old_idx in enumerate(new_order):
        new_node2pin[new_idx] = node2pin_map[old_idx]
    node2pin_list_flat = []
    node2pin_list_end = []
    cum = 0
    for i in range(n):
        pins = new_node2pin[i]
        node2pin_list_flat.extend(pins)
        cum += len(pins)
        node2pin_list_end.append(cum)
    node2pin_list_t = torch.tensor(node2pin_list_flat, dtype=torch.long)
    node2pin_list_end_t = torch.tensor(node2pin_list_end, dtype=torch.long)
    node2pin_index = torch.zeros((2, len(node2pin_list_flat)), dtype=torch.long)
    node2pin_index[0] = torch.arange(len(node2pin_list_flat), dtype=torch.long)
    node2pin_index[1] = pin_id2node_id[node2pin_list_flat]

    node_id2region_id = node_id2region_id[perm]  # all zeros, no-op effectively

    # node_type_indices: (start, end, name)
    cur = 0
    type_indices = []
    for name, count in [
        ("Mov", len(mov_ids)),
        ("FloatMov", len(float_mov_ids)),
        ("Fix", len(fix_ids)),
        ("IOPin", len(iopin_ids)),
        ("Blkg", len(blkg_ids)),
        ("FloatIOPin", len(float_iopin_ids)),
        ("FloatFix", len(float_fix_ids)),
    ]:
        type_indices.append((cur, cur + count, name))
        cur += count

    # Names
    node_names = [f"n{i}" for i in range(n)]  # use orig IDs
    node_id2node_name = [node_names[old_idx] for old_idx in new_order]
    node_id2celltype_name = list(node_id2node_name)  # each cell its own type
    net_names = [f"net{i}" for i in range(len(bench.net_nodes))]

    movable_index = (0, len(mov_ids) + len(float_mov_ids))
    connected_index = (
        0,
        len(mov_ids) + len(float_mov_ids) + len(fix_ids) + len(iopin_ids),
    )
    fixed_index = (
        len(mov_ids) + len(float_mov_ids),
        len(mov_ids) + len(float_mov_ids) + len(fix_ids) + len(iopin_ids)
        + len(blkg_ids) + len(float_iopin_ids) + len(float_fix_ids),
    )

    site_info = (1.0, 1.0)  # 1x1 sites in scaled units

    design_info = {
        "benchmark": "ispd2005",
        "dataset_path": {"design_name": bench.name, "benchmark": "ispd2005"},
        "node_names": node_names,
        "net_names": net_names,
        "pin_names": pin_names,
        "microns": int(microns),
        "node_type_indices": type_indices,
        "node_id2node_name": node_id2node_name,
        "node_id2celltype_name": node_id2celltype_name,
        "movable_index": movable_index,
        "connected_index": connected_index,
        "fixed_index": fixed_index,
        "site_info": site_info,
        "die_info": die_info,
        "node_pos": node_pos,
        "node_lpos": node_lpos,
        "node_size": node_size,
        "pin_rel_cpos": pin_rel_cpos,
        "pin_rel_lpos": pin_rel_lpos,
        "pin_size": pin_size,
        "pin_id2node_id": pin_id2node_id,
        "hyperedge_index": hyperedge_index,
        "hyperedge_list": hyperedge_list_t,
        "hyperedge_list_end": hyperedge_list_end_t,
        "node2pin_index": node2pin_index,
        "node2pin_list": node2pin_list_t,
        "node2pin_list_end": node2pin_list_end_t,
        "node_id2region_id": node_id2region_id,
        "region_boxes": region_boxes,
        "region_boxes_end": region_boxes_end,
    }
    return design_info


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("out_dir", nargs="?", default=None)
    args = ap.parse_args()

    bench_dir = find_benchmark_dir(args.bench)
    b, plc = load_benchmark_from_dir(str(bench_dir))
    print(f"[xplace-pt] {args.bench}: n={b.num_macros}, nets={b.num_nets}")

    out_dir = Path(args.out_dir or _HERE.parent / "xplace_pt" / "ispd2005")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.bench}.pt"

    info = build_design_info(b, plc)
    torch.save(info, str(out_path))
    print(f"[xplace-pt] saved {out_path}")
    print(f"  node_pos: {info['node_pos'].shape}")
    print(f"  hyperedge_index: {info['hyperedge_index'].shape}")
    print(f"  hyperedge_list: {info['hyperedge_list'].shape}")
    print(f"  pin_id2node_id: {info['pin_id2node_id'].shape}")
    print(f"  type_indices: {info['node_type_indices']}")


if __name__ == "__main__":
    main()
