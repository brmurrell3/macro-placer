"""Extract pin/net data matching canonical plc.get_routing exactly.

For each module in plc.modules_w_pins that has get_sink() non-empty (PORT or MACRO_PIN),
emit a per-net record:
  - source_pin = (macro_idx, offset_xy)
  - sink_pins  = [(macro_idx, offset_xy), ...]
  - weight     = mod.get_weight() (default 1)

Then for routing, split N-pin nets matching canonical's __split_net + __three_pin_net_routing.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, NamedTuple, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class CanonNet(NamedTuple):
    src_macro_idx: int
    src_offset: Tuple[float, float]   # offset from macro center
    sink_macro_indices: List[int]
    sink_offsets: List[Tuple[float, float]]
    weight: float


def extract_canon_nets(benchmark, plc) -> List[CanonNet]:
    """Iterate plc.modules_w_pins, extract one net per (PORT or MACRO_PIN with sinks).

    Macro indexing: PORT pins map to macro_idx=N (port-pseudo-macro at origin),
    MACRO_PIN pins map to their parent macro_idx in benchmark.
    Pin offset = pin absolute position minus parent macro center.

    Returns: list of CanonNet records, one per net in canonical's routing walk.
    """
    nets = []
    # Build name → macro_idx mapping
    n_macros = benchmark.num_macros

    # Walk all modules in plc.modules_w_pins
    for mod_idx, mod in enumerate(plc.modules_w_pins):
        ttype = mod.get_type()
        # Only PORT and MACRO_PIN types have get_sink()
        if ttype not in ("PORT", "MACRO_PIN"):
            continue
        sinks = mod.get_sink()
        if not sinks:
            continue

        weight = float(mod.get_weight()) if hasattr(mod, "get_weight") and mod.get_weight() > 1 else 1.0

        # Source = this driver pin
        if ttype == "PORT":
            # Port pseudo-macro at index n_macros (after real macros)
            src_macro = n_macros
            src_pos = mod.get_pos()
            src_off = (float(src_pos[0]), float(src_pos[1]))
        else:  # MACRO_PIN
            parent_name = mod.get_macro_name() if hasattr(mod, "get_macro_name") else None
            if parent_name is None:
                continue
            # Find parent macro index
            if hasattr(plc, "mod_name_to_indices"):
                parent_idx = plc.mod_name_to_indices.get(parent_name)
                if parent_idx is None:
                    continue
            else:
                continue
            # Pin position (absolute)
            pin_pos_fn = getattr(plc, "_PlcClient__get_pin_position", None)
            if pin_pos_fn is None:
                # Fallback: use mod's offset relative to macro
                pin_pos = mod.get_pos()
            else:
                pin_pos = pin_pos_fn(mod_idx)
            # Parent macro center
            parent_mod = plc.modules_w_pins[parent_idx] if parent_idx < len(plc.modules_w_pins) else None
            if parent_mod is None:
                continue
            macro_x, macro_y = parent_mod.get_pos()
            # Map parent macro to bench macro_idx (assumes first n_macros entries are real macros)
            src_macro = parent_idx
            src_off = (float(pin_pos[0]) - float(macro_x), float(pin_pos[1]) - float(macro_y))

        # Each (net_name → sink_list) entry is one canonical net
        for net_name, sink_list in sinks.items():
            sink_macros = []
            sink_offs = []
            for sink_name in sink_list:
                if hasattr(plc, "mod_name_to_indices"):
                    sink_idx = plc.mod_name_to_indices.get(sink_name)
                    if sink_idx is None:
                        continue
                else:
                    continue
                sink_mod = plc.modules_w_pins[sink_idx] if sink_idx < len(plc.modules_w_pins) else None
                if sink_mod is None:
                    continue
                # Sink pin position
                pin_pos_fn = getattr(plc, "_PlcClient__get_pin_position", None)
                if pin_pos_fn is None:
                    sink_pin_pos = sink_mod.get_pos()
                else:
                    sink_pin_pos = pin_pos_fn(sink_idx)
                # If sink is PORT, no parent; else parent macro center
                if sink_mod.get_type() == "PORT":
                    sink_macros.append(n_macros)
                    sink_offs.append((float(sink_pin_pos[0]), float(sink_pin_pos[1])))
                else:
                    parent_name = sink_mod.get_macro_name() if hasattr(sink_mod, "get_macro_name") else None
                    if parent_name is None:
                        continue
                    parent_idx = plc.mod_name_to_indices.get(parent_name, -1)
                    if parent_idx < 0:
                        continue
                    parent_mod = plc.modules_w_pins[parent_idx]
                    macro_x, macro_y = parent_mod.get_pos()
                    sink_macros.append(parent_idx)
                    sink_offs.append((float(sink_pin_pos[0]) - float(macro_x), float(sink_pin_pos[1]) - float(macro_y)))
            if not sink_macros:
                continue
            nets.append(CanonNet(
                src_macro_idx=src_macro,
                src_offset=src_off,
                sink_macro_indices=sink_macros,
                sink_offsets=sink_offs,
                weight=weight,
            ))
    return nets


def quick_count(bench_name: str = "ibm01"):
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.bench_paths import find_benchmark_dir
    bench, plc = load_benchmark_from_dir(str(find_benchmark_dir(bench_name)))
    nets = extract_canon_nets(bench, plc)
    print(f"{bench_name}: extracted {len(nets)} canonical nets")
    pin_count = {2: 0, 3: 0}
    pin_count_more = 0
    for n in nets:
        total_pins = 1 + len(n.sink_macro_indices)
        if total_pins in pin_count:
            pin_count[total_pins] += 1
        else:
            pin_count_more += 1
    print(f"  2-pin: {pin_count[2]}, 3-pin: {pin_count[3]}, 4+: {pin_count_more}")


if __name__ == "__main__":
    quick_count(sys.argv[1] if len(sys.argv) > 1 else "ibm01")
