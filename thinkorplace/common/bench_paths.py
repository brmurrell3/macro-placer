"""Generic benchmark-directory discovery for placer reload of PlacementCost.

Problem: the eval harness calls ``placer.place(benchmark)`` with only the
``Benchmark`` dataclass — no path, no plc. Many of our placers (CDAdaptive,
CDOnly, SDF init) need to reload the underlying ``PlacementCost`` to access
fields the dataclass doesn't carry (per-pin info, smoothing range, etc.).

This module supplies a single helper, ``find_benchmark_dir(name)``, that
locates the source directory for a benchmark in a generic way. For each
candidate root we try two layouts:

  • ``<root>/<name>/``                                    (IBM ICCAD04)
  • ``<root>/<name>/netlist/output_CT_Grouping/``        (NG45 Flows)

Candidate roots, in order:

  1. ``$BENCH_ROOT`` environment variable, if set.
  2. ``external/MacroPlacement/Testcases/ICCAD04``       (IBM)
  3. ``external/MacroPlacement/Testcases/NG45``          (reserved)
  4. ``external/MacroPlacement/Flows/NanGate45``         (public NG45)

No rglob fallback: the TILOS submodule contains test fixtures with the same
``{name}/netlist.pb.txt`` layout (e.g. ``CodeElements/Plc_client/test/
ariane133/``) that have *different* grid/routing parameters than the real
designs. Falling through to rglob silently loaded the wrong netlist.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List


from macro_place import __file__ as _infrastructure_file
_REPO_ROOT = Path(_infrastructure_file).resolve().parent.parent

# Sub-paths to try beneath each candidate root. Empty string = root itself.
_LAYOUT_SUFFIXES = ("", "netlist/output_CT_Grouping")


def _candidate_roots() -> List[Path]:
    roots: List[Path] = []
    env = os.environ.get("BENCH_ROOT")
    if env:
        roots.append(Path(env))
    mp = _REPO_ROOT / "external" / "MacroPlacement"
    roots.extend([
        mp / "Testcases" / "ICCAD04",
        mp / "Testcases" / "NG45",
        mp / "Flows" / "NanGate45",
    ])
    return roots


def find_benchmark_dir(name: str) -> Path:
    """Return the source directory for a benchmark by name.

    Raises FileNotFoundError if no candidate path resolves. The error
    message lists every path tried so misconfigurations are obvious.
    """
    tried: List[str] = []
    # Processed NG45 benchmarks append _ng45; source flow directories do not.
    # Prefer an exact directory, then the explicit suffix alias. Never recurse
    # into arbitrary fixtures with a coincidentally matching design name.
    names = (name, name[:-5]) if name.endswith("_ng45") else (name,)
    roots = _candidate_roots()
    for source_name in names:
        for root in roots:
            for suffix in _LAYOUT_SUFFIXES:
                cand = root / source_name / suffix if suffix else root / source_name
                tried.append(str(cand))
                if (cand / "netlist.pb.txt").exists():
                    return cand

    raise FileNotFoundError(
        f"Benchmark '{name}': no source dir found.\n"
        f"  Tried: {tried}\n"
        f"  Set $BENCH_ROOT to override the search root."
    )
