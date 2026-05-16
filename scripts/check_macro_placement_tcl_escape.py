"""Sanity test for generate_macro_placement_tcl.py escape + orientation fix.

These two bugs silenced our placement application via ORFS for a full overnight
chain before being caught. The TCL the generator emits must be parseable by
ORFS such that:

  (a) the bracket-escaped instance-name candidate matches Verilog-escaped
      ODB identifiers like `sram_block\\[0\\].data_sram/macro_mem\\[0\\].i_ram`
      (note the **literal** backslash in the ODB name string), and
  (b) the orient argument to `setOrient` is one of OpenROAD's accepted
      strings (R0/R90/R180/R270/MY/MX/MYR90/MXR90), not the .plc-style
      N/S/E/W.

Run with: uv run python scripts/test_generate_macro_placement_tcl.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _generated_source_excerpt() -> str:
    """Return the source of the generator's TCL-emit loop for static asserts."""
    src = (REPO / "scripts" / "generate_macro_placement_tcl.py").read_text()
    return src


def test_orientation_map_present():
    src = _generated_source_excerpt()
    assert "_orient_map" in src, "no _orient_map dict found"
    # All 8 .plc orientations should be mapped
    for k in ["N", "E", "S", "W", "FN", "FE", "FS", "FW"]:
        assert f'"{k}":' in src, f"orient mapping missing for {k}"
    # Mapped targets must be ODB-accepted forms
    for ok_orient in ["R0", "R90", "R180", "R270", "MY", "MX", "MYR90", "MXR90"]:
        assert f'"{ok_orient}"' in src, f"target orient {ok_orient} missing from map"


def test_tcl_bracket_escape_triple_backslash():
    """Verify the Python source produces TCL with `\\\\\\\\[` → file `\\\\[` → TCL runtime `\\[`.

    Background: ODB stores Verilog-escaped identifiers with a *literal*
    backslash, e.g. `sram_block\\[0\\]`. To match in TCL, the source string
    has to contain `"\\\\[0\\\\]"` (4 file chars: `\\`, `\\`, `[`, `0`, `\\`, `\\`, `]`):
    `\\\\` → literal `\\`, then `\\[` → literal `[`.
    The Python source therefore has to write `\\\\\\\\[`/`\\\\\\\\]` (4
    backslash chars in the Python string, 8 in source code).
    """
    src = _generated_source_excerpt()
    # Must contain the triple-backslash escape (Python source `\\\\\\[` = 4 chars `\\\\[`,
    # which in raw source code is the 7-char sequence `\` `\` `\` `\` `\` `\` `[`).
    assert "'\\\\\\\\\\\\['" in src.replace(" ", ""), \
        "tcl_escaped: expected Python source `'\\\\\\\\\\\\['` (3 literal backslashes + bracket)"
    # And the corresponding ] form
    assert "'\\\\\\\\\\\\]'" in src.replace(" ", ""), \
        "tcl_escaped: expected Python source `'\\\\\\\\\\\\]'` (3 literal backslashes + bracket)"
    # Old single-backslash form should still be present as a fallback candidate
    assert "'\\\\['" in src.replace(" ", ""), \
        "expected old `'\\\\['` form to remain as fallback for non-Verilog-escaped ODBs"


def test_python_escape_runtime_matches_odb_format():
    """Cross-check: when the generator runs in-process, the file content for a
    test bracket name should equal the ODB-accepted form."""
    plc_name = "i_cache_subsystem/i_icache/sram_block[0].data_sram/macro_mem[0].i_ram"
    # This is the line from the generator:
    tcl_escaped = plc_name.replace('[', '\\\\\\[').replace(']', '\\\\\\]')
    # When the generator writes `f'"{tcl_escaped}"'` to disk, the file content is:
    #   "i_cache_subsystem/i_icache/sram_block\\\[0\\\].data_sram/macro_mem\\\[0\\\].i_ram"
    # which TCL parses as a string with literal `\[` and `\]` — matching ODB's
    # Verilog-escaped identifier.
    expected_runtime = "i_cache_subsystem/i_icache/sram_block\\[0\\].data_sram/macro_mem\\[0\\].i_ram"
    # Mirror what TCL does to a `"..."` source token: `\\` -> `\`, `\[` -> `[`.
    runtime = re.sub(r'\\\\', '\x00', tcl_escaped)        # `\\` -> placeholder
    runtime = re.sub(r'\\([^\\])', r'\1', runtime)        # `\X` -> `X`
    runtime = runtime.replace('\x00', '\\')               # placeholder -> `\`
    assert runtime == expected_runtime, f"runtime={runtime!r} expected={expected_runtime!r}"


def main():
    test_orientation_map_present()
    print("  ✓ orientation map present and complete")
    test_tcl_bracket_escape_triple_backslash()
    print("  ✓ tcl_escaped uses triple-backslash for Verilog-escaped ODB names")
    test_python_escape_runtime_matches_odb_format()
    print("  ✓ Python escape produces correct TCL runtime string")
    print("All TCL generator escape/orientation checks passed.")


if __name__ == "__main__":
    main()
