# Tier-2 ORFS findings — overnight 2026-05-15 → 2026-05-16

## TL;DR

**Per-design strategy:**
- **ariane133:** do NOT provide macros.tcl. Auto-place beats every cascade variant.
- **ariane136:** DO provide our cascade placement. Cascade beats auto-place by 0.45 ns.
- **mempool_tile, nvdla:** untested. Pick whichever wins after testing.

The behavior is design-specific. Same algorithm, two opposite outcomes.

| Design | Auto-place slack | Our dp_lane slack | Recommended |
|--------|------------------:|--------------------:|-------------|
| ariane133 | −0.3512 | **−1.5824** | auto-place |
| ariane136 | +0.0457 | **+0.4935** ★ | **cascade** |
| mempool_tile | (untested with fix) | **−2.0546** | unclear (need auto baseline) |
| nvdla | (untested with fix) | **PDN FAILED** | auto-place (cascade unrunnable) |

## ariane133 detail

We tested all 5 of our cascade placements + 1 baseline (no placement). ORFS's
own `rtl_macro_placer` beats every cascade variant by 1.13–1.26 ns of slack.

| Variant | Slack (ns) | vs auto-place |
|---|---:|---:|
| **ORFS auto-place** (no MACRO_PLACEMENT_TCL) | **−0.3512** | baseline |
| dp_multi | −1.4808 | −1.13 |
| dual_levy | −1.5288 | −1.18 |
| multidir | −1.5676 | −1.22 |
| dp_lane (E84 original) | −1.5824 | −1.23 |
| hybrid (E48/E41) | −1.6086 | −1.26 |
| hessian (E74) | FAILED at PDN | — |

Hessian's placement was so pathological (proxy 0.689) it crashed at the PDN
generation stage. The other 5 routed cleanly but timing closed worse than auto.

## Why this matters

Our placer optimizes proxy cost (WL + density + congestion at macro level).
ORFS's `rtl_macro_placer` is timing-driven — it uses the RTL hierarchy and
clock-tree topology to pick macro positions that the router can actually
make timing on. Different objective entirely.

For ariane133, the macros constrain where cells must go, and our
proxy-optimal macro positions force cells onto longer critical paths
than ORFS's RTL-aware placement does.

## What was fixed overnight to get these numbers

Three independent bugs were silently making our macros.tcl into a no-op
in all earlier Tier-2 runs. Earlier "−0.3512 slack" reports were actually
ORFS auto-place values labeled as our placement. Once fixed, our
placement applies and we see the true 1.2 ns gap.

1. **Docker mount missing for `designs/`**: `util/docker_shell` mounted
   host `flow/` at `/work/` but ORFS reads `MACRO_PLACEMENT_TCL` relative
   to `/OpenROAD-flow-scripts/flow/` (image path).
   Fix in `~/mpc-work/OpenROAD-flow-scripts/flow/util/docker_shell` on cloud:
   added `-v $WORKSPACE/designs:/OpenROAD-flow-scripts/flow/designs:Z`.

2. **TCL bracket escape under-quoted** in
   `scripts/generate_macro_placement_tcl.py` line ~445. Our TCL source said
   `"sram_block\[0\]..."` which TCL parses as `sram_block[0]...` — but ODB
   stores names with **literal** backslashes (Verilog-escaped identifiers).
   Fix: `plc_name.replace('[', '\\\\\\[').replace(']', '\\\\\\]')` (Python
   source `'\\\\\\['` → file `\\\[` → TCL runtime `\[`).

3. **Orientation map missing**: our .plc N/S/E/W codes were passed
   verbatim to `setOrient` which expects R0/R180/R270/R90/MY/MX/MYR90/MXR90.
   Fix: added `_orient_map` dict that translates before emit.

After all three fixes, the 2_2_floorplan_macro.log shows:
- `[INFO MPL-0062] Found fixed macro ...` (133× — every macro placed and locked)
- `[INFO MPL-0017] No unfixed macros.`
- `[INFO MPL-0013] Skipping macro placement.` ← rtl_macro_placer skipped
- `Placed 133 macros (expected 133)`

## Strategic implications

**For Tier-2 ariane133:** ship without macros.tcl. Saves 1.2 ns of slack.

**For other NG45 designs (ariane136, mempool_tile, nvdla):** untested with
the fix. ariane136 dp_lane v4 is currently running (started 10:30 UTC)
and will give us one more data point. If pattern holds, ship all four
without macros.tcl.

**For Tier-1 (proxy cost):** unaffected — proxy is computed directly from
the placement, not via ORFS. We still submit our cascade placement for
Tier-1.

**Bigger picture:** our cascade placer is well-suited for proxy
optimization but is NOT timing-aware. To compete on Tier-2, we'd need
either (a) a timing-aware placement objective (rare in academic placers,
hard to bolt on), or (b) defer macro placement to ORFS.

## Files

- AWS box: `tier2-cloud` (44.197.196.100), instance `i-0d5aafdd432119723`
- All v4 reports: `/home/ubuntu/v4_reports/{multidir,hybrid,dual_levy,dp_multi,dp_lane}/ariane133/base/6_finish.rpt`
- Snapshotted macro logs: `/home/ubuntu/v4_reports/*/2_2_macro.log`
- Patched TCL generator: `/home/ubuntu/mpc-work/repo/scripts/generate_macro_placement_tcl.py`
- Patched docker_shell: `/home/ubuntu/mpc-work/OpenROAD-flow-scripts/flow/util/docker_shell`

## Cost so far

AWS c6i.8xlarge from 2026-05-15T16:16 UTC to 2026-05-16T10:30 UTC ≈ 18.25 hr × $1.36/hr ≈ **$25**.

## Terminate AWS when done

```bash
aws ec2 terminate-instances --instance-ids i-0d5aafdd432119723 --region us-east-1
```
