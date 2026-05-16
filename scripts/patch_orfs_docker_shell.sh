#!/bin/bash
# Patches the ORFS util/docker_shell to mount our flow/designs/ subdir
# over the docker image's /OpenROAD-flow-scripts/flow/designs/ so the
# image's hardcoded path (used by MACRO_PLACEMENT_TCL = ./designs/...)
# actually points at our edited config + macros.
#
# Apply once on a fresh ORFS install before running evaluate_with_orfs.py.
# Without this patch, our MACRO_PLACEMENT_TCL gets silently ignored and
# rtl_macro_placer runs auto-placement instead.
#
# Usage: bash patch_orfs_docker_shell.sh /path/to/OpenROAD-flow-scripts

set -e
ORFS_ROOT="${1:?Usage: $0 <path/to/OpenROAD-flow-scripts>}"

SHELL_FILE="$ORFS_ROOT/flow/util/docker_shell"
[ -f "$SHELL_FILE" ] || { echo "ERROR: $SHELL_FILE not found"; exit 1; }

# Idempotent: check if patch already applied
if grep -q 'WORKSPACE/designs:/OpenROAD-flow-scripts/flow/designs' "$SHELL_FILE"; then
    echo "Already patched: $SHELL_FILE"
    exit 0
fi

# Back up original
cp "$SHELL_FILE" "${SHELL_FILE}.orig"

# Add the designs mount after the /work mount
sed -i 's|-v "\$WORKSPACE:/work:Z"|-v "$WORKSPACE:/work:Z"\n    -v "$WORKSPACE/designs:/OpenROAD-flow-scripts/flow/designs:Z"|' "$SHELL_FILE"

# Verify
if grep -q 'WORKSPACE/designs:/OpenROAD-flow-scripts/flow/designs' "$SHELL_FILE"; then
    echo "Patched: $SHELL_FILE (original at ${SHELL_FILE}.orig)"
else
    echo "ERROR: patch failed; restoring original"
    mv "${SHELL_FILE}.orig" "$SHELL_FILE"
    exit 1
fi
