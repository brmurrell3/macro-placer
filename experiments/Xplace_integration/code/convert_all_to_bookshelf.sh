#!/usr/bin/env bash
# Convert all 17 IBM + 4 NG45 benches to bookshelf format.
# Output: experiments/E76_dreamplace_integration/bookshelf_out/<bench>/
set -e
cd "$(dirname "$0")/../../.."

CONV=experiments/E76_dreamplace_integration/code/tilos_to_bookshelf.py
OUT_ROOT=experiments/E76_dreamplace_integration/bookshelf_out

BENCHES="ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18 ariane133 ariane136 mempool_tile nvdla"

for b in $BENCHES; do
  if [ -f "$OUT_ROOT/$b/$b.aux" ]; then
    echo "[skip] $b already converted"
    continue
  fi
  echo "[convert] $b"
  uv run python $CONV $b $OUT_ROOT/$b
done
echo "[done] bookshelf conversion"
ls $OUT_ROOT/
