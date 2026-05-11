#!/bin/bash
# Run tilos_to_bookshelf.py for all 17 IBM benches + 4 NG45 designs.
# Output goes to experiments/E76_dreamplace_integration/bookshelf_out/<bench>/
set -e
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

for B in ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 \
         ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18 \
         ariane133 ariane136 mempool_tile nvdla; do
  echo "[convert_all] $B"
  uv run python experiments/E76_dreamplace_integration/code/tilos_to_bookshelf.py \
      "$B" "experiments/E76_dreamplace_integration/bookshelf_out/$B" \
      > "experiments/E76_dreamplace_integration/bookshelf_out/$B.convert.log" 2>&1 || \
      echo "  [convert_all] $B FAILED — see log"
done

echo "[convert_all] DONE — outputs in experiments/E76_dreamplace_integration/bookshelf_out/"
ls experiments/E76_dreamplace_integration/bookshelf_out/
