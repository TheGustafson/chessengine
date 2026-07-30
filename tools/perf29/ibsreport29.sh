#!/usr/bin/env bash
# perf study 29 — the SYMBOL x REGION x CACHE-LEVEL table.
#
#   bash tools/perf29/ibsreport29.sh <label>
#
# Joins three artifacts of the SAME run (addresses are per-run, ASLR moves
# them): the IBS sample dump, the FOLDREGION lines the instrumented binary
# printed, and the runtime text base from the mmap record.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
DIR=docs/research/perf-artifacts-29
BASE=$(sed -n 's/.*\[0x\([0-9a-f]*\)(.*r-xp.*/\1/p' "$DIR/ibs-$LABEL.mmap")
FOFF=$(sed -n 's/.*@ 0x\([0-9a-f]*\) <.*/\1/p' "$DIR/ibs-$LABEL.mmap")
REGIONS=$(grep '^FOLDREGION' "$DIR/ibsbench-$LABEL.txt" \
  | sed -n 's/^FOLDREGION \([a-z_]*\) lo=\(0x[0-9a-f]*\) hi=\(0x[0-9a-f]*\).*/\1:\2:\3/p' \
  | paste -sd,)
echo "# text base=0x$BASE file-off=0x$FOFF"
echo "# regions=$REGIONS"
python3 tools/perf29/ibssplit29.py "$DIR/ibs-$LABEL.script" "0x$BASE" "0x$FOFF" \
  "$DIR/$LABEL.disasm" "--regions=$REGIONS"
