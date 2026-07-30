#!/usr/bin/env bash
# perf study 25 — flat symbol attribution.
#
#   bash docs/research/perf-artifacts-25/scripts/flat25.sh <label> <bin> [frame] [event]
#
# frame: wl50 (5,001,562 nodes) | d16 (9,444,932 nodes).  A symbol share is
# meaningless without its frame (study 21 §0.2 — i8_deltas_into is 10.86% on
# wl50 and 14.88% on d16, same binary).
#
# --no-children gives cycles in a SYMBOL'S CODE REGION, INCLUDING inlined
# callees.  That is the "where are the cycles" question.  To get the "which
# function body do I edit" answer, build the noinline probe (arm P0) so each
# body is its own symbol, and read this table again.
#
# NOTE: --call-graph=none is INVALID on this perf build when combined with -e.
set -euo pipefail
LABEL=${1:?label}
BIN=${2:?bin}
FRAME=${3:-d16}
EV=${4:-cycles:u}
CPU=${PERF_CPU:-2}
case "$FRAME" in
  wl50) CMD=(bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn) ;;
  d16)  CMD=(bench 16 1 16 default depth engines/canonical/net.cenn) ;;
  *) echo "unknown frame $FRAME"; exit 1 ;;
esac
D=docs/research/perf-artifacts-25
out=$D/flat-$LABEL-$FRAME.txt
nice -n 19 perf record -q -e "$EV" -F 999 -o "$D/perf-$LABEL.data" -- \
  taskset -c "$CPU" "$BIN" "${CMD[@]}" > /dev/null 2>&1
perf report -i "$D/perf-$LABEL.data" --stdio --no-children -q --percent-limit 0.3 \
  | sed 's/[[:space:]]*$//' > "$out"
rm -f "$D/perf-$LABEL.data"
head -40 "$out"
