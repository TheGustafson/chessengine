#!/usr/bin/env bash
# perf study 21 — flat symbol attribution on THIS study's frame.
#
#   bash docs/research/perf-artifacts-21/scripts/flat21.sh <label> [event] [bin]
#
# Same fixed 5,001,562-node tree as probe21.sh so the percentages describe the
# SAME workload as the accept gate.  Default event is cycles:u — study 21 ranks
# on cycles, not instructions.
set -euo pipefail
LABEL=${1:?label}
EV=${2:-cycles:u}
BIN=${3:-target-fast21/release/chessengine}
D=docs/research/perf-artifacts-21
out=$D/flat-$LABEL.txt
nice -n 19 perf record -q -e "$EV" -F 999 -o "$D/perf-$LABEL.data" -- \
  taskset -c 6 "$BIN" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
  engines/canonical/net.cenn > /dev/null 2>&1
perf report -i "$D/perf-$LABEL.data" --stdio --no-children -q --percent-limit 0.3 \
  | sed 's/[[:space:]]*$//' > "$out"
rm -f "$D/perf-$LABEL.data"
head -40 "$out"
