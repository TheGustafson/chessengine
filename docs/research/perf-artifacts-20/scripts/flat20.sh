#!/usr/bin/env bash
# perf study 20 — flat symbol attribution on THIS study's frame.
#
#   bash docs/research/perf-artifacts-20/scripts/flat20.sh <label> [event]
#
# Same fixed 5,001,562-node tree as probe20.sh so the percentages and the i/n
# gate describe the SAME workload.  Default event is cycles:u; pass
# instructions:u for the instruction-share view (that is the one that matches
# the accept gate).
set -euo pipefail
LABEL=${1:?label}
EV=${2:-cycles:u}
BIN=target-fast20/release/chessengine
D=docs/research/perf-artifacts-20
out=$D/flat-$LABEL.txt
nice -n 19 perf record -q -e "$EV" -F 999 -o "$D/perf-$LABEL.data" -- \
  taskset -c 6 "$BIN" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
  engines/canonical/net.cenn > /dev/null 2>&1
perf report -i "$D/perf-$LABEL.data" --stdio --no-children -q --percent-limit 0.3 \
  | sed 's/[[:space:]]*$//' > "$out"
rm -f "$D/perf-$LABEL.data"
head -40 "$out"
