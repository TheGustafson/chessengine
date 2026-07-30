#!/usr/bin/env bash
# perf study 21 — flat attribution on the OTHER frame: the depth-16 bench.
#
#   bash docs/research/perf-artifacts-21/scripts/flatd16.sh <label> [event] [bin]
#
# WHY A SECOND FRAME EXISTS. The study's screening frame
# (`bench 16 1 100000 wl50.fens nodes`, 5,001,562 nodes) and this one
# (`bench 16 1 16 default depth`, 9,444,932 nodes) put VERY different weights on
# the same symbols — measured 2026-07-29, same binary, same box:
# `search::<NonPv>` is 6.75% of cycles on the wl50 frame and ~3x that on d16,
# while the fold kernels move the other way. A symbol share is meaningless
# without its frame, so quote the frame with every number.
set -euo pipefail
LABEL=${1:?label}
EV=${2:-cycles:u}
BIN=${3:-bin21/baseline}
D=docs/research/perf-artifacts-21
nice -n 19 perf record -q -e "$EV" -F 999 -o "$D/perf-$LABEL.data" -- \
  taskset -c 6 "$BIN" bench 16 1 16 default depth engines/canonical/net.cenn \
  > /dev/null 2>&1
perf report -i "$D/perf-$LABEL.data" --stdio --no-children -q --percent-limit 0.4 \
  | sed 's/[[:space:]]*$//' > "$D/flat-$LABEL.txt"
rm -f "$D/perf-$LABEL.data"
head -30 "$D/flat-$LABEL.txt"
