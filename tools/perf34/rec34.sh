#!/usr/bin/env bash
# perf study 34 — record one event into a persistent perf.data for annotation.
#
#   bash tools/perf34/rec34.sh <label> <event> [budget] [period]
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
EVENT=${2:?event}
BUDGET=${3:-100000}
PERIOD=${4:-20003}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P34_CPU:-2}
DIR=docs/research/perf-artifacts-34
mkdir -p "$DIR"
DATA="$DIR/perf-$LABEL-${EVENT%%:*}.data"
nice -n 19 perf record -q -e "$EVENT" -c "$PERIOD" -o "$DATA" -- \
  taskset -c "$CPU" "./bin34/$LABEL" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
  >/dev/null 2>"$DIR/bench-$LABEL.txt"
nodes=$(sed -n 's/^Nodes searched  : //p' "$DIR/bench-$LABEL.txt" | tail -1)
echo "$DATA  nodes=$nodes  event=$EVENT  period=$PERIOD"
perf report -q -i "$DATA" --no-children --percent-limit 0.4 2>/dev/null | head -30
