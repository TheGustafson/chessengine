#!/usr/bin/env bash
# perf study 29 — record one event into a persistent perf.data for annotation.
#
#   bash tools/perf29/rec29.sh <label> <event> [budget] [period]
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
EVENT=${2:?event}
BUDGET=${3:-100000}
PERIOD=${4:-20003}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P29_CPU:-2}
DIR=docs/research/perf-artifacts-29
mkdir -p "$DIR"
DATA="$DIR/perf-$LABEL-${EVENT%%:*}.data"
nice -n 19 perf record -q -e "$EVENT" -c "$PERIOD" -o "$DATA" -- \
  taskset -c "$CPU" "./bin29/$LABEL" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
  >/dev/null 2>"$DIR/bench-$LABEL.txt"
nodes=$(sed -n 's/^Nodes searched  : //p' "$DIR/bench-$LABEL.txt" | tail -1)
echo "$DATA  nodes=$nodes  event=$EVENT  period=$PERIOD"
