#!/usr/bin/env bash
# perf study 29 — IBS load sampling WITH DATA ADDRESSES.
#
#   bash tools/perf29/memrec29.sh <label> [budget] [period]
#
# `perf mem record -t load` on Zen 4 uses IBS, which tags a retired load with
# its data address AND its data source (L1 / L2 / L3 / DRAM). That is the one
# instrument that can say, inside a single symbol, WHICH STREAM missed — the
# accumulator (a ~1 MB stack) or a weight row (a ~100 MB table) — because the
# two live in address ranges three orders of magnitude apart.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
BUDGET=${2:-100000}
PERIOD=${3:-20011}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P29_CPU:-2}
DIR=docs/research/perf-artifacts-29
mkdir -p "$DIR"
DATA="$DIR/mem-$LABEL.data"
nice -n 19 perf mem record -t load -c "$PERIOD" -o "$DATA" -- \
  taskset -c "$CPU" "./bin29/$LABEL" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
  >/dev/null 2>"$DIR/membench-$LABEL.txt"
nodes=$(sed -n 's/^Nodes searched  : //p' "$DIR/membench-$LABEL.txt" | tail -1)
perf script -i "$DATA" -F sym,addr,data_src 2>/dev/null > "$DIR/mem-$LABEL.script"
echo "nodes=$nodes  $(wc -l < "$DIR/mem-$LABEL.script") samples -> $DIR/mem-$LABEL.script"
head -3 "$DIR/mem-$LABEL.script"
