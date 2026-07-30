#!/usr/bin/env bash
# perf study 29 — IBS op sampling with DATA ADDRESSES and DATA SOURCE.
#
#   bash tools/perf29/ibs29.sh <label> [budget] [period]
#
# `perf mem record` reports only `mem-ldst` on this kernel, so drive IBS
# directly: `-e ibs_op/swfilt=1/ -d` tags a retired op with its data address
# and the level that served it (L1 hit / L2 / L3 / DRAM). That is the one
# instrument that can say, INSIDE a single symbol, which stream missed — the
# accumulator stack (~1 MB) or a weight row (a ~100 MB table) — because the
# two live in address ranges three orders of magnitude apart.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
BUDGET=${2:-20000}
PERIOD=${3:-20011}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P29_CPU:-2}
DIR=docs/research/perf-artifacts-29
mkdir -p "$DIR"
DATA="$DIR/ibs-$LABEL.data"
nice -n 19 perf record -e ibs_op/swfilt=1/ -d -c "$PERIOD" -o "$DATA" -- \
  taskset -c "$CPU" "./bin29/$LABEL" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
  >/dev/null 2>"$DIR/ibsbench-$LABEL.txt"
nodes=$(sed -n 's/^Nodes searched  : //p' "$DIR/ibsbench-$LABEL.txt" | tail -1)
# perf script emits a FIXED column order regardless of -F order: addr, then
# data_src, then ip. Do not read the -F list as the layout.
perf script -i "$DATA" -F ip,addr,data_src 2>/dev/null > "$DIR/ibs-$LABEL.script"
# The runtime load base of the text segment, for ip -> objdump-address.
perf script -i "$DATA" --show-mmap-events -F comm,event 2>/dev/null \
  | grep -F "r-xp $PWD/bin29/$LABEL" | head -1 > "$DIR/ibs-$LABEL.mmap"
echo "nodes=$nodes  $(wc -l < "$DIR/ibs-$LABEL.script") samples -> $DIR/ibs-$LABEL.script"
cat "$DIR/ibs-$LABEL.mmap"
