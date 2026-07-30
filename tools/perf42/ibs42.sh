#!/usr/bin/env bash
# perf study 42 -- IBS op capture with DATA ADDRESSES, at a chosen Hash size.
#
#   bash tools/perf42/ibs42.sh <tag> <hashMB> [period] [binary] [l3missonly]
#
# Descends from tools/perf33/ibs33.sh. The one change that matters: the Hash
# size is a parameter, because this study's primary attribution trick is that
# THE TRANSPOSITION TABLE IS THE REGION WHOSE SIZE FOLLOWS `Hash`. Nothing else
# in the process moves when 16 MB becomes 256 MB, so the TT can be identified
# from data addresses alone -- no debug info, no symbol names, no inline
# stacks, and therefore no dependence on the line-table build whose .text is
# NOT byte-identical to release (tools/perf42/textid42.sh says so).
#
# l3missonly=1 restricts sampling to ops that missed L3, which is the only way
# to get a usable count of DRAM traffic: at 9.4 DRAM fills against 5929
# instructions per node, an unfiltered sample is ~0.1% useful.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
TAG=${1:?tag}
MB=${2:-16}
PERIOD=${3:-50021}
BIN=${4:-bin34/b42}
L3ONLY=${5:-0}
CPU=${P42_CPU:-2}
BUDGET=${P42_BUDGET:-100000}
NET=$ROOT/engines/canonical/net.cenn
FENS=$ROOT/tools/nnue/calib/wl50.fens
OUT=$ROOT/perf42data
mkdir -p "$OUT"

EV="ibs_op/swfilt=1/"
[ "$L3ONLY" = 1 ] && EV="ibs_op/swfilt=1,l3missonly=1/"

nice -n 19 perf record -e "$EV" -d -c "$PERIOD" --no-inherit -q \
  -o "$OUT/ibs-$TAG.data" -- \
  taskset -c "$CPU" "$ROOT/$BIN" bench "$MB" 1 "$BUDGET" "$FENS" nodes "$NET" \
  >/dev/null 2>"$OUT/ibs-$TAG.bench" || {
    echo "perf record failed:"; tail -5 "$OUT/ibs-$TAG.bench"; exit 1; }

# perf script emits a FIXED column order regardless of the -F order: data
# address, then data_src, then ip (perf study 29's note -- do not read the -F
# list as the layout).
perf script -i "$OUT/ibs-$TAG.data" -F ip,addr,data_src --no-demangle \
  2>/dev/null > "$OUT/ibs-$TAG.script"
perf script -i "$OUT/ibs-$TAG.data" --show-mmap-events -F comm,pid,tid,event \
  2>/dev/null | grep -F "MMAP" > "$OUT/ibs-$TAG.mmap" || true
nodes=$(sed -n 's/^Nodes searched  : //p' "$OUT/ibs-$TAG.bench" | tail -1)
echo "[ibs42] $TAG: $(wc -l < "$OUT/ibs-$TAG.script") samples, nodes=$nodes, hash=${MB}MB, l3only=$L3ONLY"
head -2 "$OUT/ibs-$TAG.script"
