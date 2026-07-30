#!/usr/bin/env bash
# perf study 33 -- IBS op sampling: precise IP + data address + data SOURCE.
#
#   bash tools/perf33/ibs33.sh <tag> [period] [binary]
#
# WHY IBS AND NOT ex_no_retire: `ex_no_retire.load_not_complete` is not a
# precise event, so inside a 15,898-byte inlined function its samples smear
# across source lines and the attribution is only as good as the region.  IBS
# tags a RETIRED op with its own IP, its data address and the level that served
# it (L1 / L2 / L3 / DRAM).  That turns "which line stalls" into "which LOAD
# misses, and how far it went" -- the mechanism, not the location.
#
# Frame and CPU pin identical to rec33.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
TAG=${1:?tag}
PERIOD=${2:-50021}
BIN=${3:-bin33/prof}
CPU=${P33_CPU:-2}
NET=$ROOT/engines/canonical/net.cenn
FENS=$ROOT/tools/nnue/calib/wl50.fens
OUT=$ROOT/perf33data
mkdir -p "$OUT"

nice -n 19 perf record -e ibs_op/swfilt=1/ -d -c "$PERIOD" --no-inherit -q \
  -o "$OUT/ibs-$TAG.data" -- \
  taskset -c "$CPU" "$ROOT/$BIN" bench 16 1 100000 "$FENS" nodes "$NET" \
  > "$OUT/ibs-$TAG.bench" 2>"$OUT/ibs-$TAG.err" || {
    echo "perf record failed:"; tail -5 "$OUT/ibs-$TAG.err"; exit 1; }

# perf script emits a FIXED column order regardless of the -F order:
# data address, then data_src, then ip (perf study 29's note -- do not read the
# -F list as the layout).
perf script -i "$OUT/ibs-$TAG.data" -F ip,addr,data_src --no-demangle \
  2>/dev/null > "$OUT/ibs-$TAG.script"
echo "[ibs33] $TAG: $(wc -l < "$OUT/ibs-$TAG.script") samples"
head -2 "$OUT/ibs-$TAG.script"
