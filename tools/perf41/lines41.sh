#!/usr/bin/env bash
# perf study 41 — how much of a hot function is actually EXECUTED.
#
#   bash tools/perf41/lines41.sh <label> [period]
#
# `hotsize41.py` reads the hot set's ALLOCATED size off the symbol table:
# 94438 bytes, 2.88x the 32 KB L1 instruction cache. That is the wrong number
# for icache pressure if a large slice of each hot function is cold code that
# is never fetched — a function's size costs nothing until the fetcher walks
# into it.
#
# So: sample the frame, bucket every sample's IP into the 64-byte cache line it
# fell in, and count the DISTINCT lines each hot symbol contributes. Lines x 64
# is the executed footprint; allocated minus executed is cold code living
# inside hot functions, which is exactly the pressure an outlining arm could
# remove.
#
# Read the executed figure as a LOWER bound: sampling cannot see a line that
# runs a handful of times in the whole frame. That makes "cold bytes inside hot
# functions" an UPPER bound on what outlining could win, which is the right way
# round for deciding whether to attempt it.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
PERIOD=${2:-4000}
NET=${P41_NET:-engines/canonical/net.cenn}
FENS=${P41_FENS:-tools/nnue/calib/wl50.fens}
CPU=${P41_CPU:-2}
OUT=docs/research/perf-artifacts-41
mkdir -p "$OUT"

nice -n 19 perf record -e cycles:u -c "$PERIOD" -o "$OUT/lines-$LABEL.data" -- \
  taskset -c "$CPU" "./bin34/$LABEL" \
  bench 16 1 "${P41_BUDGET:-40000}" "$FENS" nodes "$NET" >/dev/null 2>&1
perf script -i "$OUT/lines-$LABEL.data" -F ip,sym 2>/dev/null \
  > "$OUT/lines-$LABEL.script"
rm -f "$OUT/lines-$LABEL.data"
python3 tools/perf41/lines41.py "bin34/$LABEL" "$OUT/lines-$LABEL.script" \
  | tee "$OUT/lines-$LABEL.txt"
