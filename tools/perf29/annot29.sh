#!/usr/bin/env bash
# perf study 29 — per-INSTRUCTION attribution inside the accumulator kernels.
#
#   bash tools/perf29/annot29.sh <label> <event> [budget]
#
# A symbol share cannot see inside a symbol; a per-instruction profile can.
# Every memory operand in the fold kernels is unambiguously either the
# ACCUMULATOR STREAM (base register = the src/dst pointer) or a WEIGHT ROW
# (base = weights + feature*ACC), so sampling on a load or a miss event and
# bucketing by base register splits the two directly.
#
# Writes the raw per-address sample dump; `foldsplit29.py` does the bucketing.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
EVENT=${2:?event}
BUDGET=${3:-100000}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P29_CPU:-2}
OUT=docs/research/perf-artifacts-29/annot-$LABEL-${EVENT%%:*}.txt
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

nice -n 19 perf record -q -e "$EVENT" -c "${P29_PERIOD:-20003}" -o "$tmp/p.data" -- \
  taskset -c "$CPU" "./bin29/$LABEL" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
  >/dev/null 2>"$tmp/bench"
nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
{
  echo "# $LABEL event=$EVENT nodes=$nodes period=${P29_PERIOD:-20003}"
  # ip + symbol, one line per SAMPLE GROUP with its period sum.
  perf script -i "$tmp/p.data" -F ip,sym,period 2>/dev/null
} > "$OUT"
wc -l "$OUT"
