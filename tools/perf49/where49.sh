#!/usr/bin/env bash
# perf study 49 — per-symbol sampling of one counter on one arm binary.
#   bash tools/perf49/where49.sh <event> <period> <label> <outfile>
set -euo pipefail
cd "$(dirname "$0")/../.."
EV=${1:?event}; PERIOD=${2:?period}; LABEL=${3:?label}; OUT=${4:?out}
CPU=${P49_CPU:-2}
NET=engines/canonical/net.cenn
taskset -c "$CPU" perf record -q -e "$EV" -c "$PERIOD" -o "$OUT" -- \
  "./bin49/$LABEL" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes "$NET" >/dev/null 2>&1
echo "[where49] $EV -> $OUT"
