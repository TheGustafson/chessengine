#!/usr/bin/env bash
# perf study 48 -- attribute an event to SYMBOLS by sampling the event itself.
#
#   bash tools/perf48/where48.sh <event> <period> <label>
#
# Sampling the load-dispatch counter itself makes a symbol's share read as
# "this code issues N % of the engine's load dispatches".  Per study 36 Sec 2.3
# per-INSTRUCTION attribution of a non-precise Zen 4 counter is an artifact;
# per-SYMBOL shares survive, which is all this asks.
set -euo pipefail
cd "$(dirname "$0")/../.."
EV=${1:?event}
PER=${2:-200003}
LABEL=${3:-base48}
CPU=${P48_CPU:-2}
NET=${P48_NET:-engines/canonical/net.cenn}
FENS=${P48_FENS:-tools/nnue/calib/wl50.fens}
BIN=${P48_BIN:-bin48}
DIR=docs/research/perf-artifacts-48
mkdir -p "$DIR"
SAFE=$(echo "$EV" | tr -c 'A-Za-z0-9_' '_')
DATA=$DIR/where-$LABEL-$SAFE.data
OUT=$DIR/where-$LABEL-$SAFE.txt
perf record -e "$EV" -c "$PER" -o "$DATA" -- \
  taskset -c "$CPU" "./$BIN/$LABEL" bench 16 1 100000 "$FENS" nodes "$NET" >/dev/null 2>&1
perf report -i "$DATA" --stdio --no-children --percent-limit 0.30 -s symbol 2>/dev/null \
  | grep -vE '^#|^$|^\s*$' | head -60 > "$OUT"
echo "[where48] $EV -> $OUT"
cat "$OUT"
