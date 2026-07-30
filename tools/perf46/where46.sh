#!/usr/bin/env bash
# perf study 44 — WHERE the back-end stalls are, by symbol.
#
#   bash tools/perf46/where44.sh <event> [period] [binary-label]
#
# The census says WHICH resource; this says WHICH CODE. Sampling a stall
# counter (rather than `cycles`) attributes stall events to the instruction
# pointer that was retiring-blocked, so the profile reads as "this symbol owns
# N% of the engine's load stalls" rather than "N% of its time".
#
# Zen 4 has no PEBS-equivalent precise skid control for these events, so the
# attribution is approximate at instruction granularity but sound at SYMBOL
# granularity, which is all this study asks of it.
set -euo pipefail
cd "$(dirname "$0")/../.."
EV=${1:?event}
PERIOD=${2:-200003}
LABEL=${3:-b44}
NET=${P46_NET:-engines/canonical/net.cenn}
FENS=${P46_FENS:-tools/nnue/calib/wl50.fens}
CPU=${P46_CPU:-2}
BUDGET=${P46_BUDGET:-100000}
BIN=${P46_BIN:-bin46}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

nice -n 19 perf record -q -e "${EV}:u" -c "$PERIOD" -o "$tmp/perf.data" -- \
  taskset -c "$CPU" "./$BIN/$LABEL" \
  bench 16 1 "$BUDGET" "$FENS" nodes "$NET" >/dev/null 2>&1

echo "### $EV  (period $PERIOD, $LABEL)"
perf report -i "$tmp/perf.data" --stdio --no-children -q -F overhead,symbol 2>/dev/null \
  | head -22 \
  | sed 's/_RNv[A-Za-z0-9_]*sota_simd[0-9]*avx512[0-9]*/…avx512::/; s/_RN[A-Za-z0-9_]*//' \
  | cat
echo
