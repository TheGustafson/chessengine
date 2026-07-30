#!/usr/bin/env bash
# perf study 41 — WHERE the front end starves, by symbol.
#
#   bash tools/perf41/where41.sh <label> <event> [period]
#
# `perf stat` says how much; this says where. Sampling on a front-end event
# rather than on cycles answers a different question from every profile this
# project has taken: not "which code runs" but "which code the fetcher cannot
# keep up with".
#
# Two caveats stated up front, because a front-end profile is easy to
# over-read:
#
#  * these events are NOT precise (no PEBS-equivalent here), so the sampled RIP
#    is skidded — read this at FUNCTION granularity, never at instruction
#    granularity;
#  * an icache miss is charged where the fetch stalled, which is the VICTIM of
#    whatever evicted the line, not the culprit. A capacity problem shows up
#    smeared across the whole hot set, which is itself the diagnosis.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
EV=${2:?event}
PERIOD=${3:-20000}
NET=${P41_NET:-engines/canonical/net.cenn}
FENS=${P41_FENS:-tools/nnue/calib/wl50.fens}
CPU=${P41_CPU:-2}
OUT=docs/research/perf-artifacts-41/where-$LABEL-${EV//[:.]/_}.txt
mkdir -p "$(dirname "$OUT")"

nice -n 19 perf record -e "$EV" -c "$PERIOD" -o "$OUT.data" -- \
  taskset -c "$CPU" "./bin34/$LABEL" \
  bench 16 1 "${P41_BUDGET:-100000}" "$FENS" nodes "$NET" >/dev/null 2>&1

perf report -i "$OUT.data" --stdio --sort symbol --percent-limit 0.4 \
  --no-children 2>/dev/null | grep -vE '^\s*$|^#' | head -40 > "$OUT"
rm -f "$OUT.data"
echo "=== $LABEL / $EV ==="
cat "$OUT"
