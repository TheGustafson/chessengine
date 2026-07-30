#!/usr/bin/env bash
# perf study 43 — four-event symbol census on the study frame.
#   bash tools/perf43/prof43.sh <label> [outdir]
# Events: cycles, instructions, load-stall cycles, L1d miss.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
OUT=${2:-docs/research/perf-artifacts-43}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P43_CPU:-2}
mkdir -p "$OUT"
for ev in cycles:u instructions:u ex_no_retire.load_not_complete:u L1-dcache-load-misses:u; do
  tag=${ev%%:*}; tag=${tag//./_}
  nice -n 19 perf record -q -e "$ev" -c "${P43_PERIOD:-2000003}" \
    -o "$OUT/perf-$LABEL-$tag.data" -- \
    taskset -c "$CPU" "./bin34/$LABEL" bench 16 1 100000 "$FENS" nodes "$NET" >/dev/null 2>&1
  perf report -q -i "$OUT/perf-$LABEL-$tag.data" --percent-limit 0.20 --no-children \
    --sort symbol 2>/dev/null > "$OUT/sym-$LABEL-$tag.txt"
  echo "== $ev =="; head -14 "$OUT/sym-$LABEL-$tag.txt"
done
