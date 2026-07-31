#!/usr/bin/env bash
# perf study 49 — unmultiplexed counter groups on one arm, N reps.
#   bash tools/perf49/census49.sh <reps> <label>
# NOTE the engine writes bench progress to STDERR, so `perf stat -o` (not a
# stderr redirect) is the only way to keep the CSV clean.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPS=${1:-3}; LABEL=${2:?label}
CPU=${P49_CPU:-2}
NET=engines/canonical/net.cenn
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
EVGROUPS=(
 "cycles:u,instructions:u,ls_dispatch.ld_dispatch:u,ex_no_retire.load_not_complete:u"
 "cycles:u,instructions:u,ls_any_fills_from_sys.all:u,ls_alloc_mab_count:u"
 "cycles:u,instructions:u,ls_any_fills_from_sys.dram_io_all:u,ls_any_fills_from_sys.local_l2:u"
)
for g in "${EVGROUPS[@]}"; do
  for r in $(seq 1 "$REPS"); do
    taskset -c "$CPU" perf stat -x, -o "$TMP" -e "$g" -- \
      "./bin49/$LABEL" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes "$NET" \
      >/dev/null 2>/dev/null
    awk -F, -v L="$LABEL" -v R="$r" '$1 ~ /^[0-9]/ {print L","R","$3","$1}' "$TMP"
  done
done
