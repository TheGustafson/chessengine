#!/usr/bin/env bash
# perf study 47 -- re-derive the 512-bit op TOTAL on this study's own binary.
# It is the denominator of the whole bracket, and rule 8 says re-derive any
# number you did not set the n for.  Study 44 measured 1282.60/node, study 46
# reproduced 1283.25; this is the third independent read.
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${P47_BIN:-bin47}; CPU=${P47_CPU:-0}; REPS=${P47_REPS:-5}
NET=${P47_NET:-engines/canonical/net.cenn}; FENS=${P47_FENS:-tools/nnue/calib/wl50.fens}
EV=cycles:u,instructions:u,fp_ops_retired_by_width.pack_512_uops_retired:u,fp_ops_retired_by_width.pack_256_uops_retired:u,fp_ops_retired_by_width.all:u,fp_ops_retired_by_width.scalar_uops_retired:u
for label in "$@"; do
  for _ in $(seq 1 "$REPS"); do
    nice -n 19 perf stat -x, -e "$EV" -- taskset -c "$CPU" \
      "$BIN/$label" bench 16 1 100000 "$FENS" nodes "$NET" 2>&1 |
      grep -E '^[0-9]+,' | awk -F, -v l="$label" '{printf "%s %s %s\n", l, $3, $1}'
  done
done
