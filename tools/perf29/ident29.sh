#!/usr/bin/env bash
# perf study 29 — identity gate: every labelled binary must reproduce the same
# node count and the same wl50 eval hash. The counting build is an INSTRUMENT;
# if it changes the tree it is measuring something else.
#
#   bash tools/perf29/ident29.sh <label>...
set -euo pipefail
cd "$(dirname "$0")/../.."
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P29_CPU:-2}
for label in "$@"; do
  n=$(nice -n 19 taskset -c "$CPU" "./bin29/$label" bench 16 1 100000 "$FENS" nodes "$NET" 2>&1 >/dev/null | sed -n 's/^Nodes searched  : //p')
  h=$("./bin29/$label" nnue-eval "$NET" < "$FENS" | sha256sum | cut -c1-16)
  printf '%-10s nodes=%-10s hash=%s\n' "$label" "$n" "$h"
done
