#!/usr/bin/env bash
# perf study 34 — identity gate: node count + wl50 eval hash on the study frame.
# A counting build is an INSTRUMENT; if it moves the tree it measures something
# else. An ORDERING arm is expected to move `nodes` and must NOT move `hash`.
#
#   bash tools/perf34/ident34.sh <label>...
set -euo pipefail
cd "$(dirname "$0")/../.."
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P34_CPU:-2}
for label in "$@"; do
  n=$(nice -n 19 taskset -c "$CPU" "./bin34/$label" bench 16 1 100000 "$FENS" nodes "$NET" 2>&1 >/dev/null | sed -n 's/^Nodes searched  : //p')
  h=$("./bin34/$label" nnue-eval "$NET" < "$FENS" | sha256sum | cut -c1-16)
  printf '%-14s nodes=%-10s hash=%s\n' "$label" "$n" "$h"
done
