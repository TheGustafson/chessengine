#!/usr/bin/env bash
# perf study 40 — identity gate for the DOSE amplifier.
#
#   bash tools/perf40/identdose40.sh <label> <dose>...
#
# The amplifier's whole claim is that replaying earlier sorts on scratch
# changes nothing the search can see. A dose that moved the node count would be
# measuring a different tree, so every dose used in a panel is gated here.
set -euo pipefail
cd "$(dirname "$0")/../.."
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
LABEL=${1:?label}; shift
for d in "$@"; do
  n=$(P40_DOSE="$d" nice -n 19 taskset -c 2 "./bin34/$LABEL" \
      bench 16 1 100000 "$FENS" nodes "$NET" 2>&1 >/dev/null |
      sed -n 's/^Nodes searched  : //p')
  h=$(P40_DOSE="$d" "./bin34/$LABEL" nnue-eval "$NET" <"$FENS" | sha256sum | cut -c1-16)
  printf '%-8s dose=%-3s nodes=%-10s hash=%s\n' "$LABEL" "$d" "$n" "$h"
done
