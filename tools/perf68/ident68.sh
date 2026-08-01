#!/usr/bin/env bash
# perf study 68 — the behavioural identity check for a LAYOUT arm.
#
#   bash tools/perf68/ident68.sh <label> [label...]
#
# A code-layout change must be bit-identical in behaviour: same nodes, same
# eval. This prints, per binary, the 5 M-node frame's node total and the sha256
# of the forward-pass score over the 50 calibration FENs -- the same two
# quantities `tools/check.sh`'s NNUE gate pins, evaluated on the PGO binaries
# the study actually measures (check.sh gates a from-source non-PGO build).
set -euo pipefail
cd "$(dirname "$0")/../.."
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P68_CPU:-2}
printf '%-14s %14s  %s\n' label nodes 'eval sha256'
for L in "$@"; do
  # the per-position progress report goes to STDERR; only the summary is wanted
  n=$(taskset -c "$CPU" "engines/bin/$L" bench 16 1 100000 "$FENS" nodes "$NET" \
      2>/dev/null | awk '/Nodes searched  :/{print $NF}')
  h=$("engines/bin/$L" nnue-eval "$NET" < "$FENS" | sha256sum | cut -d' ' -f1)
  printf '%-14s %14s  %s\n' "$L" "$n" "$h"
done
