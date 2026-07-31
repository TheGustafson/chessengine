#!/usr/bin/env bash
# perf study 56 -- the NODE-IDENTITY gate. Every arm must bench at exactly
# 5,001,562 nodes on the study frame, or its instruction delta is measuring a
# different search and means nothing.
#   bash tools/perf56/nodes56.sh <label>...
set -euo pipefail
cd "$(dirname "$0")/../.."
NET=${P56_NET:-engines/canonical/net.cenn}
FENS=${P56_FENS:-tools/nnue/calib/wl50.fens}
BIN=${P56_BIN:-bin56}
WANT=5001562
rc=0
for lab in "$@"; do
  # `bench` prints its report on STDERR, not stdout.
  n=$(taskset -c "${P56_CPU:-2}" "./$BIN/$lab" bench 16 1 100000 "$FENS" nodes "$NET" \
        2>&1 >/dev/null | awk '/Nodes searched  :/ {print $4}')
  if [ "$n" = "$WANT" ]; then
    echo "[nodes56] $lab  $n  OK"
  else
    echo "[nodes56] $lab  $n  *** MISMATCH (want $WANT) ***"
    rc=1
  fi
done
exit $rc
