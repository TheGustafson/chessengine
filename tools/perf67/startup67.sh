#!/usr/bin/env bash
# perf study 67 — the STARTUP constant. Same binary, same net, 1 node per FEN
# instead of 100k, so everything that is not per-node work (magic init, the
# 138 MB net load + quantise, the bench printer) is measured on its own and
# can be subtracted from the frame.
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${1:?binary}
EV=${2:?events}
N=${3:-1}
taskset -c "${P67_CPU:-2}" perf stat -x, -e "$EV" -- \
  "$BIN" bench 16 1 "$N" tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn \
  2>&1 >/dev/null | grep ',,'
