#!/usr/bin/env bash
# perf study 67 — the frame. One binary, one workload, pinned.
#   bash tools/perf67/frame67.sh <binary> [extra perf args...]
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${1:?binary}
shift
CPU=${P67_CPU:-2}
exec taskset -c "$CPU" "$BIN" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
