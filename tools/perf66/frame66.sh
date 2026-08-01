#!/usr/bin/env bash
# perf study 66 — the study frame, run once for one binary in one data mode.
#   bash tools/perf66/frame66.sh <binary> [MODE=VALUE ...]
# Prints "nodes=<n> nps=<n>".
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${1:?binary}
shift || true
CPU=${P66_CPU:-2}
env "$@" taskset -c "$CPU" "$BIN" \
  bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn 2>&1 \
  | awk '/Nodes searched *:/{n=$NF} /Nodes\/second/{s=$NF} END{print "nodes="n" nps="s}'
