#!/usr/bin/env bash
# perf study 42 -- run the TT census at one or more Hash sizes.
#
#   bash tools/perf42/runcensus42.sh [hashMB ...]
#
# Writes docs/research/perf-artifacts-42/ttcensus-<MB>.txt and prints the
# per-node lines. The binary is bin42/census (features ...,ttcensus42), which
# `tools/perf42/ident42.sh` shows is node-identical to bin42/base -- the
# counters cost timing and nothing else.
set -euo pipefail
cd "$(dirname "$0")/../.."
CPU=${P42_CPU:-2}
BUDGET=${P42_BUDGET:-100000}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
OUT=docs/research/perf-artifacts-42
mkdir -p "$OUT"
for mb in "${@:-16}"; do
  ARCHON_TTCENSUS="$OUT/ttcensus-$mb.txt" \
    taskset -c "$CPU" ./bin42/census bench "$mb" 1 "$BUDGET" "$FENS" nodes "$NET" \
    >/dev/null 2>&1
  echo "=== Hash = ${mb} MB ==="
  cat "$OUT/ttcensus-$mb.txt"
done
