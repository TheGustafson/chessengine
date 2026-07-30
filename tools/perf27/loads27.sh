#!/usr/bin/env bash
# tools/perf27/loads27.sh <label>...
#
# TOTAL L1d loads alongside the misses -- the denominator `probe27.sh` omits.
# A rise in `L1-dcache-load-misses` means nothing on its own: it can be more
# memory work, or the SAME work with a worse hit rate, or LESS work with a
# much worse hit rate. Only the pair distinguishes them.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
NET=$ROOT/engines/canonical/net.cenn
FENS=$ROOT/tools/nnue/calib/wl50.fens
CPU=${P27_CPU:-7}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

for label in "$@"; do
  nice -n 19 perf stat -x, -e L1-dcache-loads:u,L1-dcache-load-misses:u \
    -o "$tmp/stat" -- \
    taskset -c "$CPU" "$ROOT/perf27/bin/$label" bench 16 1 100000 "$FENS" nodes "$NET" \
    >/dev/null 2>"$tmp/bench"
  nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp/bench" | tail -1)
  awk -F, -v l="$label" -v n="$nodes" '
    $3~/^L1-dcache-loads/{ld=$1} $3~/^L1-dcache-load-misses/{ms=$1}
    END{printf "%-8s loads/n=%9.2f  misses/n=%8.3f  miss-rate=%.4f%%\n", l, ld/n, ms/n, 100*ms/ld}
  ' "$tmp/stat"
done
