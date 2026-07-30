#!/usr/bin/env bash
# perf study 29 — engine-wide load totals, the denominator the census divides.
#
#   bash tools/perf29/loads29.sh <label>...
#
# `L1-dcache-loads` on Zen 4 is `ls_dispatch.ld_dispatch` — dispatched load
# OPS, so one 64-byte `vmovdqu64` counts once. That is the same unit the
# census counts, which is what makes the two comparable.
set -euo pipefail
cd "$(dirname "$0")/../.."
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P29_CPU:-2}
BUDGET=${P29_BUDGET:-100000}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for label in "$@"; do
  nice -n 19 perf stat -x, -e instructions:u,branches:u,L1-dcache-loads:u,L1-dcache-load-misses:u \
    -o "$tmp/stat" -- \
    taskset -c "$CPU" "./bin29/$label" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
    >/dev/null 2>"$tmp/bench"
  nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
  awk -F, -v l="$label" -v n="$nodes" '
    $3~/^instructions/{i=$1} $3~/^branches/{b=$1}
    $3~/^L1-dcache-loads/{ld=$1} $3~/^L1-dcache-load-misses/{ms=$1}
    END{printf "%-10s nodes=%-9d i/n=%8.2f br/n=%7.2f loads/n=%9.3f misses/n=%8.3f miss-rate=%.4f%%\n",
        l, n, i/n, b/n, ld/n, ms/n, 100*ms/ld}
  ' "$tmp/stat"
done
