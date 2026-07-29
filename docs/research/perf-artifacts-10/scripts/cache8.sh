#!/usr/bin/env bash
# perf study 8 — paired CACHE measurement for a change that trades memory for
# instructions.
#
#   bash cache8.sh <rounds> <core> <binA> <binB>
#
# Instructions alone cannot arbitrate "store the table pre-widened": it always
# wins on instruction count and can still lose on DRAM. Cache-miss counts per
# node are far less load-sensitive than cycles (a competing process pollutes
# shared LLC, but it does not change how many lines OUR access pattern needs),
# so this interleaves the two binaries and reports per-node misses for each.
set -euo pipefail
ROUNDS=${1:?rounds}
CORE=${2:?core}
A=${3:?binary A}
B=${4:?binary B}
EV=instructions:u,cycles:u,cache-references:u,cache-misses:u

for r in $(seq 1 "$ROUNDS"); do
  for b in "$A" "$B"; do
    tmp=$(mktemp)
    nice -n 19 perf stat -x, -e "$EV" -o "$tmp.perf" -- \
      taskset -c "$CORE" "$b" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
      engines/canonical/net.cenn > /dev/null 2> "$tmp.eng"
    nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
    get() { awk -F, -v e="$1" '$3==e{print $1}' "$tmp.perf"; }
    printf 'round=%s bin=%s nodes=%s instructions=%s cycles=%s cache_refs=%s cache_misses=%s\n' \
      "$r" "$(basename "$b")" "$nodes" "$(get instructions:u)" "$(get cycles:u)" \
      "$(get cache-references:u)" "$(get cache-misses:u)"
    rm -f "$tmp" "$tmp.perf" "$tmp.eng"
  done
done
