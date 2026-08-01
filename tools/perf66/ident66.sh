#!/usr/bin/env bash
# perf study 66 — the OFF-PATH gate: two binaries, the shipped feature set, an
# exact instruction comparison plus node and eval identity.
#
# `instructions:u` has a 0.000 % null with zero spread on this box (study 43
# §6.3), so it is useless as a benefit signal and exact as a cost one — which
# is precisely what is wanted for "did the shipped path move at all?".
#   bash tools/perf66/ident66.sh <bin-a> <bin-b> [reps]
set -euo pipefail
cd "$(dirname "$0")/../.."
A=${1:?bin a}
B=${2:?bin b}
REPS=${3:-3}
CPU=${P66_CPU:-2}
for bin in "$A" "$B"; do
  size=$(size -A "$bin" | awk '$1==".text"{print $2}')
  md5=$(objdump -d --section=.text "$bin" | grep -v 'file format' | md5sum | cut -c1-8)
  hash=$("$bin" nnue-eval engines/canonical/net.cenn < tools/nnue/calib/wl50.fens \
         | sha256sum | cut -c1-16)
  for _ in $(seq 1 "$REPS"); do
    perf stat -x, -e instructions:u -o /var/tmp/perf66/.id \
      taskset -c "$CPU" "$bin" \
      bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn \
      > /var/tmp/perf66/.idb 2>&1
    nodes=$(awk '/Nodes searched *:/{n=$NF} END{print n}' /var/tmp/perf66/.idb)
    ins=$(awk -F, '/instructions/{print $1}' /var/tmp/perf66/.id)
    printf '%-14s .text %s %8s  nodes %s  hash %s  instructions %s\n' \
      "$(basename "$bin")" "$md5" "$size" "$nodes" "$hash" "$ins"
  done
done
