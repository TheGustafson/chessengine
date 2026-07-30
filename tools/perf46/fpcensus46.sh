#!/usr/bin/env bash
# perf study 46 -- re-derive study 44's FP op levels on THIS study's own binary.
#
# Study 46 §5 corrects study 44's port-pressure verdict by replacing its assumed
# divisor ("Zen 4's 4 FP pipes") with a measured one. That correction is applied
# to study 44's LEVELS (1399.34 fp uops/node, 1282.60 of them 512-bit), and rule
# 8 says re-derive any number you build on -- especially one whose `n` you did
# not set. This does that.
#
#   bash tools/perf46/fpcensus46.sh [label] [reps] [cpu]
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:-base46}
REPS=${2:-5}
CPU=${3:-0}
NET=${P46_NET:-engines/canonical/net.cenn}
FENS=${P46_FENS:-tools/nnue/calib/wl50.fens}
BIN=${P46_BIN:-bin46}
EV='fp_ops_retired_by_width.all:u,fp_ops_retired_by_width.pack_512_uops_retired:u,fp_ops_retired_by_width.pack_256_uops_retired:u,cycles:u,instructions:u'
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

for r in $(seq "$REPS"); do
  perf stat -x, -e "$EV" -o "$tmp/s" -- taskset -c "$CPU" "./$BIN/$LABEL" \
    bench 16 1 100000 "$FENS" nodes "$NET" >/dev/null 2>"$tmp/b"
  n=$(sed -n 's/^Nodes searched  : //p' "$tmp/b" | tail -1)
  awk -F, -v n="$n" -v r="$r" '$1 ~ /^[0-9]+$/ && $3 != "" { printf "r%s %s %.2f\n", r, $3, $1/n }' "$tmp/s"
done | sort -k2,2 -k3,3n | awk '
  { v[$2] = v[$2] " " $3; n[$2]++ }
  END {
    printf "%-52s %10s\n", "event (per node, MEDIAN of reps)", "value"
    for (e in v) {
      c = split(v[e], a, " "); asort_done = 0
      # values already sorted ascending by the sort above
      m = (c % 2) ? a[int(c/2)+1] : (a[c/2] + a[c/2+1]) / 2
      printf "%-52s %10.2f\n", e, m
    }
  }'
