#!/usr/bin/env bash
# perf study 42 -- NODE IDENTITY across the study's arms.
#
#   bash tools/perf42/ident42.sh <labelA> <labelB> ...
#
# Every arm in this study is a CACHE HINT change: it moves when a line arrives,
# never which entry a probe finds. So all of them must reproduce the reference's
# node totals bit-for-bit, at the study frame AND at the depth-13 NNUE gate
# frame. An arm that fails this is not measuring what its name says, and no
# number from it is quotable -- print it and stop.
set -euo pipefail
cd "$(dirname "$0")/../.."
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P42_CPU:-2}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

run() { # <bin> <mode-args...>
  taskset -c "$CPU" "./$1" bench 16 1 "$2" "$FENS" "$3" "$NET" >/dev/null 2>"$tmp/o"
  sed -n 's/^Nodes searched  : //p' "$tmp/o" | tail -1
}

ref=""
for label in "$@"; do
  bin="bin42/$label"
  [ -x "$bin" ] || bin="bin34/$label"
  n=$(run "$bin" 100000 nodes)
  d=$(run "$bin" 13 depth)
  printf '%-10s nodes-frame=%-12s depth13=%-12s\n' "$label" "$n" "$d"
  if [ -z "$ref" ]; then ref="$n/$d"; else
    [ "$n/$d" = "$ref" ] || { echo "IDENTITY BROKEN: $label != reference ($ref)"; exit 1; }
  fi
done
echo "NODE-IDENTICAL across all arms ($ref)"
