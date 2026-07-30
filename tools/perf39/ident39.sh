#!/usr/bin/env bash
# perf study 39 — the IDENTITY gate, run before any panel.
#
#   bash tools/perf39/ident39.sh [budget]
#
# Two things have to be true before a single number off the panel means
# anything:
#
#  1. every arm searches the SAME NUMBER OF NODES. The probes compute wrong
#     evaluations on purpose, so a fixed-DEPTH frame would give them different
#     trees and "per node" would silently be comparing different nodes (the
#     coordinator measured a −21% wrong-sign result exactly that way). A fixed
#     NODE budget makes the frame comparable; this checks that it actually did.
#  2. `probe-addcopy` at `P39_DOSE=0` must be indistinguishable from the base
#     binary in instructions and loads — it is the same source with a dead
#     runtime loop, and it is the control the additive arms are differenced
#     against.
set -euo pipefail
cd "$(dirname "$0")/../.."
BUDGET=${1:-100000}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P39_CPU:-2}
EV=instructions:u,L1-dcache-loads:u,L1-dcache-load-misses:u
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

probe() { # label binary env...
  local label=$1 bin=$2; shift 2
  # shellcheck disable=SC2086
  nice -n 19 perf stat -x, -e "$EV" -o "$tmp/stat" -- \
    env "$@" taskset -c "$CPU" "./bin39/$bin" \
    bench 16 1 "$BUDGET" "$FENS" nodes "$NET" >/dev/null 2>"$tmp/bench"
  local nodes
  nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp/bench" | tail -1)
  awk -F, -v l="$label" -v n="$nodes" '
    $3~/^instructions/{i=$1} $3~/^L1-dcache-loads/{ld=$1}
    $3~/^L1-dcache-load-misses/{ms=$1}
    END{printf "%-12s nodes=%-9d i/n=%9.2f loads/n=%9.2f miss/n=%8.2f\n", l, n, i/n, ld/n, ms/n}
  ' "$tmp/stat"
}

probe base      base    P39_UNUSED=1
probe add-dose0 addcopy P39_DOSE=0 P39_RING=16
probe add-1x16  addcopy P39_DOSE=1 P39_RING=16
probe add-1x1   addcopy P39_DOSE=1 P39_RING=1
probe add-2x16  addcopy P39_DOSE=2 P39_RING=16
probe add-4x16  addcopy P39_DOSE=4 P39_RING=16
probe actpin    actpin  P39_UNUSED=1
probe accpin    accpin  P39_UNUSED=1
probe dstpin    dstpin  P39_UNUSED=1
