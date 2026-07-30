#!/usr/bin/env bash
# perf study 29 — the traffic panel: loads, misses, fill depth and LOAD-STALL
# cycles, interleaved round-robin so drift cannot land on one arm.
#
#   bash tools/perf29/panel29.sh <rounds> <label>...
#
# `ex_no_retire.load_not_complete` is the Zen 4 event study 28 used to
# attribute 82.8% of the per-node deficit: cycles where nothing retired
# because a load had not returned. It is the currency this study prices in;
# `cycles:u` is quoted alongside and is NOT trusted on its own (nulls on
# byte-identical binaries have run to several percent on this box).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}; shift
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P29_CPU:-2}
BUDGET=${P29_BUDGET:-100000}
EV=instructions:u,cycles:u,L1-dcache-loads:u,L1-dcache-load-misses:u,ex_no_retire.load_not_complete:u
OUT=${P29_PANEL:-docs/research/perf-artifacts-29/panel.txt}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p docs/research/perf-artifacts-29

for r in $(seq "$ROUNDS"); do
  for label in "$@"; do
    nice -n 19 perf stat -x, -e "$EV" -o "$tmp/stat" -- \
      taskset -c "$CPU" "./bin29/$label" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
      >/dev/null 2>"$tmp/bench"
    nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
    awk -F, -v l="$label" -v r="$r" -v n="$nodes" '
      $3~/^instructions/{i=$1} $3~/^cycles/{c=$1}
      $3~/^L1-dcache-loads/{ld=$1} $3~/^L1-dcache-load-misses/{ms=$1}
      $3~/^ex_no_retire.load_not_complete/{st=$1}
      END{printf "r%-2d %-8s nodes=%-9d i/n=%8.2f cyc/n=%8.2f loads/n=%9.3f miss/n=%8.3f rate=%.3f%% stall/n=%8.2f\n",
          r, l, n, i/n, c/n, ld/n, ms/n, 100*ms/ld, st/n}
    ' "$tmp/stat" | tee -a "$OUT"
  done
  # rotate the order so position cannot correlate with arm
  set -- "${@:2}" "$1"
done
