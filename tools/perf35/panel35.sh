#!/usr/bin/env bash
# perf study 35 -- the cycles panel, PAIRED WITHIN ROUND.
#
#   bash tools/perf35/panel35.sh <rounds> <label>...
#
# Every label is measured once per round, back to back, and the round order is
# rotated.  `med35.py` then reduces PER-ROUND paired deltas against the control
# rather than differencing two independent medians -- study 29 nearly shipped a
# -1.67% cycles "win" that was position-in-round, and rotation alone did not
# catch it (its byte-identical null moved +1.5% while the arms moved +1.1%).
#
# The currency is `cycles:u` (perf studies 20 and 32: instructions have twice
# been the WRONG currency on this engine).  Instructions, loads, misses and
# load-stall cycles ride along.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}; shift
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P35_CPU:-2}
BUDGET=${P35_BUDGET:-100000}
EV=instructions:u,cycles:u,L1-dcache-loads:u,L1-dcache-load-misses:u,ex_no_retire.load_not_complete:u
OUT=${P35_PANEL:-docs/research/perf-artifacts-35/panel.txt}
mkdir -p "$(dirname "$OUT")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

for r in $(seq "$ROUNDS"); do
  for label in "$@"; do
    nice -n 19 perf stat -x, -e "$EV" -o "$tmp/stat" -- \
      taskset -c "$CPU" "./bin35/$label" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
      >/dev/null 2>"$tmp/bench"
    nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
    awk -F, -v l="$label" -v r="$r" -v n="$nodes" '
      $3~/^instructions/{i=$1} $3~/^cycles/{c=$1}
      $3~/^L1-dcache-loads/{ld=$1} $3~/^L1-dcache-load-misses/{ms=$1}
      $3~/^ex_no_retire.load_not_complete/{st=$1}
      END{printf "r%-3d %-12s nodes=%-9d i/n=%8.2f cyc/n=%8.2f loads/n=%9.3f miss/n=%8.3f rate=%.3f%% stall/n=%8.2f\n",
          r, l, n, i/n, c/n, ld/n, ms/n, 100*ms/ld, st/n}
    ' "$tmp/stat" | tee -a "$OUT"
  done
  set -- "${@:2}" "$1"
done
