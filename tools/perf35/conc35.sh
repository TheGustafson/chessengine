#!/usr/bin/env bash
# perf study 35 -- CONCURRENT paired cycles: the two arms run AT THE SAME TIME
# on two physical cores, then swap cores and run again.
#
#   bash tools/perf35/conc35.sh <rounds> <control> <arm>...
#
# WHY THIS AND NOT TEMPORAL PAIRING. `panel35.sh` (sequential, rotated) put two
# BYTE-IDENTICAL binaries 1.6% apart at n=14, and `panelsmt35.sh` (sibling
# pinned, nice 0) did not fix it either. The reason is that `cycles:u` on this
# box moves with things that change on a few-second timescale and that a
# sequential design therefore CANNOT hold constant: core boost frequency (a
# fixed-latency DRAM access costs more CYCLES at a higher clock) and the D49
# trainer's memory-bandwidth phase.
#
# Running the two binaries simultaneously on two cores of the same CCX holds
# both constant BY CONSTRUCTION -- same package power state, same DRAM
# contention, same trainer phase, same second. What it introduces instead is a
# core-asymmetry confound, so every round runs the assignment BOTH WAYS and the
# two deltas are averaged; anything fixed about "core A vs core B" cancels.
#
# The SMT siblings of both cores are left idle, so the pair does not contend
# with itself.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}; shift
CTRL=${1:?control}; shift
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CA=${P35_CA:-2}
CB=${P35_CB:-4}
BUDGET=${P35_BUDGET:-100000}
EV=instructions:u,cycles:u,L1-dcache-loads:u
OUT=${P35_CONC:-docs/research/perf-artifacts-35/conc.txt}
mkdir -p "$(dirname "$OUT")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

one() { # <label> <cpu> <tag>
  perf stat -x, -e "$EV" -o "$tmp/$3.stat" -- \
    taskset -c "$2" "./bin35/$1" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
    >/dev/null 2>"$tmp/$3.bench"
}

report() { # <label> <tag> <round> <cpu>
  nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/$2.bench" | tail -1)
  awk -F, -v l="$1" -v r="$3" -v c="$4" -v n="$nodes" '
    $3~/^instructions/{i=$1} $3~/^cycles/{cy=$1} $3~/^L1-dcache-loads/{ld=$1}
    END{printf "r%-3d cpu%-2d %-12s nodes=%-9d i/n=%8.2f cyc/n=%8.2f loads/n=%9.3f\n",
        r, c, l, n, i/n, cy/n, ld/n}
  ' "$tmp/$2.stat" | tee -a "$OUT"
}

for r in $(seq "$ROUNDS"); do
  for arm in "$@"; do
    # pass 1: control on CA, arm on CB -- launched together
    one "$CTRL" "$CA" c & p1=$!
    one "$arm"  "$CB" a & p2=$!
    wait $p1 $p2
    report "$CTRL" c "$r" "$CA"
    report "$arm"  a "$r" "$CB"
    # pass 2: swapped
    one "$CTRL" "$CB" c & p1=$!
    one "$arm"  "$CA" a & p2=$!
    wait $p1 $p2
    report "$CTRL" c "$r" "$CB"
    report "$arm"  a "$r" "$CA"
  done
done
