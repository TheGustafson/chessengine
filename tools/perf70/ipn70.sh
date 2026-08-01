#!/usr/bin/env bash
# tools/perf70/ipn70.sh — instructions PER NODE for two binaries whose node
# counts differ.
#
# A functional search change moves nodes, so wall clock and total instructions
# both move for two reasons at once. `instructions:u` is the only counter that
# survives a byte-identical null on this box (CLAUDE.md), and dividing it by
# the run's own node count separates "the tree got smaller" from "each node got
# dearer" — which is the only question a move-ordering change owes the speed
# side. Rounds alternate A/B so drift cancels.
set -euo pipefail
cd "$(dirname "$0")/../.."

A=${1:?usage: ipn70.sh <binA> <binB> [rounds] [depth] [fens]}
B=${2:?}
R=${3:-5}
D=${4:-11}
F=${5:-tools/perf70/uho300.fens}
NET=${NET:-engines/canonical/net.cenn}

one() {
    local out ins nod
    out=$(perf stat -e instructions:u -x, "$1" bench 16 1 "$D" "$F" depth "$NET" 2>&1 > /dev/null)
    ins=$(printf '%s\n' "$out" | grep -m1 instructions | cut -d, -f1)
    nod=$(printf '%s\n' "$out" | sed -n 's/^Nodes searched  : //p')
    python3 -c "import sys; print(f'{int(sys.argv[1])/int(sys.argv[2]):.2f} {sys.argv[2]}')" "$ins" "$nod"
}

for r in $(seq 1 "$R"); do
    printf 'round %s  A %s   B %s\n' "$r" "$(one "$A")" "$(one "$B")"
done
