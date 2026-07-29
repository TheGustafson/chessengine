#!/usr/bin/env bash
# perf study 17 — run the probe measurement on an ARBITRARY binary (no build).
#
#   bash probeany17.sh <label> <binary> <bench-args...>
#
# Used to check whether a finding from the callgrind (AVX2) frame survives into
# the shipped (AVX-512) frame: same fixed tree, same counters, two binaries.
set -euo pipefail
LABEL=${1:?label}
BIN=${2:?binary}
shift 2
tmp=$(mktemp)
nice -n 19 perf stat -x, -e instructions:u,branches:u,branch-misses:u -o "$tmp.perf" -- \
  taskset -c 6 "$BIN" "$@" > /dev/null 2> "$tmp.eng"
ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
brs=$(awk -F, '$3=="branches:u"{print $1}' "$tmp.perf")
bmi=$(awk -F, '$3=="branch-misses:u"{print $1}' "$tmp.perf")
nodes=$(grep -oP 'Nodes searched\s+:\s*\K[0-9]+' "$tmp.eng" | tail -1)
rm -f "$tmp" "$tmp.perf" "$tmp.eng"
python3 -c "print(f'any={\"$LABEL\":26s} instr={$ins:>14,d} nodes={$nodes:>10,d} i/n={$ins/$nodes:8.1f} br/n={$brs/$nodes:7.1f} bmiss/n={$bmi/$nodes:6.2f}')"
