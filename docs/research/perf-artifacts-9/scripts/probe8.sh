#!/usr/bin/env bash
# perf study 8 — price a candidate by building a deliberately behaviour-changing
# STUB and diffing instructions:u against a control built the same way.
#
#   bash probe8.sh <core> <binary>...
#
# Non-PGO native builds are fine here: the number wanted is the DIFFERENCE
# between two builds made identically, not an absolute. A stub that changes
# search behaviour also changes the tree, so the per-node figure carries a
# tree-shape contamination — always quote it as an upper-ish bound and pair it
# with a static instruction count.
set -euo pipefail
CORE=${1:?core}
shift
for b in "$@"; do
  tmp=$(mktemp)
  nice -n 19 perf stat -x, -e instructions:u -o "$tmp.perf" -- \
    taskset -c "$CORE" "$b" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
    engines/canonical/net.cenn > /dev/null 2> "$tmp.eng"
  ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
  nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
  python3 -c "print(f'{'"'"$b"'"':44s} instr={$ins:>14,d} nodes={$nodes:>10,d} per_node={$ins/$nodes:8.1f}')"
  rm -f "$tmp" "$tmp.perf" "$tmp.eng"
done
