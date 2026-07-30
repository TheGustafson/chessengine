#!/usr/bin/env bash
# perf study 22 — identity + instruction probe (instrument inherited from
# probe20.sh, minus the build; core moved 6 -> 3 because a sibling study owns 6).
#
#   bash docs/research/perf-artifacts-22/scripts/probe22.sh <label>
#
# FRAME (quote it with every number):
#   bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
#   = 50 FENs x 100k nodes, one thread, 16 MB TT, D48X net, arch sota,arch-d48x
#   = a fixed 5,001,562-node tree.  instructions:u/node has a measured noise
#   floor of 0.0 on this box.  Node count and wl50 eval hash print on every
#   line so a tree change or an eval change cannot hide inside a speed claim.
set -euo pipefail
LABEL=${1:?label}
OUT=docs/research/perf-artifacts-22/probes-fast.txt
BIN=target-fast22/release/chessengine

tmp=$(mktemp)
nice -n 19 perf stat -x, -e instructions:u,branches:u,branch-misses:u -o "$tmp.perf" -- \
  taskset -c 2 "$BIN" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
  engines/canonical/net.cenn > /dev/null 2> "$tmp.eng"
ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
brs=$(awk -F, '$3=="branches:u"{print $1}' "$tmp.perf")
bms=$(awk -F, '$3=="branch-misses:u"{print $1}' "$tmp.perf")
nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
hash=$("$BIN" nnue-eval engines/canonical/net.cenn < tools/nnue/calib/wl50.fens | sha256sum | cut -c1-16)
rm -f "$tmp" "$tmp.perf" "$tmp.eng"

line=$(python3 -c "print(f'fast={\"$LABEL\":34s} instr={$ins:>14,d} nodes={$nodes:>10,d} per_node={$ins/$nodes:8.1f} brs_per_node={$brs/$nodes:7.1f} bmiss_per_node={$bms/$nodes:6.2f} hash=$hash')")
echo "$line" | tee -a "$OUT"
