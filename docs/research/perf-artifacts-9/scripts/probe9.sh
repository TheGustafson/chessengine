#!/usr/bin/env bash
# perf study 9 — stub-and-diff pricing (study 8 §4.8 rule 1: PRICE FIRST,
# DESIGN SECOND). Builds the CURRENT working tree as a native non-PGO binary
# and measures instructions:u on a fixed bench invocation.
#
#   bash probe9.sh <label>
#
# The number wanted is the DIFFERENCE between two builds made identically, so
# non-PGO is fine and much faster than a PGO pair. A stub that changes eval
# also changes the tree, so per-node figures carry a tree-shape contamination —
# always quote as an upper-ish bound and pair with a static instruction count.
set -euo pipefail
LABEL=${1:?label}
OUT=docs/research/perf-artifacts-9/probes.txt
free -g | awk 'NR==2 {if ($7 < 3) {print "REFUSING: only " $7 " GB available"; exit 1}}'

RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=4 \
  cargo build --release --features sota,arch-d48x --target-dir target-probe9 -q
BIN=target-probe9/release/chessengine

tmp=$(mktemp)
nice -n 19 perf stat -x, -e instructions:u,cycles:u,branches:u -o "$tmp.perf" -- \
  taskset -c 4 "$BIN" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
  engines/canonical/net.cenn > /dev/null 2> "$tmp.eng"
ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
cyc=$(awk -F, '$3=="cycles:u"{print $1}' "$tmp.perf")
nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
hash=$("$BIN" nnue-eval engines/canonical/net.cenn < tools/nnue/calib/wl50.fens | sha256sum | cut -c1-16)
rm -f "$tmp" "$tmp.perf" "$tmp.eng"

line=$(python3 -c "print(f'probe={\"$LABEL\":28s} instr={$ins:>14,d} nodes={$nodes:>10,d} per_node={$ins/$nodes:8.1f} cyc_per_node={$cyc/$nodes:8.1f} hash=$hash')")
echo "$line" | tee -a "$OUT"
