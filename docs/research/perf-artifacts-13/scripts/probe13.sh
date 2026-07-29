#!/usr/bin/env bash
# perf study 13 — pinned-profile probe (inherited from probe12.sh).
#
#   bash probe13.sh <label>
#
# Builds the CURRENT working tree as a native non-PGO binary in the PINNED
# release profile (lto = "fat", codegen-units = 1) and measures instructions:u
# on a fixed bench invocation.  Identical source + identical flags => identical
# binary, so the DIFFERENCE between two labels is exact, not statistical.
set -euo pipefail
LABEL=${1:?label}
OUT=docs/research/perf-artifacts-13/probes.txt
free -g | awk 'NR==2 {if ($7 < 6) {print "REFUSING: only " $7 " GB available (need 6)"; exit 1}}'

RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=2 \
  nice -n 10 cargo build --release --features sota,arch-d48x --target-dir target-probe13 -q
BIN=target-probe13/release/chessengine

tmp=$(mktemp)
nice -n 19 perf stat -x, -e instructions:u,cycles:u,branches:u -o "$tmp.perf" -- \
  taskset -c 6 "$BIN" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
  engines/canonical/net.cenn > /dev/null 2> "$tmp.eng"
ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
cyc=$(awk -F, '$3=="cycles:u"{print $1}' "$tmp.perf")
brs=$(awk -F, '$3=="branches:u"{print $1}' "$tmp.perf")
nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
hash=$("$BIN" nnue-eval engines/canonical/net.cenn < tools/nnue/calib/wl50.fens | sha256sum | cut -c1-16)
rm -f "$tmp" "$tmp.perf" "$tmp.eng"

line=$(python3 -c "print(f'probe={\"$LABEL\":30s} instr={$ins:>14,d} nodes={$nodes:>10,d} per_node={$ins/$nodes:8.1f} brs_per_node={$brs/$nodes:7.1f} hash=$hash')")
echo "$line" | tee -a "$OUT"
