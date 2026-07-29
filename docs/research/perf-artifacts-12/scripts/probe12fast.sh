#!/usr/bin/env bash
# perf study 12 — FAST screening probe.
#
#   bash probe12fast.sh <label>
#
# Same instrument as probe12.sh (fixed 5,001,562-node tree, instructions:u,
# wl50 eval hash + node count printed so a tree/eval change cannot hide) but
# built with thin LTO + 4 codegen units instead of the pinned
# `lto = "fat", codegen-units = 1`.  That is ~20x faster to build and a
# fraction of the peak RSS, which matters on a 15 GB box that has already lost
# one job to the global OOM killer.
#
# The absolute i/n in this frame is NOT comparable to probe12.sh's — only
# fast-to-fast deltas are.  Every arm that survives screening here is
# re-measured in the pinned profile before it is quoted.
set -euo pipefail
LABEL=${1:?label}
OUT=docs/research/perf-artifacts-12/probes-fast.txt
free -g | awk 'NR==2 {if ($7 < 6) {print "REFUSING: only " $7 " GB available (need 6)"; exit 1}}'

RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=2 \
  CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4 \
  nice -n 10 cargo build --release --features sota,arch-d48x --target-dir target-fast12 -q
BIN=target-fast12/release/chessengine

tmp=$(mktemp)
nice -n 19 perf stat -x, -e instructions:u,branches:u -o "$tmp.perf" -- \
  taskset -c 6 "$BIN" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
  engines/canonical/net.cenn > /dev/null 2> "$tmp.eng"
ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
brs=$(awk -F, '$3=="branches:u"{print $1}' "$tmp.perf")
nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
hash=$("$BIN" nnue-eval engines/canonical/net.cenn < tools/nnue/calib/wl50.fens | sha256sum | cut -c1-16)
rm -f "$tmp" "$tmp.perf" "$tmp.eng"

line=$(python3 -c "print(f'fast={\"$LABEL\":32s} instr={$ins:>14,d} nodes={$nodes:>10,d} per_node={$ins/$nodes:8.1f} brs_per_node={$brs/$nodes:7.1f} hash=$hash')")
echo "$line" | tee -a "$OUT"
