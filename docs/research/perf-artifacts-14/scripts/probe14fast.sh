#!/usr/bin/env bash
# perf study 14 — FAST screening probe (inherited from probe13fast.sh; only the
# output/target paths changed).
#
#   bash probe14fast.sh <label>
#
# Fixed 5,001,562-node tree, instructions:u, wl50 eval hash + node count printed
# so a tree/eval change cannot hide inside a speed claim.  Built with thin LTO +
# 4 codegen units: ~20x faster to build and a fraction of the peak RSS.
#
# The absolute i/n in this frame is NOT comparable to a pinned-profile build's —
# only fast-to-fast deltas are.  And study 12 §0.3: NO probe delta may be quoted
# as a result; the probe overstates PGO deltas by ~1.3-2x.  Screening only.
set -euo pipefail
LABEL=${1:?label}
OUT=docs/research/perf-artifacts-14/probes-fast.txt
# Guard in MEGABYTES, not `free -g` (which truncates).
free -m | awk 'NR==2 {if ($7 < 4500) {print "REFUSING: only " $7 " MB available (need 4500)"; exit 1}}'

RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=2 \
  CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4 \
  nice -n 10 cargo build --release --features sota,arch-d48x --target-dir target-fast14 -q
BIN=target-fast14/release/chessengine

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
