#!/usr/bin/env bash
# perf study 20 — screening probe (instrument inherited verbatim from probe18.sh).
#
#   bash docs/research/perf-artifacts-20/scripts/probe20.sh <label>
#
# FRAME (quote it with every number):
#   bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
#   = 50 FENs x 100k nodes, one thread, 16 MB TT, D48X net, arch sota,arch-d48x
#   = a fixed 5,001,562-node tree.  instructions:u / nodes is the accept gate
#   (measured noise floor 0.0 on this box).  The wl50 eval hash and the node
#   count are printed on every line so a tree change or an eval change cannot
#   hide inside a speed claim.
#
# BUILD DISCIPLINE: a GPU trainer is live and the box has 15 GB.  Every build
# runs inside its own memory-capped cgroup AND through tools/oomlast.sh so the
# kernel picks THIS build rather than the trainer.  thin LTO + 4 CGUs, -j2.
set -euo pipefail
LABEL=${1:?label}
OUT=docs/research/perf-artifacts-20/probes-fast.txt
free -m | awk 'NR==2 {if ($7 < 2500) {print "REFUSING: only " $7 " MB available (need 2500)"; exit 1}}'

systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=2 \
  CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4 \
  nice -n 10 cargo build --release --features sota,arch-d48x --target-dir target-fast20 -q
BIN=target-fast20/release/chessengine

tmp=$(mktemp)
nice -n 19 perf stat -x, -e instructions:u,branches:u -o "$tmp.perf" -- \
  taskset -c 6 "$BIN" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
  engines/canonical/net.cenn > /dev/null 2> "$tmp.eng"
ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
brs=$(awk -F, '$3=="branches:u"{print $1}' "$tmp.perf")
nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
hash=$("$BIN" nnue-eval engines/canonical/net.cenn < tools/nnue/calib/wl50.fens | sha256sum | cut -c1-16)
rm -f "$tmp" "$tmp.perf" "$tmp.eng"

line=$(python3 -c "print(f'fast={\"$LABEL\":34s} instr={$ins:>14,d} nodes={$nodes:>10,d} per_node={$ins/$nodes:8.1f} brs_per_node={$brs/$nodes:7.1f} hash=$hash')")
echo "$line" | tee -a "$OUT"
