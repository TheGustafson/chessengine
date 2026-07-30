#!/usr/bin/env bash
# perf study 26 — build + IDENTITY probe.  Descends from probe25.sh with ONE
# deliberate change: **the SHIPPED release profile, CGU=1 + fat LTO.**
#
#   bash tools/perf26/probe26.sh <label> [extra-rustflags]
#
# WHY CGU=1.  probe20/21/22/24/25 all set
# `CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4` for
# build speed.  At 4 CGUs `instructions:u` is ALSO unreliable: study 23 saw an
# arm read +48 i/n while its loop body shrank 189 -> 102 static instructions,
# with untouched code blamed for +382.  Cross-CGU inlining decisions move under
# an unrelated edit.  At CGU=1 the inert-source-change null is ~275
# instructions in 30.0 G.  The build is slower; the number is real.
#
# GATE: instructions:u AND branches:u per node, which must agree in sign.
# Node count and the wl50 eval hash must be unchanged on every arm.
#
# FRAME (quote it with every number):
#   bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
#   = 50 FENs x 100k nodes, one thread, 16 MB TT, D48X net, arch sota,arch-d48x
set -euo pipefail
LABEL=${1:?label}
EXTRA=${2:-}
OUT=docs/research/perf-artifacts-26/probes.txt
CPU=${PERF_CPU:-2}
mkdir -p docs/research/perf-artifacts-26 bin26

for _ in $(seq 40); do
  avail=$(free -m | awk 'NR==2{print $7}')
  [ "$avail" -ge 3500 ] && break
  echo "[probe26] only ${avail} MB available, waiting 120s"
  sleep 120
done

systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env RUSTFLAGS="-C target-cpu=native $EXTRA" CARGO_BUILD_JOBS=2 \
  nice -n 10 cargo build --release --features sota,arch-d48x --target-dir target-fast26 -q
BIN=target-fast26/release/chessengine

tmp=$(mktemp)
nice -n 19 perf stat -x, -e instructions:u,branches:u -o "$tmp.perf" -- \
  taskset -c "$CPU" "$BIN" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
  engines/canonical/net.cenn > /dev/null 2> "$tmp.eng"
ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
brs=$(awk -F, '$3=="branches:u"{print $1}' "$tmp.perf")
nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
hash=$("$BIN" nnue-eval engines/canonical/net.cenn < tools/nnue/calib/wl50.fens | sha256sum | cut -c1-16)
rm -f "$tmp" "$tmp.perf" "$tmp.eng"

cp "$BIN" "bin26/$LABEL"

python3 -c "print(f'cgu1={\"$LABEL\":28s} instr={$ins:>14,d} nodes={$nodes:>10,d} per_node={$ins/$nodes:8.2f} brs_per_node={$brs/$nodes:7.2f} hash=$hash')" \
  | tee -a "$OUT"
