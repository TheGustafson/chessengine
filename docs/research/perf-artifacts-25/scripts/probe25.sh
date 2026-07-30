#!/usr/bin/env bash
# perf study 25 — build + IDENTITY probe.  Descends from probe21.sh.
#
#   bash docs/research/perf-artifacts-25/scripts/probe25.sh <label> [extra-rustflags]
#
# FRAME (quote it with every number):
#   bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
#   = 50 FENs x 100k nodes, one thread, 16 MB TT, D48X net, arch sota,arch-d48x
#   = a fixed 5,001,562-node tree.
#
# What this gates is IDENTITY: node count AND the wl50 eval hash must be
# unchanged on every arm, including rejects.  instructions:u/node is exact
# (0.0 noise) but is NOT the accept gate in this study — cycles is.
#
# BUILD DISCIPLINE: a GPU trainer + three sibling studies are live on a 15 GB
# box.  Every build runs inside its own memory-capped cgroup AND through
# tools/oomlast.sh so the kernel picks THIS build.  thin LTO + 4 CGUs, -j2.
# Waits for headroom rather than racing the trainer into an OOM.
set -euo pipefail
LABEL=${1:?label}
EXTRA=${2:-}
OUT=docs/research/perf-artifacts-25/probes.txt
CPU=${PERF_CPU:-2}

for _ in $(seq 40); do
  avail=$(free -m | awk 'NR==2{print $7}')
  [ "$avail" -ge 3500 ] && break
  echo "[probe25] only ${avail} MB available, waiting 120s"
  sleep 120
done

systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env RUSTFLAGS="-C target-cpu=native $EXTRA" CARGO_BUILD_JOBS=2 \
  CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4 \
  nice -n 10 cargo build --release --features sota,arch-d48x --target-dir target-fast25 -q
BIN=target-fast25/release/chessengine

tmp=$(mktemp)
nice -n 19 perf stat -x, -e instructions:u,branches:u -o "$tmp.perf" -- \
  taskset -c "$CPU" "$BIN" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
  engines/canonical/net.cenn > /dev/null 2> "$tmp.eng"
ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
brs=$(awk -F, '$3=="branches:u"{print $1}' "$tmp.perf")
nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
hash=$("$BIN" nnue-eval engines/canonical/net.cenn < tools/nnue/calib/wl50.fens | sha256sum | cut -c1-16)
rm -f "$tmp" "$tmp.perf" "$tmp.eng"

mkdir -p bin25
cp "$BIN" "bin25/$LABEL"

python3 -c "print(f'fast={\"$LABEL\":30s} instr={$ins:>14,d} nodes={$nodes:>10,d} per_node={$ins/$nodes:8.1f} brs_per_node={$brs/$nodes:7.1f} hash=$hash')" \
  | tee -a "$OUT"
