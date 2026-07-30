#!/usr/bin/env bash
# perf study 34 — build one labelled binary into bin34/.
#
#   bash tools/perf34/build34.sh <label> [features] [target-cpu]
#
# Descends from tools/perf29/build29.sh: SHIPPED release profile (fat LTO,
# CGU = 1) — study 23 §2.2 showed `instructions:u` is a lottery at CGU=4.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
FEATURES=${2:-sota,arch-d48x}
CPUARCH=${3:-native}

for _ in $(seq 60); do
  avail=$(free -m | awk 'NR==2{print $7}')
  [ "$avail" -ge 3000 ] && break
  echo "[build34] only ${avail} MB available, waiting 120s"
  sleep 120
done

mkdir -p bin34
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env RUSTFLAGS="-C target-cpu=$CPUARCH" CARGO_BUILD_JOBS=2 \
  nice -n 10 cargo build --release --features "$FEATURES" \
  --target-dir "target34-$LABEL" -q
cp "target34-$LABEL/release/chessengine" "bin34/$LABEL"
echo "[build34] bin34/$LABEL  features=$FEATURES  target-cpu=$CPUARCH"
