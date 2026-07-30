#!/usr/bin/env bash
# perf study 29 — build one labelled binary into bin29/.
#
#   bash tools/perf29/build29.sh <label> <features> [target-cpu]
#
# Descends from tools/perf26/probe26.sh: the SHIPPED release profile (fat LTO,
# CGU = 1). probe20..25 overrode to thin/4, where `instructions:u` is itself
# unreliable (study 23). Default `target-cpu=native` (AVX-512 VBMI2 on this
# box); pass `x86-64-v3` to build the AVX2 backend the merge gate exercises.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
FEATURES=${2:?features}
CPUARCH=${3:-native}

for _ in $(seq 60); do
  avail=$(free -m | awk 'NR==2{print $7}')
  [ "$avail" -ge 3500 ] && break
  echo "[build29] only ${avail} MB available, waiting 120s"
  sleep 120
done

mkdir -p bin29
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env RUSTFLAGS="-C target-cpu=$CPUARCH" CARGO_BUILD_JOBS=2 \
  nice -n 10 cargo build --release --features "$FEATURES" \
  --target-dir "target29-$LABEL" -q
cp "target29-$LABEL/release/chessengine" "bin29/$LABEL"
echo "[build29] bin29/$LABEL  features=$FEATURES  target-cpu=$CPUARCH"
