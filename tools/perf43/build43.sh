#!/usr/bin/env bash
# perf study 43 — build one labelled binary into bin34/, SERIALISED.
#
#   bash tools/perf43/build43.sh <label> [features] [target-cpu]
#
# Descends from tools/perf34/build34.sh with two changes forced by the box on
# 2026-07-30, when swap hit 4035/4095 MB with four agents live:
#
#  * every build takes /home/nick/.cache/archon-build.lock via `flock`, so
#    concurrent agents QUEUE instead of putting 2 x 6 GB of MemoryMax against
#    9 GB of RAM with no swap headroom (the configuration that OOM-killed this
#    box before, taking the D49 trainer with it);
#  * ONE shared target dir. The crate has zero runtime dependencies, so a
#    feature change rebuilds the crate either way and per-label target dirs buy
#    nothing but disk -- and 36 GB of exactly those had just been reclaimed.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
FEATURES=${2:-sota,arch-d48x}
CPUARCH=${3:-native}
LOCK=/home/nick/.cache/archon-build.lock

for _ in $(seq 60); do
  avail=$(free -m | awk 'NR==2{print $7}')
  [ "$avail" -ge 2500 ] && break
  echo "[build43] only ${avail} MB available, waiting 120s"
  sleep 120
done

mkdir -p bin34
# No -w timeout: waiting IS the point (coordinator ruling 2026-07-30).
flock "$LOCK" \
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env RUSTFLAGS="-C target-cpu=$CPUARCH" CARGO_BUILD_JOBS=2 \
  nice -n 10 cargo build --release --features "$FEATURES" \
  --target-dir target43 -q
cp target43/release/chessengine "bin34/$LABEL"
echo "[build43] bin34/$LABEL  features=$FEATURES  target-cpu=$CPUARCH"
