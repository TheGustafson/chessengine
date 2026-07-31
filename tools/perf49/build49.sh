#!/usr/bin/env bash
# perf study 49 — build one labelled arm binary into bin49/.
#   bash tools/perf49/build49.sh <label> [extra,features]
# Same shape as study 48's build48.sh: one script builds every arm so the pair
# differs only in the feature list (study 40 §6).
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
EXTRA=${2:-}
BASE=sota,arch-d48x
FEATURES=$BASE${EXTRA:+,$EXTRA}
LOCK=/home/nick/.cache/archon-build.lock

avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
if [ "$avail" -lt 2500 ]; then
  echo "[build49] REFUSING: only ${avail} MB available (< 2500)" >&2
  exit 1
fi
echo "[build49] $LABEL  features=$FEATURES  avail=${avail}MB  (waiting for lock)"

mkdir -p bin49
flock "$LOCK" env RUSTFLAGS="-C target-cpu=native" \
  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
  --working-directory="$PWD" \
  cargo build --release --features "$FEATURES" -j2
cp target/release/chessengine "bin49/$LABEL"
# .text md5 with the objdump filename banner stripped (the commission's trap).
md5=$(objdump -d --section=.text "bin49/$LABEL" | grep -v 'file format' | md5sum | cut -c1-8)
echo "[build49] -> bin49/$LABEL  .text md5 $md5"
